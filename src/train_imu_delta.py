"""
IMU Delta 模型训练脚本 (train_imu_delta.py)
=============================================
GPU 主攻深度学习：完整的两阶段训练循环

Phase A: Adam + MSE (快速收敛)
Phase B: SGD + MAE  (鲁棒精调，降低尾部误差)

硬件适配：RTX 3070ti (8GB VRAM)，启用 AMP 混合精度

用法:
  python -m src.train_imu_delta --config configs/imu_delta_model.yml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast

# 项目导入
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.imu_delta_model import IMUDeltaNet, DeltaDistanceLoss
from src.imu_delta_dataset import (
    IMUDeltaDataset,
    build_dataset_for_sites,
    build_all_site_caches,
    load_all_cached_legs,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def euclidean_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算欧氏距离误差 (MAE of displacement)"""
    return torch.sqrt(((pred - target) ** 2).sum(dim=-1)).mean()


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """余弦退火 + 线性预热"""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    grad_clip: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_samples = 0

    for imu, aux, target in loader:
        imu = imu.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast('cuda'):
                pred = model(imu, aux)
                loss = loss_fn(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(imu, aux)
            loss = loss_fn(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        with torch.no_grad():
            mae = euclidean_error(pred, target).item()
        total_mae += mae * batch_size
        n_samples += batch_size

    return {
        'loss': total_loss / max(n_samples, 1),
        'mae_m': total_mae / max(n_samples, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    all_errors = []
    n_samples = 0

    for imu, aux, target in loader:
        imu = imu.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp:
            with autocast('cuda'):
                pred = model(imu, aux)
                loss = loss_fn(pred, target)
        else:
            pred = model(imu, aux)
            loss = loss_fn(pred, target)

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size

        errors = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
        total_mae += errors.sum().item()
        all_errors.append(errors.cpu().numpy())
        n_samples += batch_size

    all_errors = np.concatenate(all_errors)

    return {
        'loss': total_loss / max(n_samples, 1),
        'mae_m': total_mae / max(n_samples, 1),
        'median_m': float(np.median(all_errors)),
        'p90_m': float(np.percentile(all_errors, 90)),
    }


def run_training_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    phase_cfg: dict,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    model_save_path: Path,
    phase_name: str,
    patience: int,
) -> Dict[str, float]:
    """
    执行单个训练阶段 (Phase A 或 Phase B)。

    Returns
    -------
    best_metrics : 最佳验证集指标
    """
    print(f"\n{'='*60}")
    print(f"  Phase {phase_name}: {phase_cfg['optimizer'].upper()} + {phase_cfg['loss'].upper()}")
    print(f"  LR={phase_cfg['lr']}, Epochs={phase_cfg['epochs']}")
    print(f"{'='*60}")

    # 构建优化器
    if phase_cfg['optimizer'] == 'adam':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=phase_cfg['lr'],
            weight_decay=phase_cfg.get('weight_decay', 1e-4),
        )
    elif phase_cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=phase_cfg['lr'],
            momentum=phase_cfg.get('momentum', 0.9),
            weight_decay=phase_cfg.get('weight_decay', 1e-5),
        )
    else:
        raise ValueError(f"Unknown optimizer: {phase_cfg['optimizer']}")

    # 损失函数
    loss_name = phase_cfg['loss']
    if loss_name == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_name == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_name == 'huber':
        loss_fn = nn.SmoothL1Loss()
    elif loss_name == 'delta_distance':
        loss_params = phase_cfg.get('loss_params', {})
        loss_fn = DeltaDistanceLoss(
            alpha=loss_params.get('alpha', 1.0),
            beta=loss_params.get('beta', 0.5),
            gamma=loss_params.get('gamma', 0.3),
        )
        print(f"    DeltaDistanceLoss: alpha={loss_fn.alpha}, "
              f"beta={loss_fn.beta}, gamma={loss_fn.gamma}")
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    # 学习率调度器
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=phase_cfg.get('warmup_epochs', 0),
        total_epochs=phase_cfg['epochs'],
    )

    # AMP scaler
    scaler = GradScaler('cuda') if use_amp else None

    best_mae = float('inf')
    best_metrics = {}
    no_improve = 0

    for epoch in range(1, phase_cfg['epochs'] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, grad_clip, use_amp
        )
        val_metrics = validate(model, val_loader, loss_fn, device, use_amp)

        elapsed = time.time() - t0
        scheduler.step()  # 在 epoch 结束后调度 LR

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"  Epoch {epoch:03d}/{phase_cfg['epochs']:03d} "
            f"| Train Loss: {train_metrics['loss']:.4f} MAE: {train_metrics['mae_m']:.3f}m "
            f"| Val Loss: {val_metrics['loss']:.4f} MAE: {val_metrics['mae_m']:.3f}m "
            f"Med: {val_metrics['median_m']:.3f}m P90: {val_metrics['p90_m']:.3f}m "
            f"| LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_metrics['mae_m'] < best_mae:
            best_mae = val_metrics['mae_m']
            best_metrics = val_metrics.copy()
            best_metrics['best_epoch'] = epoch
            no_improve = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_mae': best_mae,
                'phase': phase_name,
                'metrics': val_metrics,
            }, model_save_path)
            print(f"    ✅ 保存最佳模型 → {model_save_path.name}  (MAE={best_mae:.3f}m)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    ⚠️ 早停：连续 {patience} 轮无提升，终止 Phase {phase_name}")
                break

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="IMU Delta Model Training")
    parser.add_argument(
        "--config", type=str, default="configs/imu_delta_model.yml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--sites", type=str, default=None,
        help="指定训练的 site_id 列表 (逗号分隔)，默认全部",
    )
    parser.add_argument(
        "--skip-phase-a", action="store_true",
        help="跳过 Phase A，直接从 Phase B 开始（需要已有 Phase A 的 checkpoint）",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从指定 checkpoint 恢复训练",
    )
    args = parser.parse_args()

    # ── 1. 加载配置 ──
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seed = config['seed']
    set_seed(seed)

    # ── 2. 环境信息 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = config['training'].get('amp', True) and device.type == 'cuda'

    print("=" * 60)
    print("  IMU Delta Model Training")
    print("=" * 60)
    print(f"  Device      : {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU         : {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  AMP         : {use_amp}")
    print(f"  Seed        : {seed}")
    print(f"  Git Commit  : {get_git_commit()}")
    print(f"  Config      : {config_path}")
    print()

    # ── 3. 构建数据集（缓存优先） ──
    data_cfg = config['data']
    train_dir = project_root / config['paths']['train_dir']
    cache_dir = project_root / config['paths'].get('leg_cache_dir', 'models/imu_delta/cache')

    site_ids = None
    if args.sites:
        site_ids = [s.strip() for s in args.sites.split(',')]

    print("📦 构建 IMU Leg 数据集...")
    t_start = time.time()

    # 检查缓存是否存在
    existing_caches = list(cache_dir.glob("*.npz")) if cache_dir.exists() else []

    if not existing_caches:
        print(f"  未发现缓存，开始逐 Site 构建并缓存到 {cache_dir}...")
        build_all_site_caches(
            train_dir=train_dir,
            cache_dir=cache_dir,
            site_ids=site_ids,
            n_time_steps=data_cfg['n_time_steps'],
            verbose=True,
        )

    print(f"  从缓存加载数据...")
    imu_all, aux_all, delta_all = load_all_cached_legs(
        cache_dir=cache_dir,
        site_ids=site_ids,
    )

    n_total = len(imu_all)
    print(f"  耗时: {time.time() - t_start:.1f}s")

    if n_total < 10:
        print(f"❌ 数据量过少 ({n_total} legs)，无法训练")
        sys.exit(1)

    # 划分训练/验证集
    n_train = int(n_total * data_cfg['train_ratio'])
    n_val = n_total - n_train

    indices = np.arange(n_total)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = IMUDeltaDataset.from_arrays(
        imu_all[train_idx], aux_all[train_idx], delta_all[train_idx]
    )
    val_dataset = IMUDeltaDataset.from_arrays(
        imu_all[val_idx], aux_all[val_idx], delta_all[val_idx]
    )

    # 释放原始大数组
    del imu_all, aux_all, delta_all, indices
    gc.collect()

    # 高斯噪声增强（仅训练集）
    noise_std = data_cfg.get('noise_std', 0.0)
    train_dataset.training = True
    train_dataset.noise_std = noise_std

    print(f"\n  训练集 : {len(train_dataset)} legs")
    print(f"  验证集 : {len(val_dataset)} legs")
    print(f"  噪声增强 : std={noise_std}")

    # DataLoader —— 自适应 batch_size，防止小数据时 drop_last 导致空 loader
    effective_bs = min(data_cfg['batch_size'], len(train_dataset))
    print(f"  Batch Size  : {effective_bs} (配置: {data_cfg['batch_size']})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(effective_bs * 2, len(val_dataset)),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False,
    )

    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── 4. 构建模型 ──
    model = IMUDeltaNet.from_config(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️ 模型参数: {n_params:,} ({n_trainable:,} 可训练)")

    # 估算显存占用
    param_mem_mb = n_params * 4 / (1024 ** 2)
    print(f"  模型参数显存: ~{param_mem_mb:.1f} MB (FP32)")

    # 如果恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ 已从 {args.resume} 恢复模型")

    # ── 5. 模型保存目录 ──
    save_dir = project_root / config['paths']['model_save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = project_root / config['paths']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config['training']
    grad_clip = train_cfg.get('gradient_clip', 5.0)
    es_cfg = train_cfg.get('early_stopping', {})
    patience = es_cfg.get('patience', 15)

    n_cycles = train_cfg.get('n_cycles', 1)
    cycle_lr_decay = train_cfg.get('cycle_lr_decay', 0.5)

    all_results = {}
    global_best_mae = float('inf')
    best_model_path = save_dir / "imu_delta_best.pt"

    # ── 6. 循环训练: N 轮 (Phase A → Phase B) ──
    for cycle in range(n_cycles):
        lr_scale = cycle_lr_decay ** cycle

        print(f"\n{'#'*60}")
        print(f"  Cycle {cycle+1}/{n_cycles}  (LR scale = {lr_scale:.3f})")
        print(f"{'#'*60}")

        # Phase A: Adam + MSE
        phase_a_cfg = dict(train_cfg['phase_a'])
        phase_a_cfg['lr'] = phase_a_cfg['lr'] * lr_scale
        # 后续 cycle 减少 epoch 数
        if cycle > 0:
            phase_a_cfg['epochs'] = max(10, int(phase_a_cfg['epochs'] * 0.7))
            phase_a_cfg['warmup_epochs'] = min(phase_a_cfg['warmup_epochs'], 2)

        phase_a_path = save_dir / f"imu_delta_cycle{cycle}_a.pt"
        phase_a_metrics = run_training_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            phase_cfg=phase_a_cfg,
            device=device,
            grad_clip=grad_clip,
            use_amp=use_amp,
            model_save_path=phase_a_path,
            phase_name=f"C{cycle+1}-A",
            patience=patience,
        )
        all_results[f'cycle{cycle}_a'] = phase_a_metrics
        print(f"\n  Cycle {cycle+1} Phase A 最佳: MAE={phase_a_metrics['mae_m']:.3f}m")

        # 加载 Phase A 最佳权重
        ckpt = torch.load(phase_a_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        if phase_a_metrics['mae_m'] < global_best_mae:
            global_best_mae = phase_a_metrics['mae_m']
            torch.save(ckpt, best_model_path)

        # Phase B: SGD + MAE（震荡跳出局部最优）
        phase_b_cfg = dict(train_cfg['phase_b'])
        phase_b_cfg['lr'] = phase_b_cfg['lr'] * lr_scale
        if cycle > 0:
            phase_b_cfg['epochs'] = max(8, int(phase_b_cfg['epochs'] * 0.7))

        phase_b_path = save_dir / f"imu_delta_cycle{cycle}_b.pt"
        phase_b_metrics = run_training_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            phase_cfg=phase_b_cfg,
            device=device,
            grad_clip=grad_clip,
            use_amp=use_amp,
            model_save_path=phase_b_path,
            phase_name=f"C{cycle+1}-B",
            patience=patience,
        )
        all_results[f'cycle{cycle}_b'] = phase_b_metrics
        print(f"\n  Cycle {cycle+1} Phase B 最佳: MAE={phase_b_metrics['mae_m']:.3f}m")

        # 加载 Phase B 最佳权重，作为下一 cycle 起点
        ckpt = torch.load(phase_b_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        if phase_b_metrics['mae_m'] < global_best_mae:
            global_best_mae = phase_b_metrics['mae_m']
            torch.save(ckpt, best_model_path)

    # ── 7. 保存实验记录 ──
    final_mae = global_best_mae
    experiment_log = {
        'seed': seed,
        'git_commit': get_git_commit(),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        'n_params': n_params,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_cycles': n_cycles,
        'noise_std': noise_std,
        'config': config,
        'results': all_results,
        'global_best_mae': global_best_mae,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    log_path = log_dir / f"train_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"  训练完成！")
    print(f"  最佳模型 : {best_model_path}")
    print(f"  实验日志 : {log_path}")
    print(f"  最终 Val MAE : {final_mae:.3f}m")
    print(f"{'='*60}")

    # 清理 GPU 显存
    del model, train_loader, val_loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
