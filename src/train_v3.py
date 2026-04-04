"""
训练主脚本 V3 - Phase 2集成版本 (train_v3.py)
================================================
Phase 2: 使用模型集成（Stacking/Weighted）

改进：
1. 支持XGBoost和CatBoost
2. 支持Stacking集成
3. 支持加权平均集成
4. 支持Optuna超参数优化

作者：Claude
版本：3.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import TRAIN_DIR
from src.features import build_bssid_vocab
from src.multimodal_fusion import (
    MultimodalFeatureFusion,
    FusionConfig,
    process_site_multimodal_train,
    save_fusion,
)
from src.models_v3 import SiteModelV3, save_site_model_v3


# ══════════════════════════════════════════════════════════════════════════════
# 核心训练函数
# ══════════════════════════════════════════════════════════════════════════════

def train_site_v3(
    site_id: str,
    n_bssid: int,
    model_dir: Path,
    train_dir: Path,
    fusion_config: FusionConfig,
    floor_params: dict,
    xy_ensemble_type: str,
    xy_model_types: List[str],
    xy_ensemble_params: dict,
    verbose: bool,
) -> Path:
    """
    使用Phase 2集成模型训练单个Site

    参数：
        site_id: Site ID
        n_bssid: BSSID词典大小
        model_dir: 模型保存目录
        train_dir: 训练数据根目录
        fusion_config: 特征融合配置
        floor_params: 楼层分类器参数
        xy_ensemble_type: XY回归集成类型（'single', 'stacking', 'weighted'）
        xy_model_types: XY回归使用的模型类型列表
        xy_ensemble_params: XY回归集成参数
        verbose: 是否打印详细日志

    返回：
        模型文件路径
    """
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"🏢  Site: {site_id}")
    print(f"{'='*65}")

    # Step 1: 构建BSSID词典
    print(f"  [1/4] 构建BSSID词典（top {n_bssid}）...")
    vocab = build_bssid_vocab(
        site_train_dir=train_dir / site_id,
        n_bssid=n_bssid,
        verbose=False,
    )
    print(f"        词典大小: {len(vocab)}")

    # Step 2: 提取多模态特征
    print(f"  [2/4] 提取多模态特征...")
    X, meta, fusion = process_site_multimodal_train(
        site_id=site_id,
        bssid_vocab=vocab,
        train_dir=train_dir,
        config=fusion_config,
        verbose=verbose,
    )

    dims = fusion.get_feature_dimensions()
    print(f"        特征维度: WiFi={dims.get('wifi', 0)}, "
          f"IMU={dims.get('imu', 0)}, "
          f"Beacon={dims.get('beacon', 0)}, "
          f"Total={dims['total']}")
    print(f"        样本数: {X.shape[0]}")
    print(f"        楼层: {sorted(meta['floor'].unique())}")
    print(f"        路径数: {meta['path_id'].nunique()}")

    # Step 3: 训练V3模型（集成）
    print(f"  [3/4] 训练模型（集成类型: {xy_ensemble_type}）...")
    print(f"        模型: {', '.join(xy_model_types)}")

    model = SiteModelV3(
        site_id=site_id,
        fusion=fusion,
        floor_params=floor_params,
        xy_ensemble_type=xy_ensemble_type,
        xy_model_types=xy_model_types,
        xy_ensemble_params=xy_ensemble_params,
    )
    model.fit(X, meta, verbose=verbose)

    # Step 4: 保存模型
    print(f"  [4/4] 保存模型...")
    out_path = save_site_model_v3(model, model_dir)

    elapsed = time.time() - t0
    print(f"  ✔ 模型已保存: {out_path.name}  [{elapsed:.1f}s]")

    return out_path


def train_all_v3(
    n_bssid: int = 1500,
    model_dir: Path = _PROJECT_ROOT / "models_v3",
    train_dir: Path = TRAIN_DIR,
    site_filter: Optional[List[str]] = None,
    ensemble_type: str = 'stacking',
    model_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> dict:
    """
    批量训练所有Site（V3版本 - 集成模型）

    参数：
        n_bssid: BSSID词典大小
        model_dir: 模型保存目录
        train_dir: 训练数据目录
        site_filter: 只训练指定的Site（None=全部）
        ensemble_type: 集成类型（'single', 'stacking', 'weighted'）
        model_types: 使用的模型类型列表（例如 ['lgbm', 'xgb', 'catboost']）
        verbose: 详细日志

    返回：
        训练结果字典 {site_id: model_path}
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 融合配置（同Phase 1）
    fusion_config = FusionConfig(
        enable_wifi=True,
        wifi_stats=True,
        wifi_spatial=True,
        wifi_temporal=True,
        enable_imu=True,
        enable_beacon=True,
        fusion_type="early",
        enable_scaling=True,
        enable_pca=False,
        wifi_window_ms=5000,
        imu_window_ms=1000,
    )

    # 楼层分类器参数（同Phase 1）
    floor_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 10,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 3,
        'colsample_bytree': 0.6,
        'subsample': 0.7,
        'subsample_freq': 5,
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'verbose': -1,
    }

    # XY回归集成参数
    if model_types is None:
        if ensemble_type == 'single':
            model_types = ['lgbm']  # 单一模型默认用LightGBM
        else:
            model_types = ['lgbm', 'xgb', 'catboost']  # 集成默认用3个模型

    xy_ensemble_params = {}
    if ensemble_type == 'stacking':
        xy_ensemble_params = {
            'n_folds': 5,
            'use_original_features': True,
        }
    elif ensemble_type == 'weighted':
        # 默认权重：LightGBM权重稍高
        xy_ensemble_params = {
            'weights': [0.4, 0.3, 0.3] if len(model_types) == 3 else None,
        }

    # 获取所有Site
    train_dir = Path(train_dir)
    all_sites = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    if site_filter:
        all_sites = [s for s in all_sites if s in site_filter]

    if not all_sites:
        raise FileNotFoundError(f"在 {train_dir} 下未找到任何Site目录")

    print(f"\n{'='*65}")
    print(f"🚀  Phase 2 集成模型训练启动")
    print(f"{'='*65}")
    print(f"Sites数量: {len(all_sites)}")
    print(f"BSSID词典: {n_bssid}")
    print(f"融合配置: {fusion_config.fusion_type} fusion")
    print(f"集成类型: {ensemble_type}")
    print(f"使用模型: {', '.join(model_types)}")
    print(f"模型目录: {model_dir}")
    print(f"{'='*65}\n")

    if ensemble_type == 'stacking':
        print("⚠️  注意: Stacking集成训练较慢（3个模型 + 5折CV）")
        print("    如需加速，可使用 --ensemble-type weighted\n")

    results = {}
    failed = []
    total_t0 = time.time()

    for i, site_id in enumerate(all_sites, 1):
        print(f"[{i}/{len(all_sites)}]", end="")
        try:
            path = train_site_v3(
                site_id=site_id,
                n_bssid=n_bssid,
                model_dir=model_dir,
                train_dir=train_dir,
                fusion_config=fusion_config,
                floor_params=floor_params,
                xy_ensemble_type=ensemble_type,
                xy_model_types=model_types,
                xy_ensemble_params=xy_ensemble_params,
                verbose=verbose,
            )
            results[site_id] = path
        except Exception as exc:
            import traceback
            print(f"\n  ⚠ Site {site_id} 训练失败: {exc}")
            if verbose:
                traceback.print_exc()
            results[site_id] = None
            failed.append(site_id)

    total_elapsed = time.time() - total_t0

    print(f"\n{'='*65}")
    print(f"✅  训练完成！")
    print(f"{'='*65}")
    print(f"总Sites: {len(all_sites)}")
    print(f"成功: {len(all_sites) - len(failed)}")
    print(f"失败: {len(failed)}")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"模型目录: {model_dir}")
    print(f"{'='*65}")

    if failed:
        print(f"\n⚠  失败的Sites:")
        for s in failed:
            print(f"    - {s}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="室内定位训练脚本 V3 - Phase 2集成模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-bssid", type=int, default=1500,
        help="BSSID词典大小（推荐1500-2000）",
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(_PROJECT_ROOT / "models_v3"),
        help="模型保存目录",
    )
    parser.add_argument(
        "--train-dir", type=str, default=str(TRAIN_DIR),
        help="训练数据根目录",
    )
    parser.add_argument(
        "--sites", nargs="*", default=None,
        help="只训练指定的Sites（例如 --sites site1 site2）",
    )
    parser.add_argument(
        "--ensemble-type", type=str, default='stacking',
        choices=['single', 'stacking', 'weighted'],
        help="集成类型: single=单一模型, stacking=Stacking集成, weighted=加权平均",
    )
    parser.add_argument(
        "--model-types", nargs="+", default=None,
        choices=['lgbm', 'xgb', 'catboost'],
        help="使用的模型类型（默认: lgbm xgb catboost）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印详细日志",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    train_all_v3(
        n_bssid=args.n_bssid,
        model_dir=Path(args.model_dir),
        train_dir=Path(args.train_dir),
        site_filter=args.sites,
        ensemble_type=args.ensemble_type,
        model_types=args.model_types,
        verbose=args.verbose,
    )
