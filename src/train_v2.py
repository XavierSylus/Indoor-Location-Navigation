"""
训练主脚本 V2 - 多模态版本 (train_v2.py)
==========================================
Phase 1: 使用多传感器特征训练模型

改进：
1. 使用features_v2模块提取高级WiFi+IMU+Beacon特征
2. 使用LightGBM代替KNN进行XY回归
3. 支持特征融合策略配置
4. 更强的模型参数

作者：Claude
版本：2.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
from src.models_v2 import SiteModelV2, save_site_model_v2


# ══════════════════════════════════════════════════════════════════════════════
# 核心训练函数
# ══════════════════════════════════════════════════════════════════════════════

def train_site_v2(
    site_id: str,
    n_bssid: int,
    model_dir: Path,
    train_dir: Path,
    fusion_config: FusionConfig,
    lgbm_params: dict,
    verbose: bool,
) -> Path:
    """
    使用多模态特征训练单个Site的模型

    参数：
        site_id: Site ID
        n_bssid: BSSID词典大小
        model_dir: 模型保存目录
        train_dir: 训练数据根目录
        fusion_config: 特征融合配置
        lgbm_params: LightGBM参数
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

    # Step 3: 训练模型
    print(f"  [3/4] 训练模型...")
    model = SiteModelV2(
        site_id=site_id,
        fusion=fusion,
        floor_params=lgbm_params['classifier'],
        xy_params=lgbm_params['regressor'],
    )
    model.fit(X, meta)

    # Step 4: 保存模型
    print(f"  [4/4] 保存模型...")
    out_path = save_site_model_v2(model, model_dir)

    elapsed = time.time() - t0
    print(f"  ✔ 模型已保存: {out_path.name}  [{elapsed:.1f}s]")

    return out_path


def train_all_v2(
    n_bssid: int = 1500,
    model_dir: Path = _PROJECT_ROOT / "models_v2",
    train_dir: Path = TRAIN_DIR,
    site_filter: list = None,
    verbose: bool = False,
) -> dict:
    """
    批量训练所有Site（V2版本）

    参数：
        n_bssid: BSSID词典大小
        model_dir: 模型保存目录
        train_dir: 训练数据目录
        site_filter: 只训练指定的Site（None=全部）
        verbose: 详细日志

    返回：
        训练结果字典 {site_id: model_path}
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 融合配置
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

    # LightGBM参数（更强的配置）
    lgbm_params = {
        'classifier': {
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
        },
        'regressor': {
            'objective': 'rmse',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'max_depth': 15,
            'learning_rate': 0.05,
            'n_estimators': 1500,
            'min_child_samples': 3,
            'colsample_bytree': 0.6,
            'subsample': 0.7,
            'subsample_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'max_bin': 255,
            'n_jobs': -1,
            'verbose': -1,
        }
    }

    # 获取所有Site
    train_dir = Path(train_dir)
    all_sites = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

    if site_filter:
        all_sites = [s for s in all_sites if s in site_filter]

    if not all_sites:
        raise FileNotFoundError(f"在 {train_dir} 下未找到任何Site目录")

    print(f"\n{'='*65}")
    print(f"🚀  多模态训练启动")
    print(f"{'='*65}")
    print(f"Sites数量: {len(all_sites)}")
    print(f"BSSID词典: {n_bssid}")
    print(f"融合配置: {fusion_config.fusion_type} fusion")
    print(f"模型目录: {model_dir}")
    print(f"{'='*65}\n")

    results = {}
    failed = []
    total_t0 = time.time()

    for i, site_id in enumerate(all_sites, 1):
        print(f"[{i}/{len(all_sites)}]", end="")
        try:
            path = train_site_v2(
                site_id=site_id,
                n_bssid=n_bssid,
                model_dir=model_dir,
                train_dir=train_dir,
                fusion_config=fusion_config,
                lgbm_params=lgbm_params,
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
        description="室内定位训练脚本 V2 - 多模态特征",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-bssid", type=int, default=1500,
        help="BSSID词典大小（推荐1500-2000）",
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(_PROJECT_ROOT / "models_v2"),
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
        "--verbose", action="store_true",
        help="打印详细日志",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    train_all_v2(
        n_bssid=args.n_bssid,
        model_dir=Path(args.model_dir),
        train_dir=Path(args.train_dir),
        site_filter=args.sites,
        verbose=args.verbose,
    )
