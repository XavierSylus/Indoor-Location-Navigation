"""
Phase 2: 多模态集成训练 + 缓存驱动 + Kaggle 适配
===================================================
融合版一站式脚本，串联所有步骤：
  Step 1: 多模态特征提取 (WiFi + IMU + Beacon) → 确定性缓存
  Step 2: 三模型集成训练 (LightGBM/XGBoost/CatBoost Stacking)
  Step 3: 集成模型推理 → submission.csv

支持特性：
  - 缓存只读模式（Kaggle 挂载 Dataset）
  - 按批次训练 (--site-batch / --site-list)
  - 自动 Kaggle 环境检测与路径适配
"""

import sys
import pickle
import gc
import time
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.kaggle_adapter import is_kaggle, resolve_paths, validate_kaggle_mounts
from src.multimodal_fusion import (
    MultimodalFeatureFusion,
    FusionConfig,
    process_site_multimodal_train,
    process_site_multimodal_test,
)
from src.models_v3 import SiteModelV3, save_site_model_v3, load_site_model_v3
from src.features import build_bssid_vocab

FLOOR_MAP = {
    'B3': -3, 'B2': -2, 'B1': -1,
    'F1': 0, 'F2': 1, 'F3': 2, 'F4': 3, 'F5': 4,
    'F6': 5, 'F7': 6, 'F8': 7, 'F9': 8,
    '1F': 0, '2F': 1, '3F': 2, '4F': 3, '5F': 4,
    '6F': 5, '7F': 6, '8F': 7, '9F': 8,
}


# ══════════════════════════════════════════════════════════════════════════════
# Site 过滤工具
# ══════════════════════════════════════════════════════════════════════════════

def filter_sites(all_site_ids: list, args) -> list:
    """
    根据 CLI 参数过滤 Site 列表。

    优先级：--site-list > --site-batch（互斥使用）
    """
    if args.site_list:
        explicit = [s.strip() for s in args.site_list.split(",") if s.strip()]
        filtered = [s for s in all_site_ids if s in explicit]
        if len(filtered) != len(explicit):
            missing = set(explicit) - set(filtered)
            print(f"  ⚠ 以下 Site 在数据中未找到: {missing}")
        return filtered

    if args.site_batch is not None:
        total = args.total_batches
        batch_idx = args.site_batch - 1  # 转 0-indexed
        if batch_idx < 0 or batch_idx >= total:
            raise ValueError(f"--site-batch 必须在 1~{total} 范围内")

        n = len(all_site_ids)
        chunk_size = (n + total - 1) // total
        start = batch_idx * chunk_size
        end = min(start + chunk_size, n)
        filtered = all_site_ids[start:end]
        print(f"  批次 {args.site_batch}/{total}: Site [{start}:{end}]，共 {len(filtered)} 个")
        return filtered

    return all_site_ids


def build_fusion_config(cfg: dict) -> FusionConfig:
    """从 YAML 配置构建 FusionConfig"""
    fc = cfg.get("fusion", {})
    return FusionConfig(
        enable_wifi=fc.get("enable_wifi", True),
        wifi_stats=fc.get("wifi_stats", True),
        wifi_spatial=fc.get("wifi_spatial", True),
        wifi_temporal=fc.get("wifi_temporal", True),
        enable_imu=fc.get("enable_imu", True),
        enable_beacon=fc.get("enable_beacon", True),
        fusion_type=fc.get("fusion_type", "early"),
        enable_scaling=fc.get("enable_scaling", True),
        enable_pca=fc.get("enable_pca", False),
        wifi_window_ms=cfg.get("features", {}).get("window_ms", 5000),
        imu_window_ms=fc.get("imu_window_ms", 1000),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 多模态特征提取 + 缓存
# ══════════════════════════════════════════════════════════════════════════════

def step1_extract_features(cfg, site_ids):
    """提取多模态训练特征并缓存。Kaggle 只读模式下完全跳过。"""
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache_readonly = cfg.get("_cache_readonly", False)

    if cache_readonly:
        # 验证缓存是否齐全
        missing = [s for s in site_ids if not (cache_dir / f"{s}_train.pkl").exists()]
        if missing:
            print(f"  ❌ 只读模式下缺少 {len(missing)} 个缓存文件:")
            for s in missing[:5]:
                print(f"     - {s}_train.pkl")
            if len(missing) > 5:
                print(f"     ... 及其余 {len(missing)-5} 个")
            raise FileNotFoundError("缓存不完整，请在本地完成特征提取后重新上传 Dataset")
        print(f"  ✔ 只读模式：{len(site_ids)} 个缓存文件全部就绪，跳过提取")
        return

    train_dir = Path(cfg["paths"]["train_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_bssid = cfg["features"]["n_bssid"]
    fusion_config = build_fusion_config(cfg)
    total = len(site_ids)

    print(f"\n{'='*60}")
    print(f"Step 1: 多模态特征提取 (n_bssid={n_bssid})")
    print(f"  {total} 个 Site → {cache_dir}")
    print(f"  模态: WiFi={fusion_config.enable_wifi} IMU={fusion_config.enable_imu} Beacon={fusion_config.enable_beacon}")
    print(f"{'='*60}")

    t0 = time.time()
    for i, site_id in enumerate(site_ids, 1):
        cache_file = cache_dir / f"{site_id}_train.pkl"
        if cache_file.exists():
            if i % 20 == 0 or i == total:
                print(f"  [{i:3d}/{total}] {site_id} (缓存命中)")
            continue

        site_dir = train_dir / site_id
        if not site_dir.exists():
            print(f"  [{i:3d}/{total}] ⚠ 跳过 {site_id}: 目录不存在")
            continue

        try:
            # 构建 BSSID 词典
            vocab = build_bssid_vocab(site_dir, n_bssid=n_bssid)

            # 提取多模态融合特征
            X, meta, fusion = process_site_multimodal_train(
                site_id=site_id,
                bssid_vocab=vocab,
                train_dir=train_dir,
                config=fusion_config,
                verbose=False,
            )

            data = {
                "vocab": vocab,
                "X": X,
                "y_f": meta["floor"].values,
                "y_x": meta["x"].values,
                "y_y": meta["y"].values,
                "fusion": fusion,
            }

            with open(cache_file, "wb") as fh:
                pickle.dump(data, fh, protocol=4)

            elapsed = time.time() - t0
            if i % 20 == 0 or i == total:
                print(f"  [{i:3d}/{total}] {X.shape[0]} samples, {X.shape[1]} dims [{elapsed:.0f}s]")

        except Exception as e:
            print(f"  [{i:3d}/{total}] ⚠ {site_id} 失败: {e}")

        gc.collect()

    print(f"  特征提取完成: {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 集成模型训练
# ══════════════════════════════════════════════════════════════════════════════

def step2_train(cfg, site_ids):
    """使用缓存训练 SiteModelV3 集成模型。"""
    cache_dir = Path(cfg["paths"]["cache_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    ensemble_cfg = cfg.get("ensemble", {})
    ensemble_type = ensemble_cfg.get("type", "stacking")
    model_types = ensemble_cfg.get("models", ["lgbm", "xgb", "catboost"])
    floor_params = cfg.get("model_params", {}).get("classifier", {})

    # 构建集成参数
    xy_ensemble_params = {}
    if ensemble_type == "stacking":
        stacking_cfg = ensemble_cfg.get("stacking", {})
        xy_ensemble_params = {
            "n_folds": stacking_cfg.get("n_folds", 5),
            "use_original_features": stacking_cfg.get("use_original_features", True),
        }
    elif ensemble_type == "weighted":
        weighted_cfg = ensemble_cfg.get("weighted", {})
        weights = weighted_cfg.get("weights")
        if weights:
            xy_ensemble_params = {"weights": weights}

    total = len(site_ids)

    print(f"\n{'='*60}")
    print(f"Step 2: 集成模型训练 (类型={ensemble_type})")
    print(f"  模型: {', '.join(model_types)}")
    print(f"  {total} 个 Site → {model_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    failed = []

    for i, site_id in enumerate(site_ids, 1):
        out_file = model_dir / f"{site_id}_v3.pkl"
        if out_file.exists():
            if i % 20 == 0 or i == total:
                print(f"  [{i:3d}/{total}] {site_id} (模型已存在)")
            continue

        cache_file = cache_dir / f"{site_id}_train.pkl"
        if not cache_file.exists():
            print(f"  [{i:3d}/{total}] ⚠ 跳过 {site_id}: 缓存不存在")
            failed.append(site_id)
            continue

        try:
            with open(cache_file, "rb") as fh:
                cache = pickle.load(fh)

            fusion = cache["fusion"]

            model = SiteModelV3(
                site_id=site_id,
                fusion=fusion,
                floor_params=floor_params,
                xy_ensemble_type=ensemble_type,
                xy_model_types=model_types,
                xy_ensemble_params=xy_ensemble_params,
            )

            meta = pd.DataFrame({
                "floor": cache["y_f"],
                "path_id": ["cached"] * len(cache["y_f"]),
                "x": cache["y_x"],
                "y": cache["y_y"],
            })

            model.fit(cache["X"], meta, verbose=(i <= 3))

            save_site_model_v3(model, model_dir)

            elapsed = time.time() - t0
            if i % 10 == 0 or i == total:
                print(f"  [{i:3d}/{total}] {len(cache['X'])} samples [{elapsed:.0f}s]")

        except Exception as e:
            print(f"  [{i:3d}/{total}] ⚠ {site_id} 训练失败: {e}")
            failed.append(site_id)

        gc.collect()

    print(f"  训练完成: {time.time()-t0:.0f}s")
    if failed:
        print(f"  失败 {len(failed)} 个: {failed[:10]}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 集成模型推理
# ══════════════════════════════════════════════════════════════════════════════

def step3_infer(cfg, out_csv):
    """使用 SiteModelV3 集成模型推理生成 submission。"""
    sub_path = Path(cfg["paths"]["sample_sub"])
    test_dir = Path(cfg["paths"]["test_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])

    sub_df = pd.read_csv(sub_path)
    if "site_path_timestamp" not in sub_df.columns:
        sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"}, inplace=True)

    # 按 site → path → (row_id, timestamp) 分组
    site_path_queries = defaultdict(lambda: defaultdict(list))
    for row_id in sub_df["site_path_timestamp"].values:
        site, path, ts = row_id.split('_')
        site_path_queries[site][path].append((row_id, int(ts)))

    total_sites = len(site_path_queries)
    print(f"\n{'='*60}")
    print(f"Step 3: 集成模型推理 ({total_sites} sites)")
    print(f"  模型目录: {model_dir}")
    print(f"  测试数据: {test_dir}")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    for i, (site_id, paths) in enumerate(site_path_queries.items(), 1):
        pkl_file = model_dir / f"{site_id}_v3.pkl"
        site_rows = sum(len(q) for q in paths.values())

        if not pkl_file.exists():
            print(f"  [{i:2d}/{total_sites}] ⚠ {site_id}: 模型不存在，填充零值")
            for path_id, queries in paths.items():
                for row_id, ts in queries:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})
            continue

        try:
            model = load_site_model_v3(pkl_file)

            for path_id, queries in paths.items():
                path_file = test_dir / f"{path_id}.txt"
                row_ids = [q[0] for q in queries]
                test_ts = np.array([q[1] for q in queries], dtype=np.int64)

                if not path_file.exists():
                    for row_id in row_ids:
                        results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})
                    continue

                # 用模型内置的 fusion 提取测试特征
                features_dict = model.fusion.extract_from_file(path_file, test_ts)
                X_test = model.fusion.fuse_features(features_dict, preprocess=True)

                floors, xy = model.predict(X_test)

                for idx, row_id in enumerate(row_ids):
                    floor_num = FLOOR_MAP.get(str(floors[idx]), 0)
                    results.append({
                        "site_path_timestamp": row_id,
                        "floor": floor_num,
                        "x": float(xy[idx, 0]),
                        "y": float(xy[idx, 1]),
                    })

        except Exception as e:
            print(f"  [{i:2d}/{total_sites}] ⚠ {site_id} 推理失败: {e}")
            for path_id, queries in paths.items():
                for row_id, ts in queries:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})

        elapsed = time.time() - t0
        if i % 5 == 0 or i == total_sites:
            print(f"  [{i:2d}/{total_sites}] Site {site_id}: {site_rows} rows [{elapsed:.1f}s]")
        del model
        gc.collect()

    out_df = pd.DataFrame(results)
    out_df = pd.merge(sub_df[["site_path_timestamp"]], out_df, on="site_path_timestamp", how="left")
    out_df.fillna(0, inplace=True)
    out_df["floor"] = out_df["floor"].astype(int)
    out_df.to_csv(out_csv, index=False)
    print(f"\n✅ 推理完成: {out_csv} ({len(out_df)} rows, {time.time()-t0:.0f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: 多模态集成训练 + Kaggle 适配",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/kaggle_train_config.yml",
                        help="YAML 配置文件路径")
    parser.add_argument("--out", default="submission_phase2_ensemble.csv",
                        help="输出 submission CSV 文件名")
    parser.add_argument("--skip-extract", action="store_true",
                        help="跳过特征提取（直接使用已有缓存）")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练（直接使用已有模型推理）")
    parser.add_argument("--skip-infer", action="store_true",
                        help="跳过推理（仅做特征提取和训练）")

    # 批次训练参数
    parser.add_argument("--site-batch", type=int, default=None,
                        help="当前批次编号 (1-indexed)，与 --total-batches 配合")
    parser.add_argument("--total-batches", type=int, default=4,
                        help="总批次数")
    parser.add_argument("--site-list", type=str, default=None,
                        help="手动指定 Site ID 列表（逗号分隔）")

    parser.add_argument("--verbose", action="store_true",
                        help="打印详细日志")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Kaggle 环境适配
    cfg = resolve_paths(cfg)

    if is_kaggle():
        if not validate_kaggle_mounts(cfg):
            print("❌ Kaggle 挂载验证失败，终止")
            sys.exit(1)

    # Site 列表获取策略：
    #   本地环境 → 从 train_dir 获取（权威来源，始终完整）
    #   Kaggle  → 从 cache_dir 获取（train_dir 可能不包含全部 Site）
    cache_dir = Path(cfg["paths"]["cache_dir"])
    train_dir = Path(cfg["paths"]["train_dir"])

    if not cfg.get("_cache_readonly", False) and train_dir.exists():
        all_site_ids = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    elif cache_dir.exists():
        all_site_ids = sorted(
            f.stem.replace("_train", "") for f in cache_dir.glob("*_train.pkl")
        )
    else:
        print(f"❌ 无法获取 Site 列表: 训练数据 ({train_dir}) 和缓存 ({cache_dir}) 均不存在")
        sys.exit(1)

    # 使用 --total-batches 从 YAML 的默认值兜底
    if args.total_batches == 4 and "batch" in cfg:
        args.total_batches = cfg["batch"].get("total_batches", 4)

    site_ids = filter_sites(all_site_ids, args)

    if not site_ids:
        print("⚠ 过滤后无 Site 需要处理，退出")
        sys.exit(0)

    print(f"\n🚀 Phase 2 集成训练流水线启动")
    print(f"   环境: {'Kaggle' if is_kaggle() else '本地'}")
    print(f"   Site 数量: {len(site_ids)} / {len(all_site_ids)}")
    print(f"   配置: {args.config}")

    # 执行流水线
    if not args.skip_extract:
        step1_extract_features(cfg, site_ids)
    else:
        print("\n⏭ 跳过 Step 1 (特征提取)")

    if not args.skip_train:
        step2_train(cfg, site_ids)
    else:
        print("\n⏭ 跳过 Step 2 (模型训练)")

    if not args.skip_infer:
        step3_infer(cfg, args.out)
    else:
        print("\n⏭ 跳过 Step 3 (推理)")
