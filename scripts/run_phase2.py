"""
Phase 2: 重新提取特征 (n_bssid=2000) + 重训 + 推理
一站式脚本，串联所有步骤。
"""

import sys
import pickle
import gc
import time
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from build_wifi_features import build_bssid_vocab, extract_path_features
from models import SiteModel

import numpy as np
import pandas as pd
from collections import defaultdict

FLOOR_MAP = {
    'B3': -3, 'B2': -2, 'B1': -1,
    'F1': 0, 'F2': 1, 'F3': 2, 'F4': 3, 'F5': 4,
    'F6': 5, 'F7': 6, 'F8': 7, 'F9': 8,
    '1F': 0, '2F': 1, '3F': 2, '4F': 3, '5F': 4,
    '6F': 5, '7F': 6, '8F': 7, '9F': 8,
}


def step1_extract_features(cfg):
    """提取训练特征并缓存。"""
    train_dir = Path(cfg["paths"]["train_dir"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_bssid = cfg["features"]["n_bssid"]
    window_ms = cfg["features"]["window_ms"]

    site_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    total = len(site_dirs)

    print(f"\n{'='*60}")
    print(f"Step 1: 特征提取 (n_bssid={n_bssid})")
    print(f"  {total} 个 site → {cache_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    for i, site_dir in enumerate(site_dirs, 1):
        site_id = site_dir.name
        cache_file = cache_dir / f"{site_id}_train.pkl"

        if cache_file.exists():
            continue

        vocab = build_bssid_vocab(site_dir, n_bssid=n_bssid)
        all_X, all_y_f, all_y_x, all_y_y = [], [], [], []

        for f in sorted(site_dir.rglob("*.txt")):
            floor_name = f.parent.name
            try:
                res = extract_path_features(f, vocab, mode="train", window_ms=window_ms)
                feats, xy = res["features"], res["xy"]
                if feats is not None and feats.shape[0] > 0:
                    all_X.append(feats)
                    all_y_f.extend([floor_name] * len(feats))
                    all_y_x.append(xy[:, 0])
                    all_y_y.append(xy[:, 1])
            except Exception:
                continue

        if not all_X:
            continue

        data = {
            "vocab": vocab,
            "X": np.vstack(all_X),
            "y_f": np.array(all_y_f),
            "y_x": np.concatenate(all_y_x),
            "y_y": np.concatenate(all_y_y),
        }

        with open(cache_file, "wb") as fh:
            pickle.dump(data, fh)

        elapsed = time.time() - t0
        if i % 20 == 0 or i == total:
            print(f"  [{i:3d}/{total}] {data['X'].shape[0]} samples [{elapsed:.0f}s]")

        del data
        gc.collect()

    print(f"  特征提取完成: {time.time()-t0:.0f}s")


def step2_train(cfg):
    """使用缓存重训 SiteModel。"""
    cache_dir = Path(cfg["paths"]["cache_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    cache_files = sorted(cache_dir.glob("*_train.pkl"))
    total = len(cache_files)

    print(f"\n{'='*60}")
    print(f"Step 2: 训练 Per-Floor KNN (k=20)")
    print(f"  {total} 个 site")
    print(f"{'='*60}")

    t0 = time.time()
    for i, cf in enumerate(cache_files, 1):
        site_id = cf.stem.replace("_train", "")
        out_file = model_dir / f"{site_id}.pkl"
        if out_file.exists():
            continue

        cache = pickle.load(open(cf, "rb"))
        meta = pd.DataFrame({
            "floor": cache["y_f"],
            "path_id": ["dummy"] * len(cache["y_f"]),
            "x": cache["y_x"],
            "y": cache["y_y"],
        })

        model = SiteModel(site_id=site_id, bssid_vocab=cache["vocab"])
        model.fit(cache["X"], meta)

        with open(out_file, "wb") as fh:
            pickle.dump(model, fh, protocol=4)

        elapsed = time.time() - t0
        if i % 20 == 0 or i == total:
            print(f"  [{i:3d}/{total}] {len(cache['X'])} samples [{elapsed:.0f}s]")

        del model, cache, meta
        gc.collect()

    print(f"  训练完成: {time.time()-t0:.0f}s")


def step3_infer(cfg, out_csv):
    """推理生成 submission。"""
    sub_path = Path(cfg["paths"]["sample_sub"])
    test_dir = Path(cfg["paths"]["test_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])

    sub_df = pd.read_csv(sub_path)
    if "site_path_timestamp" not in sub_df.columns:
        sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"}, inplace=True)

    site_path_queries = defaultdict(lambda: defaultdict(list))
    for row_id in sub_df["site_path_timestamp"].values:
        site, path, ts = row_id.split('_')
        site_path_queries[site][path].append((row_id, int(ts)))

    total_sites = len(site_path_queries)
    print(f"\n{'='*60}")
    print(f"Step 3: 推理 ({total_sites} sites)")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    for i, (site_id, paths) in enumerate(site_path_queries.items(), 1):
        pkl_file = model_dir / f"{site_id}.pkl"
        if not pkl_file.exists():
            for path_id, queries in paths.items():
                for row_id, ts in queries:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})
            continue

        model = pickle.load(open(pkl_file, "rb"))
        site_rows = sum(len(q) for q in paths.values())

        for path_id, queries in paths.items():
            path_file = test_dir / f"{path_id}.txt"
            row_ids = [q[0] for q in queries]
            test_ts = [q[1] for q in queries]

            if not path_file.exists():
                for row_id in row_ids:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})
                continue

            try:
                res = extract_path_features(
                    path_file, model.bssid_vocab, mode="test",
                    test_timestamps=test_ts, window_ms=cfg["features"]["window_ms"]
                )
                X_test = res["features"]
                path_ids_arr = np.array([path_id] * len(X_test))
                floors, xy = model.predict(X_test, path_ids=path_ids_arr)

                for idx, row_id in enumerate(row_ids):
                    floor_num = FLOOR_MAP.get(str(floors[idx]), 0)
                    results.append({
                        "site_path_timestamp": row_id,
                        "floor": floor_num,
                        "x": xy[idx, 0],
                        "y": xy[idx, 1],
                    })
            except Exception as e:
                for row_id in row_ids:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})

        elapsed = time.time() - t0
        print(f"  [{i:2d}/{total_sites}] Site {site_id}: {site_rows} rows [{elapsed:.1f}s]")
        del model

    out_df = pd.DataFrame(results)
    out_df = pd.merge(sub_df[["site_path_timestamp"]], out_df, on="site_path_timestamp", how="left")
    out_df.fillna(0, inplace=True)
    out_df["floor"] = out_df["floor"].astype(int)
    out_df.to_csv(out_csv, index=False)
    print(f"\n✅ 推理完成: {out_csv} ({len(out_df)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_2k_config.yml")
    parser.add_argument("--out", default="submission_perfloor_2k.csv")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if not args.skip_extract:
        step1_extract_features(cfg)
    if not args.skip_train:
        step2_train(cfg)
    step3_infer(cfg, args.out)
