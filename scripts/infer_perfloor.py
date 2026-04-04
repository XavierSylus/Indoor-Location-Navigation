"""
Phase 1 推理: 使用 Per-Floor KNN (k=20) SiteModel 生成 submission
"""

import sys
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "data_processing"))
from build_wifi_features import extract_path_features

FLOOR_MAP = {
    'B3': -3, 'B2': -2, 'B1': -1,
    'F1': 0, 'F2': 1, 'F3': 2, 'F4': 3, 'F5': 4,
    'F6': 5, 'F7': 6, 'F8': 7, 'F9': 8,
    '1F': 0, '2F': 1, '3F': 2, '4F': 3, '5F': 4,
    '6F': 5, '7F': 6, '8F': 7, '9F': 8,
}


def generate_submission(cfg, model_dir, out_csv):
    sub_path = Path(cfg["paths"]["sample_sub"])
    test_dir = Path(cfg["paths"]["test_dir"])

    sub_df = pd.read_csv(sub_path)
    if "site_path_timestamp" not in sub_df.columns:
        sub_df.rename(columns={sub_df.columns[0]: "site_path_timestamp"}, inplace=True)

    site_path_queries = defaultdict(lambda: defaultdict(list))
    for row_id in sub_df["site_path_timestamp"].values:
        site, path, ts = row_id.split('_')
        site_path_queries[site][path].append((row_id, int(ts)))

    results = []
    total_sites = len(site_path_queries)
    model_path = Path(model_dir)

    print("=" * 60)
    print(f"Phase 1: Per-Floor KNN Inference")
    print(f"  Model dir: {model_path}")
    print(f"  共 {total_sites} 个 Site")
    print("=" * 60)

    t0 = time.time()

    for i, (site_id, paths) in enumerate(site_path_queries.items(), 1):
        pkl_file = model_path / f"{site_id}.pkl"
        if not pkl_file.exists():
            print(f"  ⚠ 跳过 {site_id}: 无模型")
            for path_id, queries in paths.items():
                for row_id, ts in queries:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})
            continue

        with open(pkl_file, "rb") as f:
            model = pickle.load(f)

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
                    path_file=path_file,
                    bssid_vocab=model.bssid_vocab,
                    mode="test",
                    test_timestamps=test_ts,
                    window_ms=cfg["features"]["window_ms"]
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
                print(f"  Error on {path_id}: {e}")
                for row_id in row_ids:
                    results.append({"site_path_timestamp": row_id, "floor": 0, "x": 0.0, "y": 0.0})

        elapsed = time.time() - t0
        print(f"  [{i:2d}/{total_sites}] Site {site_id}: {site_rows} rows [{elapsed:.1f}s]")

        del model

    out_df = pd.DataFrame(results)
    out_df = pd.merge(sub_df[["site_path_timestamp"]], out_df, on="site_path_timestamp", how="left")
    out_df.fillna(0, inplace=True)
    out_df['floor'] = out_df['floor'].astype(int)
    out_df.to_csv(out_csv, index=False)

    print(f"\n{'=' * 60}")
    print(f"✅ 推理完成: {out_csv}")
    print(f"   总行数: {len(out_df)}, 耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_config.yml")
    parser.add_argument("--model_dir", default="models/perfloor_k20")
    parser.add_argument("--out", default="submission_perfloor.csv")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    generate_submission(cfg, args.model_dir, args.out)
