"""
Phase 1: 重训 SiteModel (Per-Floor KNN, k=20)
使用已有的训练缓存重新训练，不需要重新提取特征。
"""

import sys
import pickle
import gc
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import SiteModel

import numpy as np
import pandas as pd


def retrain_all(cache_dir: str, output_dir: str):
    cache_path = Path(cache_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cache_files = sorted(cache_path.glob("*_train.pkl"))
    total = len(cache_files)

    print(f"Phase 1: 重训 SiteModel (Per-Floor KNN, k=20)")
    print(f"  缓存目录: {cache_path}")
    print(f"  输出目录: {out_path}")
    print(f"  共 {total} 个 site")
    print("=" * 60)

    t0 = time.time()

    for i, cf in enumerate(cache_files, 1):
        site_id = cf.stem.replace("_train", "")

        out_file = out_path / f"{site_id}.pkl"
        if out_file.exists():
            continue

        with open(cf, "rb") as f:
            cache = pickle.load(f)

        X = cache["X"]
        y_f = cache["y_f"]
        y_x = cache["y_x"]
        y_y = cache["y_y"]
        vocab = cache["vocab"]

        meta = pd.DataFrame({
            "floor": y_f,
            "path_id": ["dummy"] * len(y_f),
            "x": y_x,
            "y": y_y,
        })

        model = SiteModel(site_id=site_id, bssid_vocab=vocab)
        model.fit(X, meta)

        out_file = out_path / f"{site_id}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(model, f, protocol=4)

        elapsed = time.time() - t0
        print(f"  [{i:2d}/{total}] {site_id}: {len(X)} samples, {len(set(str(f) for f in y_f))} floors [{elapsed:.1f}s]")

        del model, X, meta, cache
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"✅ 重训完成，{total} 个 SiteModel 已保存到 {out_path}")
    print(f"   总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data_processing/processed")
    parser.add_argument("--output_dir", default="models/perfloor_k20")
    args = parser.parse_args()

    retrain_all(args.cache_dir, args.output_dir)
