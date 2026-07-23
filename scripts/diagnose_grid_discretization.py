from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_beam_search_validation import (
    build_fusion_config,
    get_environment_snapshot,
    list_site_paths,
    prepare_holdout_site_bundle,
    split_holdout_paths,
)


GridMap = Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose discretization damage from training waypoint grids.")
    parser.add_argument("--config", default="configs/kaggle_train_config.yml")
    parser.add_argument("--site-list", required=True, help="Comma-separated site ids.")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--max-holdout-paths", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-metrics-out", default="data_processing/processed/grid_discretization_site_metrics.csv")
    parser.add_argument("--summary-out", default="data_processing/processed/grid_discretization_summary.json")
    return parser.parse_args()


def collect_train_waypoints(train_files: Sequence[Path]) -> np.ndarray:
    points: List[List[float]] = []
    for path_file in train_files:
        with open(path_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[1] == "TYPE_WAYPOINT":
                    try:
                        points.append([float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def build_floor_grids(train_files: Sequence[Path]) -> Dict[str, GridMap]:
    by_floor: Dict[str, List[Path]] = {}
    for path_file in train_files:
        by_floor.setdefault(path_file.parent.name, []).append(path_file)

    floor_grids: Dict[str, GridMap] = {}
    for floor_str, floor_files in by_floor.items():
        raw = collect_train_waypoints(floor_files)
        if len(raw) == 0:
            continue

        grids: GridMap = {
            "raw_all": raw,
            "raw_unique": np.unique(raw, axis=0),
            "round_0.1_unique": np.unique(np.round(raw, 1), axis=0),
            "round_0.5_unique": np.unique(np.round(raw / 0.5) * 0.5, axis=0),
            "round_1.0_unique": np.unique(np.round(raw / 1.0) * 1.0, axis=0),
        }
        floor_grids[floor_str] = grids
    return floor_grids


def nearest_distances(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(grid) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    dists = np.linalg.norm(points[:, np.newaxis, :] - grid[np.newaxis, :, :], axis=2)
    return np.min(dists, axis=1)


def evaluate_site_variants(
    bundle,
    floor_grids: Dict[str, GridMap],
) -> List[dict]:
    rows: List[dict] = []
    holdout = bundle.holdout_pred.copy()
    holdout["wifi_error"] = np.linalg.norm(
        holdout[["pred_x", "pred_y"]].to_numpy(dtype=np.float64)
        - holdout[["x", "y"]].to_numpy(dtype=np.float64),
        axis=1,
    )

    for grid_name in ["raw_all", "raw_unique", "round_0.1_unique", "round_0.5_unique", "round_1.0_unique"]:
        snap_error_sum = 0.0
        snap_move_sum = 0.0
        oracle_error_sum = 0.0
        total_points = 0

        for floor_str, floor_df in holdout.groupby("floor", sort=False):
            grid_map = floor_grids.get(str(floor_str))
            if not grid_map or grid_name not in grid_map:
                continue
            grid = grid_map[grid_name]
            pred_xy = floor_df[["pred_x", "pred_y"]].to_numpy(dtype=np.float64)
            gt_xy = floor_df[["x", "y"]].to_numpy(dtype=np.float64)

            pred_to_grid = nearest_distances(pred_xy, grid)
            gt_to_grid = nearest_distances(gt_xy, grid)

            if len(grid) == 0:
                continue
            idx = np.argmin(
                np.linalg.norm(pred_xy[:, np.newaxis, :] - grid[np.newaxis, :, :], axis=2),
                axis=1,
            )
            snapped = grid[idx]
            snap_error_sum += float(np.linalg.norm(snapped - gt_xy, axis=1).sum())
            snap_move_sum += float(pred_to_grid.sum())
            oracle_error_sum += float(gt_to_grid.sum())
            total_points += len(gt_xy)

        if total_points == 0:
            continue

        wifi_mae = float(holdout["wifi_error"].sum() / len(holdout))
        rows.append(
            {
                "site_id": bundle.site_id,
                "grid_name": grid_name,
                "n_holdout_paths": int(holdout["path_id"].nunique()),
                "n_holdout_points": int(total_points),
                "wifi_mae_m": wifi_mae,
                "snapped_pred_mae_m": snap_error_sum / total_points,
                "snap_damage_m": (snap_error_sum / total_points) - wifi_mae,
                "pred_to_grid_mae_m": snap_move_sum / total_points,
                "gt_to_grid_oracle_mae_m": oracle_error_sum / total_points,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    t0 = time.time()

    with open(PROJECT_ROOT / args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    site_ids = [site.strip() for site in args.site_list.split(",") if site.strip()]
    train_dir = PROJECT_ROOT / cfg["paths"]["train_dir"]
    fusion_config = build_fusion_config(cfg)

    all_rows: List[dict] = []
    for idx, site_id in enumerate(site_ids, 1):
        path_files = list_site_paths(train_dir / site_id)
        train_files, holdout_files = split_holdout_paths(
            path_files=path_files,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed + idx,
            max_holdout_paths=args.max_holdout_paths,
        )
        bundle = prepare_holdout_site_bundle(
            site_id=site_id,
            train_files=train_files,
            holdout_files=holdout_files,
            cfg=cfg,
            fusion_config=fusion_config,
            connect_radius=5.0,
        )
        floor_grids = build_floor_grids(train_files)
        site_rows = evaluate_site_variants(bundle, floor_grids)
        all_rows.extend(site_rows)
        print(f"[{idx}/{len(site_ids)}] {site_id}: {len(site_rows)} grid variants evaluated")

    results = pd.DataFrame(all_rows)
    site_metrics_out = PROJECT_ROOT / args.site_metrics_out
    summary_out = PROJECT_ROOT / args.summary_out
    site_metrics_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(site_metrics_out, index=False)

    summary = {
        "status": "ok",
        "elapsed_s": time.time() - t0,
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "max_holdout_paths": args.max_holdout_paths,
        "site_ids": site_ids,
        "variants": {},
        "environment": get_environment_snapshot(),
    }

    for grid_name, group in results.groupby("grid_name", sort=False):
        weights = group["n_holdout_points"].to_numpy(dtype=np.float64)
        summary["variants"][grid_name] = {
            "n_sites": int(group["site_id"].nunique()),
            "n_holdout_paths": int(group["n_holdout_paths"].sum()),
            "n_holdout_points": int(group["n_holdout_points"].sum()),
            "wifi_mae_m": float(np.sum(group["wifi_mae_m"].to_numpy(dtype=np.float64) * weights) / weights.sum()),
            "snapped_pred_mae_m": float(np.sum(group["snapped_pred_mae_m"].to_numpy(dtype=np.float64) * weights) / weights.sum()),
            "snap_damage_m": float(np.sum(group["snap_damage_m"].to_numpy(dtype=np.float64) * weights) / weights.sum()),
            "pred_to_grid_mae_m": float(np.sum(group["pred_to_grid_mae_m"].to_numpy(dtype=np.float64) * weights) / weights.sum()),
            "gt_to_grid_oracle_mae_m": float(np.sum(group["gt_to_grid_oracle_mae_m"].to_numpy(dtype=np.float64) * weights) / weights.sum()),
        }

    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Saved site metrics: {site_metrics_out}")
    print(f"Saved summary     : {summary_out}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
