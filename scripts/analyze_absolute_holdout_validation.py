from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_beam_search_validation import (
    build_fusion_config,
    floor_str_to_num,
    get_environment_snapshot,
    list_site_paths,
    prepare_holdout_site_bundle,
    split_holdout_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze absolute-position holdout validation without beam or PDR.")
    parser.add_argument("--config", default="configs/kaggle_train_config.yml")
    parser.add_argument(
        "--site-summary-source",
        default="data_processing/processed/safe_beam_param_search_summary.json",
        help="JSON file used to recover the site list for the current holdout study.",
    )
    parser.add_argument("--site-list", default="", help="Optional comma-separated override for site ids.")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--max-holdout-paths", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connect-radius", type=float, default=5.0)
    parser.add_argument(
        "--path-out",
        default="data_processing/processed/absolute_holdout_path_metrics.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="data_processing/processed/absolute_holdout_summary.json",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON payload: {path}")
    return data


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_site_ids(args: argparse.Namespace) -> List[str]:
    if args.site_list.strip():
        return [site.strip() for site in args.site_list.split(",") if site.strip()]

    summary = load_json(resolve_path(args.site_summary_source))
    site_ids = summary.get("site_ids") or []
    if not site_ids:
        raise ValueError("Could not recover site_ids from site summary source.")
    return [str(site_id) for site_id in site_ids]


def compute_waypoint_density(train_meta: pd.DataFrame) -> float:
    if train_meta.empty:
        return float("nan")

    densities: List[float] = []
    for _, floor_df in train_meta.groupby("floor", sort=True):
        unique_points = np.unique(np.round(floor_df[["x", "y"]].to_numpy(dtype=np.float64), 1), axis=0)
        if len(unique_points) == 0:
            continue
        span = unique_points.max(axis=0) - unique_points.min(axis=0)
        area = float(span[0] * span[1])
        if area <= 0.0:
            densities.append(float(len(unique_points)))
        else:
            densities.append(float(len(unique_points) / area))

    if not densities:
        return float("nan")
    return float(np.mean(densities))


def assign_quantile_buckets(dataframe: pd.DataFrame, source_column: str, target_column: str) -> pd.DataFrame:
    result = dataframe.copy()
    non_null = result[source_column].dropna()
    if len(non_null) < 3 or non_null.nunique() < 3:
        result[target_column] = "all"
        return result

    ranked = non_null.rank(method="first")
    buckets = pd.qcut(ranked, q=3, labels=["low", "medium", "high"])
    result[target_column] = "all"
    result.loc[non_null.index, target_column] = buckets.astype(str)
    return result


def point_weighted_summary(dataframe: pd.DataFrame, group_column: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group_value, group_df in dataframe.groupby(group_column, dropna=False):
        n_points = int(group_df["n_points"].sum())
        err_sum = float(group_df["wifi_error_sum_m"].sum())
        rows.append(
            {
                group_column: "nan" if pd.isna(group_value) else group_value,
                "n_paths": int(len(group_df)),
                "n_points": n_points,
                "wifi_mae_m": err_sum / max(n_points, 1),
            }
        )
    rows.sort(key=lambda item: item["wifi_mae_m"], reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    train_dir = PROJECT_ROOT / cfg["paths"]["train_dir"]
    fusion_config = build_fusion_config(cfg)
    site_ids = get_site_ids(args)

    path_rows: List[Dict[str, object]] = []

    for index, site_id in enumerate(site_ids, start=1):
        path_files = list_site_paths(train_dir / site_id)
        train_files, holdout_files = split_holdout_paths(
            path_files=path_files,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed + index,
            max_holdout_paths=args.max_holdout_paths,
        )
        if not holdout_files:
            continue

        bundle = prepare_holdout_site_bundle(
            site_id=site_id,
            train_files=train_files,
            holdout_files=holdout_files,
            cfg=cfg,
            fusion_config=fusion_config,
            connect_radius=args.connect_radius,
        )

        train_meta = bundle.train_meta.copy()
        train_floor_count = int(train_meta["floor"].astype(str).nunique())
        train_path_count = int(len(train_files))
        waypoint_density = compute_waypoint_density(train_meta)

        for path_file in bundle.holdout_files:
            path_df = bundle.holdout_pred[bundle.holdout_pred["path_id"] == path_file.stem].sort_values("timestamp")
            if path_df.empty:
                continue

            gt_xy = path_df[["x", "y"]].to_numpy(dtype=np.float64)
            pred_xy = path_df[["pred_x", "pred_y"]].to_numpy(dtype=np.float64)
            floor_str = str(path_file.parent.name)
            gt_floor_num = floor_str_to_num(floor_str)
            pred_floor_num = path_df["pred_floor_num"].to_numpy(dtype=np.int64)
            errors = np.linalg.norm(pred_xy - gt_xy, axis=1)

            path_length_m = 0.0
            if len(gt_xy) >= 2:
                path_length_m = float(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).sum())

            floor_grid = bundle.dijkstra_data.get(floor_str)
            if floor_grid is None or len(floor_grid[0]) == 0:
                pred_to_grid = np.full(len(pred_xy), np.nan, dtype=np.float64)
            else:
                wps = floor_grid[0]
                pred_to_grid = np.min(
                    np.linalg.norm(pred_xy[:, np.newaxis, :] - wps[np.newaxis, :, :], axis=2),
                    axis=1,
                )

            path_rows.append(
                {
                    "site_id": site_id,
                    "path_id": path_file.stem,
                    "floor": floor_str,
                    "n_points": int(len(path_df)),
                    "path_length_m": path_length_m,
                    "wifi_mae_m": float(errors.mean()),
                    "wifi_error_sum_m": float(errors.sum()),
                    "floor_accuracy": float(np.mean(pred_floor_num == gt_floor_num)),
                    "pred_to_grid_mae_m": float(np.nanmean(pred_to_grid)) if np.isfinite(pred_to_grid).any() else float("nan"),
                    "train_floor_count": train_floor_count,
                    "train_path_count": train_path_count,
                    "train_waypoint_density": waypoint_density,
                    "feature_source": bundle.feature_source,
                    "model_kind": bundle.model_kind,
                }
            )

    path_metrics = pd.DataFrame(path_rows)
    if path_metrics.empty:
        raise ValueError("No holdout path metrics were produced.")

    path_metrics = assign_quantile_buckets(path_metrics, "path_length_m", "path_length_bucket")
    path_metrics = assign_quantile_buckets(path_metrics, "train_waypoint_density", "waypoint_density_bucket")
    path_metrics = path_metrics.sort_values(["wifi_mae_m", "site_id", "path_id"], ascending=[False, True, True]).reset_index(drop=True)

    path_out = resolve_path(args.path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_metrics.to_csv(path_out, index=False)

    total_points = int(path_metrics["n_points"].sum())
    total_error = float(path_metrics["wifi_error_sum_m"].sum())
    summary = {
        "status": "ok",
        "target_route": "absolute_baseline",
        "site_ids": site_ids,
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "max_holdout_paths": args.max_holdout_paths,
        "n_paths": int(len(path_metrics)),
        "n_points": total_points,
        "overall_wifi_mae_m": total_error / max(total_points, 1),
        "worst_paths": path_metrics.head(10)[
            [
                "site_id",
                "path_id",
                "floor",
                "wifi_mae_m",
                "path_length_m",
                "train_floor_count",
                "train_waypoint_density",
            ]
        ].to_dict(orient="records"),
        "by_site": point_weighted_summary(path_metrics, "site_id"),
        "by_train_floor_count": point_weighted_summary(path_metrics, "train_floor_count"),
        "by_path_length_bucket": point_weighted_summary(path_metrics, "path_length_bucket"),
        "by_waypoint_density_bucket": point_weighted_summary(path_metrics, "waypoint_density_bucket"),
        "environment": get_environment_snapshot(),
    }

    summary_out = resolve_path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved path metrics: {path_out}")
    print(f"Saved summary     : {summary_out}")


if __name__ == "__main__":
    main()
