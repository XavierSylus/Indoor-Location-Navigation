from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose single-site absolute candidate recall failure.")
    parser.add_argument("--config", default="configs/kaggle_train_config.yml")
    parser.add_argument(
        "--base-summary",
        default="data_processing/processed/absolute_holdout_summary.json",
        help="Base holdout summary used to recover seed and split settings.",
    )
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--path-id", default="")
    parser.add_argument("--holdout-fraction", type=float, default=None)
    parser.add_argument("--max-holdout-paths", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--connect-radius", type=float, default=5.0)
    parser.add_argument("--catastrophic-threshold-m", type=float, default=20.0)
    parser.add_argument(
        "--point-out",
        default="data_processing/processed/single_site_recall_failure_points.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="data_processing/processed/single_site_recall_failure_summary.json",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload: {path}")
    return payload


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload: {path}")
    return payload


def get_effective_setting(args_value, base_summary: dict, key: str, default):
    if args_value is not None:
        return args_value
    return base_summary.get(key, default)


def collect_same_floor_training_waypoints(train_files: List[Path], floor: str) -> np.ndarray:
    points: List[List[float]] = []
    for path_file in train_files:
        if path_file.parent.name != floor:
            continue
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
    return np.unique(np.round(np.asarray(points, dtype=np.float64), 1), axis=0)


def nearest_neighbor_spacing(points: np.ndarray) -> Dict[str, float]:
    if len(points) < 2:
        return {"median_nn_distance_m": float("nan"), "p90_nn_distance_m": float("nan")}
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    return {
        "median_nn_distance_m": float(np.median(nearest)),
        "p90_nn_distance_m": float(np.quantile(nearest, 0.9)),
    }


def infer_site_seed(base_summary: dict, site_id: str, seed: int) -> int:
    site_ids = [str(site) for site in base_summary.get("site_ids", [])]
    if site_id not in site_ids:
        raise ValueError(f"Site {site_id} not found in base summary site_ids.")
    site_index = site_ids.index(site_id) + 1
    return seed + site_index


def build_recommendations(
    point_df: pd.DataFrame,
    oracle_rank_mean: float,
    top5_recall: float,
    pred_to_grid_mean_m: float,
    gt_to_train_oracle_mean_m: float,
    mean_bias_mag_m: float,
    same_floor_unique_waypoints: int,
    median_nn_distance_m: float,
) -> List[str]:
    items: List[str] = []
    if top5_recall <= 0.2 or oracle_rank_mean > 10.0:
        items.append("Candidate recall is failing before trajectory optimization; do not enable beam for this site yet.")
    if gt_to_train_oracle_mean_m < 0.2 and pred_to_grid_mean_m > 3.0:
        items.append("Ground-truth-adjacent waypoints exist in training, but absolute predictions are landing in the wrong candidate cluster.")
    if mean_bias_mag_m >= 4.0:
        items.append("The path shows a strong same-direction bias; test a site-specific coordinate calibration or local re-ranking stage.")
    if same_floor_unique_waypoints <= 50:
        items.append("Same-floor waypoint coverage is sparse; investigate candidate expansion or waypoint augmentation for this floor.")
    if np.isfinite(median_nn_distance_m) and median_nn_distance_m > 2.0:
        items.append("Waypoint graph density is low; sparse local support may be amplifying absolute recall errors.")
    if int((point_df["oracle_rank"] > 10).sum()) > 0:
        items.append("At least one point has the correct waypoint beyond top-10; inspect feature retrieval around those timestamps.")
    return items


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    base_summary = load_json(resolve_path(args.base_summary))

    holdout_fraction = float(get_effective_setting(args.holdout_fraction, base_summary, "holdout_fraction", 0.1))
    max_holdout_paths = int(get_effective_setting(args.max_holdout_paths, base_summary, "max_holdout_paths", 1))
    base_seed = int(get_effective_setting(args.seed, base_summary, "seed", 42))
    site_seed = infer_site_seed(base_summary, args.site_id, base_seed)

    train_dir = PROJECT_ROOT / cfg["paths"]["train_dir"]
    fusion_config = build_fusion_config(cfg)
    path_files = list_site_paths(train_dir / args.site_id)
    train_files, holdout_files = split_holdout_paths(
        path_files=path_files,
        holdout_fraction=holdout_fraction,
        seed=site_seed,
        max_holdout_paths=max_holdout_paths,
    )

    bundle = prepare_holdout_site_bundle(
        site_id=args.site_id,
        train_files=train_files,
        holdout_files=holdout_files,
        cfg=cfg,
        fusion_config=fusion_config,
        connect_radius=args.connect_radius,
    )

    holdout_pred = bundle.holdout_pred.copy()
    if holdout_pred.empty:
        raise ValueError(f"No holdout predictions found for site {args.site_id}")

    if args.path_id:
        selected_path_id = args.path_id
    else:
        path_mae = (
            holdout_pred.assign(
                point_error=np.linalg.norm(
                    holdout_pred[["pred_x", "pred_y"]].to_numpy(dtype=np.float64)
                    - holdout_pred[["x", "y"]].to_numpy(dtype=np.float64),
                    axis=1,
                )
            )
            .groupby("path_id", sort=False)["point_error"]
            .mean()
            .sort_values(ascending=False)
        )
        selected_path_id = str(path_mae.index[0])

    path_df = holdout_pred[holdout_pred["path_id"] == selected_path_id].sort_values("timestamp").reset_index(drop=True)
    if path_df.empty:
        raise ValueError(f"Path {selected_path_id} not found in reconstructed holdout for site {args.site_id}")

    floor = str(path_df["floor"].iloc[0])
    train_waypoints = collect_same_floor_training_waypoints(train_files, floor)
    if len(train_waypoints) == 0:
        raise ValueError(f"No same-floor training waypoints found for site {args.site_id}, floor {floor}")

    gt_xy = path_df[["x", "y"]].to_numpy(dtype=np.float64)
    pred_xy = path_df[["pred_x", "pred_y"]].to_numpy(dtype=np.float64)
    gt_to_train = np.linalg.norm(gt_xy[:, np.newaxis, :] - train_waypoints[np.newaxis, :, :], axis=2)
    pred_to_train = np.linalg.norm(pred_xy[:, np.newaxis, :] - train_waypoints[np.newaxis, :, :], axis=2)
    gt_nearest_idx = np.argmin(gt_to_train, axis=1)
    pred_nearest_idx = np.argmin(pred_to_train, axis=1)

    records: List[Dict[str, object]] = []
    ranks: List[int] = []
    for point_index, candidate_index in enumerate(gt_nearest_idx):
        order = np.argsort(pred_to_train[point_index])
        rank = int(np.where(order == candidate_index)[0][0]) + 1
        ranks.append(rank)

        gt_nearest_xy = train_waypoints[candidate_index]
        pred_nearest_xy = train_waypoints[pred_nearest_idx[point_index]]
        error_dx = float(pred_xy[point_index, 0] - gt_xy[point_index, 0])
        error_dy = float(pred_xy[point_index, 1] - gt_xy[point_index, 1])

        records.append(
            {
                "site_id": args.site_id,
                "path_id": selected_path_id,
                "floor": floor,
                "timestamp": int(path_df.loc[point_index, "timestamp"]),
                "gt_x": float(gt_xy[point_index, 0]),
                "gt_y": float(gt_xy[point_index, 1]),
                "pred_x": float(pred_xy[point_index, 0]),
                "pred_y": float(pred_xy[point_index, 1]),
                "error_dx": error_dx,
                "error_dy": error_dy,
                "error_m": float(np.linalg.norm([error_dx, error_dy])),
                "oracle_rank": rank,
                "gt_to_nearest_train_m": float(np.min(gt_to_train[point_index])),
                "pred_to_nearest_train_m": float(np.min(pred_to_train[point_index])),
                "gt_nearest_train_x": float(gt_nearest_xy[0]),
                "gt_nearest_train_y": float(gt_nearest_xy[1]),
                "pred_nearest_train_x": float(pred_nearest_xy[0]),
                "pred_nearest_train_y": float(pred_nearest_xy[1]),
                "top1_hit": int(rank <= 1),
                "top5_hit": int(rank <= 5),
                "top10_hit": int(rank <= 10),
                "catastrophic_outlier": int(np.linalg.norm([error_dx, error_dy]) >= args.catastrophic_threshold_m),
            }
        )

    point_df = pd.DataFrame(records)
    mean_bias = point_df[["error_dx", "error_dy"]].to_numpy(dtype=np.float64).mean(axis=0)
    oracle_rank_mean = float(point_df["oracle_rank"].mean())
    top5_recall = float(point_df["top5_hit"].mean())
    pred_to_grid_mean_m = float(point_df["pred_to_nearest_train_m"].mean())
    gt_to_train_oracle_mean_m = float(point_df["gt_to_nearest_train_m"].mean())
    same_floor_train_paths = int(sum(path.parent.name == floor for path in train_files))
    spacing = nearest_neighbor_spacing(train_waypoints)

    summary = {
        "status": "ok",
        "site_id": args.site_id,
        "path_id": selected_path_id,
        "floor": floor,
        "seed": site_seed,
        "holdout_fraction": holdout_fraction,
        "max_holdout_paths": max_holdout_paths,
        "n_points": int(len(point_df)),
        "wifi_mae_m": float(point_df["error_m"].mean()),
        "oracle_rank_mean": oracle_rank_mean,
        "oracle_rank_median": float(point_df["oracle_rank"].median()),
        "top1_recall": float(point_df["top1_hit"].mean()),
        "top5_recall": top5_recall,
        "top10_recall": float(point_df["top10_hit"].mean()),
        "catastrophic_outlier_ratio": float(point_df["catastrophic_outlier"].mean()),
        "pred_to_grid_mean_m": pred_to_grid_mean_m,
        "gt_to_train_oracle_mean_m": gt_to_train_oracle_mean_m,
        "mean_bias_x": float(mean_bias[0]),
        "mean_bias_y": float(mean_bias[1]),
        "mean_bias_mag_m": float(np.linalg.norm(mean_bias)),
        "same_floor_train_paths": same_floor_train_paths,
        "same_floor_unique_waypoints": int(len(train_waypoints)),
        "median_nn_distance_m": spacing["median_nn_distance_m"],
        "p90_nn_distance_m": spacing["p90_nn_distance_m"],
        "feature_source": bundle.feature_source,
        "model_kind": bundle.model_kind,
        "recommendations": build_recommendations(
            point_df=point_df,
            oracle_rank_mean=oracle_rank_mean,
            top5_recall=top5_recall,
            pred_to_grid_mean_m=pred_to_grid_mean_m,
            gt_to_train_oracle_mean_m=gt_to_train_oracle_mean_m,
            mean_bias_mag_m=float(np.linalg.norm(mean_bias)),
            same_floor_unique_waypoints=int(len(train_waypoints)),
            median_nn_distance_m=spacing["median_nn_distance_m"],
        ),
        "environment": get_environment_snapshot(),
    }

    point_out = resolve_path(args.point_out)
    summary_out = resolve_path(args.summary_out)
    point_out.parent.mkdir(parents=True, exist_ok=True)
    point_df.to_csv(point_out, index=False)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    del bundle
    gc.collect()

    print(f"Saved point diagnostics: {point_out}")
    print(f"Saved summary         : {summary_out}")


if __name__ == "__main__":
    main()
