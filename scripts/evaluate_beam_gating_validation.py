from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.pdr_module import PDR
from scripts.diagnose_beam_search_validation import (
    build_fusion_config,
    get_environment_snapshot,
    list_site_paths,
    prepare_holdout_site_bundle,
    split_holdout_paths,
)
from scripts.step3_infer_and_optimize import (
    beam_search_single_trajectory,
    load_imu_model,
    predict_ensemble_deltas_for_path,
)


@dataclass
class PathEvaluation:
    site_id: str
    path_id: str
    n_points: int
    grid_dist_mean_m: float
    grid_dist_p90_m: float
    wifi_err_sum_m: float
    beam_err_sum_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate path-level gating for Beam Search on strict holdout.")
    parser.add_argument("--config", default="configs/kaggle_train_config.yml")
    parser.add_argument("--imu-config", default="configs/imu_delta_model_v3.yml")
    parser.add_argument("--imu-ckpt", default="models/imu_delta_v3/imu_delta_best.pt")
    parser.add_argument("--pdr-config", default="configs/pdr_config.yml")
    parser.add_argument("--site-list", required=True, help="Comma-separated site ids.")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--max-holdout-paths", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connect-radius", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=2000)
    parser.add_argument("--n-candidates", type=int, default=100)
    parser.add_argument("--w-wifi", type=float, default=3.0)
    parser.add_argument("--w-imu-l2", type=float, default=0.5)
    parser.add_argument("--w-imu-angle", type=float, default=0.3)
    parser.add_argument("--w-dijkstra", type=float, default=1.5)
    parser.add_argument("--delta-weight-imu", type=float, default=0.67)
    parser.add_argument("--delta-weight-pdr", type=float, default=0.33)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    )
    parser.add_argument(
        "--metric",
        choices=["mean", "p90"],
        default="mean",
        help="Path-level grid distance metric used for gating.",
    )
    parser.add_argument(
        "--path-out",
        default="data_processing/processed/beam_gating_path_metrics.csv",
    )
    parser.add_argument(
        "--sweep-out",
        default="data_processing/processed/beam_gating_sweep.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="data_processing/processed/beam_gating_summary.json",
    )
    return parser.parse_args()


def nearest_distances(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(grid) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    dists = np.linalg.norm(points[:, np.newaxis, :] - grid[np.newaxis, :, :], axis=2)
    return np.min(dists, axis=1)


def evaluate_paths(
    args: argparse.Namespace,
    cfg: dict,
    fusion_config,
    imu_model,
    pdr_model,
    imu_n_time_steps: int,
    device: torch.device,
) -> List[PathEvaluation]:
    train_dir = PROJECT_ROOT / cfg["paths"]["train_dir"]
    site_ids = [site.strip() for site in args.site_list.split(",") if site.strip()]
    results: List[PathEvaluation] = []

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
            connect_radius=args.connect_radius,
        )

        for path_file in bundle.holdout_files:
            path_df = bundle.holdout_pred[bundle.holdout_pred["path_id"] == path_file.stem].sort_values("timestamp")
            if path_df.empty:
                continue

            pred_xy = path_df[["pred_x", "pred_y"]].to_numpy(dtype=np.float64)
            gt_xy = path_df[["x", "y"]].to_numpy(dtype=np.float64)
            pred_floor_str = path_df["pred_floor_str"].astype(str).tolist()
            timestamps = path_df["timestamp"].to_numpy(dtype=np.int64)

            floor_grid_distances = []
            for row_idx, floor_str in enumerate(pred_floor_str):
                data = bundle.dijkstra_data.get(floor_str)
                if data is None or len(data[0]) == 0:
                    floor_grid_distances.append(np.inf)
                    continue
                floor_grid_distances.append(
                    float(np.min(np.linalg.norm(data[0] - pred_xy[row_idx], axis=1)))
                )
            floor_grid_distances = np.asarray(floor_grid_distances, dtype=np.float64)

            ensemble_deltas = predict_ensemble_deltas_for_path(
                imu_model=imu_model,
                pdr_model=pdr_model,
                path_file=path_file,
                timestamps=timestamps.tolist(),
                floor_str=path_file.parent.name,
                site_id=site_id,
                device=device,
                imu_weight=args.delta_weight_imu,
                pdr_weight=args.delta_weight_pdr,
                n_time_steps=imu_n_time_steps,
            )
            beam_xy = beam_search_single_trajectory(
                init_xy=pred_xy,
                init_floors=pred_floor_str,
                imu_deltas=ensemble_deltas,
                dijkstra_data=bundle.dijkstra_data,
                beam_width=args.beam_width,
                n_candidates=args.n_candidates,
                w_wifi=args.w_wifi,
                w_imu_l2=args.w_imu_l2,
                w_imu_angle=args.w_imu_angle,
                w_dijkstra=args.w_dijkstra,
                cost_mode="legacy",
                use_dijkstra=True,
                candidate_mode="delta",
            )

            results.append(
                PathEvaluation(
                    site_id=site_id,
                    path_id=path_file.stem,
                    n_points=len(path_df),
                    grid_dist_mean_m=float(np.mean(floor_grid_distances)),
                    grid_dist_p90_m=float(np.quantile(floor_grid_distances, 0.9)),
                    wifi_err_sum_m=float(np.linalg.norm(pred_xy - gt_xy, axis=1).sum()),
                    beam_err_sum_m=float(np.linalg.norm(beam_xy - gt_xy, axis=1).sum()),
                )
            )

    return results


def sweep_thresholds(
    path_rows: Sequence[PathEvaluation],
    thresholds: Sequence[float],
    metric: str,
) -> pd.DataFrame:
    rows = []
    wifi_err_sum = sum(row.wifi_err_sum_m for row in path_rows)
    beam_err_sum = sum(row.beam_err_sum_m for row in path_rows)
    total_points = sum(row.n_points for row in path_rows)

    for threshold in thresholds:
        gated_err_sum = 0.0
        enabled_paths = 0
        enabled_points = 0
        for row in path_rows:
            path_metric = row.grid_dist_mean_m if metric == "mean" else row.grid_dist_p90_m
            if path_metric <= threshold:
                gated_err_sum += row.beam_err_sum_m
                enabled_paths += 1
                enabled_points += row.n_points
            else:
                gated_err_sum += row.wifi_err_sum_m

        rows.append(
            {
                "metric": metric,
                "threshold_m": float(threshold),
                "n_paths": int(len(path_rows)),
                "n_points": int(total_points),
                "beam_enabled_paths": int(enabled_paths),
                "beam_enabled_points": int(enabled_points),
                "wifi_mae_m": wifi_err_sum / max(total_points, 1),
                "always_beam_mae_m": beam_err_sum / max(total_points, 1),
                "gated_mae_m": gated_err_sum / max(total_points, 1),
                "gated_minus_wifi_m": (gated_err_sum - wifi_err_sum) / max(total_points, 1),
                "gated_minus_beam_m": (gated_err_sum - beam_err_sum) / max(total_points, 1),
            }
        )

    return pd.DataFrame(rows).sort_values("gated_mae_m").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    t0 = time.time()

    with open(PROJECT_ROOT / args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(PROJECT_ROOT / args.imu_config, "r", encoding="utf-8") as handle:
        imu_cfg = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imu_n_time_steps = int(imu_cfg["data"]["n_time_steps"])
    fusion_config = build_fusion_config(cfg)

    imu_model = load_imu_model(PROJECT_ROOT / args.imu_config, PROJECT_ROOT / args.imu_ckpt, device)
    pdr_model = PDR.from_yaml(PROJECT_ROOT / args.pdr_config)

    path_rows = evaluate_paths(
        args=args,
        cfg=cfg,
        fusion_config=fusion_config,
        imu_model=imu_model,
        pdr_model=pdr_model,
        imu_n_time_steps=imu_n_time_steps,
        device=device,
    )
    if not path_rows:
        raise ValueError("No holdout paths were evaluated.")

    path_df = pd.DataFrame([row.__dict__ for row in path_rows]).sort_values(["site_id", "path_id"]).reset_index(drop=True)
    sweep_df = sweep_thresholds(path_rows, args.thresholds, args.metric)

    path_out = PROJECT_ROOT / args.path_out
    sweep_out = PROJECT_ROOT / args.sweep_out
    summary_out = PROJECT_ROOT / args.summary_out
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_df.to_csv(path_out, index=False)
    sweep_df.to_csv(sweep_out, index=False)

    best = sweep_df.iloc[0].to_dict()
    summary = {
        "status": "ok",
        "elapsed_s": time.time() - t0,
        "metric": args.metric,
        "thresholds": list(args.thresholds),
        "site_ids": sorted(path_df["site_id"].unique().tolist()),
        "n_paths": int(len(path_df)),
        "n_points": int(path_df["n_points"].sum()),
        "best": best,
        "environment": get_environment_snapshot(),
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved path metrics: {path_out}")
    print(f"Saved sweep      : {sweep_out}")
    print(f"Saved summary    : {summary_out}")
    print()
    print(path_df.to_string(index=False))
    print()
    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
