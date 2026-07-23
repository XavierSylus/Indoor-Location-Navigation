from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

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
    floor_str_to_num,
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
class PreparedTrajectory:
    site_id: str
    path_id: str
    path_file: Path
    timestamps: np.ndarray
    gt_xy: np.ndarray
    gt_floor_nums: np.ndarray
    pred_xy: np.ndarray
    pred_floor_str: List[str]
    pred_floor_nums: np.ndarray
    dijkstra_data: Dict[str, tuple[np.ndarray, np.ndarray]]
    ensemble_deltas: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search alpha/beta_1/beta_2 for safe Beam Search.")
    parser.add_argument("--config", default="configs/kaggle_train_config.yml")
    parser.add_argument("--imu-config", default="configs/imu_delta_model_v3.yml")
    parser.add_argument("--imu-ckpt", default="models/imu_delta_v3/imu_delta_best.pt")
    parser.add_argument("--pdr-config", default="configs/pdr_config.yml")
    parser.add_argument("--site-list", required=True,
                        help="Comma-separated strict holdout sample sites.")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--max-holdout-paths", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connect-radius", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=2000)
    parser.add_argument("--n-candidates", type=int, default=100)
    parser.add_argument("--delta-weight-imu", type=float, default=0.67)
    parser.add_argument("--delta-weight-pdr", type=float, default=0.33)
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=6.0)
    parser.add_argument("--beta1-min", type=float, default=0.0)
    parser.add_argument("--beta1-max", type=float, default=2.0)
    parser.add_argument("--beta2-min", type=float, default=0.0)
    parser.add_argument("--beta2-max", type=float, default=4.0)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--results-out", default="data_processing/processed/safe_beam_param_search_results.csv")
    parser.add_argument("--summary-out", default="data_processing/processed/safe_beam_param_search_summary.json")
    return parser.parse_args()


def prepare_trajectories(
    args: argparse.Namespace,
    cfg: dict,
    fusion_config,
    imu_model,
    pdr_model,
    imu_n_time_steps: int,
    device: torch.device,
) -> List[PreparedTrajectory]:
    train_dir = PROJECT_ROOT / cfg["paths"]["train_dir"]
    site_ids = [site.strip() for site in args.site_list.split(",") if site.strip()]
    prepared: List[PreparedTrajectory] = []

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
            timestamps = path_df["timestamp"].to_numpy(dtype=np.int64)
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
            prepared.append(
                PreparedTrajectory(
                    site_id=site_id,
                    path_id=path_file.stem,
                    path_file=path_file,
                    timestamps=timestamps,
                    gt_xy=path_df[["x", "y"]].to_numpy(dtype=np.float64),
                    gt_floor_nums=path_df["floor"].map(floor_str_to_num).to_numpy(dtype=np.int64),
                    pred_xy=path_df[["pred_x", "pred_y"]].to_numpy(dtype=np.float64),
                    pred_floor_str=path_df["pred_floor_str"].astype(str).tolist(),
                    pred_floor_nums=path_df["pred_floor_num"].to_numpy(dtype=np.int64),
                    dijkstra_data=bundle.dijkstra_data,
                    ensemble_deltas=ensemble_deltas,
                )
            )

    return prepared


def evaluate_trial(
    trajectories: Sequence[PreparedTrajectory],
    args: argparse.Namespace,
    alpha: float,
    beta_1: float,
    beta_2: float,
) -> dict:
    wifi_error_sum = 0.0
    beam_error_sum = 0.0
    total_points = 0
    floor_correct = 0
    floor_total = 0

    for traj in trajectories:
        wifi_error_sum += float(np.linalg.norm(traj.pred_xy - traj.gt_xy, axis=1).sum())
        floor_correct += int((traj.pred_floor_nums == traj.gt_floor_nums).sum())
        floor_total += len(traj.pred_floor_nums)

        opt_xy = beam_search_single_trajectory(
            init_xy=traj.pred_xy,
            init_floors=traj.pred_floor_str,
            imu_deltas=traj.ensemble_deltas,
            dijkstra_data=traj.dijkstra_data,
            beam_width=args.beam_width,
            n_candidates=args.n_candidates,
            cost_mode="safe_rank2",
            alpha=alpha,
            beta_1=beta_1,
            beta_2=beta_2,
            use_dijkstra=False,
        )
        beam_error_sum += float(np.linalg.norm(opt_xy - traj.gt_xy, axis=1).sum())
        total_points += len(traj.gt_xy)

    return {
        "alpha": alpha,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "floor_accuracy": floor_correct / max(floor_total, 1),
        "wifi_mae_m": wifi_error_sum / max(total_points, 1),
        "beam_mae_m": beam_error_sum / max(total_points, 1),
        "beam_minus_wifi_m": (beam_error_sum - wifi_error_sum) / max(total_points, 1),
        "n_points": int(total_points),
    }


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

    trajectories = prepare_trajectories(
        args=args,
        cfg=cfg,
        fusion_config=fusion_config,
        imu_model=imu_model,
        pdr_model=pdr_model,
        imu_n_time_steps=imu_n_time_steps,
        device=device,
    )
    if not trajectories:
        raise ValueError("No holdout trajectories prepared for search.")

    rng = np.random.default_rng(args.seed)
    rows = []
    for trial_idx in range(1, args.n_trials + 1):
        alpha = float(rng.uniform(args.alpha_min, args.alpha_max))
        beta_1 = float(rng.uniform(args.beta1_min, args.beta1_max))
        beta_2 = float(rng.uniform(args.beta2_min, args.beta2_max))
        row = evaluate_trial(trajectories, args, alpha, beta_1, beta_2)
        row["trial"] = trial_idx
        rows.append(row)
        print(
            f"[{trial_idx:02d}/{args.n_trials}] alpha={alpha:.4f} beta_1={beta_1:.4f} beta_2={beta_2:.4f} "
            f"-> beam={row['beam_mae_m']:.4f}m delta={row['beam_minus_wifi_m']:+.4f}m"
        )

    results = pd.DataFrame(rows).sort_values("beam_mae_m").reset_index(drop=True)
    results_out = PROJECT_ROOT / args.results_out
    summary_out = PROJECT_ROOT / args.summary_out
    results_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_out, index=False)

    best = results.iloc[0].to_dict()
    summary = {
        "status": "ok",
        "elapsed_s": time.time() - t0,
        "n_trials": args.n_trials,
        "n_trajectories": len(trajectories),
        "n_points": int(sum(len(traj.gt_xy) for traj in trajectories)),
        "site_ids": sorted({traj.site_id for traj in trajectories}),
        "best": best,
        "environment": get_environment_snapshot(),
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"Saved results : {results_out}")
    print(f"Saved summary : {summary_out}")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
