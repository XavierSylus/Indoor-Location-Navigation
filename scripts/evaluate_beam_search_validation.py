from __future__ import annotations

import argparse
import gc
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models_v3 import load_site_model_v3
from scripts.step3_infer_and_optimize import (
    FLOOR_MAP,
    beam_search_single_trajectory,
    build_site_dijkstra,
    load_imu_model,
    predict_ensemble_deltas_for_path,
)
from data_processing.pdr_module import PDR


@dataclass
class TrajectorySample:
    site_id: str
    floor_str: str
    path_id: str
    path_file: Path
    timestamps: np.ndarray
    gt_xy: np.ndarray
    init_xy: np.ndarray
    init_floors: List[str]
    ensemble_deltas: np.ndarray | None


def parse_waypoints(path_file: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    floor_str = path_file.parent.name
    waypoints = []
    with open(path_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[1] == "TYPE_WAYPOINT":
                try:
                    waypoints.append((int(parts[0]), float(parts[2]), float(parts[3])))
                except (TypeError, ValueError):
                    continue
    if len(waypoints) < 2:
        return floor_str, np.array([], dtype=np.int64), np.zeros((0, 2), dtype=np.float64)

    waypoints.sort(key=lambda item: item[0])
    timestamps = np.array([item[0] for item in waypoints], dtype=np.int64)
    gt_xy = np.array([[item[1], item[2]] for item in waypoints], dtype=np.float64)
    return floor_str, timestamps, gt_xy


def infer_train_path_with_model(model, path_file: Path, timestamps: np.ndarray) -> Tuple[List[str], np.ndarray]:
    features_dict = model.fusion.extract_from_file(path_file, timestamps)
    x_test = model.fusion.fuse_features(features_dict, preprocess=True)
    floors, xy = model.predict(x_test)
    return [str(floor) for floor in floors], np.asarray(xy, dtype=np.float64)


def floor_strings_to_submission_ints(floor_strings: Sequence[str]) -> np.ndarray:
    return np.array([FLOOR_MAP.get(f, 0) if f in FLOOR_MAP else 0 for f in floor_strings], dtype=np.int64)


def euclidean_mae(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred_xy - gt_xy, axis=1)))


def choose_validation_paths(
    train_dir: Path,
    available_sites: Sequence[str],
    paths_per_site: int,
    seed: int,
) -> Dict[str, List[Path]]:
    rng = np.random.default_rng(seed)
    selected: Dict[str, List[Path]] = {}
    for site_id in available_sites:
        site_dir = train_dir / site_id
        txt_files = sorted(site_dir.rglob("*.txt"))
        if not txt_files:
            continue
        n_pick = min(paths_per_site, len(txt_files))
        chosen_idx = np.sort(rng.choice(len(txt_files), size=n_pick, replace=False))
        selected[site_id] = [txt_files[idx] for idx in chosen_idx]
    return selected


def prepare_samples_for_site(
    site_id: str,
    path_files: Sequence[Path],
    model_dir: Path,
    imu_model,
    pdr_model: PDR,
    device,
    imu_n_time_steps: int,
    delta_weights: Tuple[float, float],
) -> List[TrajectorySample]:
    pkl_file = model_dir / f"{site_id}_v3.pkl"
    if not pkl_file.exists():
        return []

    model = load_site_model_v3(pkl_file)
    samples: List[TrajectorySample] = []
    for path_file in path_files:
        floor_str, timestamps, gt_xy = parse_waypoints(path_file)
        if len(timestamps) < 2:
            continue

        init_floors, init_xy = infer_train_path_with_model(model, path_file, timestamps)
        ensemble_deltas = predict_ensemble_deltas_for_path(
            imu_model=imu_model,
            pdr_model=pdr_model,
            path_file=path_file,
            timestamps=timestamps.tolist(),
            floor_str=floor_str,
            site_id=site_id,
            device=device,
            imu_weight=delta_weights[0],
            pdr_weight=delta_weights[1],
            n_time_steps=imu_n_time_steps,
        )
        samples.append(
            TrajectorySample(
                site_id=site_id,
                floor_str=floor_str,
                path_id=path_file.stem,
                path_file=path_file,
                timestamps=timestamps,
                gt_xy=gt_xy,
                init_xy=init_xy,
                init_floors=init_floors,
                ensemble_deltas=ensemble_deltas,
            )
        )
    del model
    gc.collect()
    return samples


def evaluate_weight_grid(
    samples_by_site: Dict[str, List[TrajectorySample]],
    train_dir: Path,
    connect_radius: float,
    beam_width: int,
    n_candidates: int,
    weight_grid: List[Tuple[float, float, float, float]],
) -> pd.DataFrame:
    rows = []
    total_paths = sum(len(v) for v in samples_by_site.values())
    print(f"Evaluating {total_paths} validation trajectories across {len(weight_grid)} weight sets")

    dijkstra_cache = {}
    baseline_num = 0.0
    baseline_den = 0
    for site_samples in samples_by_site.values():
        for sample in site_samples:
            baseline_num += np.linalg.norm(sample.init_xy - sample.gt_xy, axis=1).sum()
            baseline_den += len(sample.gt_xy)
    baseline_mae = baseline_num / max(baseline_den, 1)

    for site_id, site_samples in samples_by_site.items():
        dijkstra_cache[site_id] = build_site_dijkstra(train_dir, site_id, connect_radius)

    print(f"Initial absolute-coordinate MAE on holdout paths: {baseline_mae:.6f} m")

    for idx, weights in enumerate(weight_grid, 1):
        w_wifi, w_imu_l2, w_imu_angle, w_dijkstra = weights
        err_sum = 0.0
        n_points = 0
        t0 = time.time()

        for site_id, site_samples in samples_by_site.items():
            dijkstra_data = dijkstra_cache[site_id]
            for sample in site_samples:
                opt_xy = beam_search_single_trajectory(
                    init_xy=sample.init_xy,
                    init_floors=sample.init_floors,
                    imu_deltas=sample.ensemble_deltas,
                    dijkstra_data=dijkstra_data,
                    beam_width=beam_width,
                    n_candidates=n_candidates,
                    w_wifi=w_wifi,
                    w_imu_l2=w_imu_l2,
                    w_imu_angle=w_imu_angle,
                    w_dijkstra=w_dijkstra,
                )
                err_sum += np.linalg.norm(opt_xy - sample.gt_xy, axis=1).sum()
                n_points += len(sample.gt_xy)

        mae = err_sum / max(n_points, 1)
        rows.append(
            {
                "w_wifi": w_wifi,
                "w_imu_l2": w_imu_l2,
                "w_imu_angle": w_imu_angle,
                "w_dijkstra": w_dijkstra,
                "baseline_mae_m": baseline_mae,
                "beam_mae_m": mae,
                "delta_improvement_m": baseline_mae - mae,
                "elapsed_s": time.time() - t0,
            }
        )
        print(
            f"[{idx}/{len(weight_grid)}] "
            f"wifi={w_wifi}, imu_l2={w_imu_l2}, imu_angle={w_imu_angle}, dijkstra={w_dijkstra} "
            f"-> beam_mae={mae:.6f} m"
        )

    return pd.DataFrame(rows).sort_values("beam_mae_m").reset_index(drop=True)


def build_weight_grid(args) -> List[Tuple[float, float, float, float]]:
    return list(
        itertools.product(
            args.w_wifi,
            args.w_imu_l2,
            args.w_imu_angle,
            args.w_dijkstra,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Beam Search on holdout train trajectories.")
    parser.add_argument("--config", default="configs/kaggle_train_config.yml")
    parser.add_argument("--imu-config", default="configs/imu_delta_model_v3.yml")
    parser.add_argument("--imu-ckpt", default="models/imu_delta_v3/imu_delta_best.pt")
    parser.add_argument("--pdr-config", default="configs/pdr_config.yml")
    parser.add_argument("--paths-per-site", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connect-radius", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=800)
    parser.add_argument("--n-candidates", type=int, default=60)
    parser.add_argument("--delta-weight-imu", type=float, default=0.67)
    parser.add_argument("--delta-weight-pdr", type=float, default=0.33)
    parser.add_argument("--w-wifi", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--w-imu-l2", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--w-imu-angle", nargs="+", type=float, default=[0.3, 0.8])
    parser.add_argument("--w-dijkstra", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    parser.add_argument("--out", default="data_processing/processed/beam_search_validation_results.csv")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(PROJECT_ROOT / args.imu_config, "r", encoding="utf-8") as handle:
        imu_cfg = yaml.safe_load(handle)

    train_dir = PROJECT_ROOT / cfg["paths"]["train_dir"]
    model_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available_sites = sorted(
        p.stem.replace("_v3", "")
        for p in model_dir.glob("*_v3.pkl")
    )
    selected = choose_validation_paths(
        train_dir=train_dir,
        available_sites=available_sites,
        paths_per_site=args.paths_per_site,
        seed=args.seed,
    )
    total_paths = sum(len(paths) for paths in selected.values())
    print(f"Selected {total_paths} holdout trajectories from {len(selected)} sites")

    imu_model = load_imu_model(PROJECT_ROOT / args.imu_config, PROJECT_ROOT / args.imu_ckpt, device)
    pdr_model = PDR.from_yaml(PROJECT_ROOT / args.pdr_config)

    samples_by_site: Dict[str, List[TrajectorySample]] = {}
    for idx, (site_id, path_files) in enumerate(selected.items(), 1):
        site_samples = prepare_samples_for_site(
            site_id=site_id,
            path_files=path_files,
            model_dir=model_dir,
            imu_model=imu_model,
            pdr_model=pdr_model,
            device=device,
            imu_n_time_steps=int(imu_cfg["data"]["n_time_steps"]),
            delta_weights=(args.delta_weight_imu, args.delta_weight_pdr),
        )
        if site_samples:
            samples_by_site[site_id] = site_samples
        print(f"[{idx}/{len(selected)}] {site_id}: {len(site_samples)} trajectories ready")

    weight_grid = build_weight_grid(args)
    results = evaluate_weight_grid(
        samples_by_site=samples_by_site,
        train_dir=train_dir,
        connect_radius=args.connect_radius,
        beam_width=args.beam_width,
        n_candidates=args.n_candidates,
        weight_grid=weight_grid,
    )

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    import torch

    main()
