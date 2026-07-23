from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.interpolated_wifi_source_reanchoring import (
    _path_waypoints,
    build_path_ledger,
    interpolate_scan_positions,
    load_config,
    resolve_path,
    resolve_runtime_train_root,
)


def interpolate_predictions(
    scan_timestamps: np.ndarray,
    scan_xy: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    if len(scan_timestamps) < 2:
        raise ValueError("At least two scan predictions are required for time interpolation.")
    ordered = np.argsort(scan_timestamps)
    timestamps = np.asarray(scan_timestamps, dtype=np.int64)[ordered]
    coordinates = np.asarray(scan_xy, dtype=np.float64)[ordered]
    targets = np.asarray(target_timestamps, dtype=np.int64)
    return np.column_stack(
        [np.interp(targets, timestamps, coordinates[:, axis]) for axis in range(2)]
    )


def fit_dense_scan_knn(
    train_vectors: np.ndarray, train_xy: np.ndarray, neighbors: int
) -> NearestNeighbors:
    if len(train_vectors) < neighbors:
        raise ValueError("Dense scan training rows are fewer than configured neighbors.")
    model = NearestNeighbors(n_neighbors=neighbors, metric="cosine", algorithm="brute")
    model.fit(train_vectors)
    return model


def predict_dense_scan_xy(
    model: NearestNeighbors, train_xy: np.ndarray, query_vectors: np.ndarray
) -> np.ndarray:
    distances, indices = model.kneighbors(query_vectors, return_distance=True)
    weights = 1.0 / np.maximum(distances, 1e-6)
    return (train_xy[indices] * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)


def _scan_rows(path_files: list[Path], vocab: list[str], missing_value: float):
    from scripts.analyze_hard_site_candidate_recall import _extract_basic_wifi_matrix
    from src.io_f import read_data_file

    vectors, coordinates, path_ids = [], [], []
    for path in path_files:
        sensor = read_data_file(path)
        waypoints = _path_waypoints(sensor)
        if len(waypoints) < 2 or sensor.wifi is None or sensor.wifi.empty:
            continue
        timestamps = np.sort(sensor.wifi["timestamp"].astype(np.int64).unique())
        labels = interpolate_scan_positions(timestamps, waypoints)
        if labels.empty:
            continue
        kept_timestamps = labels["wifi_timestamp"].to_numpy(dtype=np.int64)
        vectors.append(_extract_basic_wifi_matrix(sensor.wifi, kept_timestamps, vocab, 0, missing_value))
        coordinates.append(labels[["interpolated_x", "interpolated_y"]].to_numpy(dtype=np.float64))
        path_ids.extend([path.stem] * len(labels))
    if not vectors:
        raise ValueError("No dense scan training rows were constructed.")
    return np.vstack(vectors), np.vstack(coordinates), path_ids


def attach_dense_scan_unary(candidates: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    from scripts.analyze_hard_site_candidate_recall import _extract_basic_wifi_matrix
    from scripts.diagnose_beam_search_validation import build_vocab_from_files, list_site_paths
    from src.io_f import read_data_file

    train_root = resolve_runtime_train_root(config, Path("/kaggle/input").is_dir())
    data = candidates.copy()
    feature_rows, audits = [], []
    settings = config["data"]
    for site_id, site_data in data.groupby("site_id", sort=True):
        heldout_ids = set(site_data["path_id"].astype(str).unique())
        all_paths = list_site_paths(train_root / str(site_id))
        train_files = [path for path in all_paths if path.stem not in heldout_ids]
        heldout_files = [path for path in all_paths if path.stem in heldout_ids]
        if {path.stem for path in heldout_files} != heldout_ids:
            raise ValueError(f"Heldout path coverage is incomplete for site {site_id}.")
        vocab = build_vocab_from_files(train_files, n_bssid=int(settings["n_bssid"]))
        train_vectors, train_xy, _ = _scan_rows(train_files, vocab, float(settings["missing_wifi_rssi"]))
        model = fit_dense_scan_knn(train_vectors, train_xy, int(settings["knn_neighbors"]))
        predicted_by_path = {}
        for path in heldout_files:
            sensor = read_data_file(path)
            timestamps = np.sort(sensor.wifi["timestamp"].astype(np.int64).unique())
            vectors = _extract_basic_wifi_matrix(sensor.wifi, timestamps, vocab, 0, float(settings["missing_wifi_rssi"]))
            predicted_by_path[path.stem] = (timestamps, predict_dense_scan_xy(model, train_xy, vectors))
        for group_id, group in site_data.groupby("group_id", sort=False):
            path_id = str(group["path_id"].iloc[0])
            timestamp = int(group["timestamp"].iloc[0])
            scan_timestamps, scan_xy = predicted_by_path[path_id]
            predicted_xy = interpolate_predictions(scan_timestamps, scan_xy, np.array([timestamp]))[0]
            distances = np.linalg.norm(group[["candidate_x", "candidate_y"]].to_numpy(dtype=np.float64) - predicted_xy, axis=1)
            for row, distance, rank in zip(group.itertuples(index=False), distances, np.argsort(np.argsort(distances)) + 1):
                feature_rows.append({"group_id": str(group_id), "candidate_index": int(row.candidate_index), "dense_scan_unary_distance_m": float(distance), "dense_scan_unary_rank": int(rank)})
        audits.append({"site_id": str(site_id), "heldout_paths": sorted(heldout_ids), "path_overlap_count": 0, "dense_train_scans": int(len(train_vectors)), "dense_train_paths": int(len(train_files))})
    features = pd.DataFrame(feature_rows)
    if len(features) != len(data) or features.duplicated(["group_id", "candidate_index"]).any():
        raise ValueError("Dense scan unary feature coverage is incomplete.")
    return data.drop(columns=[column for column in ["dense_scan_unary_distance_m", "dense_scan_unary_rank"] if column in data]).merge(features, on=["group_id", "candidate_index"], how="left"), audits


def run_kaggle_probe(config: dict):
    from data_processing.candidate_transition_lattice import run as run_lattice

    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Dense scan probe may run only inside Kaggle.")
    candidates = pd.read_csv(resolve_path(config["inputs"]["candidate_dataset"]))
    enriched, audit = attach_dense_scan_unary(candidates, config)
    candidate_path = Path(config["outputs"]["candidate_dataset"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(candidate_path, index=False)
    lattice_config = config["lattice"]
    selections, lattice_report = run_lattice(lattice_config)
    selections.to_csv(config["outputs"]["lattice_selections"], index=False)
    with Path(config["outputs"]["lattice_summary"]).open("w", encoding="utf-8") as handle:
        json.dump(lattice_report, handle, indent=2)
    metrics = lattice_report
    report = {"status": "ok", "version": config["version"], "experiment_type": config["experiment_type"], "iteration_mode": config["iteration_mode"], "seed": config["seed"], "coverage": {"paths": int(selections["path_id"].nunique()), "groups": int(selections["group_id"].nunique()), "path_overlap_count": 0, "site_feature_audit": audit}, "metrics": {"baseline_lattice_mae_m": config["validation_gate"]["baseline_lattice_mae_m"], "dense_scan_lattice_mae_m": metrics["structured_lattice_mae_m"], "baseline_sum_excess_over_4m": config["validation_gate"]["baseline_sum_excess_over_4m"], "dense_scan_sum_excess_over_4m": metrics["structured_sum_excess_over_4m"]}, "path_ledger": build_path_ledger(selections), "lattice": lattice_report, "submission_created": False}
    report["gene_gain_passed"] = report["metrics"]["dense_scan_lattice_mae_m"] < report["metrics"]["baseline_lattice_mae_m"] and report["metrics"]["dense_scan_sum_excess_over_4m"] < report["metrics"]["baseline_sum_excess_over_4m"]
    with Path(config["outputs"]["report"]).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["kaggle"], required=True)
    args = parser.parse_args()
    print(json.dumps(run_kaggle_probe(load_config(resolve_path(args.config))), indent=2))


if __name__ == "__main__":
    main()
