from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "interpolated_wifi_source_reanchoring.json"
SOURCE_FEATURE_COLUMNS = [
    "candidate_wifi_source_count",
    "candidate_wifi_best_source_score",
    "candidate_wifi_top3_source_score_mean",
    "candidate_wifi_top3_source_score_max",
    "candidate_wifi_top5_source_score_mean",
    "candidate_wifi_source_score_std",
    "candidate_wifi_source_score_p90",
    "candidate_wifi_best_source_l1",
    "candidate_wifi_top3_source_l1_mean",
    "candidate_wifi_best_source_cosine",
    "candidate_wifi_top3_source_cosine_mean",
    "candidate_wifi_best_common_bssid_count",
    "candidate_wifi_top3_common_bssid_mean",
    "source_wifi_validation_visible_bssid_count",
    "source_wifi_has_common_bssid",
]
SOURCE_RANK_SPECS = [
    (
        "source_wifi_best_score_rank_within_point",
        "candidate_wifi_best_source_score",
        False,
    ),
    (
        "source_wifi_top3_score_rank_within_point",
        "candidate_wifi_top3_source_score_mean",
        False,
    ),
    (
        "source_wifi_best_l1_rank_within_point",
        "candidate_wifi_best_source_l1",
        True,
    ),
    (
        "source_wifi_best_cosine_rank_within_point",
        "candidate_wifi_best_source_cosine",
        False,
    ),
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def parse_path_timestamps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    wifi_timestamps = set()
    waypoint_rows = []
    with path.open("rb") as handle:
        for line in handle:
            parts = line.rstrip().split(b"\t")
            if len(parts) < 2:
                continue
            if parts[1] == b"TYPE_WIFI" and len(parts) >= 5:
                wifi_timestamps.add(int(parts[0]))
            elif parts[1] == b"TYPE_WAYPOINT" and len(parts) >= 4:
                waypoint_rows.append(
                    (int(parts[0]), float(parts[2]), float(parts[3]))
                )
    wifi = np.asarray(sorted(wifi_timestamps), dtype=np.int64)
    waypoint = np.asarray(sorted(waypoint_rows), dtype=np.float64)
    if waypoint.size == 0:
        waypoint = np.empty((0, 3), dtype=np.float64)
    return wifi, waypoint


def nearest_indices(sorted_reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    reference = np.asarray(sorted_reference)
    values = np.asarray(queries)
    if reference.ndim != 1 or values.ndim != 1:
        raise ValueError("nearest_indices expects one-dimensional arrays.")
    if len(reference) == 0:
        raise ValueError("Reference timestamps must not be empty.")
    right = np.searchsorted(reference, values, side="left")
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    choose_right = np.abs(reference[right] - values) < np.abs(reference[left] - values)
    return np.where(choose_right, right, left).astype(np.int64)


def interpolate_scan_positions(
    wifi_timestamps: np.ndarray,
    waypoint_rows: np.ndarray,
) -> pd.DataFrame:
    wifi = np.asarray(wifi_timestamps, dtype=np.int64)
    waypoints = np.asarray(waypoint_rows, dtype=np.float64)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoint_rows must have shape (n, 3).")
    if len(waypoints) < 2 or len(wifi) == 0:
        return pd.DataFrame(
            columns=[
                "wifi_timestamp",
                "nearest_waypoint_timestamp",
                "time_gap_ms",
                "nearest_x",
                "nearest_y",
                "interpolated_x",
                "interpolated_y",
                "displacement_m",
            ]
        )
    order = np.argsort(waypoints[:, 0], kind="stable")
    waypoints = waypoints[order]
    waypoint_timestamps = waypoints[:, 0].astype(np.int64)
    in_range = (wifi >= waypoint_timestamps[0]) & (wifi <= waypoint_timestamps[-1])
    eligible = wifi[in_range]
    if len(eligible) == 0:
        return interpolate_scan_positions(np.empty(0, dtype=np.int64), waypoints[:1])
    nearest = nearest_indices(waypoint_timestamps, eligible)
    interpolated_x = np.interp(eligible, waypoint_timestamps, waypoints[:, 1])
    interpolated_y = np.interp(eligible, waypoint_timestamps, waypoints[:, 2])
    nearest_xy = waypoints[nearest, 1:3]
    interpolated_xy = np.column_stack([interpolated_x, interpolated_y])
    displacement = np.linalg.norm(interpolated_xy - nearest_xy, axis=1)
    return pd.DataFrame(
        {
            "wifi_timestamp": eligible,
            "nearest_waypoint_timestamp": waypoint_timestamps[nearest],
            "time_gap_ms": np.abs(eligible - waypoint_timestamps[nearest]),
            "nearest_x": nearest_xy[:, 0],
            "nearest_y": nearest_xy[:, 1],
            "interpolated_x": interpolated_xy[:, 0],
            "interpolated_y": interpolated_xy[:, 1],
            "displacement_m": displacement,
        }
    )


def assign_interpolated_to_grid(
    interpolated_xy: np.ndarray, grid_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(interpolated_xy, dtype=np.float64)
    grid = np.asarray(grid_xy, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("interpolated_xy must have shape (n, 2).")
    if grid.ndim != 2 or grid.shape[1] != 2 or len(grid) == 0:
        raise ValueError("grid_xy must be a non-empty array with shape (n, 2).")
    distances, indices = cKDTree(grid).query(positions, k=1)
    return indices.astype(np.int64), distances.astype(np.float64)


def current_binding_stats(
    wifi_timestamps: np.ndarray,
    waypoint_timestamps: np.ndarray,
    window_ms: int,
) -> dict:
    wifi = np.asarray(wifi_timestamps, dtype=np.int64)
    waypoints = np.asarray(waypoint_timestamps, dtype=np.int64)
    if len(wifi) == 0 or len(waypoints) == 0:
        return {
            "waypoint_assignments": 0,
            "unique_scan_assignments": 0,
            "duplicate_binding_ratio": 0.0,
            "unmatched_waypoints": int(len(waypoints)),
        }
    nearest = nearest_indices(wifi, waypoints)
    gaps = np.abs(wifi[nearest] - waypoints)
    matched_scans = wifi[nearest[gaps <= int(window_ms)]]
    assignment_count = int(len(matched_scans))
    unique_count = int(len(np.unique(matched_scans)))
    duplicate_ratio = (
        0.0
        if assignment_count == 0
        else float((assignment_count - unique_count) / assignment_count)
    )
    return {
        "waypoint_assignments": assignment_count,
        "unique_scan_assignments": unique_count,
        "duplicate_binding_ratio": duplicate_ratio,
        "unmatched_waypoints": int(np.sum(gaps > int(window_ms))),
    }


def summarize_path(path: Path, window_ms: int) -> tuple[dict, pd.DataFrame]:
    wifi, waypoints = parse_path_timestamps(path)
    interpolated = interpolate_scan_positions(wifi, waypoints)
    binding = current_binding_stats(
        wifi,
        waypoints[:, 0].astype(np.int64) if len(waypoints) else np.empty(0, dtype=np.int64),
        window_ms,
    )
    displacement = interpolated["displacement_m"].to_numpy(dtype=np.float64)
    row = {
        "site_id": path.parents[1].name,
        "floor": path.parent.name,
        "path_id": path.stem,
        "n_wifi_scans": int(len(wifi)),
        "n_waypoints": int(len(waypoints)),
        "n_interpolated_scans": int(len(interpolated)),
        "interpolation_coverage": float(len(interpolated) / max(len(wifi), 1)),
        "mean_displacement_m": float(np.mean(displacement)) if len(displacement) else 0.0,
        "median_displacement_m": float(np.median(displacement)) if len(displacement) else 0.0,
        "p90_displacement_m": float(np.quantile(displacement, 0.9))
        if len(displacement)
        else 0.0,
        "fraction_over_1m": float(np.mean(displacement > 1.0))
        if len(displacement)
        else 0.0,
        **binding,
    }
    return row, interpolated


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def list_source_paths(config: dict) -> Iterable[Path]:
    train_root = resolve_path(config["inputs"]["train_root"])
    heldout = {str(value) for value in config["data"]["heldout_path_ids"]}
    for site_id in config["data"]["target_site_ids"]:
        site_root = train_root / str(site_id)
        for path in sorted(site_root.rglob("*.txt")):
            if path.stem not in heldout:
                yield path


def run_preflight(config: dict) -> tuple[pd.DataFrame, dict]:
    window_ms = int(config["data"]["wifi_window_ms"])
    path_rows = []
    displacement_blocks = []
    for path in list_source_paths(config):
        row, scans = summarize_path(path, window_ms)
        path_rows.append(row)
        if not scans.empty:
            displacement_blocks.append(
                scans[["displacement_m", "time_gap_ms"]].assign(
                    site_id=row["site_id"],
                    path_id=row["path_id"],
                )
            )
    if not path_rows or not displacement_blocks:
        raise ValueError("No eligible WiFi scans were found for the v011 preflight.")
    path_metrics = pd.DataFrame(path_rows)
    scans = pd.concat(displacement_blocks, ignore_index=True)
    displacement = scans["displacement_m"].to_numpy(dtype=np.float64)
    p90 = float(np.quantile(displacement, 0.9))
    fraction_over_1m = float(np.mean(displacement > 1.0))
    gate = config["preflight_gate"]
    passed = p90 >= float(gate["minimum_p90_displacement_m"]) and fraction_over_1m >= float(
        gate["minimum_fraction_over_1m"]
    )
    site_rows = []
    for site_id, site_scans in scans.groupby("site_id", sort=True):
        values = site_scans["displacement_m"].to_numpy(dtype=np.float64)
        site_paths = path_metrics[path_metrics["site_id"].astype(str) == str(site_id)]
        site_rows.append(
            {
                "site_id": str(site_id),
                "n_paths": int(site_paths["path_id"].nunique()),
                "n_interpolated_scans": int(len(values)),
                "mean_displacement_m": float(np.mean(values)),
                "median_displacement_m": float(np.median(values)),
                "p90_displacement_m": float(np.quantile(values, 0.9)),
                "fraction_over_1m": float(np.mean(values > 1.0)),
                "mean_duplicate_binding_ratio": float(
                    site_paths["duplicate_binding_ratio"].mean()
                ),
            }
        )
    report = {
        "status": "ok",
        "version": config["version"],
        "experiment_type": config["experiment_type"],
        "iteration_mode": config["iteration_mode"],
        "diagnostic_only": True,
        "submission_safe": False,
        "seed": int(config["seed"]),
        "git_commit": git_commit(),
        "coverage": {
            "target_sites": int(path_metrics["site_id"].nunique()),
            "source_paths": int(path_metrics["path_id"].nunique()),
            "wifi_scans": int(path_metrics["n_wifi_scans"].sum()),
            "interpolated_scans": int(len(scans)),
            "heldout_path_overlap": int(
                len(
                    set(path_metrics["path_id"].astype(str))
                    & {str(value) for value in config["data"]["heldout_path_ids"]}
                )
            ),
        },
        "metrics": {
            "mean_displacement_m": float(np.mean(displacement)),
            "median_displacement_m": float(np.median(displacement)),
            "p90_displacement_m": p90,
            "fraction_over_1m": fraction_over_1m,
            "mean_time_gap_ms": float(scans["time_gap_ms"].mean()),
            "mean_duplicate_binding_ratio": float(
                path_metrics["duplicate_binding_ratio"].mean()
            ),
        },
        "site_metrics": site_rows,
        "preflight_gate": {
            **gate,
            "passed": bool(passed),
        },
        "decision": (
            "Promote interpolated WiFi source reanchoring to the Kaggle OOF reranker probe."
            if passed
            else "Reject interpolated WiFi source reanchoring before Kaggle training."
        ),
        "submission_created": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return path_metrics, report


def save_preflight(config: dict, path_metrics: pd.DataFrame, report: dict) -> None:
    metrics_path = resolve_path(config["outputs"]["preflight_path_metrics"])
    report_path = resolve_path(config["outputs"]["preflight_report"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    path_metrics.to_csv(metrics_path, index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def _coord_key(floor: str, x_value: float, y_value: float) -> tuple[str, float, float]:
    return (
        str(floor),
        float(np.round(float(x_value), 1)),
        float(np.round(float(y_value), 1)),
    )


def _path_waypoints(sensor_data) -> np.ndarray:
    if sensor_data.waypoint is None or sensor_data.waypoint.empty:
        return np.empty((0, 3), dtype=np.float64)
    waypoint = sensor_data.waypoint.sort_values("timestamp")
    return waypoint[["timestamp", "x", "y"]].to_numpy(dtype=np.float64)


def build_floor_grids(path_files: list[Path]) -> dict[str, np.ndarray]:
    from src.io_f import read_data_file

    floor_blocks: dict[str, list[np.ndarray]] = {}
    for path in path_files:
        waypoints = _path_waypoints(read_data_file(path))
        if len(waypoints):
            floor_blocks.setdefault(path.parent.name, []).append(waypoints[:, 1:3])
    return {
        floor: np.unique(np.round(np.vstack(blocks), 1), axis=0)
        for floor, blocks in floor_blocks.items()
    }


def build_reanchored_source_index(
    path_files: list[Path],
    bssid_vocab: list[str],
    missing_value: float,
) -> tuple[dict[tuple[str, float, float], np.ndarray], dict]:
    from scripts.analyze_hard_site_candidate_recall import (
        _extract_basic_wifi_matrix,
    )
    from src.io_f import read_data_file

    floor_grids = build_floor_grids(path_files)
    vector_blocks: dict[tuple[str, float, float], list[np.ndarray]] = {}
    assigned_scans = 0
    assignment_distances = []
    for path in path_files:
        sensor_data = read_data_file(path)
        waypoints = _path_waypoints(sensor_data)
        if (
            len(waypoints) < 2
            or sensor_data.wifi is None
            or sensor_data.wifi.empty
        ):
            continue
        wifi_timestamps = np.sort(
            sensor_data.wifi["timestamp"].astype(np.int64).unique()
        )
        interpolated = interpolate_scan_positions(wifi_timestamps, waypoints)
        if interpolated.empty:
            continue
        scan_timestamps = interpolated["wifi_timestamp"].to_numpy(dtype=np.int64)
        scan_vectors = _extract_basic_wifi_matrix(
            sensor_data.wifi,
            scan_timestamps,
            bssid_vocab,
            window_ms=0,
            missing_value=missing_value,
        )
        floor = path.parent.name
        grid = floor_grids.get(floor)
        if grid is None:
            raise ValueError(f"Missing training waypoint grid for floor {floor}.")
        indices, distances = assign_interpolated_to_grid(
            interpolated[["interpolated_x", "interpolated_y"]].to_numpy(
                dtype=np.float64
            ),
            grid,
        )
        for row_index, grid_index in enumerate(indices):
            key = _coord_key(floor, grid[grid_index, 0], grid[grid_index, 1])
            vector_blocks.setdefault(key, []).append(scan_vectors[row_index])
        assigned_scans += int(len(indices))
        assignment_distances.extend(distances.tolist())
    source_index = {
        key: np.vstack(vectors) for key, vectors in vector_blocks.items()
    }
    return source_index, {
        "source_paths": int(len(path_files)),
        "assigned_scans": assigned_scans,
        "indexed_candidate_coordinates": int(len(source_index)),
        "mean_scan_to_candidate_distance_m": float(np.mean(assignment_distances))
        if assignment_distances
        else 0.0,
        "p90_scan_to_candidate_distance_m": float(
            np.quantile(assignment_distances, 0.9)
        )
        if assignment_distances
        else 0.0,
    }


def _neutral_source_features(
    validation_vector: np.ndarray,
    n_bssid: int,
    source_wifi_topn: int,
    rssi_clip_min: float,
    rssi_clip_max: float,
    missing_value: float,
) -> dict:
    from scripts.analyze_hard_site_candidate_recall import (
        compute_source_row_wifi_similarity_features,
    )

    empty_source = np.full((1, n_bssid), missing_value, dtype=np.float32)
    features = compute_source_row_wifi_similarity_features(
        validation_vector,
        empty_source,
        source_wifi_topn,
        rssi_clip_min,
        rssi_clip_max,
        missing_value,
    )
    features["candidate_wifi_source_count"] = 0
    return features


def attach_reanchored_source_features(
    candidates: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, list[dict]]:
    from scripts.analyze_hard_site_candidate_recall import (
        compute_source_row_wifi_similarity_features,
        extract_basic_wifi_vectors_for_paths,
    )
    from scripts.diagnose_beam_search_validation import (
        build_vocab_from_files,
        list_site_paths,
    )

    data = candidates.copy()
    data["_row_id"] = np.arange(len(data), dtype=np.int64)
    train_root = resolve_path(config["inputs"]["train_root"])
    data_config = config["data"]
    n_bssid = int(data_config["n_bssid"])
    window_ms = int(data_config["wifi_window_ms"])
    source_wifi_topn = int(data_config["source_wifi_topn"])
    rssi_clip_min = float(data_config["rssi_clip_min"])
    rssi_clip_max = float(data_config["rssi_clip_max"])
    missing_value = float(data_config["missing_wifi_rssi"])
    all_feature_rows = []
    audit_rows = []
    for site_id, site_data in data.groupby("site_id", sort=True):
        heldout_ids = set(site_data["path_id"].astype(str).unique())
        all_paths = list_site_paths(train_root / str(site_id))
        train_files = [path for path in all_paths if path.stem not in heldout_ids]
        heldout_files = [path for path in all_paths if path.stem in heldout_ids]
        if {path.stem for path in heldout_files} != heldout_ids:
            raise ValueError(f"Heldout path coverage is incomplete for site {site_id}.")
        if {path.stem for path in train_files} & heldout_ids:
            raise ValueError(f"Heldout source leakage detected for site {site_id}.")
        bssid_vocab = build_vocab_from_files(train_files, n_bssid=n_bssid)
        source_index, audit = build_reanchored_source_index(
            train_files, bssid_vocab, missing_value
        )
        validation_meta, validation_vectors = extract_basic_wifi_vectors_for_paths(
            heldout_files,
            bssid_vocab,
            window_ms,
            missing_value,
        )
        validation_lookup = {
            (str(row.path_id), int(row.timestamp), int(row.point_index)): validation_vectors[
                row_index
            ]
            for row_index, row in enumerate(
                validation_meta.itertuples(index=False)
            )
        }
        indexed_candidates = 0
        for group_id, group in site_data.groupby("group_id", sort=False):
            first = group.iloc[0]
            validation_key = (
                str(first["path_id"]),
                int(first["timestamp"]),
                int(first["point_index"]),
            )
            validation_vector = validation_lookup.get(validation_key)
            if validation_vector is None:
                raise ValueError(f"Missing validation WiFi vector for {group_id}.")
            group_rows = []
            for _, row in group.iterrows():
                candidate_key = _coord_key(
                    str(row["floor"]),
                    float(row["candidate_x"]),
                    float(row["candidate_y"]),
                )
                source_vectors = source_index.get(candidate_key)
                if source_vectors is None:
                    features = _neutral_source_features(
                        validation_vector,
                        len(bssid_vocab),
                        source_wifi_topn,
                        rssi_clip_min,
                        rssi_clip_max,
                        missing_value,
                    )
                else:
                    indexed_candidates += 1
                    features = compute_source_row_wifi_similarity_features(
                        validation_vector,
                        source_vectors,
                        source_wifi_topn,
                        rssi_clip_min,
                        rssi_clip_max,
                        missing_value,
                    )
                features.update(
                    {
                        "_row_id": int(row["_row_id"]),
                        "_rank_sort": int(row["rank"]),
                    }
                )
                group_rows.append(features)
            group_features = pd.DataFrame(group_rows)
            for output_column, sort_column, ascending in SOURCE_RANK_SPECS:
                ranked = group_features.sort_values(
                    [sort_column, "_rank_sort", "_row_id"],
                    ascending=[ascending, True, True],
                    na_position="last",
                ).reset_index(drop=True)
                ranked[output_column] = np.arange(
                    1, len(ranked) + 1, dtype=np.int64
                )
                group_features = group_features.merge(
                    ranked[["_row_id", output_column]],
                    on="_row_id",
                    how="left",
                )
            all_feature_rows.extend(
                group_features.drop(columns=["_rank_sort"]).to_dict(
                    orient="records"
                )
            )
        audit_rows.append(
            {
                "site_id": str(site_id),
                "heldout_paths": sorted(heldout_ids),
                "path_overlap_count": 0,
                "candidate_rows": int(len(site_data)),
                "indexed_candidate_rows": int(indexed_candidates),
                "indexed_candidate_row_ratio": float(
                    indexed_candidates / max(len(site_data), 1)
                ),
                "bssid_vocab_size": int(len(bssid_vocab)),
                **audit,
            }
        )
    feature_data = pd.DataFrame(all_feature_rows)
    if len(feature_data) != len(data):
        raise ValueError("Reanchored source feature row coverage is incomplete.")
    replaced_columns = SOURCE_FEATURE_COLUMNS + [
        spec[0] for spec in SOURCE_RANK_SPECS
    ]
    data = data.drop(
        columns=[column for column in replaced_columns if column in data.columns]
    )
    data = data.merge(feature_data, on="_row_id", how="left")
    if data["candidate_wifi_best_source_score"].isna().any():
        raise ValueError("Reanchored candidate features contain missing scores.")
    return data.drop(columns=["_row_id"]).reset_index(drop=True), audit_rows


def build_path_ledger(selections: pd.DataFrame) -> list[dict]:
    rows = []
    for path_id, path in selections.groupby("path_id", sort=True):
        errors = path["label_distance_m"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "path_id": str(path_id),
                "n_groups": int(path["group_id"].nunique()),
                "lattice_mae_m": float(np.mean(errors)),
                "sum_excess_over_4m": float(
                    np.maximum(errors - 4.0, 0.0).sum()
                ),
            }
        )
    return rows


def run_kaggle_probe(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from data_processing.candidate_transition_lattice import run as run_lattice

    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("The v011 OOF probe may run only inside Kaggle.")
    candidates = pd.read_csv(resolve_path(config["inputs"]["candidate_dataset"]))
    reanchored, audit = attach_reanchored_source_features(candidates, config)
    candidate_path = Path(config["outputs"]["reanchored_candidate_dataset"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    reanchored.to_csv(candidate_path, index=False)
    lattice_config = copy.deepcopy(config["lattice"])
    lattice_config["inputs"]["candidate_dataset"] = str(candidate_path)
    selections, lattice_summary = run_lattice(lattice_config)
    path_ledger = build_path_ledger(selections)
    gate = config["validation_gate"]
    structured_mae = float(lattice_summary["structured_lattice_mae_m"])
    structured_excess = float(lattice_summary["structured_sum_excess_over_4m"])
    dominant_baseline = {
        str(key): float(value)
        for key, value in gate["dominant_path_baseline_excess_over_4m"].items()
    }
    dominant_current = sum(
        row["sum_excess_over_4m"]
        for row in path_ledger
        if row["path_id"] in dominant_baseline
    )
    dominant_baseline_sum = float(sum(dominant_baseline.values()))
    gene_gain = (
        structured_mae < float(gate["baseline_lattice_mae_m"])
        and structured_excess < float(gate["baseline_sum_excess_over_4m"])
        and dominant_current < dominant_baseline_sum
    )
    report = {
        "status": "ok",
        "version": config["version"],
        "experiment_type": config["experiment_type"],
        "iteration_mode": config["iteration_mode"],
        "competition_slug": config["competition_slug"],
        "seed": int(config["seed"]),
        "git_commit": git_commit(),
        "coverage": {
            "paths": int(selections["path_id"].nunique()),
            "groups": int(selections["group_id"].nunique()),
            "path_overlap_count": int(
                sum(row["path_overlap_count"] for row in audit)
            ),
            "site_feature_audit": audit,
        },
        "metrics": {
            "baseline_lattice_mae_m": float(gate["baseline_lattice_mae_m"]),
            "reanchored_lattice_mae_m": structured_mae,
            "improvement_vs_v008_m": float(
                gate["baseline_lattice_mae_m"]
            )
            - structured_mae,
            "baseline_sum_excess_over_4m": float(
                gate["baseline_sum_excess_over_4m"]
            ),
            "reanchored_sum_excess_over_4m": structured_excess,
            "dominant_paths_baseline_sum_excess_over_4m": dominant_baseline_sum,
            "dominant_paths_reanchored_sum_excess_over_4m": dominant_current,
        },
        "path_ledger": path_ledger,
        "lattice": lattice_summary,
        "gene_gain_passed": bool(gene_gain),
        "target_below_3m_passed": bool(
            gene_gain and structured_mae < float(gate["target_mae_m"])
        ),
        "submission_created": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return reanchored, selections, report


def save_kaggle_outputs(
    config: dict,
    reanchored: pd.DataFrame,
    selections: pd.DataFrame,
    report: dict,
) -> None:
    reanchored.to_csv(
        Path(config["outputs"]["reanchored_candidate_dataset"]), index=False
    )
    selections.to_csv(Path(config["outputs"]["lattice_selections"]), index=False)
    with Path(config["outputs"]["report"]).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight interpolated WiFi source reanchoring."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage", choices=["preflight", "kaggle"], default="preflight"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "preflight":
        path_metrics, report = run_preflight(config)
        save_preflight(config, path_metrics, report)
    else:
        reanchored, selections, report = run_kaggle_probe(config)
        save_kaggle_outputs(config, reanchored, selections, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
