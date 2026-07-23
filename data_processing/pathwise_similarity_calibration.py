from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pathwise_similarity_calibration.json"
CALIBRATION_FORBIDDEN_COLUMNS = {
    "gt_delta_x",
    "gt_delta_y",
    "gt_x",
    "gt_y",
    "label_distance_m",
    "group_oracle_distance_at_topk",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_kaggle_runtime() -> None:
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("v009 experiment execution is Kaggle-only.")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def validate_calibration_feature_columns(columns: Sequence[str]) -> None:
    forbidden = sorted(set(columns) & CALIBRATION_FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Calibration features contain label/oracle columns: {forbidden}")


def _as_vectors(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> dict:
    source_vectors = _as_vectors(source, "source")
    target_vectors = _as_vectors(target, "target")
    if source_vectors.shape != target_vectors.shape:
        raise ValueError("source and target must have identical shapes.")
    denominator = float(np.square(source_vectors).sum())
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("source has insufficient energy for similarity calibration.")
    source_x = source_vectors[:, 0]
    source_y = source_vectors[:, 1]
    target_x = target_vectors[:, 0]
    target_y = target_vectors[:, 1]
    coefficient_a = float(np.sum(source_x * target_x + source_y * target_y) / denominator)
    coefficient_b = float(np.sum(source_x * target_y - source_y * target_x) / denominator)
    scale = float(math.hypot(coefficient_a, coefficient_b))
    rotation = float(math.atan2(coefficient_b, coefficient_a))
    return {
        "coefficient_a": coefficient_a,
        "coefficient_b": coefficient_b,
        "scale": scale,
        "rotation_rad": rotation,
        "rotation_deg": float(math.degrees(rotation)),
    }


def apply_similarity_transform(vectors: np.ndarray, transform: dict) -> np.ndarray:
    source = _as_vectors(vectors, "vectors")
    coefficient_a = float(transform["coefficient_a"])
    coefficient_b = float(transform["coefficient_b"])
    matrix = np.array(
        [
            [coefficient_a, -coefficient_b],
            [coefficient_b, coefficient_a],
        ],
        dtype=np.float64,
    )
    return source @ matrix.T


def select_unary_path(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        "group_id",
        "path_id",
        "timestamp",
        "candidate_x",
        "candidate_y",
        "predicted_score",
        "rank",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Scored candidates are missing columns: {missing}")
    return (
        scored.sort_values(["group_id", "predicted_score", "rank"])
        .drop_duplicates("group_id")
        .sort_values(["path_id", "timestamp"])
        .reset_index(drop=True)
    )


def alignment_rmse(source: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(source - target).sum(axis=1))))


def calibrate_paths(
    scored: pd.DataFrame,
    raw_delta: pd.DataFrame,
    require_exact_chain: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from data_processing.candidate_transition_lattice import build_path_delta_lookup

    validate_calibration_feature_columns(
        ["v3_delta_x", "v3_delta_y", "unary_delta_x", "unary_delta_y"]
    )
    unary = select_unary_path(scored)
    calibrated_frames = []
    transform_rows = []
    for path_id, unary_path in unary.groupby("path_id", sort=True):
        unary_path = unary_path.sort_values("timestamp").reset_index(drop=True)
        timestamps = unary_path["timestamp"].astype(np.int64).tolist()
        raw_path = raw_delta[raw_delta["path_id"].astype(str) == str(path_id)].copy()
        if raw_path.empty:
            raise ValueError(f"Missing raw delta path: {path_id}")
        lookup = build_path_delta_lookup(timestamps, raw_path, require_exact_chain)
        source_vectors = []
        target_vectors = []
        for previous, current in zip(
            unary_path.itertuples(index=False),
            unary_path.iloc[1:].itertuples(index=False),
        ):
            interval = (int(previous.timestamp), int(current.timestamp))
            if interval not in lookup:
                raise ValueError(f"Missing exact delta chain for {path_id}: {interval}")
            source_vectors.append(lookup[interval])
            target_vectors.append(
                np.array(
                    [
                        float(current.candidate_x - previous.candidate_x),
                        float(current.candidate_y - previous.candidate_y),
                    ],
                    dtype=np.float64,
                )
            )
        source = _as_vectors(np.asarray(source_vectors), "source")
        target = _as_vectors(np.asarray(target_vectors), "target")
        transform = fit_similarity_transform(source, target)
        calibrated_alignment = apply_similarity_transform(source, transform)
        raw_leg_vectors = raw_path[["v3_delta_x", "v3_delta_y"]].to_numpy(dtype=np.float64)
        calibrated_leg_vectors = apply_similarity_transform(raw_leg_vectors, transform)
        raw_path["raw_v3_delta_x"] = raw_leg_vectors[:, 0]
        raw_path["raw_v3_delta_y"] = raw_leg_vectors[:, 1]
        raw_path["v3_delta_x"] = calibrated_leg_vectors[:, 0]
        raw_path["v3_delta_y"] = calibrated_leg_vectors[:, 1]
        calibrated_frames.append(raw_path)
        transform_rows.append(
            {
                "path_id": str(path_id),
                "n_alignment_pairs": int(len(source)),
                **transform,
                "raw_alignment_rmse_m": alignment_rmse(source, target),
                "calibrated_alignment_rmse_m": alignment_rmse(calibrated_alignment, target),
            }
        )
    calibrated = pd.concat(calibrated_frames, ignore_index=True)
    transforms = pd.DataFrame(transform_rows)
    if set(calibrated["path_id"].astype(str)) != set(raw_delta["path_id"].astype(str)):
        raise ValueError("Calibrated output path coverage differs from raw delta input.")
    return calibrated, transforms


def diagnostic_delta_mae(delta: pd.DataFrame) -> float:
    required = {"gt_delta_x", "gt_delta_y", "v3_delta_x", "v3_delta_y"}
    missing = sorted(required - set(delta.columns))
    if missing:
        raise ValueError(f"Diagnostic delta metric is missing columns: {missing}")
    target = delta[["gt_delta_x", "gt_delta_y"]].to_numpy(dtype=np.float64)
    prediction = delta[["v3_delta_x", "v3_delta_y"]].to_numpy(dtype=np.float64)
    return float(np.linalg.norm(prediction - target, axis=1).mean())


def build_path_ledger(selections: pd.DataFrame) -> list[dict]:
    rows = []
    for path_id, path in selections.groupby("path_id", sort=True):
        errors = path["label_distance_m"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "path_id": str(path_id),
                "n_groups": int(path["group_id"].nunique()),
                "lattice_mae_m": float(errors.mean()),
                "sum_excess_over_4m": float(np.maximum(errors - 4.0, 0.0).sum()),
            }
        )
    return rows


def run_experiment(config: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    from data_processing.candidate_transition_lattice import (
        get_numeric_features,
        load_dataset,
        run as run_lattice,
        score_candidate_oof,
    )

    raw_delta = pd.read_csv(resolve_path(config["inputs"]["raw_delta_predictions"]))
    with resolve_path(config["inputs"]["v008_report"]).open("r", encoding="utf-8") as handle:
        v008_report = json.load(handle)
    lattice_config = copy.deepcopy(config["lattice"])
    reranker = lattice_config["reranker"]
    candidates = load_dataset(
        resolve_path(config["inputs"]["candidate_dataset"]),
        str(reranker["target"]),
        get_numeric_features(str(reranker["feature_set"])),
    )
    scored = score_candidate_oof(candidates, lattice_config)
    calibrated, transforms = calibrate_paths(
        scored,
        raw_delta,
        bool(lattice_config["transition"]["require_exact_v3_chain"]),
    )
    calibrated_path = resolve_path(config["outputs"]["calibrated_delta"])
    calibrated_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(calibrated_path, index=False)
    lattice_config["inputs"]["v3_delta_predictions"] = str(calibrated_path)
    selections, lattice_summary = run_lattice(lattice_config)
    v008_mae = float(v008_report["lattice"]["structured_lattice_mae_m"])
    calibrated_mae = float(lattice_summary["structured_lattice_mae_m"])
    improvement = v008_mae - calibrated_mae
    target_mae = float(config["success_gate"]["target_mae_m"])
    report = {
        "status": "ok",
        "version": config["version"],
        "experiment_type": config["experiment_type"],
        "iteration_mode": config["iteration_mode"],
        "competition_slug": config["competition_slug"],
        "seed": int(config["seed"]),
        "git_commit": git_commit(),
        "calibration": {
            "method": config["calibration"]["method"],
            "translation_allowed": bool(config["calibration"]["translation_allowed"]),
            "features_used": list(config["calibration"]["source_columns"])
            + list(config["calibration"]["target_columns"]),
            "forbidden_columns": sorted(CALIBRATION_FORBIDDEN_COLUMNS),
            "n_paths": int(transforms["path_id"].nunique()),
            "transforms": transforms.to_dict(orient="records"),
        },
        "metrics": {
            "raw_v008_delta_mae_m": diagnostic_delta_mae(raw_delta),
            "calibrated_delta_mae_m": diagnostic_delta_mae(calibrated),
            "raw_v008_lattice_mae_m": v008_mae,
            "calibrated_lattice_mae_m": calibrated_mae,
            "improvement_vs_v008_m": improvement,
            "structured_sum_excess_over_4m": float(
                lattice_summary["structured_sum_excess_over_4m"]
            ),
            "excess_over_4m_reduction_ratio_vs_unary": float(
                lattice_summary["excess_over_4m_reduction_ratio"]
            ),
        },
        "path_ledger": build_path_ledger(selections),
        "lattice": lattice_summary,
        "gene_gain_passed": bool(improvement > 0.0),
        "success_gate_passed": bool(improvement > 0.0 and calibrated_mae <= target_mae),
        "submission_created": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return report, {
        "calibrated_delta": calibrated,
        "transforms": transforms,
        "lattice_selections": selections,
    }


def save_outputs(config: dict, report: dict, frames: dict[str, pd.DataFrame]) -> None:
    for name in ("calibrated_delta", "transforms", "lattice_selections"):
        path = resolve_path(config["outputs"][name])
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(path, index=False)
    report_path = resolve_path(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle-only pathwise similarity calibration probe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    require_kaggle_runtime()
    config = load_config(args.config)
    report, frames = run_experiment(config)
    save_outputs(config, report, frames)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
