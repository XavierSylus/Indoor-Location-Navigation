from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.candidate_transition_lattice import run as run_lattice  # noqa: E402
from data_processing.kaggle_pathsafe_smoke import find_competition_train_root  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "path_safe_delta_oof.json"
INTERVAL_KEYS = ["path_id", "leg_index", "start_timestamp", "end_timestamp"]
PREDICTION_COLUMNS = INTERVAL_KEYS + ["v3_delta_x", "v3_delta_y"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def require_kaggle_runtime(input_root: Path) -> None:
    if input_root != Path("/kaggle/input") or not input_root.is_dir():
        raise RuntimeError("v008 training is Kaggle-only; local execution is limited to tests.")
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Missing Kaggle working directory.")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def discover_site_paths(train_root: Path, site_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for site_id in site_ids:
        site_root = train_root / str(site_id)
        if not site_root.is_dir():
            raise FileNotFoundError(f"Target site is missing: {site_id}")
        for path in sorted(site_root.glob("*/*.txt")):
            rows.append(
                {
                    "site_id": str(site_id),
                    "floor": path.parent.name,
                    "path_id": path.stem,
                    "file_path": str(path),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No trajectory files found for target sites.")
    if result["path_id"].duplicated().any():
        raise ValueError("Path IDs must be globally unique.")
    return result


def build_path_split_manifest(paths: pd.DataFrame, heldout_path_ids: Iterable[str]) -> pd.DataFrame:
    required = {"site_id", "path_id"}
    missing_columns = sorted(required - set(paths.columns))
    if missing_columns:
        raise ValueError(f"Path table is missing required columns: {missing_columns}")
    heldout = {str(value) for value in heldout_path_ids}
    available = set(paths["path_id"].astype(str))
    missing = sorted(heldout - available)
    if missing:
        raise ValueError(f"Held-out paths are missing from the selected sites: {missing}")
    result = paths.copy()
    result["path_id"] = result["path_id"].astype(str)
    result["split"] = np.where(result["path_id"].isin(heldout), "validation", "train")
    train_paths = set(result.loc[result["split"] == "train", "path_id"])
    validation_paths = set(result.loc[result["split"] == "validation", "path_id"])
    overlap = sorted(train_paths & validation_paths)
    if overlap:
        raise ValueError(f"Train/validation path overlap detected: {overlap}")
    if not train_paths or not validation_paths:
        raise ValueError("Both train and validation path sets must be non-empty.")
    return result.sort_values(["split", "site_id", "path_id"]).reset_index(drop=True)


def validate_oof_predictions(
    predictions: pd.DataFrame,
    expected_intervals: pd.DataFrame,
    train_path_ids: Iterable[str],
    validation_path_ids: Iterable[str],
) -> dict:
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {missing}")
    expected_missing = sorted(set(INTERVAL_KEYS) - set(expected_intervals.columns))
    if expected_missing:
        raise ValueError(f"Expected intervals are missing columns: {expected_missing}")
    train_paths = {str(value) for value in train_path_ids}
    validation_paths = {str(value) for value in validation_path_ids}
    overlap = sorted(train_paths & validation_paths)
    if overlap:
        raise ValueError(f"Train/validation path overlap detected: {overlap}")
    predicted_paths = set(predictions["path_id"].astype(str))
    unexpected_paths = sorted(predicted_paths - validation_paths)
    if unexpected_paths:
        raise ValueError(f"OOF predictions contain non-validation paths: {unexpected_paths}")
    if predictions.duplicated(INTERVAL_KEYS).any():
        raise ValueError("OOF predictions contain duplicated interval keys.")
    expected_keys = {
        tuple(row)
        for row in expected_intervals[INTERVAL_KEYS].astype(
            {
                "path_id": str,
                "leg_index": int,
                "start_timestamp": int,
                "end_timestamp": int,
            }
        ).itertuples(index=False, name=None)
    }
    prediction_keys = {
        tuple(row)
        for row in predictions[INTERVAL_KEYS].astype(
            {
                "path_id": str,
                "leg_index": int,
                "start_timestamp": int,
                "end_timestamp": int,
            }
        ).itertuples(index=False, name=None)
    }
    if expected_keys != prediction_keys:
        raise ValueError(
            "OOF interval coverage mismatch: "
            f"missing={len(expected_keys - prediction_keys)}, "
            f"unexpected={len(prediction_keys - expected_keys)}"
        )
    values = predictions[["v3_delta_x", "v3_delta_y"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("OOF predictions contain non-finite values.")
    return {
        "path_overlap_count": 0,
        "expected_interval_count": len(expected_keys),
        "prediction_interval_count": len(prediction_keys),
        "validation_path_count": len(validation_paths),
    }


def choose_internal_validation_paths(
    train_path_ids: Sequence[str],
    fraction: float,
    seed: int,
) -> set[str]:
    ordered = sorted({str(value) for value in train_path_ids})
    if len(ordered) < 2:
        raise ValueError("At least two non-target paths are required.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("internal_validation_fraction must be between zero and one.")
    generator = np.random.default_rng(seed)
    shuffled = np.asarray(ordered, dtype=object)
    generator.shuffle(shuffled)
    count = min(len(ordered) - 1, max(1, int(round(len(ordered) * fraction))))
    return set(str(value) for value in shuffled[:count])


def build_leg_bundle(files: Sequence[Path], n_time_steps: int) -> tuple[Any, pd.DataFrame]:
    from src.imu_delta_dataset import IMUDeltaDataset, build_legs_from_trajectory, parse_all_sensors

    all_legs = []
    metadata = []
    for file_path in files:
        legs = build_legs_from_trajectory(file_path, n_time_steps=n_time_steps) or []
        sensors = parse_all_sensors(file_path)
        waypoints = sensors["waypoint"]
        if not legs or waypoints is None:
            continue
        waypoints = waypoints.sort_values("timestamp").reset_index(drop=True)
        if len(legs) != len(waypoints) - 1:
            raise ValueError(f"Leg/waypoint mismatch for {file_path}")
        all_legs.extend(legs)
        for leg_index in range(len(legs)):
            metadata.append(
                {
                    "path_id": file_path.stem,
                    "leg_index": leg_index,
                    "start_timestamp": int(waypoints.loc[leg_index, "timestamp"]),
                    "end_timestamp": int(waypoints.loc[leg_index + 1, "timestamp"]),
                    "gt_delta_x": float(waypoints.loc[leg_index + 1, "x"] - waypoints.loc[leg_index, "x"]),
                    "gt_delta_y": float(waypoints.loc[leg_index + 1, "y"] - waypoints.loc[leg_index, "y"]),
                }
            )
    if not all_legs:
        raise ValueError("No waypoint legs were extracted.")
    return IMUDeltaDataset(all_legs), pd.DataFrame(metadata)


def make_loader(dataset: Any, batch_size: int, workers: int, shuffle: bool):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train_model(config: dict, train_dataset: Any, internal_dataset: Any):
    import torch
    from src.imu_delta_model import IMUDeltaNet
    from src.train_imu_delta import run_training_phase, set_seed

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("v008 requires a Kaggle GPU runtime.")
    data_config = config["data"]
    training = config["training"]
    train_dataset.training = True
    train_dataset.noise_std = float(data_config["noise_std"])
    train_loader = make_loader(
        train_dataset,
        int(data_config["batch_size"]),
        int(data_config["num_workers"]),
        True,
    )
    internal_loader = make_loader(
        internal_dataset,
        int(data_config["batch_size"]) * 2,
        int(data_config["num_workers"]),
        False,
    )
    model = IMUDeltaNet.from_config(config).to(device)
    use_amp = bool(training["amp"])
    checkpoint_root = Path("/tmp/path_safe_delta_oof_checkpoints")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    best_internal_mae = float("inf")
    best_state = None
    phase_history = []
    for cycle in range(int(training["n_cycles"])):
        scale = float(training["cycle_lr_decay"]) ** cycle
        for phase_key in ("phase_a", "phase_b"):
            phase = copy.deepcopy(training[phase_key])
            phase["lr"] = float(phase["lr"]) * scale
            if cycle > 0 and phase_key == "phase_a":
                phase["epochs"] = max(10, int(int(phase["epochs"]) * 0.7))
                phase["warmup_epochs"] = min(int(phase["warmup_epochs"]), 2)
            phase_name = f"cycle_{cycle + 1}_{phase_key}"
            checkpoint = checkpoint_root / f"{phase_name}.pt"
            metrics = run_training_phase(
                model=model,
                train_loader=train_loader,
                val_loader=internal_loader,
                phase_cfg=phase,
                device=device,
                grad_clip=float(training["gradient_clip"]),
                use_amp=use_amp,
                model_save_path=checkpoint,
                phase_name=phase_name,
                patience=int(training["patience"]),
            )
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            phase_history.append({"phase": phase_name, **metrics})
            if float(metrics["mae_m"]) < best_internal_mae:
                best_internal_mae = float(metrics["mae_m"])
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, device, phase_history, best_internal_mae


def predict_dataset(model, device, dataset: Any, metadata: pd.DataFrame, batch_size: int):
    import torch

    loader = make_loader(dataset, batch_size, 0, False)
    batches = []
    with torch.no_grad():
        for imu, aux, _ in loader:
            batches.append(model(imu.to(device), aux.to(device)).cpu().numpy())
    values = np.concatenate(batches, axis=0)
    if len(values) != len(metadata):
        raise ValueError("Prediction count does not match target metadata.")
    result = metadata.copy()
    result["v3_delta_x"] = values[:, 0]
    result["v3_delta_y"] = values[:, 1]
    return result


def resolve_output(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_experiment(config: dict, train_root: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_config = config["data"]
    paths = discover_site_paths(train_root, data_config["target_site_ids"])
    split_manifest = build_path_split_manifest(paths, data_config["heldout_path_ids"])
    target_rows = split_manifest[split_manifest["split"] == "validation"]
    train_rows = split_manifest[split_manifest["split"] == "train"]
    internal_paths = choose_internal_validation_paths(
        train_rows["path_id"].tolist(),
        float(data_config["internal_validation_fraction"]),
        int(config["seed"]),
    )
    fit_rows = train_rows[~train_rows["path_id"].isin(internal_paths)]
    internal_rows = train_rows[train_rows["path_id"].isin(internal_paths)]
    n_time_steps = int(data_config["n_time_steps"])
    fit_dataset, _ = build_leg_bundle(
        [Path(value) for value in fit_rows["file_path"]],
        n_time_steps,
    )
    internal_dataset, _ = build_leg_bundle(
        [Path(value) for value in internal_rows["file_path"]],
        n_time_steps,
    )
    target_dataset, target_metadata = build_leg_bundle(
        [Path(value) for value in target_rows["file_path"]],
        n_time_steps,
    )
    model, device, phase_history, internal_mae = train_model(
        config,
        fit_dataset,
        internal_dataset,
    )
    predictions = predict_dataset(
        model,
        device,
        target_dataset,
        target_metadata,
        int(data_config["batch_size"]) * 2,
    )
    safety = validate_oof_predictions(
        predictions,
        target_metadata[INTERVAL_KEYS],
        train_rows["path_id"],
        target_rows["path_id"],
    )
    prediction_path = resolve_output(config["outputs"]["predictions"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    lattice_config = copy.deepcopy(config["lattice"])
    lattice_config["inputs"]["v3_delta_predictions"] = str(prediction_path)
    selections, lattice_summary = run_lattice(lattice_config)
    target = predictions[["gt_delta_x", "gt_delta_y"]].to_numpy(dtype=np.float64)
    predicted = predictions[["v3_delta_x", "v3_delta_y"]].to_numpy(dtype=np.float64)
    target_error = np.linalg.norm(predicted - target, axis=1)
    report = {
        "status": "ok",
        "version": config["version"],
        "experiment_type": config["experiment_type"],
        "iteration_mode": config["iteration_mode"],
        "competition_slug": config["competition_slug"],
        "seed": int(config["seed"]),
        "git_commit": git_commit(),
        "data": {
            "target_site_count": int(paths["site_id"].nunique()),
            "fit_path_count": int(len(fit_rows)),
            "internal_validation_path_count": int(len(internal_rows)),
            "heldout_path_count": int(len(target_rows)),
            "fit_leg_count": int(len(fit_dataset)),
            "internal_validation_leg_count": int(len(internal_dataset)),
            "heldout_leg_count": int(len(target_dataset)),
            "heldout_path_ids": sorted(target_rows["path_id"].tolist()),
            **safety,
        },
        "training": {
            "model_family": config["model"]["family"],
            "best_internal_validation_delta_mae_m": internal_mae,
            "phase_history": phase_history,
        },
        "heldout_delta_mae_m": float(target_error.mean()),
        "lattice": lattice_summary,
        "success_gate_passed": bool(lattice_summary["success_gate_passed"]),
        "submission_created": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return report, {
        "predictions": predictions,
        "split_manifest": split_manifest,
        "lattice_selections": selections,
    }


def save_outputs(config: dict, report: dict, frames: dict[str, pd.DataFrame]) -> None:
    mapping = {
        "predictions": config["outputs"]["predictions"],
        "split_manifest": config["outputs"]["split_manifest"],
        "lattice_selections": config["outputs"]["lattice_selections"],
    }
    for name, path_value in mapping.items():
        path = resolve_output(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(path, index=False)
    report_path = resolve_output(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle-only path-safe V3 delta OOF probe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args()
    require_kaggle_runtime(args.input_root)
    config = load_config(args.config)
    train_root = find_competition_train_root(args.input_root)
    report, frames = run_experiment(config, train_root)
    save_outputs(config, report, frames)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
