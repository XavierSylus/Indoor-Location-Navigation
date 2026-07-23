from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "kaggle_pathsafe_smoke.json"
SENSOR_TYPES = (
    "TYPE_ACCELEROMETER",
    "TYPE_GYROSCOPE",
    "TYPE_MAGNETIC_FIELD",
    "TYPE_ROTATION_VECTOR",
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def find_competition_train_root(search_root: Path) -> Path:
    preferred = search_root / "indoor-location-navigation" / "train"
    if preferred.is_dir():
        return preferred
    candidates = []
    for path in search_root.rglob("train"):
        if path.is_dir() and any(path.glob("*/*/*.txt")):
            candidates.append(path)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one Indoor Location train root under {search_root}, found {len(candidates)}."
        )
    return candidates[0]


def require_kaggle_runtime(input_root: Path) -> None:
    expected = Path("/kaggle/input")
    if not expected.is_dir() or input_root != expected:
        raise RuntimeError(
            "Training smoke is Kaggle-only. Local execution may run unit tests but must not train."
        )


def parse_path_file(path: Path) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    waypoints = []
    sensors: Dict[str, list[tuple[int, float, float, float]]] = {
        sensor_type: [] for sensor_type in SENSOR_TYPES
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            timestamp = int(parts[0])
            record_type = parts[1]
            if record_type == "TYPE_WAYPOINT" and len(parts) >= 4:
                waypoints.append((timestamp, float(parts[2]), float(parts[3])))
            elif record_type in sensors and len(parts) >= 5:
                sensors[record_type].append(
                    (timestamp, float(parts[2]), float(parts[3]), float(parts[4]))
                )
    waypoint_frame = pd.DataFrame(waypoints, columns=["timestamp", "x", "y"]).sort_values("timestamp")
    sensor_frames = {
        sensor_type: pd.DataFrame(rows, columns=["timestamp", "x", "y", "z"]).sort_values("timestamp")
        for sensor_type, rows in sensors.items()
    }
    return waypoint_frame.reset_index(drop=True), sensor_frames


def summarize_sensor_interval(frame: pd.DataFrame, start: int, end: int) -> list[float]:
    if frame.empty:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    segment = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
    if segment.empty:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    xyz = segment[["x", "y", "z"]].to_numpy(dtype=np.float64)
    magnitude = np.linalg.norm(xyz, axis=1)
    return [
        float(len(segment)),
        float(np.mean(magnitude)),
        float(np.std(magnitude)),
        float(np.max(magnitude)),
        float(np.mean(xyz[:, 2])),
    ]


def build_path_legs(path: Path) -> pd.DataFrame:
    waypoints, sensors = parse_path_file(path)
    rows = []
    for index in range(max(0, len(waypoints) - 1)):
        start = waypoints.iloc[index]
        end = waypoints.iloc[index + 1]
        start_timestamp = int(start["timestamp"])
        end_timestamp = int(end["timestamp"])
        features = [max((end_timestamp - start_timestamp) / 1000.0, 0.001)]
        for sensor_type in SENSOR_TYPES:
            features.extend(
                summarize_sensor_interval(
                    sensors[sensor_type],
                    start_timestamp,
                    end_timestamp,
                )
            )
        rows.append(
            {
                "path_id": path.stem,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "features": features,
                "delta_x": float(end["x"] - start["x"]),
                "delta_y": float(end["y"] - start["y"]),
            }
        )
    return pd.DataFrame(rows)


def select_path_files(train_root: Path, max_paths: int, seed: int) -> list[Path]:
    files = sorted(train_root.glob("*/*/*.txt"))
    eligible = []
    for path in files:
        waypoint_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if "\tTYPE_WAYPOINT\t" in line:
                    waypoint_count += 1
                    if waypoint_count >= 2:
                        eligible.append(path)
                        break
    generator = random.Random(seed)
    generator.shuffle(eligible)
    return eligible[:max_paths]


def split_paths(
    path_ids: Sequence[str],
    validation_paths: int,
    minimum_train_paths: int,
) -> tuple[list[str], list[str]]:
    ordered = list(dict.fromkeys(str(path_id) for path_id in path_ids))
    if validation_paths < 1:
        raise ValueError("validation_paths must be positive.")
    if len(ordered) < validation_paths + minimum_train_paths:
        raise ValueError(
            f"Not enough paths: total={len(ordered)}, validation={validation_paths}, "
            f"minimum_train={minimum_train_paths}."
        )
    return ordered[:-validation_paths], ordered[-validation_paths:]


def stack_features(rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(rows["features"].tolist(), dtype=np.float64)


def euclidean_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target, axis=1).mean())


def run_training_smoke(config: dict, train_root: Path) -> tuple[dict, pd.DataFrame]:
    seed = int(config["seed"])
    data_config = config["data"]
    path_files = select_path_files(train_root, int(data_config["max_paths"]), seed)
    frames = []
    path_metadata = {}
    for path in path_files:
        legs = build_path_legs(path)
        if legs.empty:
            continue
        frames.append(legs)
        path_metadata[path.stem] = str(path.relative_to(train_root))
    if not frames:
        raise ValueError("No IMU waypoint legs were built from the competition input.")
    data = pd.concat(frames, ignore_index=True)
    train_paths, validation_paths = split_paths(
        data["path_id"].tolist(),
        int(data_config["validation_paths"]),
        int(data_config["minimum_train_paths"]),
    )
    train = data[data["path_id"].isin(train_paths)].reset_index(drop=True)
    validation = data[data["path_id"].isin(validation_paths)].reset_index(drop=True)
    if len(train) < int(data_config["minimum_legs"]) or validation.empty:
        raise ValueError(
            f"Insufficient legs: train={len(train)}, validation={len(validation)}, "
            f"minimum_train_legs={data_config['minimum_legs']}."
        )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(config["model"]["alpha"]))),
        ]
    )
    train_x = stack_features(train)
    train_y = train[["delta_x", "delta_y"]].to_numpy(dtype=np.float64)
    validation_x = stack_features(validation)
    validation_y = validation[["delta_x", "delta_y"]].to_numpy(dtype=np.float64)
    model.fit(train_x, train_y)
    prediction = model.predict(validation_x)
    predictions = validation[
        ["path_id", "start_timestamp", "end_timestamp", "delta_x", "delta_y"]
    ].copy()
    predictions["pred_delta_x"] = prediction[:, 0]
    predictions["pred_delta_y"] = prediction[:, 1]
    predictions["error_m"] = np.linalg.norm(prediction - validation_y, axis=1)
    overlap = sorted(set(train_paths) & set(validation_paths))
    report = {
        "status": "ok",
        "version": config["version"],
        "purpose": "Kaggle training-chain smoke only; not experiment CV and not submission evidence.",
        "seed": seed,
        "competition_slug": "indoor-location-navigation",
        "train_root": str(train_root),
        "model": {
            "type": config["model"]["type"],
            "alpha": float(config["model"]["alpha"]),
            "n_features": int(train_x.shape[1]),
        },
        "data": {
            "n_selected_paths": int(len(path_metadata)),
            "n_train_paths": int(len(train_paths)),
            "n_validation_paths": int(len(validation_paths)),
            "n_train_legs": int(len(train)),
            "n_validation_legs": int(len(validation)),
            "train_paths": train_paths,
            "validation_paths": validation_paths,
            "path_overlap": overlap,
            "path_safe_split": not overlap,
            "relative_path_files": path_metadata,
        },
        "smoke_delta_mae_m": euclidean_mae(prediction, validation_y),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "submission_created": False,
    }
    return report, predictions


def save_outputs(config: dict, report: dict, predictions: pd.DataFrame) -> None:
    report_path = Path(config["outputs"]["report"])
    predictions_path = Path(config["outputs"]["predictions"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    predictions.to_csv(predictions_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Path-safe Kaggle training-chain smoke.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args()
    require_kaggle_runtime(args.input_root)
    config = load_config(args.config)
    train_root = find_competition_train_root(args.input_root)
    report, predictions = run_training_smoke(config, train_root)
    save_outputs(config, report, predictions)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
