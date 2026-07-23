from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "floorplan_hallway_legality.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def require_columns(data: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def coordinates_to_pixels(
    coordinates: np.ndarray,
    map_width_m: float,
    map_height_m: float,
    image_width: int,
    image_height: int,
    invert_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2).")
    if map_width_m <= 0 or map_height_m <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("Map and image dimensions must be positive.")
    x_values = values[:, 0]
    y_values = map_height_m - values[:, 1] if invert_y else values[:, 1]
    pixels_x = np.rint(x_values / map_width_m * (image_width - 1)).astype(np.int64)
    pixels_y = np.rint(y_values / map_height_m * (image_height - 1)).astype(np.int64)
    in_bounds = (
        (pixels_x >= 0)
        & (pixels_x < image_width)
        & (pixels_y >= 0)
        & (pixels_y < image_height)
    )
    return pixels_x, pixels_y, in_bounds


def hallway_flags(
    image: np.ndarray,
    coordinates: np.ndarray,
    map_width_m: float,
    map_height_m: float,
    invert_y: bool,
    max_rgb_channel_exclusive: int,
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB array.")
    pixels_x, pixels_y, in_bounds = coordinates_to_pixels(
        coordinates,
        map_width_m,
        map_height_m,
        image.shape[1],
        image.shape[0],
        invert_y,
    )
    flags = np.zeros(len(coordinates), dtype=bool)
    flags[in_bounds] = (
        np.max(image[pixels_y[in_bounds], pixels_x[in_bounds]], axis=1)
        < int(max_rgb_channel_exclusive)
    )
    return flags, in_bounds


def segment_nonhallway_ratio(
    image: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    map_width_m: float,
    map_height_m: float,
    invert_y: bool,
    max_rgb_channel_exclusive: int,
) -> float:
    coordinates = np.vstack(
        [np.asarray(start_xy, dtype=np.float64), np.asarray(end_xy, dtype=np.float64)]
    )
    pixels_x, pixels_y, in_bounds = coordinates_to_pixels(
        coordinates,
        map_width_m,
        map_height_m,
        image.shape[1],
        image.shape[0],
        invert_y,
    )
    if not in_bounds.all():
        return 1.0
    sample_count = max(
        2,
        int(
            np.hypot(
                pixels_x[1] - pixels_x[0],
                pixels_y[1] - pixels_y[0],
            )
        )
        + 1,
    )
    sample_x = np.rint(np.linspace(pixels_x[0], pixels_x[1], sample_count)).astype(
        np.int64
    )
    sample_y = np.rint(np.linspace(pixels_y[0], pixels_y[1], sample_count)).astype(
        np.int64
    )
    hallway = (
        np.max(image[sample_y, sample_x], axis=1) < int(max_rgb_channel_exclusive)
    )
    return float(np.mean(~hallway))


def load_floor_map(
    metadata_root: Path, site_id: str, floor: str
) -> tuple[np.ndarray, float, float]:
    floor_root = metadata_root / site_id / floor
    image = np.asarray(Image.open(floor_root / "floor_image.png").convert("RGB"))
    with (floor_root / "floor_info.json").open("r", encoding="utf-8") as handle:
        map_info = json.load(handle)["map_info"]
    return image, float(map_info["width"]), float(map_info["height"])


def ordered_best_candidates(path: pd.DataFrame) -> pd.DataFrame:
    return (
        path.sort_values(["timestamp", "label_distance_m", "rank"])
        .drop_duplicates("group_id")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def mean_edge_ratio(
    path: pd.DataFrame,
    image: np.ndarray,
    map_width_m: float,
    map_height_m: float,
    invert_y: bool,
    threshold: int,
) -> float:
    coordinates = path[["candidate_x", "candidate_y"]].to_numpy(dtype=np.float64)
    if len(coordinates) < 2:
        return 0.0
    ratios = [
        segment_nonhallway_ratio(
            image,
            start,
            end,
            map_width_m,
            map_height_m,
            invert_y,
            threshold,
        )
        for start, end in zip(coordinates[:-1], coordinates[1:])
    ]
    return float(np.mean(ratios))


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run(config: dict) -> tuple[pd.DataFrame, dict]:
    candidates = pd.read_csv(resolve_path(config["inputs"]["candidate_dataset"]))
    selections = pd.read_csv(resolve_path(config["inputs"]["v008_selections"]))
    required = {
        "site_id",
        "path_id",
        "floor",
        "group_id",
        "timestamp",
        "candidate_x",
        "candidate_y",
        "label_distance_m",
    }
    require_columns(candidates, required | {"rank"}, "candidate dataset")
    require_columns(selections, required, "v008 selections")
    metadata_root = resolve_path(config["inputs"]["metadata_root"])
    hallway_config = config["hallway"]
    invert_y = bool(hallway_config["invert_y"])
    threshold = int(hallway_config["max_rgb_channel_exclusive"])
    rows = []
    for path_id, candidate_path in candidates.groupby("path_id", sort=True):
        selected_path = selections[
            selections["path_id"].astype(str) == str(path_id)
        ].sort_values("timestamp")
        if selected_path.empty:
            raise ValueError(f"Missing v008 selections for path {path_id}.")
        first = candidate_path.iloc[0]
        site_id = str(first["site_id"])
        floor = str(first["floor"])
        image, map_width_m, map_height_m = load_floor_map(
            metadata_root, site_id, floor
        )
        oracle_path = ordered_best_candidates(candidate_path)
        candidate_flags, candidate_bounds = hallway_flags(
            image,
            candidate_path[["candidate_x", "candidate_y"]].to_numpy(dtype=np.float64),
            map_width_m,
            map_height_m,
            invert_y,
            threshold,
        )
        oracle_flags, oracle_bounds = hallway_flags(
            image,
            oracle_path[["candidate_x", "candidate_y"]].to_numpy(dtype=np.float64),
            map_width_m,
            map_height_m,
            invert_y,
            threshold,
        )
        selected_flags, selected_bounds = hallway_flags(
            image,
            selected_path[["candidate_x", "candidate_y"]].to_numpy(dtype=np.float64),
            map_width_m,
            map_height_m,
            invert_y,
            threshold,
        )
        oracle_edge = mean_edge_ratio(
            oracle_path,
            image,
            map_width_m,
            map_height_m,
            invert_y,
            threshold,
        )
        selected_edge = mean_edge_ratio(
            selected_path,
            image,
            map_width_m,
            map_height_m,
            invert_y,
            threshold,
        )
        rows.append(
            {
                "site_id": site_id,
                "path_id": str(path_id),
                "floor": floor,
                "n_groups": int(candidate_path["group_id"].nunique()),
                "candidate_hallway_ratio": float(np.mean(candidate_flags)),
                "oracle_hallway_ratio": float(np.mean(oracle_flags)),
                "v008_hallway_ratio": float(np.mean(selected_flags)),
                "oracle_point_advantage": float(
                    np.mean(oracle_flags) - np.mean(selected_flags)
                ),
                "oracle_edge_nonhallway_ratio": oracle_edge,
                "v008_edge_nonhallway_ratio": selected_edge,
                "oracle_edge_advantage": selected_edge - oracle_edge,
                "candidate_in_bounds_ratio": float(np.mean(candidate_bounds)),
                "oracle_in_bounds_ratio": float(np.mean(oracle_bounds)),
                "v008_in_bounds_ratio": float(np.mean(selected_bounds)),
            }
        )
    metrics = pd.DataFrame(rows)
    priority_ids = {str(value) for value in hallway_config["priority_path_ids"]}
    priority = metrics[metrics["path_id"].isin(priority_ids)].copy()
    if set(priority["path_id"]) != priority_ids:
        raise ValueError("Priority path coverage is incomplete.")
    gate = config["success_gate"]
    point_threshold = float(gate["minimum_oracle_point_advantage"])
    edge_threshold = float(gate["minimum_oracle_edge_advantage"])
    priority["has_discriminative_signal"] = (
        priority["oracle_point_advantage"] >= point_threshold
    ) | (priority["oracle_edge_advantage"] >= edge_threshold)
    passed = (
        bool(priority["has_discriminative_signal"].all())
        if bool(gate["require_every_priority_path"])
        else bool(priority["has_discriminative_signal"].any())
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
        "n_paths": int(metrics["path_id"].nunique()),
        "n_groups": int(candidates["group_id"].nunique()),
        "coordinate_mapping": {
            "invert_y": invert_y,
            "max_rgb_channel_exclusive": threshold,
            "minimum_candidate_in_bounds_ratio": float(
                metrics["candidate_in_bounds_ratio"].min()
            ),
        },
        "priority_path_metrics": priority.to_dict(orient="records"),
        "success_gate_passed": passed,
        "decision": (
            "Promote floorplan legality to a Kaggle lattice probe."
            if passed
            else "Reject floorplan legality before Kaggle execution because it does not distinguish the dominant tail paths."
        ),
        "submission_created": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": Image.__version__,
        },
    }
    return metrics, report


def save_outputs(config: dict, metrics: pd.DataFrame, report: dict) -> None:
    metrics_path = resolve_path(config["outputs"]["path_metrics"])
    report_path = resolve_path(config["outputs"]["report"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Floorplan hallway-legality preflight for v008 candidates."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    metrics, report = run(config)
    save_outputs(config, metrics, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
