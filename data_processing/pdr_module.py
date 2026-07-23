from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pdr_config.yml"

WAYPOINT_TYPE = "TYPE_WAYPOINT"
ACCELEROMETER_TYPE = "TYPE_ACCELEROMETER"
GYROSCOPE_TYPE = "TYPE_GYROSCOPE"
ROTATION_VECTOR_TYPE = "TYPE_ROTATION_VECTOR"


@dataclass(frozen=True)
class SensorFrames:
    waypoint: pd.DataFrame
    accelerometer: pd.DataFrame
    gyroscope: pd.DataFrame
    rotation_vector: pd.DataFrame


@dataclass(frozen=True)
class StepSignal:
    timestamps: np.ndarray
    raw_values: np.ndarray
    smooth_values: np.ndarray
    peak_indices: np.ndarray
    peak_timestamps: np.ndarray
    peak_values: np.ndarray


def wrap_angle_rad(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def circular_mean_rad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))))


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")
    return config


def ensure_required_sections(config: dict) -> None:
    required_sections = [
        "data",
        "step_detection",
        "step_length",
        "heading",
        "paths",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise KeyError(f"Missing config sections: {missing}")


def iter_sensor_records(file_path: Path) -> Iterable[Tuple[str, List[str]]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            yield parts[1], parts


def parse_numeric_triplet(parts: Sequence[str]) -> Optional[Tuple[float, float, float]]:
    if len(parts) < 5:
        return None
    try:
        return float(parts[2]), float(parts[3]), float(parts[4])
    except (TypeError, ValueError):
        return None


def parse_waypoint(parts: Sequence[str]) -> Optional[Tuple[int, float, float]]:
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), float(parts[2]), float(parts[3])
    except (TypeError, ValueError):
        return None


def parse_trajectory_sensors(file_path: Path) -> SensorFrames:
    waypoints: List[Tuple[int, float, float]] = []
    accelerometer: List[Tuple[int, float, float, float]] = []
    gyroscope: List[Tuple[int, float, float, float]] = []
    rotation_vector: List[Tuple[int, float, float, float]] = []

    for record_type, parts in iter_sensor_records(file_path):
        if record_type == WAYPOINT_TYPE:
            waypoint = parse_waypoint(parts)
            if waypoint is not None:
                waypoints.append(waypoint)
            continue

        triplet = parse_numeric_triplet(parts)
        if triplet is None:
            continue

        try:
            timestamp = int(parts[0])
        except (TypeError, ValueError):
            continue

        if record_type == ACCELEROMETER_TYPE:
            accelerometer.append((timestamp, triplet[0], triplet[1], triplet[2]))
        elif record_type == GYROSCOPE_TYPE:
            gyroscope.append((timestamp, triplet[0], triplet[1], triplet[2]))
        elif record_type == ROTATION_VECTOR_TYPE:
            rotation_vector.append((timestamp, triplet[0], triplet[1], triplet[2]))

    return SensorFrames(
        waypoint=pd.DataFrame(waypoints, columns=["timestamp", "x", "y"]),
        accelerometer=pd.DataFrame(accelerometer, columns=["timestamp", "x", "y", "z"]),
        gyroscope=pd.DataFrame(gyroscope, columns=["timestamp", "x", "y", "z"]),
        rotation_vector=pd.DataFrame(rotation_vector, columns=["timestamp", "x", "y", "z"]),
    )


def sanitize_sensor_frame(frame: pd.DataFrame, value_columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    clean = frame.copy()
    clean = clean.sort_values("timestamp").reset_index(drop=True)

    clean["timestamp"] = pd.to_numeric(clean["timestamp"], errors="coerce")
    clean = clean[np.isfinite(clean["timestamp"].to_numpy())]
    clean["timestamp"] = clean["timestamp"].astype(np.int64)

    for column in value_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = clean[column].replace([np.inf, -np.inf], np.nan)

    if clean.empty:
        return clean.reset_index(drop=True)

    clean = clean.groupby("timestamp", as_index=False)[list(value_columns)].mean()
    for column in value_columns:
        series = clean[column].astype(np.float32)
        if series.isna().all():
            clean[column] = np.zeros(len(clean), dtype=np.float32)
            continue
        series = series.interpolate(limit_direction="both")
        series = series.ffill().bfill().fillna(0.0)
        clean[column] = series.astype(np.float32)

    return clean.reset_index(drop=True)


def sanitize_waypoints(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    clean = frame.copy()
    clean = clean.sort_values("timestamp").reset_index(drop=True)
    clean["timestamp"] = pd.to_numeric(clean["timestamp"], errors="coerce")
    clean["x"] = pd.to_numeric(clean["x"], errors="coerce")
    clean["y"] = pd.to_numeric(clean["y"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["timestamp", "x", "y"])
    clean["timestamp"] = clean["timestamp"].astype(np.int64)
    clean = clean.groupby("timestamp", as_index=False)[["x", "y"]].mean()
    return clean.reset_index(drop=True)


def estimate_sample_period_ms(timestamps: np.ndarray, fallback_ms: float) -> float:
    if timestamps.size < 2:
        return float(fallback_ms)
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return float(fallback_ms)
    return float(np.median(diffs))


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if values.size == 0 or window_size <= 1:
        return values.copy()
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(values, kernel, mode="same")


class PDR:
    def __init__(self, config: dict):
        ensure_required_sections(config)
        self.config = config

    @classmethod
    def from_yaml(cls, config_path: Path | str = DEFAULT_CONFIG_PATH) -> "PDR":
        config = load_yaml_config(Path(config_path))
        return cls(config=config)

    def parse_file(self, file_path: Path | str) -> SensorFrames:
        raw = parse_trajectory_sensors(Path(file_path))
        return SensorFrames(
            waypoint=sanitize_waypoints(raw.waypoint),
            accelerometer=sanitize_sensor_frame(raw.accelerometer, ["x", "y", "z"]),
            gyroscope=sanitize_sensor_frame(raw.gyroscope, ["x", "y", "z"]),
            rotation_vector=sanitize_sensor_frame(raw.rotation_vector, ["x", "y", "z"]),
        )

    def _build_step_signal(self, accelerometer: pd.DataFrame) -> StepSignal:
        if accelerometer.empty or len(accelerometer) < int(self.config["step_detection"]["min_samples"]):
            empty = np.array([], dtype=np.float64)
            empty_idx = np.array([], dtype=np.int64)
            return StepSignal(empty, empty, empty, empty_idx, empty, empty)

        timestamps = accelerometer["timestamp"].to_numpy(dtype=np.int64)
        values = accelerometer[["x", "y", "z"]].to_numpy(dtype=np.float64)

        alpha = float(self.config["step_detection"]["gravity_low_pass_alpha"])
        gravity = np.zeros_like(values, dtype=np.float64)
        gravity[0] = values[0]
        for index in range(1, len(values)):
            gravity[index] = alpha * gravity[index - 1] + (1.0 - alpha) * values[index]
        linear_accel = values - gravity

        source = self.config["step_detection"]["signal_source"]
        if source == "linear_accel_norm":
            raw_values = np.linalg.norm(linear_accel, axis=1)
        elif source == "acc_magnitude":
            raw_values = np.linalg.norm(values, axis=1)
        else:
            raise ValueError(f"Unsupported step_detection.signal_source: {source}")

        raw_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)

        sampling_period_ms = estimate_sample_period_ms(
            timestamps=timestamps,
            fallback_ms=float(self.config["data"]["fallback_sample_period_ms"]),
        )
        smooth_window_ms = float(self.config["step_detection"]["smooth_window_ms"])
        smooth_window_size = max(1, int(round(smooth_window_ms / max(sampling_period_ms, 1.0))))
        smooth_values = moving_average(raw_values, smooth_window_size)

        min_distance_ms = float(self.config["step_detection"]["peak_min_distance_ms"])
        min_distance = max(1, int(round(min_distance_ms / max(sampling_period_ms, 1.0))))

        peak_height_min = float(self.config["step_detection"]["peak_height_min"])
        peak_height_quantile = float(self.config["step_detection"]["peak_height_quantile"])
        quantile_height = float(np.quantile(smooth_values, peak_height_quantile)) if smooth_values.size else 0.0
        peak_height = max(peak_height_min, quantile_height)

        peak_prominence = float(self.config["step_detection"]["peak_prominence"])
        peak_indices, properties = find_peaks(
            smooth_values,
            distance=min_distance,
            prominence=peak_prominence,
            height=peak_height,
        )

        peak_timestamps = timestamps[peak_indices] if peak_indices.size else np.array([], dtype=np.int64)
        peak_values = properties["peak_heights"] if peak_indices.size else np.array([], dtype=np.float64)
        return StepSignal(
            timestamps=timestamps,
            raw_values=raw_values,
            smooth_values=smooth_values,
            peak_indices=peak_indices.astype(np.int64),
            peak_timestamps=peak_timestamps.astype(np.int64),
            peak_values=peak_values.astype(np.float64),
        )

    def _estimate_step_lengths(self, step_signal: StepSignal) -> np.ndarray:
        if step_signal.peak_indices.size == 0:
            return np.array([], dtype=np.float64)

        step_length_cfg = self.config["step_length"]
        mode = step_length_cfg["mode"]
        min_length = float(step_length_cfg["min_length_m"])
        max_length = float(step_length_cfg["max_length_m"])

        if mode == "fixed":
            lengths = np.full(step_signal.peak_indices.shape[0], float(step_length_cfg["fixed_length_m"]))
            return np.clip(lengths, min_length, max_length)

        if mode != "weinberg":
            raise ValueError(f"Unsupported step_length.mode: {mode}")

        window_ms = float(step_length_cfg["weinberg_window_ms"])
        sample_period_ms = estimate_sample_period_ms(
            timestamps=step_signal.timestamps,
            fallback_ms=float(self.config["data"]["fallback_sample_period_ms"]),
        )
        half_window = max(1, int(round(window_ms / max(sample_period_ms, 1.0) / 2.0)))
        k_value = float(step_length_cfg["weinberg_k"])

        lengths = np.zeros(step_signal.peak_indices.shape[0], dtype=np.float64)
        for idx, peak_index in enumerate(step_signal.peak_indices):
            start = max(0, int(peak_index) - half_window)
            end = min(step_signal.raw_values.size, int(peak_index) + half_window + 1)
            local_window = step_signal.raw_values[start:end]
            if local_window.size == 0:
                lengths[idx] = float(step_length_cfg["fixed_length_m"])
                continue
            amplitude = float(np.max(local_window) - np.min(local_window))
            if amplitude <= 0.0:
                lengths[idx] = float(step_length_cfg["fixed_length_m"])
            else:
                lengths[idx] = k_value * math.pow(amplitude, 0.25)

        return np.clip(lengths, min_length, max_length)

    def _heading_from_rotation_vector(
        self,
        rotation_vector: pd.DataFrame,
        query_timestamps: np.ndarray,
    ) -> Optional[np.ndarray]:
        if rotation_vector.empty or query_timestamps.size == 0:
            return None

        timestamps = rotation_vector["timestamp"].to_numpy(dtype=np.int64)
        if timestamps.size < 2:
            return None

        qx = rotation_vector["x"].to_numpy(dtype=np.float64)
        qy = rotation_vector["y"].to_numpy(dtype=np.float64)
        qz = rotation_vector["z"].to_numpy(dtype=np.float64)
        qw_sq = 1.0 - qx * qx - qy * qy - qz * qz
        qw = np.sqrt(np.clip(qw_sq, a_min=0.0, a_max=None))

        yaw = np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        yaw = np.unwrap(yaw)
        interpolated = np.interp(
            query_timestamps.astype(np.float64),
            timestamps.astype(np.float64),
            yaw.astype(np.float64),
        )
        offset = float(self.config["heading"]["yaw_offset_rad"])
        return wrap_angle_rad(interpolated + offset)

    def _heading_from_gyroscope(
        self,
        gyroscope: pd.DataFrame,
        query_timestamps: np.ndarray,
    ) -> Optional[np.ndarray]:
        if gyroscope.empty or query_timestamps.size == 0:
            return None

        timestamps = gyroscope["timestamp"].to_numpy(dtype=np.int64)
        if timestamps.size < 2:
            return None

        gz = gyroscope["z"].to_numpy(dtype=np.float64)
        delta_t = np.diff(timestamps.astype(np.float64)) / 1000.0
        max_dt = float(self.config["heading"]["max_gyro_dt_ms"]) / 1000.0
        delta_t = np.clip(delta_t, 0.0, max_dt)

        yaw = np.zeros_like(gz, dtype=np.float64)
        yaw[1:] = np.cumsum(gz[:-1] * delta_t)
        offset = float(self.config["heading"]["yaw_offset_rad"])
        interpolated = np.interp(
            query_timestamps.astype(np.float64),
            timestamps.astype(np.float64),
            yaw.astype(np.float64),
        )
        return wrap_angle_rad(interpolated + offset)

    def estimate_headings(
        self,
        rotation_vector: pd.DataFrame,
        gyroscope: pd.DataFrame,
        query_timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        primary_source = self.config["heading"]["primary_source"]
        fallback_source = self.config["heading"]["fallback_source"]
        ordered_sources = [primary_source]
        if fallback_source not in ordered_sources:
            ordered_sources.append(fallback_source)

        for source in ordered_sources:
            if source == "rotation_vector":
                headings = self._heading_from_rotation_vector(rotation_vector, query_timestamps)
            elif source == "gyroscope":
                headings = self._heading_from_gyroscope(gyroscope, query_timestamps)
            else:
                raise ValueError(f"Unsupported heading source: {source}")

            if headings is not None:
                return headings.astype(np.float64), source

        return np.zeros(query_timestamps.shape[0], dtype=np.float64), "none"

    def _predict_interval_deltas_from_sensors(
        self,
        sensors: SensorFrames,
        interval_timestamps: Sequence[int],
    ) -> pd.DataFrame:
        timestamps = np.asarray(interval_timestamps, dtype=np.int64)
        if timestamps.ndim != 1 or timestamps.size < 2:
            return pd.DataFrame(
                columns=[
                    "interval_index",
                    "start_timestamp",
                    "end_timestamp",
                    "pdr_delta_x",
                    "pdr_delta_y",
                    "step_count",
                    "distance_m",
                    "mean_heading_rad",
                    "heading_source",
                ]
            )

        step_signal = self._build_step_signal(sensors.accelerometer)
        step_lengths = self._estimate_step_lengths(step_signal)
        step_headings, heading_source = self.estimate_headings(
            rotation_vector=sensors.rotation_vector,
            gyroscope=sensors.gyroscope,
            query_timestamps=step_signal.peak_timestamps,
        )

        x_axis_sign = float(self.config["heading"]["x_axis_sign"])
        y_axis_sign = float(self.config["heading"]["y_axis_sign"])
        step_dx = x_axis_sign * step_lengths * np.sin(step_headings)
        step_dy = y_axis_sign * step_lengths * np.cos(step_headings)

        records: List[Dict[str, object]] = []
        for idx in range(timestamps.size - 1):
            start_ts = int(timestamps[idx])
            end_ts = int(timestamps[idx + 1])
            if end_ts <= start_ts:
                continue

            step_mask = (step_signal.peak_timestamps >= start_ts) & (step_signal.peak_timestamps < end_ts)
            interval_lengths = step_lengths[step_mask]
            interval_headings = step_headings[step_mask]

            records.append(
                {
                    "interval_index": idx,
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                    "pdr_delta_x": float(np.sum(step_dx[step_mask])),
                    "pdr_delta_y": float(np.sum(step_dy[step_mask])),
                    "step_count": int(np.sum(step_mask)),
                    "distance_m": float(np.sum(interval_lengths)),
                    "mean_heading_rad": circular_mean_rad(interval_headings),
                    "heading_source": heading_source,
                }
            )

        return pd.DataFrame.from_records(records)

    def predict_interval_deltas(
        self,
        file_path: Path | str,
        interval_timestamps: Sequence[int],
    ) -> pd.DataFrame:
        sensors = self.parse_file(file_path)
        return self._predict_interval_deltas_from_sensors(
            sensors=sensors,
            interval_timestamps=interval_timestamps,
        )

    def predict_leg_deltas(self, file_path: Path | str) -> pd.DataFrame:
        path = Path(file_path)
        sensors = self.parse_file(path)
        waypoints = sensors.waypoint

        if len(waypoints) < 2:
            return pd.DataFrame(
                columns=[
                    "site_id",
                    "floor",
                    "path_id",
                    "trajectory_file",
                    "leg_index",
                    "start_timestamp",
                    "end_timestamp",
                    "gt_start_x",
                    "gt_start_y",
                    "gt_end_x",
                    "gt_end_y",
                    "gt_delta_x",
                    "gt_delta_y",
                    "pdr_delta_x",
                    "pdr_delta_y",
                    "step_count",
                    "distance_m",
                    "mean_heading_rad",
                    "heading_source",
                ]
            )

        interval_table = self._predict_interval_deltas_from_sensors(
            sensors=sensors,
            interval_timestamps=waypoints["timestamp"].to_numpy(dtype=np.int64),
        )

        site_id = path.parent.parent.name if path.parent.parent != path.parent else ""
        floor = path.parent.name
        path_id = path.stem

        records: List[Dict[str, object]] = []
        for leg_index in range(len(waypoints) - 1):
            start_wp = waypoints.iloc[leg_index]
            end_wp = waypoints.iloc[leg_index + 1]
            interval_row = interval_table.iloc[leg_index] if leg_index < len(interval_table) else None

            gt_delta_x = float(end_wp["x"] - start_wp["x"])
            gt_delta_y = float(end_wp["y"] - start_wp["y"])

            records.append(
                {
                    "site_id": site_id,
                    "floor": floor,
                    "path_id": path_id,
                    "trajectory_file": str(path),
                    "leg_index": leg_index,
                    "start_timestamp": int(start_wp["timestamp"]),
                    "end_timestamp": int(end_wp["timestamp"]),
                    "gt_start_x": float(start_wp["x"]),
                    "gt_start_y": float(start_wp["y"]),
                    "gt_end_x": float(end_wp["x"]),
                    "gt_end_y": float(end_wp["y"]),
                    "gt_delta_x": gt_delta_x,
                    "gt_delta_y": gt_delta_y,
                    "pdr_delta_x": float(interval_row["pdr_delta_x"]) if interval_row is not None else 0.0,
                    "pdr_delta_y": float(interval_row["pdr_delta_y"]) if interval_row is not None else 0.0,
                    "step_count": int(interval_row["step_count"]) if interval_row is not None else 0,
                    "distance_m": float(interval_row["distance_m"]) if interval_row is not None else 0.0,
                    "mean_heading_rad": float(interval_row["mean_heading_rad"]) if interval_row is not None else 0.0,
                    "heading_source": str(interval_row["heading_source"]) if interval_row is not None else "none",
                }
            )

        result = pd.DataFrame.from_records(records)
        if self.config.get("output", {}).get("include_leg_error_columns", True) and not result.empty:
            result["delta_error_x"] = result["pdr_delta_x"] - result["gt_delta_x"]
            result["delta_error_y"] = result["pdr_delta_y"] - result["gt_delta_y"]
            result["delta_euclidean_error_m"] = np.sqrt(
                np.square(result["delta_error_x"].to_numpy(dtype=np.float64))
                + np.square(result["delta_error_y"].to_numpy(dtype=np.float64))
            )
        return result

    def predict_directory(
        self,
        data_dir: Path | str,
        output_path: Optional[Path | str] = None,
        limit_files: Optional[int] = None,
    ) -> pd.DataFrame:
        root = Path(data_dir)
        files = sorted(root.rglob("*.txt"))
        if limit_files is not None:
            files = files[: int(limit_files)]

        all_results: List[pd.DataFrame] = []
        for file_path in files:
            leg_df = self.predict_leg_deltas(file_path)
            if not leg_df.empty:
                all_results.append(leg_df)

        if all_results:
            result = pd.concat(all_results, ignore_index=True)
        else:
            result = pd.DataFrame()

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix.lower() == ".pkl":
                result.to_pickle(output)
            else:
                result.to_csv(output, index=False)

        return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDR baseline for Indoor Location & Navigation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--file", type=Path, default=None, help="Single trajectory .txt file.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing trajectory .txt files.")
    parser.add_argument("--output", type=Path, default=None, help="Output csv or pkl path.")
    parser.add_argument("--limit-files", type=int, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    pdr = PDR.from_yaml(args.config)

    if args.file is not None:
        result = pdr.predict_leg_deltas(args.file)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix.lower() == ".pkl":
                result.to_pickle(args.output)
            else:
                result.to_csv(args.output, index=False)
        print(result.head(10).to_string(index=False))
        print(f"\nrows={len(result)}")
        if "delta_euclidean_error_m" in result.columns and not result.empty:
            print(f"mean_delta_mae_m={result['delta_euclidean_error_m'].mean():.6f}")
        return

    if args.data_dir is None:
        default_dir = Path(pdr.config["paths"]["default_train_dir"])
        args.data_dir = PROJECT_ROOT / default_dir if not default_dir.is_absolute() else default_dir

    result = pdr.predict_directory(
        data_dir=args.data_dir,
        output_path=args.output,
        limit_files=args.limit_files,
    )

    print(f"processed_rows={len(result)}")
    if "delta_euclidean_error_m" in result.columns and not result.empty:
        print(f"mean_delta_mae_m={result['delta_euclidean_error_m'].mean():.6f}")


if __name__ == "__main__":
    main()
