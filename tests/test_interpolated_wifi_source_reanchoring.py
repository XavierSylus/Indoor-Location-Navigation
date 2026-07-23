from __future__ import annotations

import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from data_processing.interpolated_wifi_source_reanchoring import (
    assign_interpolated_to_grid,
    current_binding_stats,
    interpolate_scan_positions,
    nearest_indices,
    resolve_runtime_train_root,
)


class InterpolatedWifiSourceReanchoringTest(unittest.TestCase):
    def test_interpolates_wifi_scan_between_waypoints(self) -> None:
        waypoints = np.array(
            [
                [1000.0, 0.0, 0.0],
                [3000.0, 4.0, 0.0],
            ]
        )

        result = interpolate_scan_positions(np.array([2000]), waypoints)

        self.assertEqual(1, len(result))
        self.assertAlmostEqual(2.0, result.loc[0, "interpolated_x"])
        self.assertAlmostEqual(0.0, result.loc[0, "interpolated_y"])
        self.assertAlmostEqual(2.0, result.loc[0, "displacement_m"])
        self.assertEqual(1000, result.loc[0, "time_gap_ms"])

    def test_does_not_extrapolate_outside_waypoint_interval(self) -> None:
        waypoints = np.array(
            [
                [1000.0, 0.0, 0.0],
                [3000.0, 4.0, 0.0],
            ]
        )

        result = interpolate_scan_positions(
            np.array([500, 1000, 3000, 3500]), waypoints
        )

        self.assertEqual([1000, 3000], result["wifi_timestamp"].tolist())

    def test_current_binding_detects_reused_scan(self) -> None:
        stats = current_binding_stats(
            wifi_timestamps=np.array([1000, 5000]),
            waypoint_timestamps=np.array([900, 1100, 5000]),
            window_ms=5000,
        )

        self.assertEqual(3, stats["waypoint_assignments"])
        self.assertEqual(2, stats["unique_scan_assignments"])
        self.assertAlmostEqual(1.0 / 3.0, stats["duplicate_binding_ratio"])

    def test_nearest_timestamp_tie_is_deterministic(self) -> None:
        indices = nearest_indices(np.array([1000, 3000]), np.array([2000]))

        np.testing.assert_array_equal(indices, np.array([0]))

    def test_reanchoring_assigns_each_scan_to_one_nearest_grid_point(self) -> None:
        scan_positions = np.array([[0.2, 0.0], [9.7, 0.0]])
        grid = np.array([[0.0, 0.0], [10.0, 0.0]])

        indices, distances = assign_interpolated_to_grid(scan_positions, grid)

        np.testing.assert_array_equal(indices, np.array([0, 1]))
        np.testing.assert_allclose(distances, np.array([0.2, 0.3]))

    def test_direct_script_execution_can_import_project_modules(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        script = (
            project_root
            / "data_processing"
            / "interpolated_wifi_source_reanchoring.py"
        )
        config = (
            project_root
            / "configs"
            / "interpolated_wifi_source_reanchoring.json"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--config",
                str(config),
                "--stage",
                "kaggle",
            ],
            cwd=script.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertIn("may run only inside Kaggle", result.stderr)

    def test_runtime_train_root_is_selected_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_train = root / "local" / "train"
            local_train.mkdir(parents=True)
            kaggle_input = root / "input"
            mounted_train = kaggle_input / "nested-competition-mount" / "train"
            sample = mounted_train / "site" / "floor" / "path.txt"
            sample.parent.mkdir(parents=True)
            sample.write_text("", encoding="utf-8")
            config = {
                "inputs": {
                    "train_root": str(local_train),
                    "kaggle_input_root": str(kaggle_input),
                }
            }

            local = resolve_runtime_train_root(config, kaggle_runtime=False)
            kaggle = resolve_runtime_train_root(config, kaggle_runtime=True)

            self.assertEqual(local_train, local)
            self.assertEqual(mounted_train, kaggle)


if __name__ == "__main__":
    unittest.main()
