from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_processing.kaggle_pathsafe_smoke import (
    build_path_legs,
    require_kaggle_runtime,
    split_paths,
)


class KagglePathsafeSmokeTests(unittest.TestCase):
    def test_build_path_legs_uses_only_interval_sensors_and_waypoint_delta(self) -> None:
        content = "\n".join(
            [
                "100\tTYPE_WAYPOINT\t0.0\t0.0",
                "110\tTYPE_ACCELEROMETER\t1.0\t2.0\t2.0",
                "120\tTYPE_GYROSCOPE\t0.0\t0.0\t1.0",
                "200\tTYPE_WAYPOINT\t3.0\t4.0",
                "210\tTYPE_ACCELEROMETER\t9.0\t9.0\t9.0",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "path_a.txt"
            path.write_text(content, encoding="utf-8")
            legs = build_path_legs(path)
        self.assertEqual(1, len(legs))
        self.assertEqual(3.0, float(legs.iloc[0]["delta_x"]))
        self.assertEqual(4.0, float(legs.iloc[0]["delta_y"]))
        features = np.asarray(legs.iloc[0]["features"])
        self.assertEqual(21, len(features))
        self.assertEqual(1.0, features[1])

    def test_split_paths_has_no_path_overlap(self) -> None:
        train, validation = split_paths(
            ["a", "a", "b", "b", "c", "c", "d", "d"],
            validation_paths=2,
            minimum_train_paths=2,
        )
        self.assertEqual(["a", "b"], train)
        self.assertEqual(["c", "d"], validation)
        self.assertFalse(set(train) & set(validation))

    def test_split_paths_rejects_insufficient_complete_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "Not enough paths"):
            split_paths(["a", "a", "b"], validation_paths=1, minimum_train_paths=2)

    def test_runtime_guard_rejects_local_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Kaggle-only"):
                require_kaggle_runtime(Path(directory))


if __name__ == "__main__":
    unittest.main()
