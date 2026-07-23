from __future__ import annotations

import unittest

import pandas as pd

from data_processing.path_safe_delta_oof import (
    build_path_split_manifest,
    validate_oof_predictions,
)


class PathSafeDeltaOOFTest(unittest.TestCase):
    def test_split_manifest_excludes_every_heldout_path(self) -> None:
        paths = pd.DataFrame(
            {
                "site_id": ["s1", "s1", "s2", "s2"],
                "path_id": ["train-a", "hold-a", "train-b", "hold-b"],
            }
        )
        manifest = build_path_split_manifest(paths, {"hold-a", "hold-b"})

        train_paths = set(manifest.loc[manifest["split"] == "train", "path_id"])
        heldout_paths = set(manifest.loc[manifest["split"] == "validation", "path_id"])

        self.assertEqual(heldout_paths, {"hold-a", "hold-b"})
        self.assertFalse(train_paths & heldout_paths)

    def test_split_manifest_rejects_missing_heldout_path(self) -> None:
        paths = pd.DataFrame({"site_id": ["s1"], "path_id": ["train-a"]})

        with self.assertRaisesRegex(ValueError, "missing"):
            build_path_split_manifest(paths, {"missing"})

    def test_oof_validation_requires_exact_interval_coverage(self) -> None:
        expected = pd.DataFrame(
            {
                "path_id": ["hold-a", "hold-a"],
                "leg_index": [0, 1],
                "start_timestamp": [10, 20],
                "end_timestamp": [20, 30],
            }
        )
        predictions = expected.copy()
        predictions["v3_delta_x"] = [1.0, 2.0]
        predictions["v3_delta_y"] = [0.0, 1.0]

        summary = validate_oof_predictions(
            predictions=predictions,
            expected_intervals=expected,
            train_path_ids={"train-a"},
            validation_path_ids={"hold-a"},
        )

        self.assertEqual(summary["path_overlap_count"], 0)
        self.assertEqual(summary["expected_interval_count"], 2)
        self.assertEqual(summary["prediction_interval_count"], 2)

    def test_oof_validation_rejects_training_path_overlap(self) -> None:
        expected = pd.DataFrame(
            {
                "path_id": ["hold-a"],
                "leg_index": [0],
                "start_timestamp": [10],
                "end_timestamp": [20],
            }
        )
        predictions = expected.assign(v3_delta_x=1.0, v3_delta_y=0.0)

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_oof_predictions(
                predictions=predictions,
                expected_intervals=expected,
                train_path_ids={"hold-a"},
                validation_path_ids={"hold-a"},
            )


if __name__ == "__main__":
    unittest.main()
