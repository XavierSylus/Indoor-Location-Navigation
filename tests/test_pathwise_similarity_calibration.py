from __future__ import annotations

import math
import unittest

import numpy as np

from data_processing.pathwise_similarity_calibration import (
    apply_similarity_transform,
    fit_similarity_transform,
    validate_calibration_feature_columns,
)


class PathwiseSimilarityCalibrationTest(unittest.TestCase):
    def test_recovers_known_rotation_and_scale(self) -> None:
        source = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
        target = np.array([[0.0, 2.0], [-4.0, 0.0], [-2.0, 4.0]])

        transform = fit_similarity_transform(source, target)
        calibrated = apply_similarity_transform(source, transform)

        self.assertAlmostEqual(transform["scale"], 2.0, places=10)
        self.assertAlmostEqual(transform["rotation_rad"], math.pi / 2.0, places=10)
        np.testing.assert_allclose(calibrated, target, atol=1e-10)

    def test_fit_is_deterministic_without_translation(self) -> None:
        source = np.array([[1.0, 2.0], [2.0, -1.0]])
        target = source * 1.25

        first = fit_similarity_transform(source, target)
        second = fit_similarity_transform(source, target)

        self.assertEqual(first, second)
        self.assertNotIn("translation_x", first)
        self.assertNotIn("translation_y", first)

    def test_rejects_label_or_oracle_calibration_features(self) -> None:
        validate_calibration_feature_columns(
            ["v3_delta_x", "v3_delta_y", "unary_delta_x", "unary_delta_y"]
        )

        with self.assertRaisesRegex(ValueError, "label/oracle"):
            validate_calibration_feature_columns(["v3_delta_x", "gt_delta_x"])

    def test_rejects_zero_energy_source(self) -> None:
        source = np.zeros((2, 2), dtype=np.float64)
        target = np.ones((2, 2), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "energy"):
            fit_similarity_transform(source, target)


if __name__ == "__main__":
    unittest.main()
