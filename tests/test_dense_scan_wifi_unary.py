from __future__ import annotations

import unittest

import numpy as np

from data_processing.dense_scan_wifi_unary import (
    fit_dense_scan_knn,
    interpolate_predictions,
    predict_dense_scan_xy,
)
from scripts.train_topk_rerank_baseline import get_numeric_features


class DenseScanWifiUnaryTest(unittest.TestCase):
    def test_interpolates_dense_scan_predictions_at_waypoint_time(self) -> None:
        result = interpolate_predictions(
            np.array([1000, 3000]), np.array([[0.0, 0.0], [4.0, 0.0]]), np.array([2000])
        )
        np.testing.assert_allclose(result, np.array([[2.0, 0.0]]))

    def test_extends_endpoint_predictions_for_waypoints_outside_scan_range(self) -> None:
        result = interpolate_predictions(
            np.array([1000, 3000]),
            np.array([[2.0, 3.0], [8.0, 9.0]]),
            np.array([500, 3500]),
        )
        np.testing.assert_allclose(result, np.array([[2.0, 3.0], [8.0, 9.0]]))

    def test_weighted_neighbor_prediction_preserves_matching_coordinate(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        coordinates = np.array([[2.0, 3.0], [8.0, 9.0]])
        model = fit_dense_scan_knn(vectors, coordinates, 2)
        result = predict_dense_scan_xy(model, coordinates, np.array([[1.0, 0.0]]))
        np.testing.assert_allclose(result, np.array([[2.0, 3.0]]), atol=1e-5)

    def test_dense_scan_feature_set_consumes_continuous_unary_features(self) -> None:
        features = get_numeric_features("geometry_source_wifi_dense_scan")
        self.assertIn("dense_scan_unary_distance_m", features)
        self.assertIn("dense_scan_unary_rank", features)


if __name__ == "__main__":
    unittest.main()
