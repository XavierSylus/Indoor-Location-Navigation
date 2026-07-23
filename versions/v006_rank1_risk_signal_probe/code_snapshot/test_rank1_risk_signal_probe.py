import unittest

import pandas as pd

from scripts.build_rank1_risk_signal_probe import (
    build_rank1_risk_features,
    evaluate_policy,
)


class TestRank1RiskSignalProbe(unittest.TestCase):
    def test_build_rank1_risk_features_uses_inference_safe_policy_columns(self):
        candidates = pd.DataFrame(
            [
                {
                    "group_id": "g1",
                    "site_id": "s1",
                    "path_id": "p1",
                    "floor": "F1",
                    "timestamp": 1,
                    "point_index": 0,
                    "rank": 1,
                    "candidate_index": 10,
                    "is_rank1": 1,
                    "pred_to_candidate_m": 1.0,
                    "candidate_wifi_best_source_score": 0.90,
                    "candidate_wifi_top3_source_score_mean": 0.80,
                    "candidate_wifi_knn_score": 0.70,
                    "wifi_weighted_overlap_score": 0.60,
                    "candidate_temporal_smoothness_rank1_m": 0.20,
                    "label_distance_m": 0.50,
                    "group_oracle_distance_at_topk": 0.50,
                },
                {
                    "group_id": "g1",
                    "site_id": "s1",
                    "path_id": "p1",
                    "floor": "F1",
                    "timestamp": 1,
                    "point_index": 0,
                    "rank": 2,
                    "candidate_index": 11,
                    "is_rank1": 0,
                    "pred_to_candidate_m": 3.0,
                    "candidate_wifi_best_source_score": 0.50,
                    "candidate_wifi_top3_source_score_mean": 0.40,
                    "candidate_wifi_knn_score": 0.30,
                    "wifi_weighted_overlap_score": 0.20,
                    "candidate_temporal_smoothness_rank1_m": 2.00,
                    "label_distance_m": 8.00,
                    "group_oracle_distance_at_topk": 0.50,
                },
                {
                    "group_id": "g2",
                    "site_id": "s1",
                    "path_id": "p2",
                    "floor": "F1",
                    "timestamp": 2,
                    "point_index": 0,
                    "rank": 1,
                    "candidate_index": 20,
                    "is_rank1": 1,
                    "pred_to_candidate_m": 4.0,
                    "candidate_wifi_best_source_score": 0.20,
                    "candidate_wifi_top3_source_score_mean": 0.20,
                    "candidate_wifi_knn_score": 0.20,
                    "wifi_weighted_overlap_score": 0.10,
                    "candidate_temporal_smoothness_rank1_m": 4.00,
                    "label_distance_m": 9.00,
                    "group_oracle_distance_at_topk": 0.20,
                },
                {
                    "group_id": "g2",
                    "site_id": "s1",
                    "path_id": "p2",
                    "floor": "F1",
                    "timestamp": 2,
                    "point_index": 0,
                    "rank": 2,
                    "candidate_index": 21,
                    "is_rank1": 0,
                    "pred_to_candidate_m": 1.0,
                    "candidate_wifi_best_source_score": 0.95,
                    "candidate_wifi_top3_source_score_mean": 0.90,
                    "candidate_wifi_knn_score": 0.90,
                    "wifi_weighted_overlap_score": 0.80,
                    "candidate_temporal_smoothness_rank1_m": 0.50,
                    "label_distance_m": 0.20,
                    "group_oracle_distance_at_topk": 0.20,
                },
            ]
        )
        rerank = pd.DataFrame(
            [
                {
                    "group_id": "g1",
                    "fold": "path_01",
                    "heldout_key": "p1",
                    "selected_rank": 2,
                    "selected_gt_distance_m": 8.00,
                    "rank1_gt_distance_m": 0.50,
                    "oracle50_gt_distance_m": 0.50,
                    "predicted_score_margin": 0.40,
                    "selected_source_wifi_best_score": 0.50,
                    "rank1_source_wifi_best_score": 0.90,
                    "source_wifi_best_score_margin": -0.40,
                },
                {
                    "group_id": "g2",
                    "fold": "path_02",
                    "heldout_key": "p2",
                    "selected_rank": 2,
                    "selected_gt_distance_m": 0.20,
                    "rank1_gt_distance_m": 9.00,
                    "oracle50_gt_distance_m": 0.20,
                    "predicted_score_margin": 0.50,
                    "selected_source_wifi_best_score": 0.95,
                    "rank1_source_wifi_best_score": 0.20,
                    "source_wifi_best_score_margin": 0.75,
                },
            ]
        )

        features, policy_columns = build_rank1_risk_features(candidates, rerank)

        forbidden = {"label_distance_m", "selected_gt_distance_m", "rank1_gt_distance_m", "oracle50_gt_distance_m"}
        self.assertTrue(forbidden.isdisjoint(policy_columns))

        g1 = features.set_index("group_id").loc["g1"]
        g2 = features.set_index("group_id").loc["g2"]
        self.assertLess(g1["rank1_risk_score"], g2["rank1_risk_score"])
        self.assertEqual(g1["rank1_source_wifi_best_rank"], 1)
        self.assertEqual(g2["rank1_source_wifi_best_rank"], 2)
        self.assertEqual(g2["rank1_geometry_distance_rank"], 2)

    def test_evaluate_policy_scores_after_selection(self):
        features = pd.DataFrame(
            [
                {
                    "group_id": "g1",
                    "fold": "path_01",
                    "heldout_key": "p1",
                    "selected_gt_distance_m": 8.0,
                    "rank1_gt_distance_m": 0.5,
                    "oracle50_gt_distance_m": 0.5,
                },
                {
                    "group_id": "g2",
                    "fold": "path_02",
                    "heldout_key": "p2",
                    "selected_gt_distance_m": 0.2,
                    "rank1_gt_distance_m": 9.0,
                    "oracle50_gt_distance_m": 0.2,
                },
            ]
        )

        metrics = evaluate_policy(features, pd.Series([False, True], index=features.index))

        self.assertEqual(metrics["n_groups"], 2)
        self.assertAlmostEqual(metrics["selected_mae"], 0.35)
        self.assertAlmostEqual(metrics["improvement_vs_rank1"], 4.4)
        self.assertAlmostEqual(metrics["sum_excess_over_4m"], 0.0)


if __name__ == "__main__":
    unittest.main()
