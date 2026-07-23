from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data_processing.candidate_transition_lattice import (
    build_path_delta_lookup,
    decode_candidate_path,
    validate_oof_candidate_keys,
    validate_transition_feature_columns,
)


class CandidateTransitionLatticeTests(unittest.TestCase):
    def test_transition_recovers_consistent_path_when_unary_prefers_wrong_candidate(self) -> None:
        candidates = pd.DataFrame(
            [
                {"group_id": "g1", "timestamp": 100, "candidate_index": 1, "candidate_x": 0.0, "candidate_y": 0.0, "predicted_score": 0.2},
                {"group_id": "g1", "timestamp": 100, "candidate_index": 2, "candidate_x": 10.0, "candidate_y": 0.0, "predicted_score": 0.0},
                {"group_id": "g2", "timestamp": 200, "candidate_index": 3, "candidate_x": 1.0, "candidate_y": 0.0, "predicted_score": 0.2},
                {"group_id": "g2", "timestamp": 200, "candidate_index": 4, "candidate_x": 20.0, "candidate_y": 0.0, "predicted_score": 0.0},
            ]
        )
        selected = decode_candidate_path(
            candidates,
            {(100, 200): np.array([1.0, 0.0])},
            alpha=4.0,
            pairwise_cap_m=10.0,
        )
        self.assertEqual([1, 3], selected["candidate_index"].tolist())

    def test_exact_delta_chain_accumulates_only_prediction_columns(self) -> None:
        v3 = pd.DataFrame(
            [
                {"start_timestamp": 100, "end_timestamp": 150, "v3_delta_x": 1.0, "v3_delta_y": 2.0},
                {"start_timestamp": 150, "end_timestamp": 200, "v3_delta_x": 3.0, "v3_delta_y": 4.0},
            ]
        )
        lookup = build_path_delta_lookup([100, 200], v3, require_exact_chain=True)
        np.testing.assert_allclose(lookup[(100, 200)], np.array([4.0, 6.0]))

    def test_missing_exact_chain_raises(self) -> None:
        v3 = pd.DataFrame(
            [{"start_timestamp": 100, "end_timestamp": 150, "v3_delta_x": 1.0, "v3_delta_y": 2.0}]
        )
        with self.assertRaisesRegex(ValueError, "No exact V3 interval chain"):
            build_path_delta_lookup([100, 200], v3, require_exact_chain=True)

    def test_label_or_oracle_transition_feature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "label/oracle"):
            validate_transition_feature_columns(["v3_delta_x", "gt_delta_x"])

    def test_candidate_index_may_repeat_across_groups(self) -> None:
        scored = pd.DataFrame(
            [
                {"group_id": "g1", "candidate_index": 1},
                {"group_id": "g2", "candidate_index": 1},
            ]
        )
        validate_oof_candidate_keys(scored)


if __name__ == "__main__":
    unittest.main()
