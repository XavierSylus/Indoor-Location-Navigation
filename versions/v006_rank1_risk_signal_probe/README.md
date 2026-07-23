# v006_rank1_risk_signal_probe

## Goal
Create and diagnose inference-safe rank1 risk signals after v005 safety gating failed to beat the GSW reranker.

## Base
`v005_reranker_safety_policy`

## Data Flow
- Input: `hard_site_candidate_rerank_dataset.csv` and `topk_rerank_gsw_strong_cv_predictions.csv`.
- Processing: compute group-level rank1 support/risk features from candidate geometry, source WiFi, WiFi KNN, temporal smoothness, and reranker confidence margins.
- Model: none.
- Postprocess: evaluate deterministic policies that decide whether to use rank1 or the GSW reranker.
- Output: risk features, policy summary, and by-fold policy diagnostics.

## Changes
- Experiment type: `cv_probe`
- Iteration mode: `add_module`
- Route family: `rank1_risk_signal`
- Added `scripts/build_rank1_risk_signal_probe.py`.
- Added `tests/test_rank1_risk_signal_probe.py` with red-green coverage for leakage-safe feature declaration and diagnostic policy scoring.
- Added `configs/rank1_risk_signal_probe.yml`.
- No submission file and no Kaggle submission.

## Validation
- Holdout definition: hard-site Top-50 leave-one-path-id-out; 59 groups / 5 paths / 5 sites.
- Local metric: best policy `score_margin_positive`, MAE `7.524908`; best no-regression risk policy `risk_ge_3_rank_le_10_score_margin_positive`, MAE `8.055353`; oracle@50 MAE `0.583326`.
- Public LB: not applicable.
- Private LB: not applicable.
- CV/LB gap or explanation: CV-only diagnostic. The 59-group hard-site set is too small to treat as LB expectation.

## Files
- Config: `versions/v006_rank1_risk_signal_probe/config.yml`
- Script: `scripts/build_rank1_risk_signal_probe.py`
- Test: `tests/test_rank1_risk_signal_probe.py`
- Metrics: `data_processing/processed/rank1_risk_signal_policy_summary.json`
- Feature report: `data_processing/processed/rank1_risk_signal_features.csv`
- By-fold report: `data_processing/processed/rank1_risk_signal_policy_by_fold.csv`
- Validation report: `versions/v006_rank1_risk_signal_probe/validation_summary.json`
- Submission: none.

## Policy Results
- Rank1 baseline MAE: `9.268067`
- GSW always-rerank MAE: `7.524908`
- v005 score-margin-positive MAE: `7.524908`, bad-fold regression count `1`
- Best v006 policy: `score_margin_positive`, MAE `7.524908`
- Best no-regression v006 risk policy: `risk_ge_3_rank_le_10_score_margin_positive`, MAE `8.055353`
- Oracle@50 MAE: `0.583326`

## Leakage Boundary
- Policy features use only candidate/reranker inference-time fields.
- GT, label, and oracle distances are excluded from policy feature columns.
- GT-derived distances are used only after a policy selects rank1 or reranker output, for diagnostic scoring.
- Thresholds selected on this 59-group probe are diagnostic-only.

## Judgement
Rejected as a score-improving route. The new rank1 risk signal did not beat v005 safely. The only best-MAE policy remains the existing score-margin policy, and it still has one bad fold regression. A stricter no-regression risk policy removes fold regression but worsens MAE to `8.055353`.

This is useful negative evidence: current candidate-level confidence fields are not enough to identify when rank1 is safe. The next route should create stronger candidate/path context, not tune this policy further.

## What Worked
- The script produces explicit inference-safe policy columns and reports forbidden GT/oracle fields.
- The test verifies leakage-safe feature declaration and post-selection diagnostic scoring.
- The result clarifies that current rank1 risk fields are insufficient.

## What Failed
- No rank1-risk policy beat v005's `7.524908` MAE.
- The best-MAE policy still has one bad fold regression.
- Policies that remove fold regression give up too much reranker gain.

## Next
Run `v007_candidate_context_expansion`. The next information source should expand candidate/path context, for example local path-neighbor candidates, richer same-site/floor WiFi neighborhood context, or path-level consistency features that are available at inference.
