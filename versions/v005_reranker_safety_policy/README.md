# v005_reranker_safety_policy

## Goal
Evaluate whether the current Top-50 reranker can be made safer with inference-available gating rules after v004 showed that selection failure dominates the 4m gap.

## Base
`v004_error_source_score_waterfall`

## Data Flow
- Input: `hard_site_candidate_rerank_dataset.csv`
- Processing: train leave-one-path-id-out rerankers, emit group-level CV selections, then compare policy decisions against rank1/rerank/oracle only after selection.
- Model: LightGBM regressor with strong regularization, seed `42`.
- Postprocess: policy diagnostics for rank, predicted score margin, WiFi margin, and diagnostic site/floor blacklist.
- Output: named CV predictions, reranker summaries, and `reranker_safety_policy_comparison.json`.

## Changes
- Experiment type: `cv_probe`
- Iteration mode: `tune_module`
- Route family: `reranker_safety_policy`
- Compared two rerankers:
  - `geometry_source_wifi` strong
  - `geometry_wifi_source_temporal_core` strong
- No submission file, no beam/PDR change, no official model saved.

## Validation
- Holdout definition: hard-site Top-50 leave-one-path-id-out; 59 groups / 5 paths / 5 sites.
- Local metric: best submission-safe MAE `7.524908`; best diagnostic blacklist MAE `7.361871`; oracle@50 MAE `0.583326`.
- Public LB: not applicable.
- Private LB: not applicable.
- CV/LB gap or explanation: CV-only diagnostic on a very small hard-site set; not a direct LB expectation.

## Files
- Config: `versions/v005_reranker_safety_policy/config.yml`
- Script: `scripts/train_topk_rerank_baseline.py`
- Metrics: `data_processing/processed/reranker_safety_policy_comparison.json`
- Prediction outputs:
  - `data_processing/processed/topk_rerank_gsw_strong_cv_predictions.csv`
  - `data_processing/processed/topk_rerank_temporal_strong_cv_predictions.csv`
  - `data_processing/processed/topk_rerank_gsw_strong_blacklist_diag_cv_predictions.csv`
  - `data_processing/processed/topk_rerank_temporal_strong_blacklist_diag_cv_predictions.csv`
- Summary outputs:
  - `data_processing/processed/topk_rerank_gsw_strong_summary.json`
  - `data_processing/processed/topk_rerank_temporal_strong_summary.json`
  - `data_processing/processed/topk_rerank_gsw_strong_blacklist_diag_summary.json`
  - `data_processing/processed/topk_rerank_temporal_strong_blacklist_diag_summary.json`
- Validation report: `versions/v005_reranker_safety_policy/validation_summary.json`
- Submission: none.

## Policy Results
- Rank1 baseline: MAE `9.268067`, hit@3 `0.135593`, hit@5 `0.288136`.
- GSW always-rerank: MAE `7.524908`, hit@3 `0.288136`, hit@5 `0.457627`.
- GSW score-margin-positive: MAE `7.524908`, usage `0.745763`; same MAE as always-rerank with lower usage.
- GSW diagnostic site/floor blacklist: MAE `7.361871`, hit@3 `0.305085`, hit@5 `0.474576`, usage `0.949153`.
- Temporal always-rerank: MAE `8.030249`, hit@3 `0.220339`, hit@5 `0.423729`.
- Temporal diagnostic site/floor blacklist: MAE `8.023926`, usage `0.949153`.

## Leakage Boundary
- Inference-safe policy rules use only selected rank, predicted score, score margins, WiFi margins, site/floor identifiers, and other prediction-time fields.
- GT, labels, and oracle distances are used only for diagnostic scoring after a policy selects rank1 or reranker output.
- The site/floor blacklist is post-hoc and diagnostic-only. It must not be used for submission.

## Judgement
GSW remains the best reranker, but the simple safety policies do not solve the fold-specific regression. The only result that beats GSW always-rerank is the diagnostic blacklist, which proves there is recoverable error but does not provide a generalizable submission-safe rule.

This means the next useful iteration is not another reranker hyperparameter pass. The missing piece is an inference-safe rank1/rerank risk signal that can identify when rank1 is already good and when deeper reranker choices are justified.

## What Worked
- GSW strong reproduced the current best rerank MAE: `7.524908`.
- Score-margin-positive is a conservative tie with lower usage than always-rerank.
- Post-hoc blacklist recovered the known regression fold and lowered diagnostic MAE to `7.361871`.

## What Failed
- `rank <= 10`, `rank <= 20`, WiFi margin, and combined rank/WiFi rules did not beat GSW always-rerank.
- The known bad fold `5da138364db8ce0c98bc00f1|F3` is not separable by the current simple policy features.
- The best diagnostic policy is not submission-safe.

## Next
Run `v006_rank1_risk_signal_probe`. It should create a new inference-safe signal for rank1 confidence / rerank risk, rather than tuning the same reranker again.
