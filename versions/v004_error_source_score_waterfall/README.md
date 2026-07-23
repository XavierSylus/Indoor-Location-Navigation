# v004_error_source_score_waterfall

## Goal
Build a 4m-oriented score waterfall and excess-over-4m error ledger. This version answers where the current hard-site validation error comes from before creating the next information source.

## Base
`v003_site_selective_beam_report` plus current Top-50 candidate recall and reranker CV diagnostics.

## Data Flow
- Input: `hard_site_candidate_topk_points.csv`, `hard_site_candidate_rerank_dataset.csv`, `topk_rerank_baseline_cv_predictions.csv`, absolute holdout reports, beam reports, and PDR metric reports.
- Processing: bridge incompatible metric scopes, compute selected/rank1/oracle ceilings, classify candidate missing vs selection failure, and aggregate `excess_over_4m = max(error - 4, 0)`.
- Model: none.
- Postprocess: site/floor, path, point, floor, and path-shift diagnostic summaries.
- Output: CV-only reports under `data_processing/processed/`.

## Changes
- Experiment type: `cv_probe`
- Iteration mode: `add_module`
- Route family: `error_source_score_waterfall`
- Added a diagnostic ledger; no submission file, no model tuning, no beam/PDR change.

## Validation
- Holdout definition: hard-site Top-50 diagnostic holdout; 59 groups / 5 paths / 5 sites.
- Local metric: rank1 MAE `9.268067`; current reranker MAE `8.030249`; oracle@50 MAE `0.583326`.
- Public LB: not applicable.
- Private LB: not applicable.
- CV/LB gap or explanation: diagnostic-only CV probe. Results are not a direct LB expectation because the validation set has only 59 groups and includes oracle/GT-only analyses.

## Files
- Config: `versions/v004_error_source_score_waterfall/config.yml`
- Script: `scripts/build_error_source_score_waterfall.py`
- Metrics: `data_processing/processed/error_source_score_waterfall.json`
- Reports:
  - `data_processing/processed/metric_bridge.csv`
  - `data_processing/processed/oracle_ceiling_table.csv`
  - `data_processing/processed/error_source_by_site_floor.csv`
  - `data_processing/processed/error_source_by_path.csv`
  - `data_processing/processed/error_source_by_point.csv`
  - `data_processing/processed/floor_error_report.csv`
  - `data_processing/processed/path_geometry_repairability.csv`
- Validation report: `versions/v004_error_source_score_waterfall/validation_summary.json`
- Submission: none.

## Score Waterfall
- raw prediction MAE: `9.148229`, sum excess over 4m: `319.651142`
- rank1 MAE: `9.268067`, sum excess over 4m: `340.487957`
- current reranker CV MAE: `8.030249`, sum excess over 4m: `290.185569`
- oracle@50 MAE: `0.583326`, sum excess over 4m: `10.078551`
- path-shift oracle MAE: `7.907911`, gain vs absolute points: `1.240319`

## Error Contribution
- `selection_failure`: 46 / 59 groups, rank1 excess `250.589584`, oracle excess `0.000000`
- `candidate_missing`: 3 / 59 groups, rank1 excess `89.898373`, oracle excess `10.078551`
- `low_value`: 10 / 59 groups, rank1 excess `0.000000`

## Leakage Boundary
- Oracle candidate, path-shift, residual, and bias rows use GT and are diagnostic-only.
- No oracle or label-derived result may be converted into a submission rule without out-of-fold or submission-safe validation.
- PDR delta-leg metrics are marked non-comparable to waypoint-level submission metrics in `metric_bridge.csv`.
- Point-level floor prediction columns are not available in the current artifact, so floor error is reported only from available aggregate floor metrics.

## Judgement
The next route should be `v005_reranker_safety_policy`. The Top-50 candidate pool already contains near-ground-truth options for almost all groups, so the main 4m gap is not candidate recall on this diagnostic set. The dominant loss is selection failure: rank1 is often wrong, and the reranker improves average MAE but still damages some rank1-good groups.

## What Worked
- The metric bridge prevents mixing LB MPE, waypoint MAE, reranker CV MAE, and PDR delta-leg metrics.
- The oracle ceiling shows a strong reachable upper bound: oracle@50 MAE is far below 4m.
- The excess-over-4m ledger identifies selection failure as the largest tail-error source.

## What Failed
- The current reranker is not regression-safe. It reduces mean MAE but still leaves high tail error and can worsen rank1-good folds.
- Path-shift oracle gives only limited gain here and is GT-dependent.
- Floor diagnostics are incomplete because point-level predicted floor is not present in the current prediction artifacts.

## Next
Run `v005_reranker_safety_policy` as a CV-only policy probe. It should compare strong `geometry_source_wifi` and strong `geometry_wifi_source_temporal_core`, evaluate inference-safe gating rules, and select a default-rank1 override policy only if it improves MAE without fold-specific regression.
