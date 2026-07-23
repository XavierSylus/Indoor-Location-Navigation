# Experiment Ledger

Last updated: 2026-04-20

This file is the persistent experiment ledger for submission-facing work.
Do not create a new default submission path unless it is recorded here together with its validation evidence.

## Required Fields For Every New Entry

- date
- submission file
- route family
- config path
- validation report
- git commit
- seed
- holdout definition
- public leaderboard score
- private leaderboard score
- decision
- notes

## Normalization Rules

- If a field is not known for a historical run, record `unknown` instead of guessing.
- If a submission was uploaded but exact leaderboard values were not preserved, say `submitted, not better than current best`.
- `decision` must be one of: `mainline`, `exploratory`, `rejected`, `historical`.
- Validation reports must point to a durable file, not a console log.

## Current Best Reference

| date | submission file | route family | config path | validation report | git commit | seed | holdout definition | public LB | private LB | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| unknown | `submission_step3_beamsearch.csv` | `step3 global beam` | `configs/kaggle_train_config.yml` | `unknown` | `unknown` | `unknown` | `unknown` | 6.64 | 7.90 | `historical` | Best recorded leaderboard result still in use |

## Recent Submission Records

| date | submission file | route family | config path | validation report | git commit | seed | holdout definition | public LB | private LB | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-19 to 2026-04-20 | `submission_step3_optimized_ensemble.csv` | `step3 V3+PDR exploratory` | `configs/pdr_v3_ensemble.yml` | `data_processing/processed/pdr_v3_ensemble_metrics.json` | `a8e918d419a7ef92c48dc4672b3ea41aaa11e0fa` | unknown | validation legs in `pdr_v3_ensemble_metrics.json` | unknown | submitted, not better than current best | `rejected` | Local delta ensemble is worse than V3 alone: `2.215767 m` vs `2.065458 m` |
| 2026-04-19 to 2026-04-20 | `submission_step3_gated_mean15.csv` | `step3 beam gating exploratory` | `configs/kaggle_train_config.yml` | `data_processing/processed/beam_gating_summary.json` | `a8e918d419a7ef92c48dc4672b3ea41aaa11e0fa` | 42 | `5` holdout paths / `40` points | unknown | submitted, not better than current best | `rejected` | Best local gain is only `-0.073917 m`; evidence too small for private-LB promotion |
| 2026-05-09 | `versions/v001_historical_step3_beamsearch/submission.csv` | `v001_historical_step3_beamsearch` | `versions/v001_historical_step3_beamsearch/config.yml` | `versions/v001_historical_step3_beamsearch/validation_summary.json` | `a8e918d419a7ef92c48dc4672b3ea41aaa11e0fa` | `unknown` | `unknown` | 6.64 | 7.90 | `historical` | Backfilled current best recorded historical submission into versioned workflow; original exact command/seed/holdout unknown. |
| 2026-05-09 | `none` | `v002_unified_validation_baseline` | `versions/v002_unified_validation_baseline/config.yml` | `data_processing/processed/unified_validation_report.json` | `a8e918d419a7ef92c48dc4672b3ea41aaa11e0fa` | `42` | `configured in configs/unified_validation.yml` | unknown | unknown | `exploratory` | Created reproducible unified validation baseline; no Kaggle submission. |
| 2026-05-09 | `none` | `v003_site_selective_beam_report` | `versions/v003_site_selective_beam_report/config.yml` | `data_processing/processed/unified_validation_report.json` | `a8e918d419a7ef92c48dc4672b3ea41aaa11e0fa` | `42` | `same comparable holdout as safe_beam and beam_gating; site-selective coverage 2 paths / 13 points` | unknown | unknown | `exploratory` | Site-selective beam improves the whitelist subset but is not promotable because evidence covers only 2 paths / 13 points. |
| 2026-05-09 | `none` | `v004_error_source_score_waterfall` | `versions/v004_error_source_score_waterfall/config.yml` | `data_processing/processed/error_source_score_waterfall.json` | `a8e918d419a7ef92c48dc4672b3ea41aaa11e0fa` | `42` | `hard-site top50 diagnostic holdout; 59 groups / 5 paths / 5 sites` | unknown | unknown | `exploratory` | Score waterfall shows selection failure is the dominant excess-over-4m source; recommended v005 is reranker_safety_policy. CV-only diagnostic, no submission. |

## Supporting Validation References

| artifact | purpose | key conclusion |
| --- | --- | --- |
| `data_processing/processed/pdr_v3_ensemble_metrics.json` | compare V3 delta vs PDR delta vs weighted ensemble | weighted V3 + PDR is worse than V3 alone |
| `data_processing/processed/safe_beam_param_search_summary.json` | global beam parameter search | best global beam setting still regresses against WiFi baseline |
| `data_processing/processed/beam_gating_summary.json` | beam gating sweep | gating evidence is too weak to define a new mainline |
| `data_processing/processed/grid_discretization_summary.json` | grid snapping / candidate gap diagnosis | absolute candidate quality is the main bottleneck |

## Entry Template

Copy this row for the next serious run:

```text
| YYYY-MM-DD | `submission_name.csv` | `route family` | `configs/...yml` | `path/to/report.json` | `<git commit>` | `<seed>` | `<holdout>` | <public> | <private> | `mainline/exploratory/rejected/historical` | short conclusion |
```
