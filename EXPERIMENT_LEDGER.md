# Experiment Ledger

Last updated: 2026-07-23

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
| 2026-05-12 | `none` | `v005_reranker_safety_policy` | `versions/v005_reranker_safety_policy/config.yml` | `data_processing/processed/reranker_safety_policy_comparison.json` | `a8e918d419a7ef92c48dc4672b3ea41aaa11e0fa` | `42` | `hard-site Top-50 leave-one-path-id-out; 59 groups / 5 paths / 5 sites` | unknown | unknown | `exploratory` | Inference-safe simple policies do not solve fold regression; post-hoc blacklist is diagnostic-only. Next v006 should create rank1-risk signal. |
| 2026-05-12 | `none` | `v006_rank1_risk_signal_probe` | `versions/v006_rank1_risk_signal_probe/config.yml` | `data_processing/processed/rank1_risk_signal_policy_summary.json` | `a8e918d419a7ef92c48dc4672b3ea41aaa11e0fa` | `42` | `hard-site Top-50 leave-one-path-id-out; 59 groups / 5 paths / 5 sites` | unknown | unknown | `rejected` | Rank1 risk signal did not beat v005 safely. Best remains score_margin_positive, with one bad fold regression. Next route should create stronger candidate/path context. |
| 2026-07-23 | `none` | `v007_candidate_transition_lattice` | `versions/v007_candidate_transition_lattice/config.yml` | `data_processing/processed/candidate_transition_lattice_summary.json` | `3e2c70e1fdf94df6003ebdf2d5305c63fd84d52e` | `42` | `hard-site Top-50 leave-one-path-out; 59 groups / 5 paths; nested alpha calibration` | unknown | unknown | `exploratory` | `dna:v007_candidate_transition_lattice`; unary `7.524908m` -> lattice `1.355896m`, excess-over-4m reduced `91.3658%`. Promote the transition-lattice gene, but existing V3 checkpoint is not path-safe; next is `v008_path_safe_delta_oof`, no submission. |
| 2026-07-23 | `none` | `v008_path_safe_delta_oof` | `configs/path_safe_delta_oof.json` | `versions/v008_path_safe_delta_oof/validation_summary.json` | `459ef04a79e2437ff3a9f11770c9f6177ff7de24` | `42` | `five complete held-out paths; 59 groups; 54/54 path-safe delta intervals; nested alpha calibration` | unknown | unknown | `exploratory` | `dna:v008_path_safe_delta_oof,parent:v007_candidate_transition_lattice`; unary `7.524908m` -> path-safe lattice `4.346602m`, excess-over-4m reduced `54.4469%`, but `<3m` gate failed. Promote the transition gene, reject the score, no submission. Next: pathwise rotation/scale calibration. |
| 2026-07-23 | `none` | `v009_pathwise_rotation_scale_calibration` | `configs/pathwise_similarity_calibration.json` | `versions/v009_pathwise_rotation_scale_calibration/validation_summary.json` | `e65a2e42ea1ab8a9ec4d0815e40c95d891860466` | `42` | `same frozen five path-safe paths / 59 groups as v008; per-path transform uses OOF unary geometry without GT` | unknown | unknown | `rejected` | `dna:v009_pathwise_rotation_scale_calibration,parent:v008_path_safe_delta_oof`; delta MAE `3.318868m` -> `3.784095m`, lattice MAE `4.346602m` -> `8.068362m`. Unary geometry is too noisy to calibrate relative motion; stop this route and return to error diagnosis. No submission. |
| 2026-07-23 | `none` | `v010_floorplan_hallway_legality_probe` | `configs/floorplan_hallway_legality.json` | `versions/v010_floorplan_hallway_legality_probe/validation_summary.json` | `8e8367f2513cdf840e4d077ec08f9209c2db370f` | `42` | `five diagnostic paths / 59 groups; information gate on the two paths dominating v008 excess-over-4m` | unknown | unknown | `rejected` | `dna:v010_floorplan_hallway_legality_probe,parent:v008_path_safe_delta_oof,genes:floorplan_hallway_legality`; dominant-path point advantages were `-0.0500/0.0000`, edge advantages `-0.060293/0.013610`, below the frozen `0.05` gate. Rejected before Kaggle execution; no submission or quota consumed. |
| 2026-07-23 | `none` | `v011_interpolated_wifi_source_reanchoring` | `configs/interpolated_wifi_source_reanchoring.json` | `versions/v011_interpolated_wifi_source_reanchoring/validation_summary.json` | `af4f66206b375e2f05ed7b5fbee11d6f499699cf` | `42` | `five complete held-out paths / 59 groups; path overlap 0` | unknown | unknown | `rejected` | `dna:v011_interpolated_wifi_source_reanchoring,parent:v008_path_safe_delta_oof,genes:nearest_grid_reanchored_wifi_fingerprint`; Kaggle computation completed but regressed lattice MAE `4.346602m -> 5.352491m` and tail excess `125.194641 -> 157.252901`. Kernel terminal error was only a missing unused wrapper output; no submission. |

## Supporting Validation References

| artifact | purpose | key conclusion |
| --- | --- | --- |
| `data_processing/processed/pdr_v3_ensemble_metrics.json` | compare V3 delta vs PDR delta vs weighted ensemble | weighted V3 + PDR is worse than V3 alone |
| `data_processing/processed/safe_beam_param_search_summary.json` | global beam parameter search | best global beam setting still regresses against WiFi baseline |
| `data_processing/processed/beam_gating_summary.json` | beam gating sweep | gating evidence is too weak to define a new mainline |
| `data_processing/processed/grid_discretization_summary.json` | grid snapping / candidate gap diagnosis | absolute candidate quality is the main bottleneck |
| `data_processing/processed/floorplan_hallway_legality_report.json` | floor-image point and transition legality preflight | floorplan legality does not distinguish the correct trajectory on both dominant v008 tail paths; reject before lattice integration |

## Entry Template

Copy this row for the next serious run:

```text
| YYYY-MM-DD | `submission_name.csv` | `route family` | `configs/...yml` | `path/to/report.json` | `<git commit>` | `<seed>` | `<holdout>` | <public> | <private> | `mainline/exploratory/rejected/historical` | short conclusion |
```
