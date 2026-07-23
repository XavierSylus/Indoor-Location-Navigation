# v003_site_selective_beam_report

## Goal
Materialize site-selective beam as a durable waypoint-level diagnostic report.

## Base
v002_unified_validation_baseline

## Data Flow
- Input: `safe_beam_param_search_summary.json`, `beam_gating_summary_whitelist.json`, and `beam_gating_path_metrics_whitelist.csv`.
- Processing: combine full-holdout absolute baseline error with whitelist beam path-level error sums.
- Model: no new model is trained in this version.
- Postprocess: no submission postprocess is run.
- Output: `data_processing/processed/unified_validation_report.json`.

## Changes
- Experiment type: `cv_probe`
- Iteration mode: `assemble_module`
- Route family: `site-selective beam diagnostic`

## Validation
- Holdout definition: same comparable holdout as safe beam and beam gating; full holdout has `5` paths / `40` points.
- Local metric: site-selective beam diagnostic MAE `10.953045`; delta vs absolute baseline `-0.665994 m`.
- Public LB: not submitted.
- Private LB: not submitted.
- CV/LB gap or explanation: CV-only diagnostic; evidence covers only `2` whitelist paths / `13` points.

## Files
- Config: `versions/v003_site_selective_beam_report/config.yml`
- Script: `scripts/build_unified_validation_report.py`
- Metrics: `data_processing/processed/unified_validation_report.json`
- Validation report: `data_processing/processed/unified_validation_report.json`
- Manifest: `versions/v003_site_selective_beam_report/run_manifest.json`
- Reproduce: `versions/v003_site_selective_beam_report/reproduce.ps1`
- Submission: none

## Side Effects
Updated `data_processing/processed/unified_validation_report.json`. No model, submission pipeline, beam logic, or PDR logic was changed.

## Judgement
Site-selective beam has a real positive diagnostic signal: replacing absolute baseline with beam on the whitelist subset improves the comparable holdout estimate from `11.619039` to `10.953045`. It is not promotable yet because the positive evidence covers only `2` paths / `13` points, so it cannot be treated as a stable LB expectation.

## What Worked
- Whitelist subset beam improves from WiFi MAE `11.906022` to beam MAE `9.856810`.
- Estimated full comparable holdout delta is `-0.665994 m` versus absolute baseline.
- The route is now represented as a durable diagnostic report instead of a missing route.

## What Failed
- Coverage is too small for promotion: only `2` whitelist paths / `13` points.
- The current report does not yet prove fewer regressing sites under broader holdout coverage.

## Next
- Expand site-selective validation coverage or construct a regression-safe policy before generating any submission candidate.
- Compare this signal against Top-50 reranker safety gating; only assemble a submission candidate after both risks are controlled.
