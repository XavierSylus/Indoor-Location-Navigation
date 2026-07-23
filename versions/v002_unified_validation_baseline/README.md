# v002_unified_validation_baseline

## Goal
Build a reproducible unified validation baseline for absolute, beam, PDR, and site-selective comparisons.

## Base
v001_historical_step3_beamsearch as historical leaderboard reference; current validation evidence in data_processing/processed/

## Data Flow
- Input: durable validation artifacts in `data_processing/processed/` plus `configs/unified_validation.yml`.
- Processing: `scripts/build_unified_validation_report.py` compares existing validation summaries under one reporting contract.
- Model: no new model is trained in this version.
- Postprocess: no submission postprocess is run.
- Output: `data_processing/processed/unified_validation_report.json`.

## Changes
- Experiment type: `cv_probe`
- Iteration mode: `assemble_module`
- Route family: `unified validation baseline`

## Validation
- Holdout definition: configured in `configs/unified_validation.yml`; comparable beam/gating holdout has `5` paths and `40` points.
- Local metric: unified validation report.
- Public LB: not submitted.
- Private LB: not submitted.
- CV/LB gap or explanation: CV-only diagnostic version; no leaderboard claim.

## Files
- Config: `versions/v002_unified_validation_baseline/config.yml`
- Script: `scripts/build_unified_validation_report.py`
- Metrics: `data_processing/processed/unified_validation_report.json`
- Validation report: `data_processing/processed/unified_validation_report.json`
- Manifest: `versions/v002_unified_validation_baseline/run_manifest.json`
- Reproduce: `versions/v002_unified_validation_baseline/reproduce.ps1`
- Submission: none

## Side Effects
Updated `data_processing/processed/unified_validation_report.json`. No model, submission pipeline, beam logic, or PDR logic was changed.

## Judgement
The unified report keeps `absolute_baseline` as the recommended default route. Global beam regresses on the comparable holdout, PDR is still reported on a non-comparable delta-leg metric, and site-selective beam is still missing as a durable waypoint-level report.

## What Worked
- Unified report generation is reproducible through `reproduce.ps1`.
- Existing comparable holdout artifacts agree on the same 5-path / 40-point signature.
- The report clearly separates comparable waypoint MAE from non-comparable delta-leg metrics.

## What Failed
- This version has no submission by design, so the current `check-version` gate fails on submission-required checks.
- Site-selective beam remains missing as a durable report.

## Next
- Update `check-version` to support CV-only versions with `--allow-no-submission`.
- Build the missing site-selective beam report before promoting any beam policy.
