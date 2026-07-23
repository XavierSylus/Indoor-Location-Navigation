# v001_historical_step3_beamsearch

## Goal
Backfill the best recorded historical step3 beam-search submission into the versioned workflow.

## Base
historical best submission_step3_beamsearch.csv, Public 6.64 m, Private 7.90 m

## Data Flow
- Input: historical repository artifacts and `submission_step3_beamsearch.csv`.
- Processing: historical backfill only; original exact generation command is unknown.
- Model: step3 global beam historical route.
- Postprocess: historical beam-search submission output.
- Output: `versions/v001_historical_step3_beamsearch/submission.csv` and `submissions/v001_historical_step3_beamsearch.csv`.

## Changes
- Experiment type: `cv_probe`
- Iteration mode: `assemble_module`
- Route family: `step3 global beam historical`

## Validation
- Holdout definition: unknown for the original run.
- Local metric: unknown for the original run.
- Public LB: `6.64 m`.
- Private LB: `7.90 m`.
- CV/LB gap or explanation: historical leaderboard record; local validation details were not preserved.

## Files
- Config: `versions/v001_historical_step3_beamsearch/config.yml`
- Script: historical route references `scripts/step3_infer_and_optimize.py` and `scripts/run_phase2.py`; exact original command unknown.
- Metrics: historical leaderboard record in `README.md` and `EXPERIMENT_LEDGER.md`.
- Validation report: `versions/v001_historical_step3_beamsearch/validation_summary.json`
- Manifest: `versions/v001_historical_step3_beamsearch/run_manifest.json`
- Reproduce: `versions/v001_historical_step3_beamsearch/reproduce.ps1`
- Submission: `versions/v001_historical_step3_beamsearch/submission.csv`

## Side Effects
Backfilled existing historical submission into the versioned workflow. No model, submission pipeline, beam logic, or PDR logic was changed.

## Judgement
This is the best recorded historical leaderboard result in the repository and should remain the reference until a new version has comparable validation evidence and a better Kaggle result.

## What Worked
- The CSV matches `indoor-location-navigation/sample_submission.csv` by columns, row count, and `site_path_timestamp` order.
- Public/Private leaderboard record remains `6.64 m / 7.90 m`.

## What Failed
- Original exact generation command, seed, and holdout definition were not preserved, so this version is a historical reference rather than a fully reproducible rerun.

## Next
- Build a fully reproducible current baseline under the new workflow before promoting any new route.
