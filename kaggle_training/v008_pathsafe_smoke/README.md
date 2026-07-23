# Kaggle v008 Path-safe Training Chain

## Outcome

The direct Kaggle training chain is operational.

- Private kernel: `kiivii/indoor-v008-path-safe-training-smoke`
- Validated kernel version: `2`
- Remote status: `COMPLETE`
- Competition source: `indoor-location-navigation`
- Repository commit: `9be867918e92170552e1f9fe9b71ceec15e92fd4`
- Leaderboard submissions: none

## Data Flow

1. Kaggle mounts the official competition input.
2. The private kernel clones the locked repository commit into `/tmp`.
3. The kernel builds IMU leg features from 12 paths.
4. Ten complete paths train a Ridge delta smoke model.
5. Two disjoint paths provide validation.
6. Kaggle writes only the report and prediction CSV under `/kaggle/working`.
7. The CLI monitors the run and downloads the two output files.

## Remote Validation

- Train paths: `10`
- Validation paths: `2`
- Path overlap: `0`
- Train legs: `52`
- Validation legs: `9`
- Prediction rows: `9`
- Non-finite values: `0`
- Remote smoke delta MAE: `6.317891 m`
- `submission.csv`: not created

The smoke MAE validates execution only. It is not comparable to v007 candidate-lattice MAE, submission MPE, or leaderboard results.

## Files

- Kernel entrypoint: `kernel.py`
- Kernel metadata: `kernel-metadata.json`
- Remote validation: `remote_validation.json`
- Report: `output_v2/training_smoke_report.json`
- Predictions: `output_v2/training_smoke_predictions.csv`
- Log: `output_v2/indoor-v008-path-safe-training-smoke.log`

## Next

Reuse this private kernel and monitoring path for `v008_path_safe_delta_oof`. Replace the smoke Ridge trainer with the complete path-isolated V3 delta training job while preserving:

- Kaggle-only training guard
- fixed Git commit
- private kernel
- official competition input
- `/tmp` code checkout
- `/kaggle/working` output contract
- no leaderboard submission without explicit authorization
