# Indoor Location & Navigation

> Multi-stage indoor positioning pipeline for the Kaggle Indoor Location & Navigation competition.

This repository currently contains two distinct layers of work:

1. A verified mainline centered on absolute per-site positioning with the V3 ensemble.
2. Several exploratory trajectory-constraint branches (`beam`, `PDR`, `gating`) that are retained for analysis but are not the recommended submission path.

---

## Current Verified Status

### Best leaderboard result still in use

| Metric | Score |
| --- | ---: |
| Public MPE | 6.64 m |
| Private MPE | 7.90 m |
| Submission family | `step3` beam-search line |

These remain the best recorded leaderboard scores in this repository.
The newer `submission_step3_optimized_ensemble.csv` and `submission_step3_gated_mean15.csv` variants did not beat that result and are therefore not treated as the new baseline.

### Verified local findings

| Topic | Evidence | Conclusion |
| --- | --- | --- |
| IMU delta V3 | `data_processing/processed/pdr_v3_ensemble_metrics.json` | `v3_mae_m = 2.0655`, currently the strongest delta signal in repo |
| PDR delta | `data_processing/processed/pdr_v3_ensemble_metrics.json` | `pdr_mae_m = 3.3527`, weaker than V3 delta |
| V3 + PDR weighted delta | `data_processing/processed/pdr_v3_ensemble_metrics.json` | `ensemble_mae_m = 2.2158`, worse than V3 alone |
| Global beam search parameter search | `data_processing/processed/safe_beam_param_search_summary.json` | best beam result is still worse than WiFi baseline by `+1.1355 m` |
| Beam gating | `data_processing/processed/beam_gating_summary.json` | best gain is only `-0.0739 m` on `5` holdout paths / `40` points |
| Grid discretization gap | `data_processing/processed/grid_discretization_summary.json` | `pred_to_grid_mae_m ~= 5.06`, `gt_to_grid_oracle_mae_m ~= 1.57` |

### Practical interpretation

- The current bottleneck is not "missing more trajectory logic".
- The current bottleneck is the quality of the absolute candidate position before any trajectory constraint is applied.
- `PDR`, global `beam`, and simple `gating` are exploratory branches until they show stable holdout gains under a unified validation split.

---

## Recommended Mainline

The project should now be advanced in this order:

1. Correct documentation and experiment bookkeeping.
2. Rebuild a unified holdout validation protocol.
3. Analyze absolute-position error without beam or PDR.
4. Reintroduce beam only for sites that repeatedly improve under fixed validation splits.

### What is not recommended now

- Do not continue submitting global `PDR + beam + gating` variants as the default path.
- Do not treat isolated smoke runs as proof of improvement.
- Do not mix results from different holdout sets when comparing approaches.

---

## System Components

### Mainline components

| Module | Status | Notes |
| --- | --- | --- |
| Absolute per-site ensemble | Active mainline | strongest current submission base |
| IMU Delta V3 | Active supporting model | useful as a relative signal, but not enough to justify global beam rollout |

### Exploratory components

| Module | Status | Notes |
| --- | --- | --- |
| Beam Search | Exploratory | strong site variance; some sites improve, some regress |
| PDR delta fusion | Exploratory | currently worse than V3 delta alone |
| Beam gating | Exploratory | evidence too weak to promote |
| Pseudo-labeling | Planned | not started as a mainline task |

---

## Repository Structure

```text
Indoor Location & Navigation/
|-- configs/
|-- data_processing/
|   |-- processed/
|   `-- *.py
|-- models/
|-- scripts/
|-- src/
|-- tests/
|-- README.md
|-- PROJECT_STATUS.md
`-- CONTINUATION_GUIDE.md
```

Key paths for current decision-making:

- `scripts/step3_infer_and_optimize.py`
- `scripts/evaluate_pdr_v3_ensemble.py`
- `scripts/evaluate_beam_search_validation.py`
- `scripts/evaluate_beam_gating_validation.py`
- `data_processing/processed/*.json`

---

## Execution Guidance

### Recommended workflow now

1. Keep the current submission entrypoint unchanged.
2. Treat the absolute ensemble output as the reference baseline.
3. Run all future comparisons on the same holdout split.
4. Promote a beam-based variant only if it improves both:
   - overall MAE
   - number of regressing sites

### Mandatory experiment metadata

Every new experiment should record:

- random seed
- environment snapshot
- git commit
- config path
- output file path
- holdout definition
- public / private leaderboard result when submitted

---

## Reproducibility Notes

- Hyperparameters should remain in config files rather than hardcoded.
- Experiments should be tied to a single commit and a single validation report.
- Repository decisions should follow verified results in `data_processing/processed/`, not outdated roadmap text.
