# v007_candidate_transition_lattice

## Goal
Test whether V3 vector-consistent candidate-to-candidate path decoding attacks the dominant selection failure.

## Iteration Type
- Innovation
- Experiment type: `cv_probe`
- Iteration mode: `add_module`

## Error Source
v004 showed that 46 of 59 hard validation groups were selection failures: Top-50 usually contained a near-ground-truth candidate, but the pointwise reranker selected the wrong coordinate. v005-v006 also showed that pointwise confidence gates could not reliably identify those failures.

## Attack Plan
Replace independent point selection with a candidate transition lattice. The unary cost is the existing path-OOF `geometry_source_wifi` reranker score. The pairwise cost is the capped vector error between each adjacent candidate displacement and the accumulated V3 IMU delta. Alpha is calibrated without the held-out path.

## Base
v006_rank1_risk_signal_probe

## Data Flow
- Input: `hard_site_candidate_rerank_dataset.csv` and `v3_all_train_delta.csv`.
- Processing: reproduce path-level OOF candidate scores, build exact timestamp-to-timestamp V3 delta chains, and decode the lowest-cost candidate sequence.
- Model: the frozen v005 LightGBM reranker specification provides unary scores; no new learned model family.
- Postprocess: dynamic programming over the complete Top-50 candidate lattice.
- Output: selected candidate per group and a diagnostic validation summary.

## Changes
- Experiment type: `cv_probe`
- Iteration mode: `add_module`
- Route family: `candidate_transition_lattice`
- Added candidate-to-candidate vector consistency. Existing temporal features only compared a candidate with raw/rank1 neighbor positions and did not jointly optimize the full path.

## Validation
- Holdout definition: hard-site Top-50 leave-one-path-out; 59 groups / 5 paths. Each held-out path uses an alpha selected on the other four paths.
- Unary reranker MAE: `7.524908 m`.
- Structured lattice MAE: `1.355896 m`.
- Improvement: `6.169012 m`.
- Excess-over-4m reduction: `91.3658%`.
- Hit@3: `0.762712`; Hit@5: `0.898305`.
- Public LB: not applicable.
- Private LB: not applicable.
- CV/LB gap: not measurable. The V3 checkpoint used leg-level random splitting, so the V3 transition signal is not path-safe for this holdout.

## Files
- Config: `versions/v007_candidate_transition_lattice/config.yml`
- Script: `data_processing/candidate_transition_lattice.py`
- Test: `tests/test_candidate_transition_lattice.py`
- Metrics: `data_processing/processed/candidate_transition_lattice_summary.json`
- Selections: `data_processing/processed/candidate_transition_lattice_selections.csv`
- Validation report: `versions/v007_candidate_transition_lattice/validation_summary.json`
- Submission: none.

## Side Effects
- Two diagnostic artifacts were created under `data_processing/processed/`.
- No model, submission, Kaggle run, or LB attempt was created.

## Judgement
Promote the gene, not the score. Candidate-to-candidate path consistency is the first post-v004 mechanism to close most of the selection gap and is a strong innovation direction. The numeric result cannot support a submission because the existing V3 model may have trained on legs from the held-out paths.

## What Worked
- Nested path calibration selected `alpha=4.0` for every held-out path.
- Four of five paths improved strongly; the two longest paths reached `1.980885 m` and `0.693075 m`.
- The lattice selected candidates as deep as rank 50, confirming that useful candidates existed beyond the pointwise reranker choice.

## What Failed
- One three-point path remained at `6.251040 m`.
- The existing V3 checkpoint is not path-safe because its original split was performed over legs instead of complete paths.
- The diagnostic covers only five hard paths and is not comparable to submission-wide MPE.

## Next
Create `v008_path_safe_delta_oof` as a separate `add_module` CV probe. Generate V3 delta predictions from models that exclude the complete held-out path, then rerun this frozen lattice without changing its feature set or alpha grid. Only a retained gain can justify all-site expansion or Kaggle training.
