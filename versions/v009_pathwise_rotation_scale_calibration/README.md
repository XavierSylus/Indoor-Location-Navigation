# v009 Pathwise Rotation and Scale Calibration

## Why

v008 proved that path-safe relative motion cuts excess-over-4m by `54.4469%`, but its
lattice MAE remained `4.346602 m`. A public high-ranking solution corrected the rotation
of relative motion by minimizing disagreement with an absolute predicted path. v009 tests
that idea as one isolated gene.

Source:
`https://cocoinit23.com/kaggle-indoor-location-navigation-15th-place-solution/`

## What changed

- inherited the complete v008 delta predictions and frozen lattice;
- selected an OOF unary WiFi candidate for each diagnostic group;
- fitted one origin-constrained rotation plus uniform scale for each path;
- transformed every v008 leg delta with the fitted path transform;
- did not add translation, clipping, manual thresholds, iteration or GT features.

Target-path labels and oracle distances were used only after selection for diagnostic
metrics.

## Result

- Kaggle kernel: `kiivii/indoor-v009-pathwise-similarity-calibration`, version 1,
  `COMPLETE`.
- Delta rows: `54`; duplicate keys: `0`; non-finite predictions: `0`.
- Transform rows: `5`.
- Raw v008 delta MAE: `3.318868 m`.
- Calibrated delta MAE: `3.784095 m`.
- Raw v008 lattice MAE: `4.346602 m`.
- Calibrated lattice MAE: `8.068362 m`.
- Change versus v008: `-3.721760 m`.
- Structured excess-over-4m: `274.986013`.
- Gene gain gate: failed.
- Target `<3 m` gate: failed.
- Submission created: no.

## Root cause

The inferred transform followed noise in the unary WiFi geometry instead of correcting
relative motion. The two short paths produced extreme transforms from only two alignment
pairs, while the long path `5dd0e5...` retained a `14.844 m` alignment residual and its
lattice MAE increased to `12.232596 m`.

## Decision

Reject this gene. Do not add clipping, path blacklists or post-hoc gates to make this result
look positive. The next version must return to the v008 error ledger and create a genuinely
independent information source rather than calibrating delta against the noisy unary path.
