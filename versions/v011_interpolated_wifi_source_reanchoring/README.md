# v011 Interpolated WiFi Source Reanchoring

## Why

The public 15th-place solution uses linearly interpolated waypoint coordinates at WiFi scan
timestamps. v011 tested a narrower adaptation: reanchor source scans to the nearest training
candidate grid coordinate, then use the resulting fingerprint signal in the retained v008
path-safe transition lattice.

## What changed

- inherited the v008 candidate dataset, path-safe delta OOF, and transition lattice;
- interpolated source-path coordinates at WiFi scan timestamps;
- excluded every held-out path from the source fingerprint index;
- assigned each retained source scan to its nearest candidate-grid coordinate.

The remote probe used five complete held-out paths and 59 groups, with path overlap zero.
No submission file was generated.

## Result

Kaggle completed the probe at commit `af4f662`. The wrapper marked the kernel as failed only
after all probe artifacts were written because it expected an unused summary filename.
The measured result is nevertheless decisive:

- v008 lattice MAE: `4.346602m`;
- v011 lattice MAE: `5.352491m`;
- v008 excess-over-4m: `125.194641`;
- v011 excess-over-4m: `157.252901`.

## Decision

Reject the nearest-grid WiFi reanchoring gene. It degrades both aggregate and tail error, so
v012 must not inherit it. The useful public-solution principle remains dense scan-time absolute
prediction followed by time interpolation, which is a distinct gene.
