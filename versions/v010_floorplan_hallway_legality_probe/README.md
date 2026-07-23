# v010 Floorplan Hallway Legality Probe

## Why

The public 15th-place solution used floor-image pixels to move predictions out of walls.
v010 tests the prerequisite for that idea: whether hallway membership or wall-crossing
separates the correct candidate path from v008 on the two paths that dominate its
excess-over-4m.

Source:
`https://cocoinit23.com/kaggle-indoor-location-navigation-15th-place-solution/`

## What changed

- inherited v008 as the parent DNA;
- mapped candidate coordinates to `floor_image.png` using `floor_info.json`;
- measured point hallway membership and segment non-hallway ratios;
- kept all candidate coordinates and v008 selections unchanged;
- used oracle candidates only to measure diagnostic separation after selection.

No model was trained and no Kaggle kernel or submission was created.

## Result

All candidate coordinates were inside the mapped images. The information-value gate failed
on both dominant tail paths:

- `5dd0e5...`: point advantage `-0.0500`, edge advantage `-0.060293`;
- `5dd374...`: point advantage `0.0000`, edge advantage `0.013610`;
- required advantage: `0.05` on every priority path.

For the 20-point path, the oracle path was less hallway-compliant than v008 under both
measures. For the 31-point path, both paths already lay almost entirely on hallway pixels.

## Decision

Reject the floorplan-legality gene before Kaggle execution. Do not rescue it with threshold
tuning, coordinate movement, snap-to-grid, or path-specific gates. The next experiment must
create a candidate-selection signal that is independent of geometry already encoded by the
training waypoint candidates.
