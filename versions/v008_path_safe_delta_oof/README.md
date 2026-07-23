# v008 Path-safe Delta OOF

## Why

v007 reduced structured candidate MAE from `7.524908 m` to `1.355896 m`, but its V3
delta checkpoint used a leg-random split. The held-out paths could therefore influence the
transition signal. Public high-ranking solutions also treat complete paths as the validation
boundary, so v008 tests whether the transition gene survives that boundary.

## What changed

Only the delta source changed:

- retained the v007 candidate set, unary reranker, alpha grid and pairwise cap;
- trained the same V3 Conv1d + bidirectional-GRU model family on Kaggle;
- limited training to the five relevant sites;
- excluded all five diagnostic paths from fitting and checkpoint selection;
- selected checkpoints on 130 different complete paths;
- generated exactly one path-safe delta prediction for each of the 54 held-out legs.

No Kalman filter, rotation/scale correction, pseudo labeling, hallway map, ensemble or
leaderboard submission was added.

## Result

- Kaggle kernel: `kiivii/indoor-v008-path-safe-delta-oof`, version 2, `COMPLETE`.
- Train/held-out path overlap: `0`.
- Held-out interval coverage: `54/54`.
- Internal validation delta MAE: `2.806851 m`.
- Held-out delta MAE: `3.318868 m`.
- Unary reranker MAE: `7.524908 m`.
- Path-safe lattice MAE: `4.346602 m`.
- Improvement over unary: `3.178306 m`.
- Excess-over-4m reduction: `54.4469%`.
- Target `<3 m` gate: failed.
- Submission created: no.

The transition gene is real, but the v007 score was materially optimistic. The largest
remaining tail is concentrated in the two long paths: `5dd374...` contributes
`83.120252` excess meters and `5dd0e5...` contributes `35.321270`.

## Decision

Promote the path-consistency gene, not the score. v008 must not become a submission base
without another validated component.

The unique next direction is `v009_pathwise_rotation_scale_calibration`, an `add_module`
CV probe derived from public high-ranking solutions. It should estimate rotation and scale
from inference-safe absolute-path geometry, then replace only the raw v008 delta vectors.
The v008 candidate set, lattice and validation split must remain frozen so the added gene
has an attributable effect.
