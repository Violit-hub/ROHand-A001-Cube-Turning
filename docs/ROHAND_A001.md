# ROHand A001 adaptation

Based on wuji-technology/wuji-mjlab commit `b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9`. The existing `WujiHand2_Reorient_50Hz` registration is adapted to A001; it has not been replaced with a new task or a SuperMimic training algorithm.

## Model and control

The right-hand A001 mechanism retains 18 physical joints, six position actuators, native axes and reference angles, and six closed-linkage connect constraints. The natural action order is:

| Action | Range (rad) |
|---|---|
| Thumb root rotation | 0–1.57 |
| Thumb bending drive | 0–1.00 |
| Index bending | 0–1.44 |
| Middle bending | 0–1.44 |
| Ring bending | 0–1.44 |
| Little bending | 0–1.44 |

The XML actuator declaration order differs from the natural joint/action order; mjlab's transmission mapping is retained. `linkage.npz` provides consistent passive joint positions at reset only. Physical linkages remain dynamically simulated.

The reset/first-0.4-second pose is `[0,0,0.3,0.3,0.3,0.3]`. The policy reference is `[0.8,0.65,0.9,0.8,0.9,1.1]`. The original controller is reference + 0.5 × clipped action, clamped to joint limits and EMA-smoothed with alpha=0.3. Thus the policy's effective target range is narrower than the full hardware range.

The 54 mm / 120 g cube starts at `[-0.151,0.0072,0.5616]` with identity quaternion and zero initial velocity. Original random palm pitch `[-0.4,0.1]` radians remains active. The static jig is a mounting visual, not the source of the random tilt.

## Changes from upstream

- A001 assets, six controlled-joint observations/actions, sensor-name adaptation and closed-linkage resets.
- Physics timestep 5 ms, four substeps per 50 Hz control step. Solver settings and original PPO hyperparameters are retained.
- Primitive hand-geometry size randomization is disabled for this mesh-only hand. Other original randomization and curricula remain.
- Saved cube/reset pose above.
- Stagnation penalty: first 0.4 s excluded; 2 s of no meaningful cube rotation/progress gives a 2 s linear ramp to -10 points/s. Motion/progress threshold is 2 degrees. This can be avoided by exploration-induced movement; it is not a guarantee against deterministic policy stagnation.
- A001 training isolation: detect nonfinite/excessive physics and feedback, reset affected worlds, discard the pending PPO rollout and restore its normalization state. Invalid optimizer updates are rolled back. This protects learning data but does not repair the underlying physical mechanism.

## Known limitations

Rare large passive-joint velocities remain, previously localized to the little-finger connecting linkage. Isolation mitigates training contamination; it is not a dynamics fix. The supplied intermediate checkpoint also retains extreme critic normalization variance from earlier training; see `pretrained/README.md`.

Upstream Wuji deployment scripts are retained for provenance and are not adapted to A001 hardware. This repository's verified A001 workflow is simulation training and playback.

## Source guide

- `src/wuji_mjlab/assets/robots/rohand_a001/`: MJCF, meshes, control ranges, linkage data.
- `src/wuji_mjlab/tasks/reorient/config/wuji_hand2/`: adapted original task and PPO configuration.
- `src/wuji_mjlab/tasks/reorient/mdp/rohand_a001.py`: reset and controlled-joint observation adaptation.
- `src/wuji_mjlab/tasks/reorient/mdp/stagnation.py`: stagnation term.
- `src/wuji_mjlab/rl/isolation.py`: invalid rollout/update isolation.
- `scripts/rohand/`: portable training/resume and playback entry points.

Wuji task code remains under its original Apache-2.0 license with its copyright/NOTICE. ROHand model resources retain their manufacturer provenance and included vendor README; the upstream software license is not a new license grant for third-party model assets.
