# ROH-A001 right-hand asset

The configured device is **ROH-A001 right**. OYMotion publishes this geometry
and mechanism in a shared A001/A002 MuJoCo package. See `VENDOR_README.md` and
`provenance.json` for the unmodified source identity and SHA-256.

The model retains 18 native physical joints, six position actuators, all six
connect equalities, original axes and reference angles, masses, inertias,
gravity compensation, actuator gains, and native collision exclusions.
Native palm normal -Z is aligned with the task's +Y axis; the unchanged Wuji
mounting rotation then makes the palm face world +Z.

The six action joints in mjlab's natural order are thumb rotation, thumb
flexion, index, middle, ring, and little. Their physical and command ranges
are [0,1.57], [0,1.00], and four copies of [0,1.44] rad. The original XML
actuator declaration order differs; mjlab retains its transmission mapping.

`linkage.npz` supplies 513 independently solved samples for each bending
linkage. It is used **only to assemble consistent joint positions on reset**.
During actual simulation, MuJoCo/Warp integrates all native joints and solves
the original connect equalities; the table does not drive passive joints on
each physics step and does not replace them with linear mimic constraints.

The policy reference pose (natural action order) is
`[0.80, 0.65, 0.90, 0.80, 0.90, 1.10]` rad. It was checked against the unchanged
54 mm / 120 g task cube. Reset and the first 0.4 seconds instead use
`[0, 0, 0.3, 0.3, 0.3, 0.3]`. These poses are separate from learned actions.
The upstream absolute default+0.5*action controller, EMA, and warmup are kept.

Fingertip sites are geometric reference points derived from the terminal
STL ends. They are not additional physical tactile sensors. Mechanical
connecting bars are named as linkages, not as fictional middle phalanges.

The upstream mesh configuration option disables only primitive hand-size
randomization (`geom_size_dr_geoms=None`). The remaining original task,
reward, curriculum, tilt, and simulation parameters are retained.


Training integration additionally uses a 5 ms physics substep: the original
10 ms setting produced divergent A001 linkage states under PPO exploration.
The 20 Hz base and 50 Hz training control periods are unchanged (10 and 4
substeps respectively). Solver iterations, rewards, action scale, EMA and PPO
settings remain upstream values. See `docs/ROHAND_A001.md` and
`docs/PUBLICATION_VERIFICATION.md` in the repository root for the current
configuration, validation and known limitations.
