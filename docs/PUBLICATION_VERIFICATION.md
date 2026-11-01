# Publication verification — 2026-11-01

The publication source was extracted to an isolated directory on the H100 host, using the existing installed CUDA dependencies and the same pinned Wuji description assets. This is source/entry-point verification, not a new dependency installation or a convergence evaluation.

| Check | Result |
|---|---|
| A001 stagnation and isolation tests, including real GPU fault injection | 11 passed |
| Bundled checkpoint deterministic headless inference | 200 steps, finite actions/qpos/qvel |
| Portable training entry point | 32 environments, 2 PPO updates completed |
| Portable resume entry point | Bundled checkpoint, 32 environments, 1 PPO update completed |
| Restored optimizer history | Step advanced by 128, matching 4 epochs × 32 minibatches |
| Runtime curriculum metadata | Restored and saved; disturbance scale retained at 500,000 steps |
| Bundled checkpoint integrity after training probes | SHA-256 unchanged |
| Portable Viser playback entry point | Started successfully; HTTP 200 |
| Python static checks on portable entry points and A001 tests | Passed |

These probes used separate output directories and did not stop or resume the ongoing formal training process. The checkpoint remains an intermediate training snapshot with the normalization limitations documented in `pretrained/README.md`.
