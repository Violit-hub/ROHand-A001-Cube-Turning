# A001 playback checkpoint

`model_11100.pt` is the intermediate PPO snapshot used for the current A001 visualization. It contains actor, critic, observation normalization, optimizer state, iteration and environment step state. Its SHA-256 and task details are in `model_11100.json`.

The policy is not claimed to have converged or to be the best checkpoint. The critic normalization inherits unusually large variance from earlier training (maximum standard deviation approximately 2.97e14). The checkpoint is supplied unchanged for reproducible playback; resuming it also preserves those statistics. Starting from scratch creates fresh statistics.

Run `pixi run python scripts/rohand/play.py` from the repository root. The policy uses its deterministic mean; the environment clips actions to [-1,1].

The portable resume entry point additionally reads `model_11100.json` to restore the two runtime curriculum scalars omitted from the upstream checkpoint. Newly saved checkpoints include these scalars in their own `infos.rohand_training_state`.
