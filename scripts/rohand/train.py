"""Portable entry point for the existing A001 task and upstream PPO trainer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=12228)
    parser.add_argument('--iterations', type=int, default=5000,
                        help='Number of additional PPO updates.')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--run-name', default='A001_stall10_isolated')
    parser.add_argument('--curriculum-steps', type=int, default=500000,
                        help='Preserve the original disturbance ramp when extending training.')
    options, overrides = parser.parse_known_args()
    checkpoint = options.checkpoint.resolve() if options.checkpoint else None
    if checkpoint and not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    if min(options.num_envs, options.iterations, options.curriculum_steps) < 1:
        parser.error('Environment count, iterations and curriculum steps must be positive.')
    os.chdir(ROOT)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

    import torch
    from wuji_mjlab.rl.runner import WujiOnPolicyRunner

    original_load = WujiOnPolicyRunner.load
    original_save = WujiOnPolicyRunner.save
    original_learn = WujiOnPolicyRunner.learn

    def load(self, _path, *args, **kwargs):
        result = original_load(self, str(checkpoint), *args, **kwargs)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        metadata = (state.get('infos') or {}).get('rohand_training_state')
        if metadata is None:
            sidecar = checkpoint.with_suffix('.json')
            if sidecar.exists():
                metadata = json.loads(sidecar.read_text())['training_state']
        if metadata:
            for key in ('_adaptive_episode_curriculum', '_reorient_success_curriculum'):
                setattr(self.env.unwrapped, key, metadata[key])
            self.env.unwrapped.max_common_steps = int(metadata['max_common_steps'])
        return result

    def save(self, path, infos=None):
        payload = dict(infos or {})
        env = self.env.unwrapped
        payload['rohand_training_state'] = {
            'max_common_steps': env.max_common_steps,
            '_adaptive_episode_curriculum': float(getattr(env, '_adaptive_episode_curriculum', .05)),
            '_reorient_success_curriculum': float(getattr(env, '_reorient_success_curriculum', 0.)),
        }
        return original_save(self, path, payload)

    def learn(self, *args, **kwargs):
        # The number of requested future updates must not stretch the DR ramp.
        if checkpoint is None:
            self.env.unwrapped.max_common_steps = options.curriculum_steps
        return original_learn(self, *args, **kwargs)

    WujiOnPolicyRunner.save = save
    WujiOnPolicyRunner.learn = learn
    if checkpoint:
        WujiOnPolicyRunner.load = load
    # The upstream CLI resolves resume paths inside logs before runner.load().
    # Override only that resolver so a checkpoint can live anywhere.
    import mjlab.utils.os as os_utils
    if checkpoint:
        os_utils.get_checkpoint_path = lambda *args, **kwargs: str(checkpoint)
    sys.argv = [str(ROOT / 'scripts/train/train_rsl_rl.py'),
                '--task', 'WujiHand2_Reorient_50Hz',
                '--agent.logger', 'tensorboard', '--agent.upload-model', 'False',
                f'env.scene.num_envs={options.num_envs}',
                f'agent.max_iterations={options.iterations}',
                f'agent.run_name={options.run_name}',
                f'agent.resume={checkpoint is not None}', *overrides]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except KeyboardInterrupt:
        print('\nTraining interrupted; completed periodic checkpoints remain under logs/.')


if __name__ == '__main__':
    main()
