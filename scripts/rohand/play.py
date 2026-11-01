"""Play an A001 checkpoint, or run a short deterministic inference probe."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'pretrained/model_11100.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8779)
    parser.add_argument('--steps', type=int, default=0,
                        help='Positive: run a headless probe and exit. Zero: open Viser.')
    options = parser.parse_args()
    if options.steps < 0:
        parser.error('--steps must be nonnegative')
    checkpoint = options.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import numpy as np
    import torch
    import wuji_mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    task = 'WujiHand2_Reorient_50Hz'
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device=options.device)
    wrapped = None
    try:
        agent = load_rl_cfg(task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(task)(wrapped, asdict(agent), device=options.device)
        runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=options.device)
        policy = runner.get_inference_policy(device=options.device)
        if options.steps:
            with torch.no_grad():
                for _ in range(options.steps):
                    action = policy(wrapped.get_observations())
                    assert bool(torch.isfinite(action).all())
                    wrapped.step(action)
                    assert bool(torch.isfinite(env.sim.data.qpos).all())
                    assert bool(torch.isfinite(env.sim.data.qvel).all())
            print(json.dumps({'checkpoint': checkpoint.name, 'probe_steps': options.steps,
                              'passed': True, 'mode': 'deterministic_inference'}))
            return
        import viser
        from mjlab.viewer import ViserPlayViewer

        server = viser.ViserServer(host=options.host, port=options.port, label='ROHand A001')
        server.gui.add_markdown(
            f'### ROHand A001\n**{checkpoint.name}**\n\n'
            '确定性策略回放，动作截断 −1～1。Play 播放；Reset Environment 更换目标和倾角。')
        status = server.gui.add_markdown('等待播放。')
        goal = server.scene.add_box('/policy_goal', dimensions=(.054, .054, .054),
                                    wireframe=True, color=(80, 255, 110))
        axes = server.scene.add_frame('/policy_cube_axes', axes_length=.04, axes_radius=.001)

        class Viewer(ViserPlayViewer):
            def sync_env_to_viewer(self):
                super().sync_env_to_viewer()
                if self._counter % 10:
                    return
                obj = env.scene['object']
                command = env.command_manager.get_term('reorient_command')
                goal.position = obj.data.root_link_pos_w[0].detach().cpu().numpy() + np.array([0, .13, 0])
                goal.wxyz = command.goal_quat_w[0].detach().cpu().numpy()
                axes.position = obj.data.root_link_pos_w[0].detach().cpu().numpy()
                axes.wxyz = obj.data.root_link_quat_w[0].detach().cpu().numpy()
                error, _ = command._policy_status_for_env(0)
                status.content = (f'姿态误差：**{np.degrees(error):.1f}°**　'
                                  f'本回合目标达成：**{int(command.goal_reach_count[0])}次**')

        viewer = Viewer(wrapped, policy, viser_server=server)
        viewer.pause()
        viewer.run()
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == '__main__':
    main()
