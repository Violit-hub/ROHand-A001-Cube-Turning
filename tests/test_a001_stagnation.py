"""Stagnation timing, units, goal/reset handling and physical-motion gates."""
import math
from types import SimpleNamespace

import pytest
import torch
import wuji_mjlab.tasks  # noqa: F401 -- register tasks
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.tasks.registry import load_env_cfg
from wuji_mjlab.tasks.reorient.mdp.stagnation import StagnationPenalty


def quat(angle):
  return torch.tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])


def make_env(dt=.02):
  command = SimpleNamespace(goal_quat_w=quat(math.pi / 2).repeat(2, 1),
                            cfg=SimpleNamespace(success_threshold=.2),
                            in_success_window=torch.zeros(2, dtype=torch.bool))
  obj = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=quat(0).repeat(2, 1)))
  env = SimpleNamespace(num_envs=2, device='cpu', step_dt=dt,
                        scene={'object': obj},
                        command_manager=SimpleNamespace(get_term=lambda _name: command))
  return env, command, obj


def manager_for(env):
  return RewardManager({'stagnation': RewardTermCfg(func=StagnationPenalty, weight=-10.,
                       params={'warmup_time_s': .4, 'grace_time_s': 2., 'ramp_time_s': 2.})}, env)


@pytest.mark.parametrize('dt', [.02, .05])
def test_timing_and_per_second_strength(dt):
  env, _, _ = make_env(dt)
  manager = manager_for(env)
  warmup = round(.4 / dt)
  grace = round(2. / dt)
  ramp = round(2. / dt)
  for _ in range(warmup + grace):
    assert torch.equal(manager.compute(dt), torch.zeros(2))
  for _ in range(ramp // 2):
    value = manager.compute(dt)
  assert torch.allclose(value, torch.full((2,), -5 * dt), atol=1e-6)
  for _ in range(ramp - ramp // 2):
    value = manager.compute(dt)
  assert torch.allclose(value, torch.full((2,), -10 * dt), atol=1e-6)
  full_second = sum(manager.compute(dt).clone() for _ in range(round(1. / dt)))
  assert torch.allclose(full_second, torch.full((2,), -10.), atol=1e-5)


def full_penalty():
  env, command, obj = make_env()
  cfg = RewardTermCfg(func=StagnationPenalty, weight=-10., params={})
  penalty = StagnationPenalty(cfg, env)
  for _ in range(220):
    penalty(env)
  assert torch.equal(penalty(env), torch.ones(2))
  return env, command, obj, penalty


def test_cube_motion_clears_penalty_even_away_from_goal():
  env, _, obj, penalty = full_penalty()
  obj.data.root_link_quat_w[:] = quat(math.radians(-3))
  assert torch.equal(penalty(env), torch.zeros(2))


def test_goal_change_success_and_success_window_are_exempt():
  env, command, obj, penalty = full_penalty()
  command.goal_quat_w[:] = quat(math.pi)
  assert torch.equal(penalty(env), torch.zeros(2))
  obj.data.root_link_quat_w[:] = command.goal_quat_w
  for _ in range(250):
    assert torch.equal(penalty(env), torch.zeros(2))
  obj.data.root_link_quat_w[:] = quat(0)
  command.in_success_window[:] = True
  for _ in range(250):
    assert torch.equal(penalty(env), torch.zeros(2))


def test_antipodal_quaternion_is_not_a_goal_switch_and_reset_is_scoped():
  env, command, _, penalty = full_penalty()
  command.goal_quat_w *= -1
  assert torch.equal(penalty(env), torch.ones(2))
  penalty.reset(torch.tensor([0]))
  assert torch.equal(penalty(env), torch.tensor([0., 1.]))


def test_invalid_orientation_cannot_produce_nonfinite_penalty():
  env, _, obj, penalty = full_penalty()
  obj.data.root_link_quat_w[0, 0] = float('nan')
  assert torch.equal(penalty(env), torch.tensor([0., 1.]))


def test_only_a001_recipe_has_the_new_penalty():
  cfg = load_env_cfg('WujiHand2_Reorient_50Hz')
  assert cfg.rewards['stagnation'].weight == -10.
  assert cfg.rewards['orientation_alignment'].weight == 15.
  assert cfg.rewards['action_hf'].weight == -30.
  assert cfg.rewards['action_rate'].weight == 0.
  assert 'stagnation' not in load_env_cfg('WujiHand_Reorient').rewards
