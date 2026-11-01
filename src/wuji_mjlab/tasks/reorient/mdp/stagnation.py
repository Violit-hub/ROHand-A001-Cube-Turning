"""Delayed cube-stagnation penalty, independent of finger action magnitude."""
from __future__ import annotations

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_error_magnitude

from .actions import warmup_steps


class StagnationPenalty(ManagerTermBase):
  """Return [0, 1]; RewardManager applies the configured weight and dt."""

  def __init__(self, cfg, env):
    super().__init__(env)
    params = cfg.params
    self._command_name = params.get('command_name', 'reorient_command')
    self._motion_threshold = float(params.get('motion_threshold_rad', math.radians(2)))
    self._progress_threshold = float(params.get('progress_threshold_rad', math.radians(2)))
    self._warmup_steps = warmup_steps(float(params.get('warmup_time_s', .4)), env.step_dt)
    self._grace_steps = warmup_steps(float(params.get('grace_time_s', 2.)), env.step_dt)
    self._ramp_steps = warmup_steps(float(params.get('ramp_time_s', 2.)), env.step_dt)
    if self._motion_threshold <= 0 or self._progress_threshold <= 0 or self._ramp_steps < 1:
      raise ValueError('Stagnation thresholds and ramp duration must be positive')
    self._anchor_quat = torch.zeros((env.num_envs, 4), device=env.device)
    self._anchor_quat[:, 0] = 1
    self._last_goal = self._anchor_quat.clone()
    self._anchor_error = torch.zeros(env.num_envs, device=env.device)
    self._age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._idle_steps = torch.zeros_like(self._age)
    self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self._age[ids] = 0
    self._idle_steps[ids] = 0
    self._initialized[ids] = False

  def __call__(self, env, **_params):
    command = env.command_manager.get_term(self._command_name)
    actual = env.scene['object'].data.root_link_quat_w
    goal = command.goal_quat_w
    valid = torch.isfinite(actual).all(-1) & torch.isfinite(goal).all(-1)
    valid &= (actual.norm(dim=-1) > 1e-6) & (goal.norm(dim=-1) > 1e-6)
    identity = torch.zeros_like(actual)
    identity[:, 0] = 1
    q = torch.where(valid[:, None], actual, identity)
    g = torch.where(valid[:, None], goal, identity)
    error = quat_error_magnitude(q, g)
    motion = quat_error_magnitude(q, self._anchor_quat)
    changed_goal = quat_error_magnitude(g, self._last_goal) > 1e-5
    self._age += 1
    clear = (~valid | ~self._initialized | changed_goal
             | (self._age <= self._warmup_steps)
             | (error <= command.cfg.success_threshold)
             | command.in_success_window
             | (motion >= self._motion_threshold)
             | (self._anchor_error - error >= self._progress_threshold))
    self._idle_steps = torch.where(clear, torch.zeros_like(self._idle_steps), self._idle_steps + 1)
    self._anchor_quat = torch.where(clear[:, None], q, self._anchor_quat)
    self._anchor_error = torch.where(clear, error, self._anchor_error)
    self._last_goal.copy_(g)
    self._initialized.copy_(valid)
    return ((self._idle_steps - self._grace_steps).float() / self._ramp_steps).clamp(0., 1.)
