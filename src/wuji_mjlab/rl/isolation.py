"""Quarantine numerically invalid A001 rollouts before they reach PPO updates.

Normal rollouts use the upstream sampler, GAE, losses and optimizer unchanged.
On a fault, discard the pending rollout and restore its normalizer baseline.
No substitute transition or zero reward is trained on. Already committed
updates and polluted statistics from an older checkpoint cannot be undone.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

import torch
from mjlab.rl import RslRlVecEnvWrapper


@dataclass(frozen=True)
class IsolationConfig:
  # Conservative failure bounds, NOT new action or task reward limits.
  max_abs_qpos: float = 1000.0
  max_abs_qvel: float = 1000.0
  max_abs_actuator_force: float = 1000.0
  max_object_distance: float = 5.0
  max_abs_observation: float = 10000.0
  max_abs_step_reward: float = 10000.0
  max_abs_action: float = 10000.0
  max_consecutive_discards: int = 8


class QuarantinedRollout(RuntimeError):
  def __init__(self, event: dict):
    super().__init__(event.get('reason', 'numerical isolation'))
    self.event = event


def needs_a001_isolation(env) -> bool:
  robot = env.scene.entities.get('robot')
  required = {'th_root_link', 'th_slider_connecting_link', 'if_proximal_link',
              'mf_proximal_link', 'rf_proximal_link', 'lf_proximal_link'}
  return robot is not None and required.issubset(set(robot.joint_names))


def _invalid_rows(value: torch.Tensor, limit: float | None = None):
  flat = value.reshape(value.shape[0], -1)
  bad = ~torch.isfinite(flat).all(-1)
  if limit is not None:
    bad |= flat.abs().amax(-1) > limit
  return bad


def _json_safe(value):
  if isinstance(value, dict):
    return {k: _json_safe(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_json_safe(v) for v in value]
  if isinstance(value, float) and not math.isfinite(value):
    return str(value)
  return value


class A001IsolationWrapper(RslRlVecEnvWrapper):
  def __init__(self, env, clip_actions=None, *, output_dir=None, config=None):
    super().__init__(env, clip_actions=clip_actions)
    self.isolation = self
    self.isolation_config = config or IsolationConfig()
    self.output_dir = Path(output_dir) if output_dir is not None else None
    self.discarded_rollouts = 0
    self._bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._event = None
    self._original_sim_step = env.sim.step
    self._previous_qpos = env.sim.data.qpos.clone()
    self._previous_qvel = env.sim.data.qvel.clone()
    env.sim.step = self._physics_step
    if self.output_dir:
      self.output_dir.mkdir(parents=True, exist_ok=True)
      (self.output_dir / 'config.json').write_text(json.dumps(asdict(self.isolation_config), indent=2) + '\n')

  def _physics_step(self):
    self._original_sim_step()
    data = self.unwrapped.sim.data
    cfg = self.isolation_config
    bad = _invalid_rows(data.qpos, cfg.max_abs_qpos)
    bad |= _invalid_rows(data.qvel, cfg.max_abs_qvel)
    bad |= _invalid_rows(data.actuator_force, cfg.max_abs_actuator_force)
    for name in ('qacc', 'qacc_warmstart', 'sensordata'):
      bad |= _invalid_rows(getattr(data, name))
    fresh = bad & ~self._bad
    if bool(fresh.any()) and self._event is None:
      ids = fresh.nonzero().flatten()[:8]
      self._event = dict(reason='physics', env_ids=ids.cpu().tolist(),
                         qpos=data.qpos[ids].cpu().tolist(), qvel=data.qvel[ids].cpu().tolist(),
                         previous_qpos=self._previous_qpos[ids].cpu().tolist(),
                         previous_qvel=self._previous_qvel[ids].cpu().tolist(),
                         actuator_force=data.actuator_force[ids].cpu().tolist())
      model = self.unwrapped.sim.model
      fields = {}
      for name in ('body_mass', 'body_inertia', 'dof_armature', 'actuator_gainprm',
                   'actuator_biasprm', 'geom_solref', 'geom_solimp'):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor) and value.shape[0] == self.num_envs:
          fields[name] = value[ids].cpu().tolist()
      self._event['model_fields'] = fields
    self._bad |= bad
    self._previous_qpos.copy_(torch.where(self._bad[:, None], self._previous_qpos, data.qpos))
    self._previous_qvel.copy_(torch.where(self._bad[:, None], self._previous_qvel, data.qvel))

  def _observation_bad(self, obs):
    bad = torch.zeros_like(self._bad)
    for value in obs.values():
      bad |= _invalid_rows(value, self.isolation_config.max_abs_observation)
    return bad

  def get_observations(self):
    obs = super().get_observations()
    if bool(self._observation_bad(obs).any()):
      raise RuntimeError('Invalid observations at rollout start; no update was performed')
    return obs

  def _reset_and_quarantine(self, bad, reason, rewards=None, obs=None):
    ids = bad.nonzero().flatten()
    event = self._event or dict(reason=reason, env_ids=ids[:8].cpu().tolist())
    event['affected_environment_count'] = int(ids.numel())
    event['common_step_counter'] = int(self.unwrapped.common_step_counter)
    if rewards is not None:
      event['raw_rewards'] = rewards[ids[:8]].detach().cpu().tolist()
    if obs is not None:
      event['raw_observations'] = {key: value[ids[:8]].detach().cpu().tolist() for key, value in obs.items()}
    clean, _ = self.unwrapped.reset(env_ids=ids)
    if bool(self._observation_bad(clean).any()):
      self.record_discard({**event, 'reason': 'reset_failed'})
      raise RuntimeError('Numerical reset did not produce valid observations; refusing to train')
    # Raise BEFORE PPO.process_env_step(), its normalizer updates and storage.
    raise QuarantinedRollout(event)

  def step(self, actions):
    self._bad.zero_()
    self._event = None
    bad_action = _invalid_rows(actions, self.isolation_config.max_abs_action)
    if bool(bad_action.any()):
      self._reset_and_quarantine(bad_action, 'action')
    obs, rewards, dones, extras = super().step(actions)
    cfg = self.isolation_config
    bad = self._bad | self._observation_bad(obs)
    bad |= _invalid_rows(rewards, cfg.max_abs_step_reward)
    obj = self.unwrapped.scene['object'].data.root_link_pos_w
    palm = self.unwrapped.scene['robot'].data.root_link_pos_w
    bad |= ~torch.isfinite(obj).all(-1) | ((obj - palm).norm(dim=-1) > cfg.max_object_distance)
    if bool(bad.any()):
      self._reset_and_quarantine(bad, 'feedback', rewards, obs)
    return obs, rewards, dones, extras

  def record_discard(self, event):
    self.discarded_rollouts += 1
    event = _json_safe({**event, 'discarded_rollouts': self.discarded_rollouts})
    if self.output_dir:
      with (self.output_dir / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
    print('A001_ISOLATED_ROLLOUT', json.dumps({k: v for k, v in event.items()
          if k in ('reason', 'env_ids', 'common_step_counter', 'affected_environment_count', 'discarded_rollouts')}), flush=True)

  def close(self):
    self.unwrapped.sim.step = self._original_sim_step
    return super().close()


def _clone_tree(value):
  if isinstance(value, torch.Tensor):
    return value.detach().clone()
  if isinstance(value, dict):
    return {k: _clone_tree(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_clone_tree(v) for v in value]
  if isinstance(value, tuple):
    return tuple(_clone_tree(v) for v in value)
  return deepcopy(value)


def _finite_tree(value):
  if isinstance(value, torch.Tensor):
    return bool(torch.isfinite(value).all())
  if isinstance(value, dict):
    return all(_finite_tree(v) for v in value.values())
  if isinstance(value, (list, tuple)):
    return all(_finite_tree(v) for v in value)
  return not isinstance(value, float) or math.isfinite(value)


def learn_with_isolation(runner, original_learn, num_learning_iterations, init_at_random_ep_len):
  """Retry a discarded rollout, reusing the upstream learning implementation."""
  alg = runner.alg
  guard = runner.env.isolation
  if alg.is_multi_gpu or alg.actor.is_recurrent or alg.critic.is_recurrent or alg.rnd or alg.symmetry:
    raise ValueError('A001 isolation supports the original feedforward PPO configuration on one GPU only')
  original_act = alg.act
  original_update = alg.update
  original_writer_init = runner.logger.init_logging_writer
  initial_iteration = runner.current_learning_iteration
  completed = 0
  writer_initialized = False
  baseline = None

  def init_writer_once():
    nonlocal writer_initialized
    if not writer_initialized:
      original_writer_init()
      writer_initialized = True

  def tracked_act(_self, obs):
    nonlocal baseline
    if alg.storage.step == 0:
      baseline = [_clone_tree(model.obs_normalizer.state_dict()) for model in (alg.actor, alg.critic)]
    return original_act(obs)

  def transactional_update(_self):
    nonlocal completed, consecutive_discards
    storage = alg.storage
    for value in (storage.rewards, storage.values, storage.returns, storage.advantages):
      if not _finite_tree(value):
        raise QuarantinedRollout(dict(reason='invalid_return_or_advantage', env_ids=[]))
    # Loss/gradient overflow can occur with finite inputs. Restore BOTH model
    # and optimizer if the original update did not remain numerically finite.
    snapshot = _clone_tree(alg.save())
    learning_rate = alg.learning_rate
    try:
      losses = original_update()
    except Exception:
      with torch.inference_mode():
        alg.load(snapshot, None, strict=True)
      alg.learning_rate = learning_rate
      alg.optimizer.zero_grad(set_to_none=True)
      raise
    if not _finite_tree(losses) or not _finite_tree(alg.save()):
      with torch.inference_mode():
        alg.load(snapshot, None, strict=True)
      alg.learning_rate = learning_rate
      alg.optimizer.zero_grad(set_to_none=True)
      raise QuarantinedRollout(dict(reason='invalid_optimizer_update', env_ids=[], losses=losses))
    completed += 1
    consecutive_discards = 0
    return losses

  alg.act = MethodType(tracked_act, alg)
  alg.update = MethodType(transactional_update, alg)
  runner.logger.init_logging_writer = init_writer_once
  first_attempt = True
  consecutive_discards = 0
  try:
    while completed < num_learning_iterations:
      try:
        original_learn(num_learning_iterations - completed,
                       init_at_random_ep_len=init_at_random_ep_len if first_attempt else False)
        break
      except QuarantinedRollout as failure:
        if baseline is None:
          raise RuntimeError('No normalizer rollback baseline was captured') from failure
        # Upstream replaces normalizer._std inside inference_mode while
        # collecting. Its destination must also be restored in that mode.
        with torch.inference_mode():
          for model, state in zip((alg.actor, alg.critic), baseline, strict=True):
            model.obs_normalizer.load_state_dict(state, strict=True)
        alg.storage.clear()
        alg.transition.clear()
        runner.current_learning_iteration = initial_iteration + completed
        runner.logger.ep_extras.clear()
        # These are only display statistics; clearing avoids reporting partial
        # discarded episodes as complete, while healthy physical worlds stay.
        runner.logger.cur_reward_sum.zero_()
        runner.logger.cur_episode_length.zero_()
        guard.record_discard(failure.event)
        if runner.logger.writer is not None:
          runner.logger.writer.add_scalar('Isolation/discarded_rollouts',
                                         guard.discarded_rollouts, runner.current_learning_iteration)
        consecutive_discards += 1
        if consecutive_discards >= guard.isolation_config.max_consecutive_discards:
          raise RuntimeError('Repeated invalid A001 rollouts; stopped without committing them') from failure
        first_attempt = False
  finally:
    alg.act = original_act
    alg.update = original_update
    runner.logger.init_logging_writer = original_writer_init
