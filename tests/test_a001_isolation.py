"""Isolation contracts: upstream parity, rollback, and actual GPU fault injection."""
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import wuji_mjlab.tasks  # noqa: F401 -- register tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from tensordict import TensorDict
from wuji_mjlab.rl.isolation import (
  A001IsolationWrapper,
  IsolationConfig,
  QuarantinedRollout,
  learn_with_isolation,
  needs_a001_isolation,
)
from wuji_mjlab.rl.runner import WujiOnPolicyRunner


def config():
  cfg = asdict(load_rl_cfg('WujiHand2_Reorient_50Hz'))
  cfg.update(logger='tensorboard', upload_model=False, max_iterations=2,
             num_steps_per_env=4)
  cfg['actor']['hidden_dims'] = (16, 16)
  cfg['critic']['hidden_dims'] = (16, 16)
  cfg['algorithm'].update(num_learning_epochs=2, num_mini_batches=4)
  return cfg


class FakeEnv:
  def __init__(self, fault=False):
    self.num_envs, self.num_actions, self.device = 8, 6, 'cpu'
    self.cfg = {}
    self.unwrapped = self
    self.episode_length_buf = torch.zeros(8, dtype=torch.long)
    self.max_episode_length = 1000
    self.state = torch.zeros(8, 4)
    self.calls = 0
    self.fault = fault
    self.events = []
    self.isolation = SimpleNamespace(isolation_config=IsolationConfig(), discarded_rollouts=0)
    def record(event):
      self.events.append(event)
      self.isolation.discarded_rollouts += 1
      if hasattr(self, 'on_discard'):
        self.on_discard()
    self.isolation.record_discard = record

  def get_observations(self):
    return TensorDict({'policy': self.state, 'critic': self.state}, batch_size=[8])

  def step(self, actions):
    self.calls += 1
    self.state = self.state * .9 + actions[:, :4] * .1
    self.episode_length_buf += 1
    if self.fault and self.calls == 3:
      self.state[0].zero_()
      raise QuarantinedRollout(dict(reason='injected_feedback', env_ids=[0]))
    return self.get_observations(), -self.state.square().sum(-1), torch.zeros(8, dtype=torch.long), {}


def assert_trees_equal(left, right):
  if isinstance(left, torch.Tensor):
    assert torch.equal(left, right)
  elif isinstance(left, dict):
    assert left.keys() == right.keys()
    for key in left:
      assert_trees_equal(left[key], right[key])
  elif isinstance(left, (tuple, list)):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
      assert_trees_equal(a, b)
  else:
    assert left == right


def test_healthy_training_matches_upstream_exactly():
  torch.manual_seed(100)
  ordinary = WujiOnPolicyRunner(FakeEnv(), config(), device='cpu')
  ordinary.learn(2, False)
  torch.manual_seed(100)
  guarded = WujiOnPolicyRunner(FakeEnv(), config(), device='cpu')
  learn_with_isolation(guarded, super(WujiOnPolicyRunner, guarded).learn, 2, False)
  assert_trees_equal(ordinary.alg.save(), guarded.alg.save())
  assert ordinary.current_learning_iteration == guarded.current_learning_iteration == 1


def test_partial_rollout_and_normalizers_are_rolled_back():
  torch.manual_seed(101)
  env = FakeEnv(fault=True)
  runner = WujiOnPolicyRunner(env, config(), device='cpu')
  def on_discard():
    assert int(runner.alg.actor.obs_normalizer.count) == 0
    assert int(runner.alg.critic.obs_normalizer.count) == 0
    assert runner.alg.storage.step == 0
    assert runner.alg.optimizer.state_dict()['state'] == {}
  env.on_discard = on_discard
  learn_with_isolation(runner, super(WujiOnPolicyRunner, runner).learn, 2, False)
  assert len(env.events) == 1
  assert int(runner.alg.actor.obs_normalizer.count) == 2 * 4 * 8
  assert runner.current_learning_iteration == 1


def test_bad_update_restores_model_and_optimizer():
  torch.manual_seed(102)
  env = FakeEnv()
  runner = WujiOnPolicyRunner(env, config(), device='cpu')
  original_update = runner.alg.update
  update_calls = 0
  committed_state = None
  def update_with_overflow():
    nonlocal update_calls, committed_state
    update_calls += 1
    loss = original_update()
    if update_calls == 1:
      committed_state = deepcopy(runner.alg.save())
    if update_calls == 2:
      with torch.no_grad():
        next(runner.alg.actor.parameters()).fill_(float('nan'))
      loss['value'] = float('inf')
    return loss
  runner.alg.update = update_with_overflow
  def on_discard():
    assert committed_state is not None
    assert committed_state['optimizer_state_dict']['state']
    assert_trees_equal(committed_state, runner.alg.save())
  env.on_discard = on_discard
  learn_with_isolation(runner, super(WujiOnPolicyRunner, runner).learn, 2, False)
  assert len(env.events) == 1 and env.events[0]['reason'] == 'invalid_optimizer_update'
  assert all(bool(torch.isfinite(v).all()) for v in runner.alg.actor.state_dict().values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual A001 GPU simulation required')
def test_actual_a001_faults_are_isolated_before_learning(tmp_path):
  root = Path(__file__).resolve().parents[1]
  manifest = json.loads((root / 'pretrained/model_11100.json').read_text())
  best = root / 'pretrained/model_11100.pt'
  original_hash = hashlib.sha256(best.read_bytes()).hexdigest()
  cfg = load_env_cfg('WujiHand2_Reorient_50Hz')
  cfg.scene.num_envs = 32
  cfg.seed = 42
  env = ManagerBasedRlEnv(cfg, device='cuda:0')
  assert needs_a001_isolation(env)
  wrapped = A001IsolationWrapper(env, output_dir=tmp_path / 'isolation')
  agent = config()
  agent['num_steps_per_env'] = 8
  runner = WujiOnPolicyRunner(wrapped, agent, device='cuda:0')
  env.max_common_steps = 500000
  simulation_step = wrapped._original_sim_step
  raw_step = env.step
  substeps = 0
  controls = 0
  def inject_physics():
    nonlocal substeps
    simulation_step()
    substeps += 1
    if substeps == 3 * cfg.decimation:
      env.sim.data.qpos[0, -7] = 1e12
  def inject_feedback(actions):
    nonlocal controls
    result = raw_step(actions)
    controls += 1
    if controls == 3:
      result[1][1] = -1e30
      result[0]['policy'][2, 36] = 1e12
      result[0]['critic'][3, 93] = float('nan')
    return result
  wrapped._original_sim_step = inject_physics
  env.step = inject_feedback
  try:
    runner.learn(2, False)
    events = [json.loads(line) for line in (tmp_path / 'isolation/events.jsonl').read_text().splitlines()]
    assert wrapped.discarded_rollouts == 1
    assert events[0]['affected_environment_count'] == 4
    assert events[0]['reason'] == 'physics'
    assert events[0]['qpos'][0][-7] > 1e11
    # Counts must exclude ALL 2 earlier steps of the discarded rollout.
    assert int(runner.alg.actor.obs_normalizer.count) == 2 * 8 * 32
    assert int(runner.alg.critic.obs_normalizer.count) == 2 * 8 * 32
    assert float(runner.alg.actor.obs_normalizer.std.max()) < 1000
    assert float(runner.alg.critic.obs_normalizer.std.max()) < 1000
    for sd in [runner.alg.actor.state_dict(), runner.alg.critic.state_dict()]:
      assert all(bool(torch.isfinite(value).all()) for value in sd.values())
    assert hashlib.sha256(best.read_bytes()).hexdigest() == original_hash == manifest['sha256']
    report = dict(passed=True, num_envs=32, completed_test_updates=2,
                  isolated_rollouts=wrapped.discarded_rollouts, injected_environments=4,
                  normalizer_sample_count=int(runner.alg.actor.obs_normalizer.count),
                  actor_std_max=float(runner.alg.actor.obs_normalizer.std.max()),
                  critic_std_max=float(runner.alg.critic.obs_normalizer.std.max()),
                  best_checkpoint_unchanged=True, training_started=False)
    report_out = tmp_path / 'verification'
    report_out.mkdir(exist_ok=True)
    (report_out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (report_out / 'injected_event.json').write_text(json.dumps(events[0], indent=2) + '\n')
  finally:
    env.step = raw_step
    wrapped._original_sim_step = simulation_step
    wrapped.close()
