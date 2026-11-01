"""Small A001 interfaces; the upstream PPO, EMA controller and rewards are reused."""

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wuji_mjlab.assets.robots.rohand_a001.rohand_a001_cfg import (
  CONTROL_JOINTS,
  CONTROL_MAX,
  linkage_data,
)
from wuji_mjlab.tasks.reorient.mdp._env_utils import resolve_env_ids


def joint_target_error(env, asset_cfg=SceneEntityCfg("robot")):
  robot = env.scene[asset_cfg.name]
  action = env.action_manager.get_term("joint_pos")
  ids = action.target_ids
  position = robot.data.joint_pos[:, ids]
  limits = robot.data.soft_joint_pos_limits[:, ids]
  center = limits.mean(-1)
  half = (limits[..., 1] - limits[..., 0]) * 0.5 + 1e-6
  position = ((position - center) / half).clamp(-1, 1)
  target = ((action.processed_action - center) / half).clamp(-1, 1)
  return position - target


def _lookup(env, robot):
  cache = getattr(env, "_rohand_a001_linkage", None)
  if cache is None:
    source = linkage_data()
    if tuple(robot.joint_names) != tuple(source["joint_names"]):
      raise ValueError("A001 linkage joint order differs from the compiled model")
    ids, names = robot.find_joints(CONTROL_JOINTS)
    if tuple(names) != CONTROL_JOINTS:
      raise ValueError(f"A001 action order mismatch: {names}")
    cache = {"active_ids": torch.tensor(ids, device=env.device), "groups": []}
    for prefix in ("th", "if", "mf", "rf", "lf"):
      cache["groups"].append(
        (
          int(source[f"{prefix}_control_column"].item()),
          torch.as_tensor(source[f"{prefix}_passive_ids"], device=env.device),
          torch.as_tensor(
            source[f"{prefix}_passives"], device=env.device, dtype=torch.float32
          ),
          float(source[f"{prefix}_values"][-1]),
        )
      )
    env._rohand_a001_linkage = cache
  return cache


def reset_joints(
  env,
  env_ids,
  asset_cfg=SceneEntityCfg("robot"),
  position_offset=(-0.3, -0.1),
  control_positions=None,
):
  ids = resolve_env_ids(env, env_ids)
  if ids.numel() == 0:
    return
  robot = env.scene[asset_cfg.name]
  cache = _lookup(env, robot)
  q = robot.data.default_joint_pos[ids].clone()
  controls = q[:, cache["active_ids"]].clone()
  if control_positions is not None:
    controls[:] = controls.new_tensor(control_positions)
  perturb = torch.empty_like(controls).uniform_(*position_offset)
  perturb[:, 0] = 0.0  # Preserve opposition, as upstream keeps abduction fixed.
  upper = controls.new_tensor(CONTROL_MAX)
  controls = (controls + perturb).clamp(min=0.0)
  controls = torch.minimum(controls, upper)
  q[:, cache["active_ids"]] = controls
  for col, passive_ids, table, maximum in cache["groups"]:
    coordinate = (controls[:, col] / maximum * (len(table) - 1)).clamp(
      0.0, len(table) - 1
    )
    lo = coordinate.floor().long().clamp(max=len(table) - 2)
    weight = (coordinate - lo).unsqueeze(-1)
    q[:, passive_ids] = table[lo] * (1 - weight) + table[lo + 1] * weight
  robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)


def torque_saturation_ratio(env, asset_cfg=SceneEntityCfg("robot"), threshold_frac=0.9):
  robot = env.scene[asset_cfg.name]
  ctrl_ids = robot.indexing.ctrl_ids[asset_cfg.actuator_ids]
  force = robot.data.actuator_force[:, asset_cfg.actuator_ids]
  limits = env.sim.model.actuator_forcerange[:, ctrl_ids, :].abs().amax(-1)
  flags = torch.as_tensor(env.sim.mj_model.actuator_forcelimited, device=env.device)[
    ctrl_ids
  ].bool()
  saturated = (force.abs() > threshold_frac * limits) & flags
  return saturated.float().sum(-1) / flags.float().sum().clamp_min(1.0)
