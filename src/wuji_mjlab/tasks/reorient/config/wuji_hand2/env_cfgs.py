# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Original Hand 2 reorient task with the user's ROH-A001 hardware overlay."""

from dataclasses import replace
from functools import partial

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.rohand_a001.rohand_a001_cfg import (
  CONTROL_JOINTS,
  get_rohand_a001_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_MOUNT_POS,
  REORIENT_HAND2_MOUNT_ROT,
  REORIENT_HAND2_PALM_NORMAL_AXIS,
  REORIENT_HAND2_ROOT_POS,
  REORIENT_HAND2_ROOT_ROT,
  hand2_cube_init_state,
  inject_hand2_reorient_rig,
)
from wuji_mjlab.tasks.reorient.mdp.rohand_a001 import (
  joint_target_error,
  reset_joints,
  torque_saturation_ratio,
)
from wuji_mjlab.tasks.reorient.mdp.stagnation import StagnationPenalty
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg

HAND2_TIP_COLLISION_GEOMS = (".*_finger[1-5]_link4_col",)
HAND2_TIP_CONTACT_BODY_NAMES = (".*_finger[1-5]_link4",)
# ROHand connecting bars are linkages, not additional distal phalanges.
HAND2_DISTAL_FINGER_OBJECT_BODY_NAMES = HAND2_TIP_CONTACT_BODY_NAMES
HAND2_FINGER_CONTACT_PARAM_GEOMS = (r".*finger[2-5]_link[14]_col",)


def _a001_rig_spec(base_spec_fn):
  spec = base_spec_fn()
  spec.body("right_palm_link").add_body(
    name="right_mount", pos=REORIENT_HAND2_MOUNT_POS, quat=REORIENT_HAND2_MOUNT_ROT
  )
  return inject_hand2_reorient_rig(spec)


def get_wuji_hand2_rig_cfg(hand_side: str = "right") -> EntityCfg:
  """Retain the original rig and coordinate interfaces around the A001 asset."""
  robot = get_rohand_a001_cfg(
    REORIENT_HAND2_ROOT_POS, REORIENT_HAND2_ROOT_ROT, hand_side
  )
  robot.spec_fn = partial(_a001_rig_spec, robot.spec_fn)
  return robot


def wuji_hand2_reorient_env_cfg(
  hand_side: str = "right", play: bool = False, num_envs: int = 8192
) -> ManagerBasedRlEnvCfg:
  if hand_side != "right":
    raise ValueError("The ROH-A001 model is right-handed")
  cfg = make_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    palm_subtree_body="right_palm_link",
    soft_pad_geoms=(
      "right_palm_collision",
      "right_finger1_link2_col",
      "right_finger1_link4_col",
    ),
    # Upstream explicitly provides None for mesh-only hands. Keep all other DR.
    geom_size_dr_geoms=None,
    tip_collision_geoms=HAND2_TIP_COLLISION_GEOMS,
    tip_contact_body_names=HAND2_TIP_CONTACT_BODY_NAMES,
    distal_finger_object_body_names=HAND2_DISTAL_FINGER_OBJECT_BODY_NAMES,
    finger_contact_param_geoms=HAND2_FINGER_CONTACT_PARAM_GEOMS,
    cage_up_axis=REORIENT_HAND2_PALM_NORMAL_AXIS,
    robot_tilt_pivot_in_root=REORIENT_HAND2_MOUNT_POS,
  )
  # A001 closed linkages became numerically unstable at 10 ms during PPO.
  # Preserve the upstream 20 Hz base control period; the 50 Hz recipe below
  # derives its substep count from this finer internal physics timestep.
  control_period = cfg.decimation * cfg.sim.mujoco.timestep
  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = round(control_period / cfg.sim.mujoco.timestep)
  cfg.scene.entities = {
    "robot": get_wuji_hand2_rig_cfg(hand_side),
    "object": get_inhand_object_cfg(),
  }
  # User-selected initialization, saved 2026-10-28. Keep original random palm tilt.
  cfg.scene.entities["object"].init_state = replace(
    hand2_cube_init_state(hand_side),
    pos=(-0.151, 0.0072, 0.5616),
    rot=(1.0, 0.0, 0.0, 0.0),
  )
  cfg.actions["joint_pos"].warmup_joint_pos = (0.0, 0.0, 0.3, 0.3, 0.3, 0.3)
  cfg.events["reset_object_pose"].params.update(
    pos_noise=0.0, randomize_orientation=False
  )
  cfg.viewer.body_name = "right_palm_link"
  # Keep all seven upstream sensors; remap nonexistent Wuji middle links.
  cfg.scene.sensors = tuple(
    replace(
      sensor,
      primary=ContactMatch(
        mode="body",
        entity="robot",
        pattern=(".*_palm_link", ".*_finger[1-5]_link1", "right_finger1_link2"),
      ),
    )
    if sensor.name == "undesired_object_contact"
    else sensor
    for sensor in cfg.scene.sensors
  )
  for group_name, angles_name in (
    ("policy", "noisy_joint_angles"),
    ("critic", "joint_angles"),
  ):
    terms = cfg.observations[group_name].terms
    terms[angles_name].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=CONTROL_JOINTS
    )
    terms["qpos_error"].func = joint_target_error
  cfg.events["reset_robot_joints"].func = reset_joints
  cfg.events["reset_robot_joints"].params = {
    "position_offset": (0.0, 0.0),
    "control_positions": (0.0, 0.0, 0.3, 0.3, 0.3, 0.3),
  }
  if "encoder_bias" in cfg.events:
    cfg.events["encoder_bias"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=CONTROL_JOINTS
    )
  cfg.metrics["torque_saturation_ratio"].func = torque_saturation_ratio
  cfg.rewards["stagnation"] = RewardTermCfg(
    func=StagnationPenalty,
    weight=-10.0,
    params={
      "warmup_time_s": 0.4,
      "grace_time_s": 2.0,
      "ramp_time_s": 2.0,
    },
  )
  return cfg
