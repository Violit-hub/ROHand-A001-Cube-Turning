"""ROH-A001 right-hand asset. Native closed linkages and joint axes are retained."""

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ASSET_DIR = Path(__file__).parent
XML_PATH = ASSET_DIR / "a001_right.xml"
# This is mjlab's natural joint order, not XML actuator declaration order.
CONTROL_JOINTS = (
  "th_root_link",
  "th_slider_connecting_link",
  "if_proximal_link",
  "mf_proximal_link",
  "rf_proximal_link",
  "lf_proximal_link",
)
CONTROL_MAX = (1.57, 1.00, 1.44, 1.44, 1.44, 1.44)
DEFAULT_CONTROLS = (0.80, 0.65, 0.90, 0.80, 0.90, 1.10)
CONTROL_LABELS = ("拇指根部旋转", "拇指弯曲", "食指", "中指", "无名指", "小指")
BODY_NAMES = {
  "base_link": "right_palm_link",
  "th_root_link": "right_finger1_link1",
  "th_slider_connecting_link": "right_finger1_drive_link",
  "th_slider_link": "right_finger1_slider_link",
  "th_connecting_link": "right_finger1_linkage",
  "th_distal_link": "right_finger1_link4",
  "th_proximal_link": "right_finger1_link2",
}
for finger, prefix in enumerate(("if", "mf", "rf", "lf"), 2):
  BODY_NAMES.update(
    {
      f"{prefix}_proximal_link": f"right_finger{finger}_link1",
      f"{prefix}_distal_link": f"right_finger{finger}_link4",
      f"{prefix}_connecting_link": f"right_finger{finger}_linkage",
    }
  )


def get_rohand_a001_spec():
  spec = mujoco.MjSpec.from_file(str(XML_PATH))
  spec.assets = {
    f"meshes/{p.name}": p.read_bytes() for p in (ASSET_DIR / "meshes").glob("*.STL")
  }
  return spec


@lru_cache(maxsize=1)
def linkage_data():
  with np.load(ASSET_DIR / "linkage.npz", allow_pickle=False) as data:
    return {k: v.copy() for k, v in data.items()}


def joint_positions(controls):
  """Native assembly branch, interpolated from independently solved linkages."""
  data = linkage_data()
  q = data["reference"].copy()
  for name, value in zip(CONTROL_JOINTS, controls, strict=True):
    q[list(data["joint_names"]).index(name)] = value
  for prefix in ("th", "if", "mf", "rf", "lf"):
    value = controls[int(data[f"{prefix}_control_column"].item())]
    indices = data[f"{prefix}_passive_ids"]
    for col, jid in enumerate(indices):
      q[jid] = np.interp(
        value, data[f"{prefix}_values"], data[f"{prefix}_passives"][:, col]
      )
  return q


def get_rohand_a001_cfg(root_pos, root_rot, hand_side="right"):
  if hand_side != "right":
    raise ValueError("This A001 asset is the native right-hand model")
  data = linkage_data()
  q = joint_positions(DEFAULT_CONTROLS)
  return EntityCfg(
    spec_fn=get_rohand_a001_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=root_pos,
      rot=root_rot,
      joint_pos=dict(zip(map(str, data["joint_names"]), map(float, q), strict=True)),
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=CONTROL_JOINTS),),
      soft_joint_pos_limit_factor=1.0,
    ),
  )
