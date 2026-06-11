from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def box_too_far_from_robot(
    env: ManagerBasedRLEnv,
    max_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("carry_box"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    carry_box: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(robot.data.root_pos_w.torch[:, :3] - carry_box.data.root_pos_w.torch[:, :3], dim=1)
    return distance > max_distance
