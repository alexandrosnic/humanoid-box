from __future__ import annotations

import copy
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ObservationGroupCfg, ObservationManager
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class FrozenLocomotionPolicyActionCfg(ActionTermCfg):
    """Run a frozen locomotion policy and apply only the selected joints."""

    class_type: type | str = f"{__name__}:FrozenLocomotionPolicyAction"

    asset_name: str = MISSING
    policy_path: str = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
    full_joint_names: list[str] = MISSING
    controlled_joint_names: list[str] = MISSING
    low_level_decimation: int = 4
    policy_output_scale: float = 0.5


class FrozenLocomotionPolicyAction(ActionTerm):
    cfg: FrozenLocomotionPolicyActionCfg

    def __init__(self, cfg: FrozenLocomotionPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._env = env

        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(
                f"Frozen locomotion policy '{cfg.policy_path}' does not exist. "
                "Export the stage-1 H1 locomotion policy first."
            )
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._counter = 0
        self._all_joint_ids, self._all_joint_names = self.robot.find_joints(cfg.full_joint_names)
        self._controlled_joint_ids, self._controlled_joint_names = self.robot.find_joints(cfg.controlled_joint_names)
        self._low_level_full_actions = torch.zeros(self.num_envs, len(self._all_joint_ids), device=self.device)

        low_level_obs_cfg = copy.deepcopy(cfg.low_level_observations)

        def last_low_level_action():
            if hasattr(env, "episode_length_buf"):
                self._low_level_full_actions[env.episode_length_buf == 0, :] = 0
            return self._low_level_full_actions

        # Build original locomotion default joint positions (where arms default is 0.0)
        loco_defaults = torch.zeros((env.num_envs, len(self._all_joint_ids)), device=env.device)
        for idx, name in enumerate(self._all_joint_names):
            if "hip_pitch" in name:
                loco_defaults[:, idx] = -0.28
            elif "knee" in name:
                loco_defaults[:, idx] = 0.79
            elif "ankle" in name:
                loco_defaults[:, idx] = -0.52
            else:
                loco_defaults[:, idx] = 0.0

        def correct_joint_pos(dummy_env):
            # Calculate positions relative to original locomotion defaults, NOT custom carry defaults
            pos = self.robot.data.joint_pos[:, self._all_joint_ids]
            return pos - loco_defaults

        low_level_obs_cfg.actions.func = lambda dummy_env: last_low_level_action()
        low_level_obs_cfg.actions.params = dict()
        low_level_obs_cfg.velocity_commands.func = lambda dummy_env: self._raw_actions
        low_level_obs_cfg.velocity_commands.params = dict()
        low_level_obs_cfg.joint_pos.func = correct_joint_pos
        low_level_obs_cfg.joint_pos.params = dict()
        self._low_level_obs_manager = ObservationManager({"ll_policy": low_level_obs_cfg}, env)

        action_name_to_index = {name: index for index, name in enumerate(self._all_joint_names)}
        self._controlled_action_indices = torch.tensor(
            [action_name_to_index[name] for name in self._controlled_joint_names], device=self.device, dtype=torch.long
        )
        self._controlled_joint_offsets = self.robot.data.default_joint_pos.torch[:, self._controlled_joint_ids].clone()
        self._policy_output_scale = torch.tensor(cfg.policy_output_scale, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        # Scale actions from [-1, 1] to target locomotion velocity ranges
        # vx: [-1, 1] -> [-0.05, 0.40] m/s (maps to actions[:, 0] * 0.225 + 0.175)
        self._raw_actions[:, 0] = actions[:, 0] * 0.225 + 0.175
        # vy: [-1, 1] -> [-0.10, 0.10] m/s
        self._raw_actions[:, 1] = actions[:, 1] * 0.10
        # wz: [-1, 1] -> [-0.25, 0.25] rad/s
        self._raw_actions[:, 2] = actions[:, 2] * 0.25

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self._low_level_full_actions[:] = self.policy(low_level_obs)
            self._counter = 0

        controlled_joint_actions = self._low_level_full_actions.index_select(1, self._controlled_action_indices)
        controlled_joint_targets = controlled_joint_actions * self._policy_output_scale + self._controlled_joint_offsets
        self.robot.set_joint_position_target_index(target=controlled_joint_targets, joint_ids=self._controlled_joint_ids)
        self._counter += 1
