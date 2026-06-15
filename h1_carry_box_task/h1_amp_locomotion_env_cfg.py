from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1Rewards

from .mdp import rewards as carry_rewards


@configclass
class H1AmpLocomotionRewards(H1Rewards):
    # Add kinematic tracking reward for gait mocap
    track_gait_mocap = RewTerm(
        func=carry_rewards.track_gait_mocap,
        params={"std": 0.5},
        weight=3.0,
    )


@configclass
class H1AmpLocomotionEnvCfg(H1FlatEnvCfg):
    rewards: H1AmpLocomotionRewards = H1AmpLocomotionRewards()

    def __post_init__(self):
        super().__post_init__()
        
        # We keep H1FlatEnvCfg's velocity tracking and regularization, 
        # but add the mocap imitation reward to force a natural human gait.
        # We can adjust the weights or std here if needed.
        pass
