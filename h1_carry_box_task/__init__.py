from __future__ import annotations

import gymnasium as gym

from . import agents

TRAIN_TASK_ID = "Isaac-H1-Carry-Box-v0"
PLAY_TASK_ID = "Isaac-H1-Carry-Box-Play-v0"
HOLD_TRAIN_TASK_ID = "Isaac-H1-Carry-Box-Hold-v0"
HOLD_PLAY_TASK_ID = "Isaac-H1-Carry-Box-Hold-Play-v0"


gym.register(
    id=TRAIN_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_carry_box_env_cfg:H1CarryBoxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1CarryBoxPPORunnerCfg",
    },
)


gym.register(
    id=PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_carry_box_env_cfg:H1CarryBoxEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1CarryBoxPPORunnerCfg",
    },
)


gym.register(
    id=HOLD_TRAIN_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_carry_box_env_cfg:H1CarryBoxHoldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1CarryBoxPPORunnerCfg",
    },
)


gym.register(
    id=HOLD_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_carry_box_env_cfg:H1CarryBoxHoldEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1CarryBoxPPORunnerCfg",
    },
)
