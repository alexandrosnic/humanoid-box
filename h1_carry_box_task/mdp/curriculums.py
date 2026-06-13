from __future__ import annotations

from collections.abc import Sequence

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase


class increase_box_mass_range(ManagerTermBase):
    """Increase the randomized carry-box mass range over training."""

    def __init__(self, cfg: CurriculumTermCfg, env):
        super().__init__(cfg, env)
        self._term_cfg = env.event_manager.get_term_cfg(cfg.params["term_name"])

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        term_name: str,
        stages: Sequence[tuple[int, tuple[float, float]]],
    ) -> float:
        target_range = tuple(self._term_cfg.params["mass_distribution_params"])
        for step_threshold, mass_range in stages:
            if env.common_step_counter >= step_threshold:
                target_range = tuple(mass_range)

        if tuple(self._term_cfg.params["mass_distribution_params"]) != target_range:
            self._term_cfg.params["mass_distribution_params"] = target_range
            env.event_manager.set_term_cfg(term_name, self._term_cfg)

        return float(self._term_cfg.params["mass_distribution_params"][1])


class ramp_reward_weight(ManagerTermBase):
    """Gradually ramp up a reward weight over training steps."""

    def __init__(self, cfg: CurriculumTermCfg, env):
        super().__init__(cfg, env)
        self._term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(self._term_name)

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        term_name: str,
        start_step: int,
        end_step: int,
        target_weight: float,
    ) -> float:
        current_step = env.common_step_counter
        if current_step < start_step:
            weight = 0.0
        elif current_step > end_step:
            weight = target_weight
        else:
            # Linear interpolation
            weight = target_weight * (current_step - start_step) / (end_step - start_step)

        if self._term_cfg.weight != weight:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(self._term_name, self._term_cfg)

        return weight

