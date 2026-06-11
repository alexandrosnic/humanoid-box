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
