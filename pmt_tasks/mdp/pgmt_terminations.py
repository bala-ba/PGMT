"""Native manager lifecycle for Stage2 sustained tracking failures."""
from isaaclab.managers import ManagerTermBase

from pmt_tasks.pgmt_terminations import ConsecutiveFailureTimer


class SustainedTrackingFailure(ManagerTermBase):
    """Keep the raw predicate; terminate only after consecutive failed steps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.timer = ConsecutiveFailureTimer(
            env.num_envs, dt=env.step_dt, duration_s=cfg.params["duration_s"],
            device=env.device, initial_step=env.common_step_counter)

    def __call__(self, env, predicate, predicate_params, duration_s):
        return self.timer.update(predicate(env, **predicate_params), env.common_step_counter)

    def reset(self, env_ids=None):
        self.timer.reset(env_ids, step=self._env.common_step_counter)
