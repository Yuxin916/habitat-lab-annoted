import importlib
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np

import habitat
from habitat import Dataset
from habitat.gym.gym_wrapper import HabGymWrapper

if TYPE_CHECKING:
    from omegaconf import DictConfig


RLTaskEnvObsType = Union[np.ndarray, Dict[str, np.ndarray]]

class RLTaskEnv(habitat.RLEnv):
    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        super().__init__(config, dataset)
        self._reward_measure_name = self.config.task.reward_measure
        self._success_measure_name = self.config.task.success_measure
        self._slack_reward = self.config.task.slack_reward
        self._success_reward = self.config.task.success_reward
        self._end_on_success = self.config.task.end_on_success
        assert (
            self._reward_measure_name is not None
        ), "The key task.reward_measure cannot be None"
        assert (
            self._success_measure_name is not None
        ), "The key task.success_measure cannot be None"

    def reset(
        self, *args, return_info: bool = False, **kwargs
    ) -> Union[RLTaskEnvObsType, Tuple[RLTaskEnvObsType, Dict]]:
        return super().reset(*args, return_info=return_info, **kwargs)

    def step(
        self, *args, **kwargs
    ) -> Tuple[RLTaskEnvObsType, float, bool, dict]:
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward = self._slack_reward

        reward += current_measure

        if self._episode_success():
            reward += self._success_reward

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            print("DEBUG - env_RL_task - get_done -> Episode Over")
            done = True
        if self._end_on_success and self._episode_success():
            print("DEBUG - env_RL_task - get_done -> Episode Success")
            done = True
        return done

    def get_info(self, observations):
        return self._env.get_metrics()




### Register the environment
@habitat.registry.register_env(name="GymHabitatEnv")
class GymHabitatEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        base_env = RLTaskEnv(config=config, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)
