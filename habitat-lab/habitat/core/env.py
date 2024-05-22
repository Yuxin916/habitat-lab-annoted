#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import gym
import numba
import numpy as np
from gymnasium import spaces

from habitat.config import read_write
from habitat.core.dataset import BaseEpisode, Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks.registration import make_task
from habitat.utils import profiling_wrapper

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Env:
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied task with simulator
    is abstracted inside

    :ref:`Env`. Acts as a base for other derived environment classes. 基础环境
    :ref:`Env` consists of three major components:
            ``dataset`` (`episodes`)
            ``simulator`` (:ref:`sim`)
             :ref:`task`
    and connects all the three components together.
    """

    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: "DictConfig"
    _dataset: Optional[Dataset[Episode]]
    number_of_episodes: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator[Episode]]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool
    _episode_from_iter_on_reset: bool
    _episode_force_changed: bool

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset[Episode]] = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``. 所有config参数
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        if "habitat" in config:
            config = config.habitat
        self._config = config
        self._dataset = dataset
        # 在没有dataset传进来的时候
        # 根据config文件中的dataset配置，创建dataset
        if self._dataset is None and config.dataset.type:
            # @registry.register_dataset(name="ObjectNav-v1")
            # /habitat-lab/habitat-lab/habitat/datasets/object_nav/object_nav_dataset.py
            # 获取self.episode
            self._dataset = make_dataset(
                id_dataset=config.dataset.type, # ObjectNav-v1
                config=config.dataset
            )

        self._current_episode = None
        self._episode_iterator = None
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        # TODO: 后面出问题了再看
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"

            self._setup_episode_iterator()
            self.current_episode = next(self.episode_iterator)
            with read_write(self._config):
                self._config.simulator.scene_dataset = (
                    self.current_episode.scene_dataset_config
                )
                self._config.simulator.scene = self.current_episode.scene_id

            self.number_of_episodes = len(self.episodes)
        else:
            self.number_of_episodes = None

        # 通过config文件中的simulator配置，创建simulator
        # 通过registry，创建simulator
        # @registry.register_simulator(name="Sim-v0")
        # /habitat-lab/habitat-lab/habitat/sims/habitat_simulator line 270
        self._sim = make_sim(
            id_sim=self._config.simulator.type,  # "Sim-v0"
            config=self._config.simulator
        )

        # 通过config文件中的task配置，创建task
        # 通过registry，创建task
        # @registry.register_task(name="Nav-v0")
        # /habitat-lab/habitat-lab/habitat/core/embodied_task.py
        self._task = make_task(
            self._config.task.type,
            config=self._config.task,
            sim=self._sim,
            dataset=self._dataset,
        )

        # 合并sim和task的状态空间
            # sim里面只包含RGB，Depth等sensor
            # gps, compass等sensor在task里面
        # self.observation_space = spaces.Dict(
        #     {
        #         **self._sim.sensor_suite.observation_spaces.spaces,
        #         **self._task.sensor_suite.observation_spaces.spaces,
        #     }
        # )

        # 从task中读取动作空间
        self.action_space = self._task.action_space

        # 一个episode的最长时间 wallclock time
        self._max_episode_seconds = (
            self._config.environment.max_episode_seconds
        )
        # 一个episode的最大步数
        self._max_episode_steps = self._config.environment.max_episode_steps
        # 一个episode的agent数量
        self.num_agents = self._config.simulator.num_agents

        # 一些数据
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    def _setup_episode_iterator(self):
        assert self._dataset is not None
        iter_option_dict = {
            k.lower(): v
            for k, v in self._config.environment.iterator_options.items()
        }
        iter_option_dict["seed"] = self._config.seed

        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Episode) -> None:
        self._current_episode = episode
        # This allows the current episode to be set here
        # and then reset be called without the episode changing
        self._episode_from_iter_on_reset = False
        self._episode_force_changed = True

    @property
    def episode_iterator(self) -> Iterator[Episode]:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator[Episode]) -> None:
        self._episode_iterator = new_iter
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def episodes(self) -> List[Episode]:
        return (
            self._dataset.episodes
            if self._dataset
            else cast(List[Episode], [])
        )

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        assert (
            self._dataset is not None
        ), "Environment must have a dataset to set episodes"
        self._dataset.episodes = episodes
        self._setup_episode_iterator()
        self._current_episode = None
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        # 指向EmbodiedTask
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        return (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ) or (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        )

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        ############################################################
        # reset一些数据 - episode开始时间，episode结束标志，episode步数
        self._reset_stats()
        ############################################################

        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        if (
            self._episode_iterator is not None
            and self._episode_from_iter_on_reset
        ):
            self._current_episode = next(self._episode_iterator)

        # This is always set to true after a reset that way
        # on the next reset an new episode is taken (if possible)
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        assert self._current_episode is not None, "Reset requires an episode"

        # 在每次reset环境的时候 覆盖_task里面的config
        self.reconfigure(self._config)

        # 在embodied_task.py中reset
        """
        observations是一个list of 2 agents
        Each agent has a dict of observations (all ndarray)
            'rgb': (480, 640, 3)
            'depth': (480, 640, 1)
            'objectgoal': (1,)
            'compass': (1,)
            'gps': (2,)

        两个agent初始化在一个点 但是朝向不同
        """
        observations = self.task.reset(episode=self.current_episode)

        # reset所有measurements
        self._task.measurements.reset_measures(
            episode=self.current_episode,
            task=self.task,
            observations=observations,
        )

        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """

        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"
        assert (
            not self._episode_force_changed
        ), "Episode was changed either by setting current_episode or changing the episodes list. Call reset before stepping the environment again."

        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer)):
            action = {"action": action}

        try:
            observations = self.task.step(
                action=action, episode=self.current_episode)
        except Exception as e:
            print(f"Error in env.py step: {e}")

        try:
            self._task.measurements.update_measures(
                episode=self.current_episode,
                action=action,
                task=self.task,
                observations=observations,
            )
        except Exception as e:
            print(f"Error in env.py update_measures: {e}")

        self._update_step_stats()

        return observations

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: "DictConfig") -> None:
        # 在每次reset环境的时候
        # 根据当前的episode信息
        # 覆盖_task里面的config

        self._config = self._task.overwrite_sim_config(
            config, self.current_episode
        )
        # habitat_simulator 根据当前的episode信息
        # 更新sim_config和agent_state
        self._sim.reconfigure(self._config.simulator, self.current_episode)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over :ref:`Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: :ref:`get_reward_range()`, :ref:`get_reward()`,
    :ref:`get_done()`, :ref:`get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config to construct :ref:`Env`
        :param dataset: dataset to construct :ref:`Env`.
        """
        if "habitat" in config:
            config = config.habitat
        self._core_env_config = config
        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

    @property
    def config(self) -> "DictConfig":
        return self._core_env_config

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Episode]:
        return self._env.episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        self._env.episodes = episodes

    def current_episode(self, all_info: bool = False) -> BaseEpisode:
        r"""Returns the current episode of the environment.

        :param all_info: If true, all the information in the episode
                         will be provided. Otherwise, only episode_id
                         and scene_id will be included.
        :return: The BaseEpisode object for the current episode.
        """
        if all_info:
            return self._env.current_episode
        else:
            return BaseEpisode(
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id,
            )

    @profiling_wrapper.RangeContext("RLEnv.reset")
    def reset(
        self, *, return_info: bool = False, **kwargs
    ) -> Union[Observations, Tuple[Observations, Dict]]:
        observations = self._env.reset()
        if return_info:
            return observations, self.get_info(observations)
        else:
            return observations

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the :ref:`step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    @profiling_wrapper.RangeContext("RLEnv.step")
    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
