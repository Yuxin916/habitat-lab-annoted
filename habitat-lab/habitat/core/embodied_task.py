#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements task and measurements needed for training and benchmarking of
``habitat.Agent`` inside ``habitat.Env``.
"""

import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import numpy as np
from omegaconf import OmegaConf

from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Observations, SensorSuite, Simulator
from habitat.core.spaces import ActionSpace, EmptySpace, Space

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Action:
    r"""
    SimulatorTaskAction类继承自Action类

    An action that can be performed by an agent solving a task in environment.
    For example for navigation task action classes will be:
    ``MoveForwardAction, TurnLeftAction, TurnRightAction``. The action can
    use ``Task`` members to pass a state to another action, as well as keep
    own state and reset when new episode starts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return

    def reset(self, *args: Any, **kwargs: Any) -> None:
        r"""Reset method is called from ``Env`` on each reset for each new
        episode. Goal of the method is to reset ``Action``'s state for each
        episode.
        """
        raise NotImplementedError

    def step(self, *args: Any, **kwargs: Any) -> Observations:
        r"""Step method is called from ``Env`` on each ``step``. Can call
        simulator or task method, change task's state.

        :param kwargs: optional parameters for the action, like distance/force.
        :return: observations after taking action in the task, including ones
            coming from a simulator.
        """
        raise NotImplementedError

    @property
    def action_space(self) -> Space:
        r"""a current Action's action space."""
        raise NotImplementedError


class SimulatorTaskAction(Action):
    r"""
    所有的动作都会继承这个类
    An ``EmbodiedTask`` action that is wrapping simulator action.
    """

    def __init__(
            self, *args: Any, config: "DictConfig", sim: Simulator, **kwargs: Any
    ) -> None:
        self._config = config
        self._sim = sim

    @property
    def action_space(self):
        return EmptySpace()

    def reset(self, *args: Any, **kwargs: Any) -> None:
        return None

    def step(self, *args: Any, **kwargs: Any) -> Observations:
        r"""Step method is called from ``Env`` on each ``step``."""
        raise NotImplementedError


class Measure:
    r"""Represents a measure that provides measurement on top of environment
    and task.

    :data uuid: universally unique id.
    :data _metric: metric for the :ref:`Measure`, this has to be updated with
        each :ref:`step() <env.Env.step()>` call on :ref:`env.Env`.

    This can be used for tracking statistics when running experiments. The
    user of this class needs to implement the :ref:`reset_metric()` and
    :ref:`update_metric()` method and the user is also required to set the
    :ref:`uuid <Measure.uuid>` and :ref:`_metric` attributes.

    .. (uuid is a builtin Python module, so just :ref:`uuid` would link there)
    """

    _metric: Any
    uuid: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.uuid = self._get_uuid(*args, **kwargs)
        self._metric = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Reset :ref:`_metric`, this method is called from :ref:`env.Env` on
        each reset.
        """
        raise NotImplementedError

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Update :ref:`_metric`, this method is called from :ref:`env.Env`
        on each :ref:`step() <env.Env.step()>`
        """
        raise NotImplementedError

    def get_metric(self):
        r"""..

        :return: the current metric for :ref:`Measure`.
        """
        # 获得当前计算的数值
        return self._metric


class Metrics(dict):
    r"""Dictionary containing measurements."""

    def __init__(self, measures: Dict[str, Measure]) -> None:
        """Constructor

        :param measures: list of :ref:`Measure` whose metrics are fetched and
            packaged.
        """
        data = [
            (uuid, measure.get_metric()) for uuid, measure in measures.items()
        ]
        super().__init__(data)


class Measurements:
    r"""Represents a set of Measures, with each :ref:`Measure` being
    identified through a unique id.
    """

    measures: Dict[str, Measure]

    def __init__(self, measures: Iterable[Measure]) -> None:
        """Constructor
        整合所有以Measure为基类的类

        :param measures: list containing :ref:`Measure`, uuid of each
            :ref:`Measure` must be unique.
        """
        self.measures = OrderedDict()
        for measure in measures:
            assert (
                    measure.uuid not in self.measures
            ), "'{}' is duplicated measure uuid".format(measure.uuid)
            self.measures[measure.uuid] = measure

    def reset_measures(self, *args: Any, **kwargs: Any) -> None:
        """
        Reset所有的measure
        """
        for measure in self.measures.values():
            # 重置每一个measure
            measure.reset_metric(*args, **kwargs)

    def update_measures(self, *args: Any, task, **kwargs: Any) -> None:
        """
        更新所有的measure
        """
        for measure in self.measures.values():
            t_start = time.time()
            # nav.py update_metric function
            try:
                measure.update_metric(*args, task=task, **kwargs)
            except Exception as e:
                print(f"Error in measure {measure.uuid}: {e}")
                print('embodied_task.py: update_measures')
            measure_name = measure._get_uuid(*args, task=task, **kwargs)
            task.add_perf_timing(f"measures.{measure_name}", t_start)

    def get_metrics(self) -> Metrics:
        r"""Collects measurement from all :ref:`Measure`\ s and returns it
        packaged inside :ref:`Metrics`.
        """
        return Metrics(self.measures)

    def _get_measure_index(self, measure_name):
        return list(self.measures.keys()).index(measure_name)

    def check_measure_dependencies(
            self, measure_name: str, dependencies: List[str]
    ):
        r"""Checks if dependencies measures are enabled and calculatethat the measure
        :param measure_name: a name of the measure for which has dependencies.
        :param dependencies: a list of a measure names that are required by
        the measure.
        :return:
        """
        measure_index = self._get_measure_index(measure_name)
        for dependency_measure in dependencies:
            assert (
                    dependency_measure in self.measures
            ), f"""{measure_name} measure requires {dependency_measure}
                listed in the measures list in the config."""

        for dependency_measure in dependencies:
            assert measure_index > self._get_measure_index(
                dependency_measure
            ), f"""{measure_name} measure requires be listed after {dependency_measure}
                in the measures list in the config."""


class EmbodiedTask:
    r"""Base class for embodied task.

    包含一些重要的任务属性：
        observation space,
        action space,
        measurements,
        用的simulator

    ``Env``里的:ref:`reset` and :ref:`step` 指向这里的reset和step方法。

    nav.py里的NavigationTask类继承自EmbodiedTask类。

    Args:
        config: config for the task.
        sim: reference to the simulator for calculating task observations.
        dataset: reference to dataset for task instance level information.

    :data measurements: set of task measures.
    :data sensor_suite: suite of task sensors.
    """

    _config: Any
    _sim: Optional[Simulator]
    _dataset: Optional[Dataset]
    _is_episode_active: bool
    measurements: Measurements
    sensor_suite: SensorSuite

    def __init__(
            self,
            config: "DictConfig",
            sim: Simulator,
            dataset: Optional[Dataset] = None,
    ) -> None:
        from habitat.core.registry import registry

        # sim里面只包含RGB，Depth等sensor
        # gps, compass等sensor在task里面

        self._config = config
        self._sim = sim
        self._dataset = dataset
        self._physics_target_sps = config.physics_target_sps
        assert (
                self._physics_target_sps > 0
        ), "physics_target_sps must be positive"

        # 根据一系列metrics(config.measurements)
        # 分别初始化对应的measurements
        # 在nav.py里的NavigationTask类中，
        # 比如 config.measurements = ['distance_to_goal', 'spl', 'success']
        # 就会初始化对应的measurements函数 (在nav.py里的NavigationTask类中)
        # 最后一个Measurements整合起来
        self.measurements = Measurements(
            self._init_entities(
                entities_configs=config.measurements,
                register_func=registry.get_measure,
            ).values()
        )

        # 根据一系列sensors(config.lab_sensors)
        # 分别初始化对应的sensors
        # 在nav.py里的NavigationTask类中，
        # 比如 config.lab_sensors = ['objectgoal', 'gps', 'compass']
        # 就会初始化对应的sensors函数 (在object_nav_task.py, 和nav.py里)
        # 最后一个SensorSuite整合起来
        self.sensor_suite = SensorSuite(
            self._init_entities(
                entities_configs=config.lab_sensors,
                register_func=registry.get_sensor,
            ).values()
        )

        # 根据一系列actions(config.actions)
        # 分别初始化对应的actions
        # 在nav.py里的NavigationTask类中，
        # 比如 config.actions = ['move_forward', 'turn_left', 'turn_right']
        # 就会初始化对应的actions函数 (在nav.py里的NavigationTask类中)
        self.actions = self._init_entities(
            entities_configs=config.actions,
            register_func=registry.get_task_action,
        )
        self._action_keys = list(self.actions.keys())

        self._is_episode_active = False

    def add_perf_timing(self, *args, **kwargs):
        if hasattr(self._sim, "add_perf_timing"):
            self._sim.add_perf_timing(*args, **kwargs)

    def _init_entities(self, entities_configs, register_func) -> OrderedDict:
        """
        批量初始化
        - measurements, (Measure)
        - sensors, (Sensor)
        - actions (Action)

        """
        entities = OrderedDict()
        for entity_name, entity_cfg in entities_configs.items():
            entity_cfg = OmegaConf.create(entity_cfg)
            if "type" not in entity_cfg:
                raise ValueError(f"Could not find type in {entity_cfg}")
            entity_type = register_func(entity_cfg.type)
            assert (
                    entity_type is not None
            ), f"invalid {entity_name} type {entity_cfg.type}"
            entities[entity_name] = entity_type(
                sim=self._sim,
                config=entity_cfg,
                dataset=self._dataset,
                task=self,
                name=entity_name,
            )
        return entities

    def reset(self, episode: Episode):
        # 从simulator里面获取observations
        observations = self._sim.reset()

        if self._sim.habitat_config.num_agents == 1:
            observations.update(
                self.sensor_suite.get_observations(
                    observations=observations,
                    episode=episode,
                    task=self,
                    should_time=True,
                )
            )
        elif self._sim.habitat_config.num_agents > 1:
            [observations[i].update(
                self.sensor_suite.get_observations(
                    observations=observations[i],
                    episode=episode,
                    task=self,
                    agent_id=i,
                    should_time=True,
                )
            )
                for i in range(len(self._sim.habitat_config.agents))]

        # 重置所有的action
        # nav.py里的所有action
        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        self._is_episode_active = True

        return observations

    def _step_single_action(
            self,
            action_name: Any,
            action: Dict[str, Any],
            episode: Episode,
    ):
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
                action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."
        task_action = self.actions[action_name]
        return task_action.step(
            **action["action_args"],
            task=self,
        )

    def step(self, action: Dict[str, Any], episode: Episode):
        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer, List)):
            action = {"action": action}

        action_name = action["action"]
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        observations: Optional[Any] = None
        # if isinstance(action_name, tuple):  # there are multiple actions
        if isinstance(action_name, List):  # there are multiple actions
            # for a_name in action_name:
            #     observations = self._step_single_action(
            #         a_name,
            #         action,
            #         episode,
            #     )

            # if 0 in action_name:
            #     task_action = self.actions["stop"]
            #     task_action.step(**action["action_args"], task=self)
            #     obs = [self._sim.get_sensor_observations(agent_ids=i) for i in
            #            range(len(action_name))]
            # else:
            try:
                obs = self._sim.step(action_name)
            except Exception as e:
                print('action_name: ', action_name)
                print("Error in embodied_task: ", e)
                print('EmbodiedTask.py: step')

            [obs[i].update(
                self.sensor_suite.get_observations(
                    observations=obs[i],
                    episode=episode,
                    agent_id=i,
                    task=self,
                )
            ) for i in range(len(self._sim.habitat_config.agents))]

            if 0 in action_name:
                task_action = self.actions["stop"]
                task_action.step(**action["action_args"], task=self)

            # 分别检测两个agent是否active
            self._is_episode_active = self._check_episode_is_active(
                observations=obs[0], action=action, episode=episode
            ) and self._check_episode_is_active(
                observations=obs[1], action=action, episode=episode
            )

            # 直接返回obs 不进入下面
            return obs

        # else:
        #     observations = self._step_single_action(
        #         action_name, action, episode
        #     )

        # self._sim.step_physics(1.0 / self._physics_target_sps)  # type:ignore
        #
        # if observations is None:
        #     observations = self._sim.step(None)
        #
        # observations.update(
        #     self.sensor_suite.get_observations(
        #         observations=observations,
        #         episode=episode,
        #         action=action,
        #         task=self,
        #         should_time=True,
        #     )
        # )
        # self._is_episode_active = self._check_episode_is_active(
        #     observations=observations, action=action, episode=episode
        # )
        # return observations

    def get_action_name(self, action_index: Union[int, np.integer]):
        if action_index >= len(self.actions):
            raise ValueError(f"Action index '{action_index}' is out of range.")
        return self._action_keys[action_index]

    @property
    def action_space(self) -> Space:
        return ActionSpace(
            {
                action_name: action_instance.action_space
                for action_name, action_instance in self.actions.items()
            }
        )

    def overwrite_sim_config(
            self, sim_config: "DictConfig", episode: Episode
    ) -> "DictConfig":
        r"""Update config merging information from :p:`sim_config` and
        :p:`episode`.

        :param sim_config: config for simulator.
        :param episode: current episode.
        """
        return sim_config

    def _check_episode_is_active(
            self,
            *args: Any,
            action: Union[int, Dict[str, Any]],
            episode: Episode,
            **kwargs: Any,
    ) -> bool:
        raise NotImplementedError

    @property
    def is_episode_active(self):
        return self._is_episode_active

    def seed(self, seed: int) -> None:
        return
