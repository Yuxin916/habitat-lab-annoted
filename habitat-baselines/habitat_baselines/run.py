#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import sys
from typing import TYPE_CHECKING
from notes_data.utils.get_config import register_plugins
from notes_il_train.utils.get_config import register_plugins_baseline

import os
import hydra
from omegaconf import open_dict
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch

from habitat.config.read_write import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

"""
Working Directory:
    habitat-lab/

Command:
    train
        --config-name="mp3d_few_filtered_il_baseline_single_node.yaml"
        --config-path="/home/tsaisplus/mrs_llm/vis_nav_v2/notes_il_train/configs/"
        habitat_baselines.evaluate=False
    eval

Environment Variables:
    HABITAT_ENV_DEBUG=1;GLOG_minloglevel=2;MAGNUM_LOG=quiet;HABITAT_SIM_LOG=quiet;
"""

def patch_exp_name(cfg, hydra_cfg, exp_name):
    with read_write(hydra_cfg):
        hydra_cfg.run.dir = hydra_cfg.run.dir.format(exp_name=exp_name)
        hydra_cfg.job.name = hydra_cfg.job.name.format(exp_name=exp_name)
        hydra_cfg.runtime.output_dir = hydra_cfg.runtime.output_dir.format(exp_name=exp_name)

    with read_write(cfg):
        cfg.habitat_baselines.tensorboard_dir = cfg.habitat_baselines.tensorboard_dir.format(exp_name=exp_name)
        cfg.habitat_baselines.video_dir = cfg.habitat_baselines.video_dir.format(exp_name=exp_name)
        cfg.habitat_baselines.checkpoint_folder = cfg.habitat_baselines.checkpoint_folder.format(exp_name=exp_name)
        cfg.habitat_baselines.log_file = cfg.habitat_baselines.log_file.format(exp_name=exp_name)

    # Evaluation phase
    if cfg.habitat_baselines.evaluate:
        with read_write(cfg):
            cfg.habitat_baselines.eval_ckpt_path_dir = cfg.habitat_baselines.eval_ckpt_path_dir.format(
                exp_name=exp_name,
                no=cfg.habitat_baselines.ckpt_no
            )
    return cfg

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg: "DictConfig"):

    # get the config name you read and mark it as the experiment name for logging
    hydra_cfg = HydraConfig.get()
    config_name_with_ext = hydra_cfg.job.config_name
    exp_name = os.path.splitext(config_name_with_ext)[0]

    # insert the experiment name into the config
    cfg = patch_exp_name(cfg, hydra_cfg, exp_name)

    cfg = patch_config(cfg)

    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


def execute_exp(config: "DictConfig", run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    from habitat_baselines.common.baseline_registry import baseline_registry

    # get registered trainer
    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"

    # initialize trainer
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    # register my custom plugin to hydra (habitat)
    register_plugins()
    # regiser my custom plugin to hydra (habitat-baselines) BC
    register_plugins_baseline()
    if "--exp-config" in sys.argv or "--run-type" in sys.argv:
        raise ValueError(
            "The API of run.py has changed to be compatible with hydra.\n"
            "--exp-config is now --config-name and is a config path inside habitat-baselines/habitat_baselines/config/. \n"
            "--run-type train is replaced with habitat_baselines.evaluate=False (default) and --run-type eval is replaced with habitat_baselines.evaluate=True.\n"
            "instead of calling:\n\n"
            "python -u -m habitat_baselines.run --exp-config habitat-baselines/habitat_baselines/config/<path-to-config> --run-type train/eval\n\n"
            "You now need to do:\n\n"
            "python -u -m habitat_baselines.run --config-name=<path-to-config> habitat_baselines.evaluate=False/True\n"
        )
    main()
