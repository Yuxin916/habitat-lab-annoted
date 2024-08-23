#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import sys
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

"""
train
--config-name=objectnav/ddppo_objectnav_hm3d.yaml habitat_baselines.trainer_name=ver habitat_baselines.evaluate=False

eval
--config-name=objectnav/ddppo_objectnav_hm3d.yaml habitat_baselines.trainer_name=ver habitat_baselines.num_environments=1 habitat_baselines.evaluate=True

add HABITAT_ENV_DEBUG=1 to environment variables to enable debug logging
HABITAT_ENV_DEBUG=1;GLOG_minloglevel=2;MAGNUM_LOG=quiet;HABITAT_SIM_LOG=quiet;HF_HOME=/home/tsaisplus/mrs_llm/myproject/spatial_bot_test
"""

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg: "DictConfig"):
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

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
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
