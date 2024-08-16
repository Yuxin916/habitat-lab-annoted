#!/usr/bin/env python3
import time
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# VLM requirement
from torchvision.transforms import ToPILImage
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from habitat.utils.constant import category_to_id, parse_tensor_value

"""
Download Dataset:
    huggingface-cli download google/siglip-so400m-patch14-384 --local-dir spatial_bot_test/siglip --local-dir-use-symlinks False
My token:
    HUGGINGFACE_TOKEN=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
"""
# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

if TYPE_CHECKING:
    from omegaconf import DictConfig

try:
    import clip
except ImportError:
    clip = None

@baseline_registry.register_policy
class SpatialBotPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        backbone: str = "BunnyPhiForCausalLM",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Keyword arguments:
        rnn_type: RNN layer type; one of ["GRU", "LSTM"]
        backbone: Visual encoder backbone;
        one of ["BunnyPhiForCausalLM"]
        """
        assert backbone in ["BunnyPhiForCausalLM"], \
            f"{backbone} backbone is not recognized."

        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            ObjectNavSpatialNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names = [
            sensor.uuid
            for sensor in
            config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
        )

class ObjectNavSpatialNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        rnn_input_size = 0

        self.discrete_actions = discrete_actions
        # self._n_prev_action = 32
        # if discrete_actions:
        #     self.prev_action_embedding = nn.Embedding(
        #         action_space.n + 1, self._n_prev_action
        #     )
        # else:
        #     num_actions = get_num_actions(action_space)
        #     self.prev_action_embedding = nn.Linear(
        #         num_actions, self._n_prev_action
        #     )
        # # self._n_prev_action = 32
        # rnn_input_size += self._n_prev_action  # test

        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            # removing keys that correspond to goal sensors
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]
        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1
        ]
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )

        # if (
        #     IntegratedPointGoalGPSAndCompassSensor.cls_uuid
        #     in observation_space.spaces
        # ):
        #     n_input_goal = (
        #         observation_space.spaces[
        #             IntegratedPointGoalGPSAndCompassSensor.cls_uuid
        #         ].shape[0]
        #         + 1
        #     )
        #     self.tgt_embeding = nn.Linear(n_input_goal, 32)
        #     rnn_input_size += 32

        # if ObjectGoalSensor.cls_uuid in observation_space.spaces:
        #     self._n_object_categories = (
        #         int(
        #             observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
        #         )
        #         + 1
        #     )
        #     self.obj_categories_embedding = nn.Embedding(
        #         self._n_object_categories, 32
        #     )
        #     rnn_input_size += 32

        # if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
        #     input_gps_dim = observation_space.spaces[
        #         EpisodicGPSSensor.cls_uuid
        #     ].shape[0]
        #     self.gps_embedding = nn.Linear(input_gps_dim, 32)
        #     rnn_input_size += 32
        #
        # if PointGoalSensor.cls_uuid in observation_space.spaces:
        #     input_pointgoal_dim = observation_space.spaces[
        #         PointGoalSensor.cls_uuid
        #     ].shape[0]
        #     self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
        #     rnn_input_size += 32
        #
        # if HeadingSensor.cls_uuid in observation_space.spaces:
        #     input_heading_dim = (
        #         observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
        #     )
        #     assert input_heading_dim == 2, "Expected heading with 2D rotation."
        #     self.heading_embedding = nn.Linear(input_heading_dim, 32)
        #     rnn_input_size += 32
        #
        # if ProximitySensor.cls_uuid in observation_space.spaces:
        #     input_proximity_dim = observation_space.spaces[
        #         ProximitySensor.cls_uuid
        #     ].shape[0]
        #     self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
        #     rnn_input_size += 32
        #
        # if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
        #     assert (
        #         observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
        #             0
        #         ]
        #         == 1
        #     ), "Expected compass with 2D rotation."
        #     input_compass_dim = 2  # cos and sin of the angle
        #     self.compass_embedding = nn.Linear(input_compass_dim, 32)
        #     rnn_input_size += 32

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )

        if backbone.startswith("BunnyPhiForCausalLM"):
            with torch.no_grad():
                self.visual_encoder = SpatialVLMEncoder(
                    use_obs_space,
                )

            if not self.visual_encoder.is_blind:
                self.adapter = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        1458, hidden_size
                    ),
                    # nn.LeakyReLU(True),
                    nn.Linear(
                        hidden_size, hidden_size
                    ),
                    nn.LeakyReLU(True),
                )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                ObjectNavSpatialNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                # batch(n_env) x 1 x token_embedding
                visual_feats = observations[
                    ObjectNavSpatialNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.adapter(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # if len(self._fuse_keys_1d) != 0:
        #     fuse_states = torch.cat(
        #         [observations[k] for k in self._fuse_keys_1d], dim=-1
        #     )
        #     x.append(fuse_states.float())
        #
        # if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
        #     goal_observations = observations[
        #         IntegratedPointGoalGPSAndCompassSensor.cls_uuid
        #     ]
        #     if goal_observations.shape[1] == 2:
        #         # Polar Dimensionality 2
        #         # 2D polar transform
        #         goal_observations = torch.stack(
        #             [
        #                 goal_observations[:, 0],
        #                 torch.cos(-goal_observations[:, 1]),
        #                 torch.sin(-goal_observations[:, 1]),
        #             ],
        #             -1,
        #         )
        #     else:
        #         assert (
        #             goal_observations.shape[1] == 3
        #         ), "Unsupported dimensionality"
        #         vertical_angle_sin = torch.sin(goal_observations[:, 2])
        #         # Polar Dimensionality 3
        #         # 3D Polar transformation
        #         goal_observations = torch.stack(
        #             [
        #                 goal_observations[:, 0],
        #                 torch.cos(-goal_observations[:, 1])
        #                 * vertical_angle_sin,
        #                 torch.sin(-goal_observations[:, 1])
        #                 * vertical_angle_sin,
        #                 torch.cos(goal_observations[:, 2]),
        #             ],
        #             -1,
        #         )
        #
        #     x.append(self.tgt_embeding(goal_observations))
        #
        # if PointGoalSensor.cls_uuid in observations:
        #     goal_observations = observations[PointGoalSensor.cls_uuid]
        #     x.append(self.pointgoal_embedding(goal_observations))
        #
        # if ProximitySensor.cls_uuid in observations:
        #     sensor_observations = observations[ProximitySensor.cls_uuid]
        #     x.append(self.proximity_embedding(sensor_observations))
        #
        # if HeadingSensor.cls_uuid in observations:
        #     sensor_observations = observations[HeadingSensor.cls_uuid]
        #     sensor_observations = torch.stack(
        #         [
        #             torch.cos(sensor_observations[0]),
        #             torch.sin(sensor_observations[0]),
        #         ],
        #         -1,
        #     )
        #     x.append(self.heading_embedding(sensor_observations))
        #
        # if ObjectGoalSensor.cls_uuid in observations:
        #     object_goal = observations[ObjectGoalSensor.cls_uuid].long()
        #     x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))
        #
        # if EpisodicCompassSensor.cls_uuid in observations:
        #     compass_observations = torch.stack(
        #         [
        #             torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
        #             torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
        #         ],
        #         -1,
        #     )
        #     x.append(
        #         self.compass_embedding(compass_observations.squeeze(dim=1))
        #     )
        #
        # if EpisodicGPSSensor.cls_uuid in observations:
        #     x.append(
        #         self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
        #     )
        #
        # for uuid in [
        #     ImageGoalSensor.cls_uuid,
        #     InstanceImageGoalSensor.cls_uuid,
        # ]:
        #     if uuid in observations:
        #         goal_image = observations[uuid]
        #
        #         goal_visual_encoder = getattr(self, f"{uuid}_encoder")
        #         goal_visual_output = goal_visual_encoder({"rgb": goal_image})
        #
        #         goal_visual_fc = getattr(self, f"{uuid}_fc")
        #         x.append(goal_visual_fc(goal_visual_output))
        #
        # if self.discrete_actions:
        #     prev_actions = prev_actions.squeeze(-1)
        #     start_token = torch.zeros_like(prev_actions)
        #     # The mask means the previous action will be zero, an extra dummy action
        #     prev_actions = self.prev_action_embedding(
        #         torch.where(masks.view(-1), prev_actions + 1, start_token)
        #     )
        # else:
        #     prev_actions = self.prev_action_embedding(
        #         masks * prev_actions.float()
        #     )
        #
        # x.append(prev_actions)

        try:
            out = torch.cat(x, dim=1)
        except Exception as e:
            print(f"Error: {e}")
            # Handle the error by inspecting tensor shapes
            for i, tensor in enumerate(x):
                print(f"Tensor {i} shape: {tensor.shape}")
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state

class SpatialVLMEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        model_name: str = 'spatial_bot_test/',
    ):
        super().__init__()

        # get current working directory
        working_dir = os.path.dirname(os.getcwd())
        model_name = os.path.join(working_dir, model_name)

        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]
        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if not self.is_blind:
            # load vlm model
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16, # float32 for cpu
                device_map='auto',
                trust_remote_code=True).to(torch.float16).eval()
            self.delete_lm_head = self.backbone.lm_head
            self.backbone.lm_head = nn.Sequential()
            self.backbone.get_vision_tower().to('cuda')
            self.backbone.model.vision_tower = self.backbone.model.vision_tower.to('cuda')

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True)

            self.output_shape = (
                1, # number of tokens  # TODO
                729*2  # token embedding dimension
            )

            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.model.parameters():
                param.requires_grad = False
            for param in self.backbone.model.vision_tower.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

            self.offset_bos = 0

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def pre_process_image(self, image):
        """
        normalize a batch of RGB / Depth images such that each image in the batch
        - Calculate the minimum and maximum values for each RGB image in the batch.
        - Normalize each image in the batch individually to have values between 0 and 1.
        - Rescale the normalized images to the range [0, 255] and convert to uint8.
        """
        # batch(n_env) x channel x height x width
        # Get the dimensions
        B, C, H, W = image.size()

        # Calculate the min and max value for each image in the batch
        min_vals = image.view(B, C, -1).min(dim=-1, keepdim=True)[0]
        max_vals = image.view(B, C, -1).max(dim=-1, keepdim=True)[0]

        # Normalize each image in the batch to the range [0, 1]
        image_normalized = (image - min_vals.view(B, C, 1, 1)) / (
            max_vals.view(B, C, 1, 1) - min_vals.view(B, C, 1, 1))

        # Rescale to [0, 255] and convert to uint8
        image_rescaled = (image_normalized * 255).clamp(0, 255).to(
            torch.uint8)

        if C == 1:
            # Repeat the single-channel depth image to make it a three-channel image
            image_rescaled = image_rescaled.repeat(1, 3, 1, 1)

        # # Verify the normalization by checking the new max value (should be 255)
        # new_max_vals = \
        #     image_rescaled.view(B, C, -1).max(dim=-1, keepdim=True)[0]
        # print("New max values per batch:", new_max_vals)

        pil_image = [ToPILImage()(img) for img in image_rescaled]

        return pil_image

    def form_prompt(self, observations):
        goal_name = observations['objectgoal']
        str_gps = parse_tensor_value(observations['gps'])
        str_compass = parse_tensor_value(observations['compass'])
        robot_position = f"GPS: {str_gps}  Compass: {str_compass}"

        prompt_template = """
        Objective:
        Control a robot equipped with RGB-D sensors to locate a specific target object

        Representation Details:
        - RGB-D images are provided as input
        - Robot Position: GPS and Compass relative to the starting position

        Strategy:
        Take note all detected objects and their spatial relationships,
        including proximity and layout, based on depth information.
        Commonsense reasoning about the target object and detected objects.

        [input:]
        Task: Locate the {GOAL_NAME}

        Position:
        {ROBOT_POSITION}
        """
        prompt = prompt_template.format(GOAL_NAME=category_to_id[goal_name[0]],
                                        ROBOT_POSITION=robot_position)
        return prompt

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore

        if self.is_blind:
            return None

        # cnn_input = []
        # for k in self.visual_keys:
        #     obs_k = observations[k]
        #     # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        #     obs_k = obs_k.permute(0, 3, 1, 2)
        #     if self.key_needs_rescaling[k] is not None:
        #         obs_k = (
        #             obs_k.float() * self.key_needs_rescaling[k]
        #         )  # normalize
        #     cnn_input.append(obs_k)

        # text prompt
        # prompt = (
        #     "Ignore walls, floors, ceilings, and windows. "
        #     "List all detected objects and describe their spatial relationships, "
        #     "including proximity and layout, based on depth information. "
        #     "Focus on the 2D layout of the objects, considering their relative positions and distances to "
        #     "support navigation tasks. Extract features that reflect these 3D spatial relationships."
        # )
        # batched_input_id = []
        # for k in range(len(observations['gps'])):
        #     prompt = self.form_prompt(observations[k])
        #
        #     text = (f"A chat between a curious user and an artificial intelligence assistant. "
        #             f"The assistant gives helpful, detailed, "
        #             f"and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:")
        #     text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        #     input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][self.offset_bos:], dtype=torch.long).unsqueeze(0).to('cuda')
        #     batched_input_id.append(input_ids)

        # batch(n_env) x channel x height x width
        rgb_image = observations['rgb'].permute(0, 3, 1, 2)
        depth_image = observations['depth'].permute(0, 3, 1, 2)
        n_env = rgb_image.size(0)

        # self.backbone.get_vision_tower().to('cuda')
        # self.backbone.eval()
        # Pre-process the images
        rgb_pil_images = self.pre_process_image(rgb_image)
        depth_pil_images = self.pre_process_image(depth_image)

        # (rgb_batch(n_env) + rgb_batch(n_env)) x channel x height x width
        image_tensor = self.backbone.process_images(rgb_pil_images + depth_pil_images,
                                               self.backbone.config).to(dtype=self.backbone.dtype, device='cuda')
        self.backbone.get_vision_tower().to('cuda')
        self.backbone.eval()

        # batch(n_env) x channel x height x width
        rgb_image_tensor = image_tensor[:n_env, :, :, :]
        depth_image_tensor = image_tensor[n_env:, :, :, :]

        # if True:
        #     self.debug_image_tensor(image_tensor)

        output = torch.zeros(n_env, 2, 729, device='cuda')

        with torch.no_grad():
            start = time.time()
            for i in range(n_env):
                # concatenate the rgb and depth images
                image_tensor = torch.cat([rgb_image_tensor[i].unsqueeze(0),
                                          depth_image_tensor[i].unsqueeze(0)], dim=0)
                # output_ids = self.backbone(
                #     batched_input_id[i],
                #     images=image_tensor,  # 2 x 3 x 384 x 384
                #     # max_new_tokens=250,
                #     use_cache=False,
                #     # repetition_penalty=1.0 # increase this to avoid chattering
                # )[0]
                output_image = self.backbone.get_vision_tower()(image_tensor)

                # x = self.delete_lm_head(output_ids)

                # # Max pooling along the token dimension (dim=1)
                x = F.adaptive_max_pool1d(output_image,
                                            output_size=1).squeeze()

                # or
                # x = output_ids[:, -1, :].unsqueeze(0)

                output[i] = x

            # logging.info(f"Time taken for VLM forward pass: {time.time() - start:.2f}s for {n_env} environments")
        output = output.view(n_env, 1, -1)

        return output

    def debug_image_tensor(self, image_tensor):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        # Iterate over 4 images
        for i in range(4):
            test_image = image_tensor.to('cpu').to(torch.float32)
            img = test_image[i].permute(1, 2, 0).numpy()
            axes[i // 2, i % 2].imshow(img)
            axes[i // 2, i % 2].axis('off')
        plt.tight_layout()

        plt.show()
