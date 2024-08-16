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
from torchvision import transforms

# VLM requirement
from torchvision.transforms import ToPILImage
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from collections import Counter

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

IMAGE_TOKEN_INDEX = [-201, -202]
IGNORE_INDEX = -100

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
        # previous action embedding
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test
        self.discrete_actions = discrete_actions

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

        # Add goal sensor embeddings
        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        # GPS sensor embeddings
        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        # Compass sensor embeddings
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

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
                self.test = nn.Sequential(
                    # (1, 1, 2560)
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=128,
                        kernel_size=(1, 6),

                    ),
                    nn.ReLU(),
                    # (128, 1458, 1)
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=(10, 1),
                        stride=(10, 1)
                    ),
                    # (256, 145, 1)
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=(10, 1),
                        stride=(10, 1)
                    ),
                    # (512, 14, 1)
                    nn.ReLU(),
                )

                self.adapter = nn.Sequential(
                    nn.Linear(
                        2560, hidden_size
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_size, hidden_size
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_size, hidden_size
                    ),
                    nn.ReLU(),
                )
        else:
            raise ValueError(f"Invalid backbone: {backbone}")

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

            # visual_feats = self.test(visual_feats.unsqueeze(1)).view(
            #     visual_feats.size(0), -1)
            visual_feats = self.adapter(visual_feats.squeeze(1))
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observations:
                goal_image = observations[uuid]

                goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                goal_visual_output = goal_visual_encoder({"rgb": goal_image})

                goal_visual_fc = getattr(self, f"{uuid}_fc")
                x.append(goal_visual_fc(goal_visual_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

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
        # ['rgb', 'depth']
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]

        # Count total # of channels for all visual observations
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if not self.is_blind:
            # load vlm model
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,  # path to huggingface download
                torch_dtype=torch.float16,  # float32 for cpu
                device_map='auto',
                trust_remote_code=True).to(torch.float16).eval()
            self.vision_tower = self.backbone.get_vision_tower()
            if not self.vision_tower.is_loaded:
                self.vision_tower.load_model()
            # get self.backbone parameter size
            self.backbone_size = sum(p.numel() for p in self.backbone.parameters())

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True)

            # for batch inference
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = self.backbone.generation_config.pad_token_id
            padding_max_length = 128  # customize for your circumstance
            self.tokenizer.add_tokens(['<image>'])
            image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')

            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.lm_head = nn.Sequential()
            self.backbone.prepare_inputs_labels_for_multimodal = override.__get__(self.backbone)

            self.output_shape = (
                1,
                self.backbone.config.hidden_size
            )
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

        # transform to RGBA model PIL image list
        to_pil = transforms.ToPILImage()
        pil_images = [
            to_pil(image_rescaled[i]).convert('RGBA') for i in range(B)
        ]

        # Optional: Display or save the images
        # for i, image in enumerate(pil_images):
        #     image.show()  # Display the image

        return pil_images

    def form_prompt(self, observations):
        goal_name = observations['objectgoal']
        str_gps = parse_tensor_value(observations['gps'])
        str_compass = parse_tensor_value(observations['compass'])
        robot_position = f"GPS: {str_gps}  Compass: {str_compass}"

        prompt_template = """
        Objective:
        Control a robot equipped with RGB-D sensors to locate {GOAL_NAME}

        Strategy:
        Take note all detected objects and their spatial relationships,
        including proximity and layout, based on depth information.
        Commonsense reasoning about the relationship between target object and detected objects.

        Task: Locate the {GOAL_NAME}
        Position: You are at {ROBOT_POSITION}
        """
        prompt = prompt_template.format(GOAL_NAME=category_to_id[goal_name[0]],
                                        ROBOT_POSITION=robot_position)
        return prompt

    def visualize_tensor_preprocess(self, image_tensor):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        image_tensor = image_tensor.to(torch.float32)
        for i in range(3):  # Loop over the 3 environments
            for j in range(2):  # Loop over the 2 images per environment
                # Select the corresponding image and transpose it to (384, 384, 3) for visualization
                image = image_tensor[i, j].permute(1, 2, 0).cpu().numpy()
                axs[i, j].imshow(image)
                axs[i, j].axis('off')  # Turn off axis labels

        plt.tight_layout()
        plt.show()

    def forward(self, observations: Dict[
        str, torch.Tensor]) -> torch.Tensor:  # type: ignore

        if self.is_blind:
            return None

        # text prompt
        batched_input_id = []
        for k in range(len(observations['gps'])):
            prompt = self.form_prompt(observations[k])

            text = (f"A chat between a curious user and an artificial intelligence assistant. "
                    f"The assistant gives helpful, detailed, "
                    f"and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:")
            text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
            input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][self.offset_bos:], dtype=torch.long).unsqueeze(0).to('cuda')
            batched_input_id.append(input_ids)

        max_length = max(ids.shape[-1] for ids in batched_input_id)
        padded_input_ids_batch = torch.stack([
            torch.cat([ids, torch.full((1, max_length - ids.shape[-1]),
                                       self.tokenizer.pad_token_id,
                                       dtype=torch.long,
                                       device=ids.device
                                       )], dim=-1)
            for ids in batched_input_id
        ]).to('cuda').squeeze(1)

        # batch(n_env) x channel x height x width
        rgb_image = observations['rgb'].permute(0, 3, 1, 2)
        depth_image = observations['depth'].permute(0, 3, 1, 2)
        n_env = rgb_image.size(0)

        # Pre-process the images
        rgb_pil_images = self.pre_process_image(rgb_image)
        depth_pil_images = self.pre_process_image(depth_image)

        # (rgb_batch(n_env) + rgb_batch(n_env)) x channel x height x width
        # first time need to load the visual encoder weights
        processed_images = self.backbone.process_images(
            # element by element concatenation
            [val for pair in zip(rgb_pil_images, depth_pil_images) for val in pair],
            self.backbone.config).to(dtype=self.backbone.dtype, device='cuda').view(n_env, 2, 3, 384, 384)
        self.backbone.get_vision_tower().to('cuda')

        # visualize the pre-processed images
        # self.visualize_tensor_preprocess(processed_images)

        with torch.no_grad():
            # image_all = torch.concatenate([rgb_image_tensor,
            #                                depth_image_tensor
            #                                ], dim=0)

            # 2, 1590, 2560
            output_ids = self.backbone(
                        padded_input_ids_batch, # n_env x input_length
                        images=processed_images,  # n_env x 2 x 3 x 384 x 384
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True
                    ).hidden_states[-1]


            pooled_output = torch.max(output_ids, dim=1).values.unsqueeze(1)

        return pooled_output

    def debug_image_tensor(self, image_tensor):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        # Iterate over 4 images
        for i in range(6):
            test_image = image_tensor.to('cpu').to(torch.float32)
            img = test_image[i].permute(1, 2, 0).numpy()
            axes[i // 2, i % 2].imshow(img)
            axes[i // 2, i % 2].axis('off')
        plt.tight_layout()

        plt.show()


def override(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images
):
    # input_ids: 2 x 128
    # position_ids: 2 x 128
    # attention_mask: 2 x 128
    # images: 2 x 2 x 3 x 384 x 384
    # rest all are None
    vision_tower = self.get_vision_tower()

    if images.ndim == 5:
        # n_env x 2 x 3 x 384 x 384 -> 4 x 3 x 384 x 384
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        # list of n_env, each one's embedding is 2x729x 2560 (RGB Embedding and Depth Embedding)
        image_features = torch.split(image_features, split_sizes, dim=0)
        # list of n_env, each one's embedding is 1458 x 2560 (RGB Embedding and Depth Embedding)
        # image_features = [x.to(self.device) for x in image_features]
    else:
        image_features = self.encode_images(images).to(self.device)

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        # Tensor n_env x 128
        attention_mask = attention_mask.bool()
    if position_ids is None:
        # Tensor n_env x 128
        position_ids = torch.arange(0, input_ids.shape[1],
                                    dtype=torch.long,
                                    device=input_ids.device)
    if labels is None:
        # Tensor n_env x 128
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # input_ids_temp = input_ids  # points to the actual input_ids tensor

    # remove the padding using attention_mask
    # List of n_env, each one is 128,
    input_ids = [cur_input_ids[cur_attention_mask] for
                 cur_input_ids, cur_attention_mask in
                 zip(input_ids, attention_mask)]
    # List of n_env, each one is 128,
    labels = [cur_labels[cur_attention_mask] for
              cur_labels, cur_attention_mask in
              zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0

    # iterate over this one element list
    for batch_idx, cur_input_ids in enumerate(input_ids):
        # List 128
        input_ids_list = cur_input_ids.tolist()
        # Counter 34
        input_ids_counter = Counter(input_ids_list)
        # 2 images
        num_images = sum(
            input_ids_counter[element] for element in IMAGE_TOKEN_INDEX)

        common_elements_positions = [index for index, value in
                                     enumerate(input_ids_list) if
                                     value in set(IMAGE_TOKEN_INDEX)]
        image_token_indices = [-1] + common_elements_positions + [
            cur_input_ids.shape[0]]

        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[
                                      image_token_indices[i] + 1:
                                      image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:
                                              image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(
            torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes,
                                             dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[batch_idx][
                    cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full((cur_image_features.shape[0],),
                               IGNORE_INDEX, device=cur_labels.device,
                               dtype=cur_labels.dtype))

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
        cur_image_idx = 0

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config,
                                         'tokenizer_model_max_length',
                                         None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in
                            new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX,
                                   dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len),
                               dtype=position_ids.dtype,
                               device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(
        zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side',
                   'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len,
                                                          dtype=position_ids.dtype,
                                                          device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len,
                                                         dtype=position_ids.dtype,
                                                         device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
