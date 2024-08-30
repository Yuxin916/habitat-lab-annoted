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
import argparse
"""
Download Dataset:
    huggingface-cli download google/siglip-so400m-patch14-384 --local-dir hf_spatialbot/siglip --local-dir-use-symlinks False
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

import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    # ===================================================================
    multi_image_tower: Optional[str] = field(default=None)
    # ===================================================================

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    # ================================================
    tokenizer_model_max_length: Optional[int] = None
    mul_image_enable: bool = False
    # ================================================



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
        prompt: Optional[str] = None,
        visualize_prompt: bool = False,
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
                prompt=prompt,
                visualize_prompt=visualize_prompt,
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
            prompt = config.habitat_baselines.prompt,
            visualize_prompt = config.habitat_baselines.visualize_prompt,
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
        prompt: Optional[str] = None,
        visualize_prompt: bool = False,
    ):
        super().__init__()
        rnn_input_size = 0
        self.discrete_actions = discrete_actions
        self._hidden_size = hidden_size

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
        rnn_input_size += self._n_prev_action

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
                    prompt,
                    visualize_prompt,
                )

            if not self.visual_encoder.is_blind:
                self.adapter = nn.Sequential(
                    nn.Linear(
                        self.visual_encoder.backbone.config.hidden_size,
                        hidden_size
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
                ).to(torch.float16)
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
            visual_feats = self.adapter(visual_feats.squeeze(1).to(torch.float16))
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # Object goal
        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        # Compass
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

        # GPS
        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )


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

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state

class SpatialVLMEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        prompt: Optional[str] = None,
        visualize_prompt: bool = False,
        model_name: str = 'hf_spatialbot/',

    ):
        super().__init__()

        def parse_arguments():
            parser = argparse.ArgumentParser(
                description="Training script for fine-tuning models.")

            # Adding all the options with their default values
            parser.add_argument('--lora_enable', type=bool, default=True,
                                help='Enable LoRA')
            parser.add_argument('--lora_r', type=int, default=128,
                                help='Rank of LoRA')
            parser.add_argument('--lora_alpha', type=int, default=256,
                                help='Scale of LoRA')
            parser.add_argument('--mm_projector_lr', type=float, default=2e-5,
                                help='Learning rate for multimodal projector')
            parser.add_argument('--deepspeed',
                                default='./script/deepspeed/zero3.json',
                                help='Deepspeed configuration file')
            parser.add_argument('--model_name_or_path',
                                default='../merged_model',
                                help='Path to the pretrained model')
            parser.add_argument('--model_type', default='default_model_type',
                                help='Type of model')
            parser.add_argument('--version', default='bunny',
                                help='Version tag')
            parser.add_argument('--data_path',
                                default='./data/finetune/SpatialQA.json',
                                help='Data path for training')
            parser.add_argument('--image_folder',
                                default='./data/finetune/images',
                                help='Folder containing training images')
            parser.add_argument('--vision_tower',
                                default='../hf_spatialbot/siglip',
                                help='Vision tower setting')
            parser.add_argument('--mm_projector_type', default='mlp2x_gelu',
                                help='Type of multimodal projector')
            parser.add_argument('--image_aspect_ratio', default='pad',
                                help='Image aspect ratio processing')
            parser.add_argument('--group_by_modality_length', type=bool,
                                default=False,
                                help='Group by modality length flag')
            parser.add_argument('--bf16', type=bool, default=True,
                                help='Use bf16 precision')
            parser.add_argument('--output_dir',
                                default='./checkpoints-default_model_type/default_output_dir',
                                help='Output directory for checkpoints and logs')
            parser.add_argument('--num_train_epochs', type=int, default=1,
                                help='Number of training epochs')
            parser.add_argument('--per_device_train_batch_size', type=int,
                                default=8,
                                help='Training batch size per device')
            parser.add_argument('--per_device_eval_batch_size', type=int,
                                default=4,
                                help='Evaluation batch size per device')
            parser.add_argument('--gradient_accumulation_steps', type=int,
                                default=2,
                                help='Number of gradient accumulation steps')
            parser.add_argument('--evaluation_strategy', default='no',
                                help='Evaluation strategy')
            parser.add_argument('--save_strategy', default='steps',
                                help='Checkpoint save strategy')
            parser.add_argument('--save_steps', type=int, default=500,
                                help='Steps between saves')
            parser.add_argument('--save_total_limit', type=int, default=1,
                                help='Maximum number of checkpoints to keep')
            parser.add_argument('--learning_rate', type=float, default=2e-4,
                                help='Learning rate')
            parser.add_argument('--weight_decay', type=float, default=0.0,
                                help='Weight decay')
            parser.add_argument('--warmup_ratio', type=float, default=0.03,
                                help='Warmup ratio')
            parser.add_argument('--lr_scheduler_type', default='cosine',
                                help='Type of learning rate scheduler')
            parser.add_argument('--logging_steps', type=int, default=1,
                                help='Number of logging steps')
            parser.add_argument('--tf32', type=bool, default=True,
                                help='Use TensorFlow 32 precision')
            parser.add_argument('--model_max_length', type=int, default=2048,
                                help='Maximum model input length')
            parser.add_argument('--gradient_checkpointing', type=bool,
                                default=True,
                                help='Enable gradient checkpointing')
            parser.add_argument('--dataloader_num_workers', type=int,
                                default=4,
                                help='Number of workers for data loading')
            parser.add_argument('--lazy_preprocess', type=bool, default=True,
                                help='Enable lazy preprocessing')
            parser.add_argument('--report_to', default='none',
                                help='Reporting configuration')

            args = parser.parse_args()
            return args

        parser = parse_arguments()

        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        local_rank = training_args.local_rank
        compute_dtype = (torch.float16 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32))
        bnb_model_from_pretrained_args = {}
        assert model_args.vision_tower is not None

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
        if tokenizer.unk_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **bnb_model_from_pretrained_args
        )

        self.prompt = prompt
        self.visualize_prompt = visualize_prompt
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
                trust_remote_code=True).to(torch.float16).eval()
            # load vision tower weights
            self.vision_tower = self.backbone.get_vision_tower().to(self.backbone.device)
            if not self.vision_tower.is_loaded:
                self.vision_tower.load_model()
            self.vision_tower = self.backbone.get_vision_tower().to(self.backbone.device)

            # Override the function in the backbone model
            self.backbone.prepare_inputs_labels_for_multimodal = override.__get__(
                self.backbone)

            # get self.backbone parameter size
            self.backbone_size = sum(p.numel() for p in self.backbone.parameters())
            # logging.info(f"Backbone size: {self.backbone_size}")

            # check each layer's device
            # for name, param in self.backbone.named_parameters():
            #     logging.info(f"Parameter: {name} is on device: {param.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # for batch inference
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = self.backbone.generation_config.pad_token_id

            for param in self.backbone.parameters():
                param.requires_grad = False
            self.deleted_lm_head = self.backbone.lm_head
            self.backbone.lm_head = nn.Sequential()

            self.output_shape = (
                1,
                self.backbone.config.hidden_size
            )
            self.offset_bos = 0

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def pre_process_rgb_image(self, image):
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

    def pre_process_depth_image(self, depth_image):
        # Assuming depth_image is a numpy array with shape (batch_size, 1, 480, 640)
        batch_size, C, height, width = depth_image.shape

        # Reshape to (batch_size, 480, 640) for easier manipulation
        depth_image = depth_image.view(batch_size, height, width)

        # Normalize depth values to [0, 1]
        min_depth = depth_image.min()
        max_depth = depth_image.max()
        normalized_depth = (depth_image - min_depth) / (max_depth - min_depth)

        # Initialize a list to hold the PIL images
        pil_images = []

        # Loop through each tensor in the batch
        for i in range(batch_size):
            # Rescaling to uint16 range
            depth_image = (
                    normalized_depth[i] * 65535).short()  # Convert to int16 to fit into uint16 range
            assert depth_image.ndim == 2, "Depth must be 2D"

            # Convert to numpy
            depth_array = depth_image.cpu().numpy().astype('uint16')

            # Create PIL Image
            pil_image = Image.fromarray(depth_array,
                                        mode='I;16')  # 'I;16' mode for 16-bit grayscale images

            assert len(pil_image.getbands()) == 1, "Depth image must be single-channel"

            # To visualize the depth image, you can use the following code:
            # import matplotlib.pyplot as plt
            # import matplotlib.colors as mcolors
            # # Convert PIL Image to numpy array
            # depth_array = np.array(pil_image)
            # # Normalize the image to the range [0, 1]
            # normalized_image = depth_array / np.max(depth_array)
            # # Display the image with a colormap
            # plt.figure(figsize=(10, 5))
            # plt.imshow(normalized_image,
            #            cmap='viridis')  # 'viridis' is a good colormap for depth data
            # plt.colorbar(label='Normalized Depth')
            # plt.title('Depth Image Visualization')
            # plt.axis('off')  # Turn off axis numbers and ticks
            # plt.show()

            if len(pil_image.getbands()) == 1:
                img = np.array(pil_image)
                height, width = img.shape
                three_channel_array = np.zeros((height, width, 3),
                                               dtype=np.uint8)
                three_channel_array[:, :, 0] = (img // 1024) * 4
                three_channel_array[:, :, 1] = (img // 32) * 8
                three_channel_array[:, :, 2] = (img % 32) * 8
                image2 = Image.fromarray(three_channel_array, 'RGB')

            # Append the PIL image to the list
            pil_images.append(image2)

        return pil_images

    def form_prompt(self, observations):
        goal_name = observations['objectgoal']
        str_gps = parse_tensor_value(observations['gps'])
        str_compass = parse_tensor_value(observations['compass'])
        robot_position = f"GPS: {str_gps}  Compass: {str_compass}"

        prompt_template = self.prompt
        prompt = prompt_template.format(GOAL_NAME=category_to_id[goal_name[0]],
                                        ROBOT_POSITION=robot_position)
        return prompt

    def visualize_tensor_preprocess(self, image_tensor):
        import matplotlib.pyplot as plt

        # first and second axis concatenation
        image_tensor = image_tensor.view(-1, 3, 384, 384)

        # Ensure the tensor is on the CPU and in float32 for visualization
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()

        # Convert to float32 if it's in float16
        if image_tensor.dtype == torch.float16:
            image_tensor = image_tensor.to(torch.float32)

        # Normalize the tensor to the [0, 1] range for visualization (optional)
        # This step is often necessary if your tensor has values outside the [0, 1] range
        image_tensor = (image_tensor - image_tensor.min()) / (
                image_tensor.max() - image_tensor.min())

        batch_size, channels, height, width = image_tensor.shape

        # If the tensor has more than one image, visualize them one by one
        for i in range(batch_size):
            img = image_tensor[i]  # Select the i-th image

            if channels == 1:
                # If the image is grayscale, remove the channel dimension
                img = img.squeeze(0)
                plt.imshow(img, cmap='gray')
            else:
                # Transpose to [H, W, C] for RGB display
                img = img.permute(1, 2, 0)
                plt.imshow(img)

            plt.axis('off')  # Hide axis labels
            plt.show()

    def prepare_input_ids(self, observations):
        batch_size = observations['gps'].shape[0]
        # text prompt
        # Formulate text prompts for all observations at once
        self.prompts = [
            self.form_prompt(
                {key: value[i] for key, value in observations.items()})
            for i in range(batch_size)
        ]
        # logging.info('Prompts:' + str(prompts))

        texts = [
            (
                f"A chat between a curious user and an artificial intelligence assistant. "
                f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
            )
            for prompt in self.prompts
        ]

        # Tokenize the texts and create input_ids tensors
        tokenized_chunks = [
            [self.tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
            for text in texts
        ]

        # Combine tokenized chunks with special tokens and create input tensors
        input_ids_list = [
            torch.tensor(chunk[0] + [-201] + [-202] + chunk[1][self.offset_bos:], dtype=torch.long).unsqueeze(0).to(self.backbone.device)
            for chunk in tokenized_chunks
        ]

        # Find the maximum length for padding
        max_length = max(ids.shape[-1] for ids in input_ids_list)

        # Pad the sequences to the maximum length and stack into a batch
        padded_input_ids_batch = torch.stack([
            torch.cat([ids, torch.full((1, max_length - ids.shape[-1]),
                                       self.tokenizer.pad_token_id,
                                       dtype=torch.long, device=ids.device)], dim=-1)
            for ids in input_ids_list
        ]).squeeze(1).to(self.backbone.device)

        return padded_input_ids_batch

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore

        self.backbone.eval()
        # get current rank
        # logging.info(f"vision tower device at the begining of forward loop: {str(self.backbone.get_vision_tower().device)}")

        if self.is_blind:
            return None

        # batch(n_env) x input_length
        padded_input_ids_batch = self.prepare_input_ids(observations)

        # batch(n_env) x channel x height x width
        rgb_image = observations['rgb'].permute(0, 3, 1, 2)
        depth_image = observations['depth'].permute(0, 3, 1, 2)

        # batch_size
        n_env = rgb_image.size(0)

        # Pre-process the images
        rgb_pil_images = self.pre_process_rgb_image(rgb_image)
        depth_pil_images = self.pre_process_depth_image(depth_image)

        # (rgb_batch(n_env) + rgb_batch(n_env)) x channel x height x width
        # first time need to load the visual encoder weights
        processed_images = self.backbone.process_images(
            # element by element concatenation
            [val for pair in zip(rgb_pil_images, depth_pil_images) for val in pair],
            self.backbone.config).to(dtype=self.backbone.dtype,
                                     device=self.backbone.device).view(n_env, 2, 3, 384, 384)

        # visualize the pre-processed images
        # self.visualize_tensor_preprocess(processed_images)

        with torch.no_grad():
            last_hidden_layer = self.backbone(
                padded_input_ids_batch,  # n_env x input_length
                images=processed_images,  # n_env x 2 x 3 x 384 x 384
                output_hidden_states=True,
                use_cache=True,
            ).hidden_states[-1]  # Final layer hidden states

            max_pooled_hidden_state = torch.max(last_hidden_layer, dim=1).values.unsqueeze(1)
            # max_pooled_hidden_state = F.max_pool1d(last_hidden_layer.permute(0, 2, 1),
            #                                        kernel_size=last_hidden_layer.size(1)).permute(0, 2, 1)


        if self.visualize_prompt:
            self.backbone.lm_head = self.deleted_lm_head
            logging.info(self.prompts)
            # batch x output_id
            start_time = time.time()
            outputs = self.backbone.generate(
                padded_input_ids_batch, # n_env x input_length
                images=processed_images,  # n_env x 2 x 3 x 384 x 384
                max_new_tokens=150,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
                repetition_penalty=1.0,
                temperature=0,
                # output_attentions=True
            )
            logging.info(f"Time taken for generation: {time.time() - start_time:.2f}s")
            # The generated sequences
            generated_sequences = outputs.sequences

            # logging.info([ans.strip() for ans in self.tokenizer.batch_decode(generated_sequences[:, padded_input_ids_batch.shape[1]:],
            #                                                           skip_special_tokens=True)])
            for ans in self.tokenizer.batch_decode(
                generated_sequences[:, padded_input_ids_batch.shape[1]:],
                skip_special_tokens=True):
                logging.info(ans.strip())
            self.backbone.lm_head = nn.Sequential()

        return max_pooled_hidden_state

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


    # Ensure device consistency across all tensors
    if input_ids is not None:
        input_ids_device = input_ids.device
    if position_ids is not None:
        position_ids_device = position_ids.device
    if attention_mask is not None:
        attention_mask_device = attention_mask.device
    if labels is not None:
        labels_device = labels.device
    if images is not None:
        images_device = images.device

    # Get the vision tower (assuming it's a part of the model and resides on a specific device)
    vision_tower = self.get_vision_tower()
    # Ensure images are on the same device as the vision tower
    vision_tower_device = next(vision_tower.parameters()).device

    # Determine the device of the embed_tokens layer
    embed_tokens_device = next(self.get_model().embed_tokens.parameters()).device

    # Ensure all input tensors are on the same device as the embed_tokens layer
    if input_ids is not None and input_ids_device != embed_tokens_device:
        input_ids = input_ids.to(embed_tokens_device)
    if position_ids is not None and position_ids_device != embed_tokens_device:
        position_ids = position_ids.to(embed_tokens_device)
    if attention_mask is not None and attention_mask_device != embed_tokens_device:
        attention_mask = attention_mask.to(embed_tokens_device)
    if labels is not None and labels_device != embed_tokens_device:
        labels = labels.to(embed_tokens_device)

    # Check for conditions for auto-regressive generation
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        # auto-regressive generation, input_ids is a single token
        if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
            1] == 1:
            target_shape = past_key_values[-1][-1].shape[-2] + 1
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                ),
                dim=1
            )
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    # Handling images and feature extraction
    if images.ndim == 5:
        # n_env x 2 x 3 x 384 x 384 -> 4 x 3 x 384 x 384
        if images_device != vision_tower_device:
            images = images.to(vision_tower_device)
        concat_images = torch.cat([image.to(vision_tower_device) for image in images], dim=0)

        image_features = self.encode_images(concat_images)

        split_sizes = [image.shape[0] for image in images]
        # list of n_env, each one's embedding is 2x729x 2560 (RGB Embedding and Depth Embedding)
        image_features = torch.split(image_features, split_sizes, dim=0)
        # list of n_env, each one's embedding is 1458 x 2560 (RGB Embedding and Depth Embedding)
        # image_features = [x.to(self.device) for x in image_features]
    else:
        image_features = self.encode_images(images)

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

    # Iterate over input batches and handle embedding and image features
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
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])

        split_sizes = [x.shape[0] for x in cur_labels_noim]

        # Make sure all tensors are on the correct device
        cur_input_embeds = self.get_model().embed_tokens(
            torch.cat(cur_input_ids_noim).to(embed_tokens_device)
        )
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

        cur_new_input_embeds = []
        cur_new_labels = []
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[batch_idx][cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
        cur_image_idx = 0

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
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
