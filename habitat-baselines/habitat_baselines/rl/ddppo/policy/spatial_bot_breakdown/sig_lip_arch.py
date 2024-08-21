from transformers import AutoConfig, AutoModelForCausalLM
from abc import ABC, abstractmethod
from collections import Counter

'''
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
'''

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch.utils.checkpoint
from torch import nn
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (convert_to_rgb, normalize, rescale, resize, to_channel_dimension_format, )
from transformers.image_utils import (ChannelDimension, PILImageResampling, to_numpy_array, )
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from .config import SigLipVisionConfig
from .siglip_component import SigLipPreTrainedModel, SigLipVisionTransformer

class SigLipImageProcessor:
    def __init__(self,
                 image_mean=(0.5, 0.5, 0.5),
                 image_std=(0.5, 0.5, 0.5),
                 size=(384, 384),
                 crop_size: Dict[str, int] = None,
                 resample=PILImageResampling.BICUBIC,
                 rescale_factor=1 / 255,
                 data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)

class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
            self,
            pixel_values,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()

        # whether the weights are loaded or not
        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        # pre-process the image input
        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self):
        if self.is_loaded:
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name)

        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2






