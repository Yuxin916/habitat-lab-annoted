from .config import SigLipVisionConfig, BunnyPhiConfig
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from abc import ABC, abstractmethod
from collections import Counter
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

# Model Constants
IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_INDEX = [
    -201, -202
]

class BunnyMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and \
                input_ids.shape[
                    1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0],
                     target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(
                    -1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3]
        images_minibatch = torch.stack(
            [images[idx] for idx in image_idx]) if len(image_idx) > 0 else []
        tmp_image_features = [None] * (len(image_idx))
        if getattr(images_minibatch, 'ndim', 0) == 4:
            if vision_tower is not None:
                image_features_minibatch = self.encode_images(
                    images_minibatch).to(self.device)
            else:
                image_features_minibatch = torch.randn(1).to(self.device)

            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]

        new_tmp = []
        for image in tmp_image_features:
            if isinstance(image, list):
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
            else:
                new_tmp.append(image)
        image_features = new_tmp

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1],
                                        dtype=torch.long,
                                        device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for
                     cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for
                  cur_labels, cur_attention_mask in
                  zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            input_ids_list = cur_input_ids.tolist()
            input_ids_counter = Counter(input_ids_list)
            num_images = sum(
                input_ids_counter[element] for element in IMAGE_TOKEN_INDEX)

            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

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
                    cur_image_features = image_features[cur_image_idx]
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


class BunnyPhiForCausalLM(PhiForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyPhiConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = BunnyPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_images(self, images, model_cfg):
        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == 'pad':
            for image in images:
                image = self.expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors='pt')['pixel_values']
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

AutoConfig.register("bunny-phi", BunnyPhiConfig)
AutoModelForCausalLM.register(BunnyPhiConfig, BunnyPhiForCausalLM)
