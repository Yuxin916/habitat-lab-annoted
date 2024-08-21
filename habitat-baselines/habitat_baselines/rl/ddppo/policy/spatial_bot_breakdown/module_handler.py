from .sig_lip_arch import SigLipVisionTower
import re
import torch.nn as nn
from transformers import AutoConfig

device = 'cuda'  # or cpu

def load_config(pretrained_model_name_or_path):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    return config

def build_vision_tower(config, **kwargs): # checked
    model_path = getattr(config, 'mm_vision_tower', getattr(config, 'vision_tower', None))

    return SigLipVisionTower(model_path, **kwargs).to(device=device)


def build_vision_projector(config):
    # mlp2x_gelu
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
