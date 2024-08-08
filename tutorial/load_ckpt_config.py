import torch

checkpoint_path = 'data/ddppo_checkpoints_minival/ckpt.26.pth'

ckpt_dict = torch.load(checkpoint_path, map_location='cpu')

print(ckpt_dict.keys())
# dict_keys(['state_dict', 'config', 'extra_state'])

print('end of script')
