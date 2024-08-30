import torch

path = 'ckpt/vlm_adapter/ckpt.24.pth'
config = torch.load(path, map_location="cpu")
print(config['state_dict'].keys())

path2 = 'ckpt/reproduce_1_ep/ckpt.4.pth'
config2 = torch.load(path, map_location="cpu")
print(config2['state_dict'].keys())

# compare the two keys
config_keys = config['state_dict'].keys()
config2_keys = config2['state_dict'].keys()
print(set(config_keys) - set(config2_keys))

# compare DictConfig
habitat_config1 = config['config']['habitat']
habitat_config2 = config2['config']['habitat']
# compare the values



# path2 = 'ckpt/vlm_adapter/ckpt.24.pth'
# config2 = torch.load(path2, map_location="cpu")
pass
