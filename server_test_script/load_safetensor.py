from transformers import AutoModel
from safetensors import safe_open
import torch

# Define the paths to the model files
model_path_1 = "../merged_model/model-00001-of-00002.safetensors"
model_path_2 = "../merged_model/model-00002-of-00002.safetensors"

# Initialize an empty dictionary to store model weights
model_weights_1 = {}
model_weights_2 = {}

# Load weights from the first part
with safe_open(model_path_1, framework="pt", device="cpu") as f:
    for key in f.keys():
        model_weights_1[key] = f.get_tensor(key)

# Load weights from the second part
with safe_open(model_path_2, framework="pt", device="cpu") as f:
    for key in f.keys():
        model_weights_2[key] = f.get_tensor(key)

pass
