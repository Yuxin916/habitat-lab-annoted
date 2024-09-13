import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np
import time
import json
import shutil
"""
This file contains quick start inference of SpatialBot-3B
Used to test the installation of model, and inference speed
"""

"""
Download Dataset:
    huggingface-cli download google/siglip-so400m-patch14-384 --local-dir hf_spatialbot/siglip --local-dir-use-symlinks False
My token:
    HUGGINGFACE_TOKEN=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
Run in Server:
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5

"""
debug = False


# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set model path
model_name = '../merged_1pXnuDYAj8r/'
with open(model_name + 'config.json', 'r') as file:
    data = json.load(file)

# Add the "auto_map" entry to the existing JSON data
data["auto_map"] = {
    "AutoConfig": "configuration_bunny_phi.BunnyPhiConfig",
    "AutoModelForCausalLM": "modeling_bunny_phi.BunnyPhiForCausalLM"
}

# Write the modified data back to the JSON file
with open(model_name + 'config.json', 'w') as file:
    json.dump(data, file, indent=4)

# copy "configuration_bunny_phi.py" and "modeling_bunny_phi.py" to the model directory from ../hf_spatialbot using python
shutil.copy2('../hf_spatialbot/configuration_bunny_phi.py', model_name)
shutil.copy2('../hf_spatialbot/modeling_bunny_phi.py', model_name)


offset_bos = 0

# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

# text prompt
# prompt = "could the tv monitor be found in this scene? Justify your reasoning based on the objects present and the environment."
prompt = "You are an intelligent robot equipped with an RGB-D sensor, navigating a home scene environment. Your current task is to locate and navigate to the chair. Your available actions are: {move forward, turn left, turn right, look up, look down, stop}. Your response should be formatted as a valid JSON file, structured as follows: \n{\n\"observation\": \"{Describe spatial relationships of all detected movable objects as inferred from the depth image in one or two sentence.}\", \n\"thought\": \"{Would the chair be found in this scene? Why or why not? }\", \n\"action\": \"{Choose one of the admissible actions listed above. }\" \n}"

text = (f"A chat between a curious user and an artificial intelligence assistant. "
        f"The assistant gives helpful, detailed, "
        f"and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:")

text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]

input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(model.device)

image1 = Image.open('examples/images/select_2/1pXnuDYAj8r/A1BZNPQ0H7ZSER:3JPSL1DZ5U2877WSMKVCE8TQK8HNAY_chair/images/rgb/0000.png')
image2 = Image.open('examples/images/select_2/1pXnuDYAj8r/A1BZNPQ0H7ZSER:3JPSL1DZ5U2877WSMKVCE8TQK8HNAY_chair/images/depth/0000.png')

# TODO: Now is just a very naive evaluation way, better loop with the sim

channels = len(image2.getbands())
if channels == 1:
    img = np.array(image2)
    height, width = img.shape
    three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
    three_channel_array[:, :, 0] = (img // 1024) * 4
    three_channel_array[:, :, 1] = (img // 32) * 8
    three_channel_array[:, :, 2] = (img % 32) * 8
    image2 = Image.fromarray(three_channel_array, 'RGB')

image_tensor = model.process_images([image1,image2], model.config).to(dtype=model.dtype, device=model.device)

if debug:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Iterate over the two images
    for i in range(2):
        test_image = image_tensor.to('cpu').to(torch.float32)
        img = test_image[i].permute(1, 2, 0).numpy()  # Change dimensions for visualization
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off axis

    # Adjust layout
    plt.tight_layout()
    plt.show()

# patch for model device
model.model.vision_tower = model.model.vision_tower.to(model.device)

# image_tensor = torch.concatenate((image_tensor.unsqueeze(0), image_tensor.unsqueeze(0)), dim=0).to(device)
# generate
start_time = time.time()
output_ids = model.generate(
    input_ids, # 1 x 64
    images=image_tensor, # 2 x 3 x 384 x 384
    max_new_tokens=250,
    # output_hidden_states=True,
    # return_dict_in_generate=True,
    use_cache=True,
    repetition_penalty=1.0 # increase this to avoid chattering
)[0]
print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
print(f"Time taken: {time.time() - start_time:.2f}s")

