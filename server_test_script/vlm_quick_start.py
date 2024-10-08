import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np
import time

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
model_name = '../hf_spatialbot/'

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
prompt = "Describe spatial relationships of all detected movable objects as inferred from the depth image in one or two sentence."

text = (f"A chat between a curious user and an artificial intelligence assistant. "
        f"The assistant gives helpful, detailed, "
        f"and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:")

text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]

input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(model.device)

image1 = Image.open('examples/images/20_episodes/32324_tv_monitor/images/rgb/0010.png')
image2 = Image.open('examples/images/20_episodes/32324_tv_monitor/images/depth/0010.png')

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

