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
    huggingface-cli download google/siglip-so400m-patch14-384 --local-dir spatial_bot_test/siglip --local-dir-use-symlinks False
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
model_name = '../spatial_bot_test/'

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
prompt = (
    "Ignore wall, floor, ceiling, and window. "
    "List all objects detected and output them in order of proximity. "
    "Answer from RGB and depth map. "
    "Use the following proximity levels: 'close', 'medium', 'far'. "
    "Output the detected objects in the following JSON format: "
    '{"objects": [{"name": "object_name", "proximity": "proximity_level"}, ...]}'
)
# {"objects": [{"name": "chair", "proximity": "medium"}, {"name": "table", "proximity": "medium"}, {"name": "window", "proximity": "far"}, {"name": "fan", "proximity": "far"}]}
prompt = (
    "Ignore wall, floor, ceiling, and window. "
    "List all objects detected and output their depth values. "
    "Answer from RGB and depth map. "
    "Output the detected objects in the following JSON format: "
    '{"objects": [{"name": "object_name", "depth": "depth_value"}, ...]}'
)
# {"objects": [{"name": "chair", "depth": 3200}, {"name": "table", "depth": 3200}, {"name": "window", "depth": 4100}, {"name": "fan", "depth": 4100}]}

prompt = (
    "Ignore wall, floor, ceiling, and window. "
    "List all objects detected and describe their spatial relationships. "
    "Answer using the RGB and depth map. "
    "Output the spatial relationships in the following JSON format: "
    '{"objects": [{"name": "object_name", "depth": depth_value, "position": {"x": x_value, "y": y_value, "z": z_value}, "relationships": {"closer_than": ["object1", "object2"], "further_than": ["object3", "object4"]}}, ...]}'
)
prompt = (
    "Ignore wall, floor, ceiling, and window. "
    "List all objects detected and describe their spatial relationships. "
    "Answer using the RGB and depth map. "
    "If the same type of object appears in multiple locations, append a unique identifier to the object's name (e.g., chair_1, chair_2). "
    "Output the spatial relationships in the following JSON format: "
    '{"objects": [{"name": "object_name", "depth": depth_value, "position": {"x": x_value, "y": y_value, "z": z_value}, "relationships": {"closer_than": ["object1", "object2"], "further_than": ["object3", "object4"]}}, ...]}'
)

prompt = (
    "Ignore wall, floor, ceiling, and window. "
    "List all objects detected and What is the spatial relationship among the objects? "
    )
#The objects detected are a table, four chairs, a bench, a fan, and a light fixture.
# The table is in the center, with four chairs around it.
# The bench is to the right of the table. The fan is hanging from the ceiling,
# and the light fixture is also hanging from the ceiling.

# prompt = (
# "Ignore walls, floors, ceilings, and windows. "
# "List all detected objects and describe their spatial relationships, "
# "including proximity and layout, based on depth information. "
# "Focus on the 2D layout of the objects, considering their relative positions and distances to "
# "support navigation tasks. Extract features that reflect these 3D spatial relationships."
# )

text = (f"A chat between a curious user and an artificial intelligence assistant. "
        f"The assistant gives helpful, detailed, "
        f"and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:")

text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]

input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(model.device)

image1 = Image.open('../images/home_rgb.png')
image2 = Image.open('../images/home_d.png')

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

