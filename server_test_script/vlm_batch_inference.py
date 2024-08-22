import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np
from habitat_baselines.rl.ddppo.policy.foundation_policy import override
from torchvision import transforms
import matplotlib.pyplot as plt
import logging
import time

"""
This file contains batch inference of SpatialBot-3B
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

class vis_utils():
    def visualize_batch(batch):
        # Ensure the batch is in the correct shape (B, C, H, W)
        assert len(batch.shape) == 4, "Expected batch shape [batch_size, channels, height, width]"

        batch_size, channels, height, width = batch.shape

        # If the batch is on the GPU, move it to CPU for visualization
        if batch.is_cuda:
            batch = batch.cpu()

        # Normalize if the tensor is not in the [0, 1] range (optional)
        # batch = (batch - batch.min()) / (batch.max() - batch.min())

        # Create a grid to display the images
        fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 4, 4))

        for i in range(batch_size):
            img = batch[i]  # Select the i-th image

            if channels == 1:
                # If the image is grayscale, remove the channel dimension
                img = img.squeeze(0)
                axes[i].imshow(img, cmap='gray')
            else:
                # Transpose to [H, W, C] for RGB display
                img = img.permute(1, 2, 0)
                axes[i].imshow(img)

            axes[i].axis('off')  # Turn off axis labels

        plt.show()


    def visualize_tensor(image_tensor):
        # Ensure the tensor is on the CPU and in float32 for visualization
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()

        # Convert to float32 if it's in float16
        if image_tensor.dtype == torch.float16:
            image_tensor = image_tensor.to(torch.float32)

        # Normalize the tensor to the [0, 1] range for visualization (optional)
        # This step is often necessary if your tensor has values outside the [0, 1] range
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

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


# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()
# load vision tower weights
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()

# get self.backbone parameter size
model_size = sum(p.numel() for p in model.parameters())
logging.info(f"model size: {model_size}")

# override prepare_inputs_labels_for_multimodal
model.prepare_inputs_labels_for_multimodal = override.__get__(model)

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

def prepare_batch_padded_input_id(model, tokenizer):
    # Define the prompts
    prompts = [
        "Ignore wall, floor, ceiling, and window. List all objects detected.",
        "What is the spatial relationship among the objects? List the objects in order of proximity.",
        "Ignore wall, floor, ceiling, and window. List all objects detected.",
        "What is the spatial relationship among the objects? List the objects in order of proximity.",
    ]

    # Set tokenizer padding side and pad token id
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = model.generation_config.pad_token_id

    # Prepare the batched input IDs
    texts = [
        f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        for prompt in prompts
    ]

    # Tokenize the texts and create input_ids tensors
    tokenized_chunks = [
        [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        for text in texts
    ]

    # Combine tokenized chunks with special tokens and create input tensors
    input_ids_list = [
        torch.tensor(chunk[0] + [-201] + [-202] + chunk[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(model.device)
        for chunk in tokenized_chunks
    ]

    # Find the maximum length for padding
    max_length = max(ids.shape[-1] for ids in input_ids_list)

    # Pad the sequences to the maximum length and stack into a batch
    padded_input_ids_batch = torch.stack([
        torch.cat([ids, torch.full((1, max_length - ids.shape[-1]), tokenizer.pad_token_id,
                                   dtype=torch.long, device=ids.device)], dim=-1)
        for ids in input_ids_list
    ]).squeeze(1).to(model.device)

    return padded_input_ids_batch

input_ids = prepare_batch_padded_input_id(model, tokenizer)


def pre_process_image(image):
    """
    normalize a batch of RGB / Depth images such that each image in the batch
    - Calculate the minimum and maximum values for each RGB image in the batch.
    - Normalize each image in the batch individually to have values between 0 and 1.
    - Rescale the normalized images to the range [0, 255] and convert to uint8.
    """
    # batch(n_env) x channel x height x width
    image = image[:, :3]  # Ensure only RGB channels are considered
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

paths = [
    "../images/home_rgb.png",
    "../images/home_d.png",
    "../images/home_rgb.png",
    "../images/home_d.png",
    "../images/home_rgb.png",
    "../images/home_d.png",
    "../images/home_rgb.png",
    "../images/home_d.png",
]

def load_and_prepare_images(paths):
    images = []
    transform = transforms.ToTensor()  # Converts images to [C, H, W] format

    # Load and transform each image
    for path in paths:
        image = Image.open(path).convert('RGBA')  # Ensure 3 channels (RGB)
        image_tensor = transform(image)
        images.append(image_tensor)

    # Determine the max height and width to pad images accordingly
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    # Pad images to the same size
    padded_images = []
    for img in images:
        _, h, w = img.shape
        padding = (0, max_width - w, 0, max_height - h)  # Padding (left, right, top, bottom)
        padded_img = torch.nn.functional.pad(img, padding)
        padded_images.append(padded_img)

    # Stack images to form a batch
    batch = torch.stack(padded_images)

    return batch

batch = load_and_prepare_images(paths)
if debug:
    vis_utils.visualize_batch(batch)

batch = pre_process_image(batch)

# 4 x channel x height x width
image_tensor = model.process_images(batch, model.config).to(dtype=model.dtype, device=model.device)
model.get_vision_tower().to(model.device)

if debug:
    vis_utils.visualize_tensor(image_tensor)

image_tensor = image_tensor.view(int(len(paths)/2), 2, 3, 384, 384)

start_time = time.time()

model.config.output_hidden_states = True
# batch x output_id
outputs = model.generate(
    input_ids,  # batch x 55
    images=image_tensor,  # batch x 2 x 3 x 384 x 384
    max_new_tokens=250,
    output_hidden_states=True,
    return_dict_in_generate=True,
    use_cache=True,
    repetition_penalty=1.0  # increase this to avoid chattering
)
print(f"Time taken: {time.time() - start_time:.2f}s")

# The generated sequences
generated_sequences = outputs.sequences

for ans in tokenizer.batch_decode(generated_sequences[:, input_ids.shape[1]:], skip_special_tokens=True):
    print(ans.strip())
# print([ans.strip() for ans in tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)])
