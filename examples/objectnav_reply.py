import numpy as np
from PIL import Image
import json
import os
import shutil
import base64

import numpy as np
from PIL import Image
from gradio.themes.builder_app import history

import habitat
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image, \
    images_to_video, append_text_underneath_image
from openai import OpenAI
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np
import time

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos",
                    video_name=prefix)


def run_reference_replay(max_episodes=10):
    config = habitat.get_config(
        config_path="benchmark/nav/objectnav/objectnav_mp3d.yaml",
        overrides=[
            "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"
        ],
    )

    # with SimpleRLEnv(config=config) as env:
    with habitat.Env(config=config) as env:
        total_success = 0
        spl = 0

        print("Environment creation successful")
        print("How many episodes are in the environment?", len(env.episodes))
        # OUTPUT_DIR = config.habitat.dataset.split + '_' + str(
        #     len(env.episodes))

        # print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            observation_list = []
            env.reset()
            step_index = 1
            episode = env.current_episode

            rgb_images = []
            depth_images = []
            human_actions = []
            images = []

            episode_id = episode.episode_id

            scene_id = episode.scene_id.split('/')[3]
            object_categories = episode.object_category

            print(
                f"Agent stepping around inside environment. "
                f"Scene id: {scene_id}. "
                f"Episode id: {episode_id}. Final Goal: {object_categories}"
            )

            dirname = os.path.join(
                IMAGE_DIR,
                # OUTPUT_DIR,
                scene_id,
                f"{episode_id}_{object_categories}"
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)

            for data in env.current_episode.reference_replay[step_index:]:
                if data.action == "STOP":
                    action = 0
                elif data.action == "MOVE_FORWARD":
                    action = 1
                elif data.action == "TURN_LEFT":
                    action = 2
                elif data.action == "TURN_RIGHT":
                    action = 3
                elif data.action == "LOOK_UP":
                    action = 4
                elif data.action == "LOOK_DOWN":
                    action = 5
                else:
                    raise ValueError("Invalid action")
                action_name = env.task.get_action_name(
                    action
                )
                human_actions.append(action_name)

                observations = env.step(action=action)[0]
                info = env.get_metrics()

                rgb_images.append(observations["rgb"])
                depth_images.append(observations["depth"])
                top_down_map = draw_top_down_map(info,
                                                 observations["rgb"].shape[0])

                # this is a concat of rgb and topdownmap
                output_im = np.concatenate((observations["rgb"], top_down_map),
                                           axis=1)
                images.append(output_im)

                frame = observations_to_image({"rgb": observations["rgb"]},
                                              info)
                frame = append_text_underneath_image(frame,
                                                     "Find and go to {}".format(
                                                         episode.object_category))

                observation_list.append(frame)
                if action_name == "stop":
                    break
            if 2 < len(rgb_images) < 300:
                images_to_video(images, dirname, "trajectory")
                save_images(depth_images, rgb_images, human_actions, dirname,
                            env.current_episode.goals_key,
                            env.current_episode.object_category,
                            OUTPUT_DIR
                            )
            else:
                # delete the folder if the episode is too short
                shutil.rmtree(dirname)
                print("Episode finished without taking any action or too long")

            # make_videos([observation_list], "example/demo", ep_id)
            # print("Total reward for trajectory: {}".format(total_reward))

            # if len(episode.reference_replay) <= 500:
            #     total_success += info["success"]
            #     spl += info["spl"]


def save_images(depth_images, rgb_images, best_actions, output_dir, json_id,
                goal_object, OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # create a image folder
    os.makedirs(os.path.join(output_dir, "images"))
    output_dir = os.path.join(output_dir, "images")

    # create separate folders for depth and rgb images
    os.makedirs(os.path.join(output_dir, "depth"))
    os.makedirs(os.path.join(output_dir, "rgb"))

    # create a json file at the previous stage of output_dir
    with open(
        os.path.join(os.path.dirname(output_dir),"QA.json"),
        "w") as f:
        # write in template first
        empty_data = [
                         {
                             "id": "",
                             "image": [],
                             "conversations": [
                                                  {
                                                      "from": "",
                                                      "value": ""
                                                  }
                                              ] * 2
                             # how many 回合 of conversation (by default one from human one from gpt)
                         }] * len(depth_images)
        # fill in the data
        json.dump(empty_data, f, indent=4)

    for i, (depth_image, rgb_image) in enumerate(
        zip(depth_images, rgb_images)):
        depth_image_path = os.path.join(output_dir, "depth", f"{i:04d}.png")
        rgb_image_path = os.path.join(output_dir, "rgb", f"{i:04d}.png")

        # preprocess the rgb image before saving
        # double check to 0-255 and unit 8 (already are)
        rgb_image = Image.fromarray(rgb_image, 'RGB')
        rgb_image.save(rgb_image_path)

        # preprocess the depth image before saving
        # normalize to 0-1
        depth_image = (depth_image - np.min(depth_image)) / (
                np.max(depth_image) - np.min(depth_image))
        # Convert depth from meters to millimeters
        depth_mm = depth_image * 1000  # Convert from meters to millimeters

        # Convert depth to 16-bit unsigned integer
        depth_mm_uint16 = depth_mm.astype(np.uint16)

        # Squeeze the last dimension to have a 2D depth image
        depth_mm_uint16 = np.squeeze(depth_mm_uint16)

        # Save the depth image as a 16-bit single channel image
        depth_image = Image.fromarray(depth_mm_uint16, mode='I;16')
        depth_image.save(depth_image_path)

        def visualize_depth(depth_image_path):
            import matplotlib.pyplot as plt
            # Convert the depth image to a numpy array
            depth_image = Image.open(depth_image_path)
            depth_array = np.array(depth_image)

            # Normalize the depth image for visualization
            depth_normalized = (depth_array - np.min(depth_array)) / (
                np.max(depth_array) - np.min(depth_array))

            # Plot the depth image
            plt.imshow(depth_normalized,
                       cmap='viridis')  # Use colormap suitable for depth images
            plt.colorbar(label='Depth (normalized)')
            plt.title('Depth Image Visualization')
            plt.show()

        # visualize_depth(depth_image_path)
        # or visualize using ImageJ

        CoT = get_CoT_from_gpt(rgb_image_path, goal_object)
        spatial = get_Spatial_from_Spatialbot(rgb_image_path, depth_image_path, goal_object)


        with (open(
            os.path.join(os.path.dirname(output_dir) + "/QA.json"),
            "r") as f):
            data = json.load(f)
        data[i]["id"] = json_id + f"_{i:04d}"
        data[i]["image"] = [os.path.join(*rgb_image_path.split(os.sep)[2:]),
                            os.path.join(*depth_image_path.split(os.sep)[2:])]
        data[i]["conversations"][0]["from"] = "human"
        data[i]["conversations"][0]["value"] = (
                "<image 1>\n<image 2>\nYou are an intelligent robot equipped with an RGB-D sensor, navigating a home scene environment. "
                "Your current task is to locate and navigate to the " + goal_object + ". "
                # + history +
                "Your available actions are: {move forward, turn left, turn right, look up, look down, stop}. "
                "Your response should be formatted as a valid JSON file, structured as follows: \n"
                "{\n"
                "\"observation\": \"{Describe spatial relationships of all detected movable objects as inferred from the depth image in one or two sentence.}\", \n"
                "\"thought\": \"{Would the " + goal_object + " be found in this scene? Why or why not? }\", \n"
                "\"action\": \"{Choose one of the admissible actions listed above. }\" \n}"
                )
        data[i]["conversations"][1]["from"] = "gpt"
        data[i]["conversations"][1]["value"] = (
            "{\n"
            "\"observation\": " + spatial + ", \n"
            "\"thought\": " + CoT + ", \n"
            "\"action\": " + best_actions[i] + ", \n}"
        )

        # if this json file already exists, delete it
        if os.path.exists(os.path.join(os.path.dirname(output_dir) + "/QA.json")):
            os.remove(os.path.join(os.path.dirname(output_dir) + "/QA.json"))
        # Open the JSON file for writing to save the modified data
        with open(
            os.path.join(os.path.dirname(output_dir) + "/QA.json"),
            "w") as f:
            json.dump(data, f,
                      indent=4)  # Write the modified data back to the file



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
prompt = "Describe spatial relationships of all detected movable objects as inferred from the depth image in one or two sentence."

text = (f"A chat between a curious user and an artificial intelligence assistant. "
        f"The assistant gives helpful, detailed, "
        f"and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:")

text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]

input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(model.device)


def get_Spatial_from_Spatialbot(rgb_image_path, depth_image_path, target_object):
    image1 = Image.open(rgb_image_path)
    image2 = Image.open(depth_image_path)

    channels = len(image2.getbands())
    if channels == 1:
        img = np.array(image2)
        height, width = img.shape
        three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
        three_channel_array[:, :, 0] = (img // 1024) * 4
        three_channel_array[:, :, 1] = (img // 32) * 8
        three_channel_array[:, :, 2] = (img % 32) * 8
        image2 = Image.fromarray(three_channel_array, 'RGB')

    image_tensor = model.process_images([image1, image2], model.config).to(
        dtype=model.dtype, device=model.device)

    # patch for model device
    model.model.vision_tower = model.model.vision_tower.to(model.device)

    # image_tensor = torch.concatenate((image_tensor.unsqueeze(0), image_tensor.unsqueeze(0)), dim=0).to(device)
    # generate
    start_time = time.time()
    output_ids = model.generate(
        input_ids,  # 1 x 64
        images=image_tensor,  # 2 x 3 x 384 x 384
        max_new_tokens=250,
        use_cache=True,
        repetition_penalty=1.0  # increase this to avoid chattering
    )[0]

    history = tokenizer.decode(output_ids[input_ids.shape[1]:],
                               skip_special_tokens=True).strip()
    return history



def get_CoT_from_gpt(rgb_image_path, target_object):

    base64_image = encode_image(rgb_image_path)

    human_question = f"Would a {target_object} be found here? Why or why not? Answer in two or three sentences."
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": human_question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    CoT = response.choices[0].message.content
    return CoT


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--path", type=str, default="replays/demo_1.json.gz"
    # )
    # parser.add_argument(
    #     "--output-prefix", type=str, default="examples/demo"
    # )
    # parser.add_argument(
    #     "--num-episodes", type=int, default=10
    # )
    # args = parser.parse_args()
    # cfg = config

    run_reference_replay()


if __name__ == "__main__":
    main()
