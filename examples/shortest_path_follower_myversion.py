#!/usr/bin/env python3
import json
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import numpy as np

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from PIL import Image


IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


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


def shortest_path_example():
    config = habitat.get_config(
        config_path="benchmark/nav/objectnav/objectnav_mp3d.yaml",
        overrides=[
            "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"
        ],
    )

    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.habitat.simulator.forward_step_size
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        print("Environment creation successful")
        print("How many episodes are in the environment?", len(env.episodes))
        OUTPUT_DIR = config.habitat.dataset.split  + '_' + str(len(env.episodes))
        # if len(env.episodes) > 100:
        #     raise ValueError("The number of episodes is too large. Please reduce the number of episodes to 100 or less.")
        for episode in range(len(env.episodes)):
            env.reset()
            episode_id = env.habitat_env.current_episode.episode_id
            scene_id = env.habitat_env.current_episode.scene_id.split('/')[4]
            object_categories = env.habitat_env.current_episode.object_category
            print(
                f"Agent stepping around inside environment. "
                f"Scene id: {scene_id}. "
                f"Episode id: {episode_id}. Final Goal: {object_categories}"
            )

            dirname = os.path.join(
                IMAGE_DIR,
                OUTPUT_DIR,
                scene_id,
                f"{int(episode_id):02d}_{object_categories}"
            )

            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)

            images = []
            depth_images = []
            rgb_images = []
            best_actions = []
            while not env.habitat_env.episode_over:
                # integer index of the action
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                best_actions.append(best_action)
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                im = observations[0]["rgb"]
                dep = observations[0]["depth"]
                top_down_map = draw_top_down_map(info, im.shape[0])
                output_im = np.concatenate((im, top_down_map), axis=1)
                # this is a concat of rgb and topdownmap
                images.append(output_im)

                # collect depth and rgb images for video
                depth_images.append(dep)
                rgb_images.append(im)

            if len(images) > 2:
                images_to_video(images, dirname, "trajectory")
                save_images(depth_images, rgb_images, best_actions, dirname,
                            env.habitat_env.current_episode.goals_key,
                            env.habitat_env.current_episode.object_category,
                            OUTPUT_DIR
                            )
                print("Episode finished")
            else:
                # delete the folder if the episode is too short
                shutil.rmtree(dirname)
                print("Episode finished without taking any action")


def save_images(depth_images, rgb_images, best_actions, output_dir, json_id, goal_object, OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # create a image folder
    os.makedirs(os.path.join(output_dir, "images"))
    output_dir = os.path.join(output_dir, "images")

    # create separate folders for depth and rgb images
    os.makedirs(os.path.join(output_dir, "depth"))
    os.makedirs(os.path.join(output_dir, "rgb"))

    # create a json file at the previous stage of output_dir
    with open(os.path.join(os.path.dirname(output_dir), OUTPUT_DIR + "_QA.json"),
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
                ] * 2  # how many 回合 of conversation (by default one from human one from gpt)
            }] * len(depth_images)
        # fill in the data
        json.dump(empty_data, f, indent=4)

    for i, (depth_image, rgb_image) in enumerate(zip(depth_images, rgb_images)):
        depth_image_path = os.path.join(output_dir, "depth", f"{i:04d}.png")
        rgb_image_path = os.path.join(output_dir, "rgb", f"{i:04d}.png")

        # preprocess the rgb image before saving
        # double check to 0-255 and unit 8 (already are)
        rgb_image = Image.fromarray(rgb_image, 'RGB')
        rgb_image.save(rgb_image_path)

        # preprocess the depth image before saving
        # normalize to 0-1
        depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
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

        with (open(os.path.join(os.path.dirname(output_dir), OUTPUT_DIR + "_QA.json"),
                  "r") as f):
            data = json.load(f)
        data[i]["id"] = json_id + f"_{i:04d}"
        data[i]["image"] = [os.path.join(*rgb_image_path.split(os.sep)[2:]), os.path.join(*depth_image_path.split(os.sep)[2:])]
        data[i]["conversations"][0]["from"] = "human"
        data[i]["conversations"][0]["value"] = ("<image 1>\n<image 2>\nYou are an intelligent robot equipped with an RGB-D sensor, navigating a home scene environment. "
                                                "Your current task is to locate and navigate to the " + goal_object + ". "
                                                "During your navigation, you have encountered the following objects: \n"
                                                "1. TV monitor\n"
                                                "2. Sofa\n"
                                                "Your available actions are: move forward, turn left, turn right, stop. "
                                                "Your response should be formatted as a valid JSON file, structured as follows: \n"
                                                "{\n"
                                                "\"observation\": \"{First, describe the movable objects you observe in the RGB image and their spatial relationships as inferred from the Depth image.}\", \n"
                                                "\"thought\": \"{Second, carefully consider: could the " + goal_object + " be found in this scene? Justify your reasoning based on the objects present and the environment.}\", \n"
                                                "\"action\": \"{Choose one of the admissible actions listed above}\""
                                                                                                                         "\n}")
        data[i]["conversations"][1]["from"] = "gpt"
        data[i]["conversations"][1]["value"] = {
            "observation: "
            "thought: "
            "action: " + str(best_actions[i])
        }
        # Open the JSON file for writing to save the modified data

        with open(os.path.join(os.path.dirname(output_dir), OUTPUT_DIR + "_QA.json"),
                  "w") as f:
            json.dump(data, f,
                      indent=4)  # Write the modified data back to the file




def main():
    shortest_path_example()


if __name__ == "__main__":
    main()
