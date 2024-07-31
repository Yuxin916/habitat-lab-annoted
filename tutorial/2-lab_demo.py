import random
import matplotlib.pyplot as plt

import habitat

config = habitat.get_config(config_paths="configs/tasks/objectnav_hm3d_multi.yaml")
# # To change config, defrost it first
# config.defrost()
# config.DATASET.SPLIT = "val"
# # freeze the config
# config.freeze()

env = habitat.Env(config=config)
# Initializing dataset ObjectNav-v1
# initializing sim Sim-v0
# Initializing task ObjectNav-v1

"""
Scene semantic annotations
"""
def print_scene_recur(scene, limit_output=10):
    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None

# Print semantic annotation information (id, category, bounding box details)
# for the current scene in a hierarchical fashion
scene = env.sim.semantic_annotations()
#TODO: did not find the annotations, but data exists
print_scene_recur(scene, limit_output=15)

env.close()
# Note: Since only one OpenGL is allowed per process,
# you have to close the current env before instantiating a new one.

"""
Actions and Sensors
"""
import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

def display_sample(rgb_obs, semantic_obs, depth_obs):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")

    arr = [rgb_img, semantic_img, depth_img]

    titles = ['rgb', 'semantic', 'depth']
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()

config = habitat.get_config(config_paths="configs/tasks/objectnav_hm3d.yaml")
config.defrost()
config.DATASET.SPLIT = "val"
config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
config.freeze()

env = habitat.Env(config=config)
env.episodes = random.sample(env.episodes, 2)

max_steps = 4

action_mapping = {
    0: 'stop',
    1: 'move_forward',
    2: 'turn left',
    3: 'turn right'
}

for i in range(len(env.episodes)):
    observations = env.reset()

    display_sample(observations[0]['rgb'],
                   observations[0]['semantic'],
                   np.squeeze(observations[0]['depth']))

    count_steps = 0
    while count_steps < max_steps:
        action = random.choice(list(action_mapping.keys()))
        print(action_mapping[action])
        observations = env.step([action])
        display_sample(observations[0]['rgb'],
                       observations[0]['semantic'],
                       np.squeeze(observations[0]['depth']))

        count_steps += 1
        if env.episode_over:
            break

env.close()
