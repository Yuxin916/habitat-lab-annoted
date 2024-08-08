"""
This is the test file for single agent environment construction.
It aligns with how the environment is constructed in the habitat-baselines. (PPO, DDPPO, ...)
Objective:
    1. add more log in infos, done information..
    2. add more visualization in the environment

Load the model from checkpoint.
and run step by step to see the output of the model.
"""

import torch

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    batch_obs,
)
from habitat_baselines.utils.env_utils import construct_envs
from typing import Any, Dict, List, Optional
import tqdm

# Load the model from checkpoint
checkpoint_path = 'data/ddppo_checkpoints_minival/ckpt.26.pth'
ckpt_dict = torch.load(checkpoint_path, map_location='cpu')

exp_config = "habitat_baselines/env_test/env_test_ddppo_objectnav.yaml"
config = get_config(exp_config, None)
logger.info(f"env config: {config}")

####################### initialize the environment ##########################
# vector env -> ObjNavRLEnv -> RL Env -> Env
envs = construct_envs(
    config,
    # 获取register好的env class 封装好 - environments.py
    get_env_class(config.ENV_NAME),
    workers_ignore_signals=False,
)
# spaces
observation_space = envs.observation_spaces[0]
action_space = envs.action_spaces[0]

######################## device ##############################################
if torch.cuda.is_available():
    device = torch.device("cuda", config.TORCH_GPU_ID)
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

######################## initialize the agent ################################
logger.add_filehandler(config.LOG_FILE)
policy = baseline_registry.get_policy(config.RL.POLICY.name)

# initialize actor critic network (using either initialzation or class method)
# actor_critic = policy.from_config(
#     config=config,
#     observation_space=observation_space,
#     action_space=action_space,
# )
actor_critic = policy(
    observation_space=observation_space,
    action_space=action_space,
    hidden_size=config.RL.PPO.hidden_size,
    rnn_type=config.RL.DDPPO.rnn_type,
    num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
    backbone=config.RL.DDPPO.backbone,
    normalize_visual_inputs="rgb" in observation_space.spaces,
    force_blind_policy=config.FORCE_BLIND_POLICY,
)
actor_critic.to(device)
# Whether or not the visual encoder backbone will be trained.
if not config.RL.DDPPO.train_encoder:
    # set the visual encoder backbone to not trainable
    for param in actor_critic.net.visual_encoder.parameters():
        param.requires_grad_(False)

# initialize the agent
agent = DDPPO(
    actor_critic=actor_critic,
    clip_param=config.RL.PPO.clip_param,
    ppo_epoch=config.RL.PPO.ppo_epoch,
    num_mini_batch=config.RL.PPO.num_mini_batch,
    value_loss_coef=config.RL.PPO.value_loss_coef,
    entropy_coef=config.RL.PPO.entropy_coef,
    lr=config.RL.PPO.lr,
    eps=config.RL.PPO.eps,
    max_grad_norm=config.RL.PPO.max_grad_norm,
    use_normalized_advantage=config.RL.PPO.use_normalized_advantage,
)
# load each layers weights from the checkpoint
agent.load_state_dict(ckpt_dict["state_dict"])
logger.info("agent architecture: \n" + str(agent))

######################## initialize the agent ################################
observations = envs.reset()
batch = batch_obs(
    observations, device=device, cache=ObservationBatchingCache()
)

current_episode_reward = torch.zeros(config.NUM_ENVIRONMENTS,
                                     1,
                                     device="cpu")
test_recurrent_hidden_states = torch.zeros(config.NUM_ENVIRONMENTS,
                                           actor_critic.net.num_recurrent_layers,
                                           config.RL.PPO.hidden_size,
                                           device=device)
prev_actions = torch.zeros(config.NUM_ENVIRONMENTS,
                           1,
                           device=device,
                           dtype=torch.long)
not_done_masks = torch.zeros(config.NUM_ENVIRONMENTS,
                             1,
                             device=device,
                             dtype=torch.bool,)
stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
rgb_frames = [[] for _ in range(config.NUM_ENVIRONMENTS)]  # type: List[List[np.ndarray]]

number_of_eval_episodes = sum(envs.number_of_episodes)

pbar = tqdm.tqdm(total=number_of_eval_episodes)
actor_critic.eval()

while (len(stats_episodes) < number_of_eval_episodes and config.NUM_ENVIRONMENTS > 0):
    current_episodes = envs.current_episodes()

    with torch.no_grad():
        (
            _,
            actions,
            _,
            test_recurrent_hidden_states,
        ) = actor_critic.act(
            batch,
            test_recurrent_hidden_states,
            prev_actions,
            not_done_masks,
            deterministic=False,
        )

        prev_actions.copy_(actions)  # type: ignore

    # NB: Move actions to CPU.  If CUDA tensors are
    # sent in to env.step(), that will create CUDA contexts
    # in the subprocesses.
    # For backwards compatibility, we also call .item() to convert to
    # an int
    step_data = [a.item() for a in actions.to(device="cpu")]

    outputs = envs.step(step_data)

    observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

    batch = batch_obs(
        observations,
        device=device,
        cache=ObservationBatchingCache(),
    )

    not_done_masks = torch.tensor(
        [[not done] for done in dones],
        dtype=torch.bool,
        device="cpu",
    )

    rewards = torch.tensor(
        rewards_l, dtype=torch.float, device="cpu"
    ).unsqueeze(1)
    current_episode_reward += rewards
    next_episodes = envs.current_episodes()
    envs_to_pause = []
    for i in range(config.NUM_ENVIRONMENTS):
        if (
            next_episodes[i].scene_id,
            next_episodes[i].episode_id,
        ) in stats_episodes:
            envs_to_pause.append(i)

        # episode ended
        if not not_done_masks[i].item():
            pbar.update()
            episode_stats = {}
            episode_stats["reward"] = current_episode_reward[i].item()
            episode_stats.update(
                _extract_scalars_from_info(infos[i])
            )
            current_episode_reward[i] = 0
            # use scene_id + episode_id as unique id for storing stats
            stats_episodes[
                (
                    current_episodes[i].scene_id,
                    current_episodes[i].episode_id,
                )
            ] = episode_stats

    not_done_masks = not_done_masks.to(device=device)
    (
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    ) = _pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    )


print('stop here')
