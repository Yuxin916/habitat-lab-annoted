#!/bin/bash
# Job script example
###PBS -q moreGPU-express
#PBS -l select=1:ncpus=40:ngpus=8:mem=500gb:host=dgx04
#PBS -N one_eps_multi_gpu
#PBS -l software=nvidia-smi
#PBS -l walltime=01:00:00
#PBS -m abe
#PBS -M stucaiy@i2r.a-star.edu.sg
#PBS -o /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/server_test_bash/vlm_one_eps.out
#PBS -e /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/server_test_bash/one_eps_multi_gpu.error



### single GPU
#python habitat-baselines/habitat_baselines/run.py \
#--config-name=objectnav/saved_config/vlm_rl_all_input.yaml \
#habitat_baselines.num_environments=8 \
#habitat_baselines.evaluate=False

### Multi GPU
set -x
python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=3 \
--use_env \
habitat-baselines/habitat_baselines/run.py \
--config-name=objectnav/saved_config/vlm_rl_all_input.yaml \
habitat.dataset.train_eps=20 \
habitat_baselines.num_environments=5 \
habitat_baselines.trainer_name=ddppo \
habitat_baselines.log_interval=5 \
habitat_baselines.rl.ppo.num_steps=32 \
habitat_baselines.tensorboard_dir="log/tb/VLM_20_eps" \
habitat_baselines.video_dir="log/video_dir/VLM_20_eps" \
habitat_baselines.eval_ckpt_path_dir="ckpt/VLM_20_eps/latest.pth" \
habitat_baselines.checkpoint_folder="ckpt/VLM_20_eps/" \
habitat_baselines.log_file="log/log/VLM_20_eps.log" \
habitat_baselines.prompt="'List all detected movable objects in provided home scene RGB-D images and describe their spatial relationships, including proximity and layout.'"
