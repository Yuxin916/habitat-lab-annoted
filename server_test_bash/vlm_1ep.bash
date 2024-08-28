#!/bin/bash
# Job script example
###PBS -q lessGPU
#PBS -l select=1:ncpus=15:ngpus=3:mem=256gb:host=dgx02
#PBS -N vlm_1_eps
#PBS -l software=nvidia-smi
#PBS -l walltime=60:00:00
#PBS -m abe
#PBS -M stucaiy@i2r.a-star.edu.sg
#PBS -o /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/server_test_bash/one_eps_multi_gpu.out
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
habitat.dataset.train_eps=1 \
habitat_baselines.num_environments=6 \
habitat_baselines.trainer_name=ddppo \
habitat_baselines.log_interval=5 \
habitat_baselines.rl.ppo.num_steps=64 \
habitat_baselines.tensorboard_dir="log/tb/vlm_one_eps" \
habitat_baselines.video_dir="log/video_dir/vlm_one_eps" \
habitat_baselines.eval_ckpt_path_dir="ckpt/vlm_one_eps/latest.pth" \
habitat_baselines.checkpoint_folder="ckpt/vlm_one_eps/" \
habitat_baselines.log_file="log/log/vlm_one_eps.log" \
habitat_baselines.prompt="'List all detected movable objects in provided home scene RGB-D images and describe their spatial relationships, including proximity and layout.'"
