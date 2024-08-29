#!/bin/bash
# Job script example
###PBS -q moreGPU
#PBS -l select=1:ncpus=10:ngpus=2:host=dgx02
#PBS -N baseline_multi_gpu
#PBS -l software=nvidia-smi
#PBS -l walltime=01:00:00
#PBS -m abe
#PBS -M stucaiy@i2r.a-star.edu.sg
#PBS -o /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/server_test_bash/baseline_multi_gpu.out
#PBS -e /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/server_test_bash/baseline_multi_gpu.error

# Get the number of available GPUs
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# Define global variable
LOG_DIR="baseline_debug"

# Check if only one GPU is available
if [ "$num_gpus" -eq 1 ]; then
    echo "Running on Single GPU！！！！！！"
    export HABITAT_ENV_DEBUG=0

    python habitat-baselines/habitat_baselines/run.py \
    --config-name=objectnav/saved_config/baseline_fix.yaml \
    habitat_baselines.num_environments=6 \
    habitat.dataset.train_eps=1 \
    habitat_baselines.trainer_name=ddppo \
    habitat_baselines.log_interval=5 \
    habitat_baselines.rl.ppo.num_steps=128 \
    habitat_baselines.rl.ppo.num_steps=64 \
    habitat_baselines.tensorboard_dir="log/tb/${LOG_DIR}" \
    habitat_baselines.video_dir="log/video_dir/${LOG_DIR}" \
    habitat_baselines.eval_ckpt_path_dir="ckpt/${LOG_DIR}/latest.pth" \
    habitat_baselines.checkpoint_folder="ckpt/${LOG_DIR}/" \
    habitat_baselines.log_file="log/log/${LOG_DIR}.log" \
    habitat_baselines.evaluate=False
else
    echo "Running on $num_gpus GPUs"
    export HABITAT_ENV_DEBUG=0

    set -x
    python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=$num_gpus \
    --use_env \
    habitat-baselines/habitat_baselines/run.py \
    --config-name=objectnav/saved_config/baseline_fix.yaml \
    habitat.dataset.train_eps=1 \
    habitat_baselines.num_environments=6 \
    habitat_baselines.trainer_name=ddppo \
    habitat_baselines.log_interval=5 \
    habitat_baselines.rl.ppo.num_steps=128 \
    habitat_baselines.rl.ppo.num_steps=64 \
    habitat_baselines.tensorboard_dir="log/tb/${LOG_DIR}" \
    habitat_baselines.video_dir="log/video_dir/${LOG_DIR}" \
    habitat_baselines.eval_ckpt_path_dir="ckpt/${LOG_DIR}/latest.pth" \
    habitat_baselines.checkpoint_folder="ckpt/${LOG_DIR}/" \
    habitat_baselines.log_file="log/log/${LOG_DIR}.log" \
    habitat_baselines.evaluate=False
fi
