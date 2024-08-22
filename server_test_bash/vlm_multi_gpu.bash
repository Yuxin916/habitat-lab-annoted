#!/bin/bash
# Job script example
###PBS -q moreGPU
#PBS -l select=1:ncpus=10:ngpus=2:host=dgx02
#PBS -N baseline_multi_gpu
#PBS -l software=nvidia-smi
#PBS -l walltime=01:00:00
#PBS -m abe
#PBS -M stucaiy@i2r.a-star.edu.sg
#PBS -o /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/server_test_bash/vlm_multi_gpu.out
#PBS -e /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/server_test_bash/vlm_multi_gpu.error

### Start of commands to be run
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6


### single GPU
#python habitat-baselines/habitat_baselines/run.py \
#--config-name=objectnav/saved_config/vlm_rl_all_input.yaml \
#habitat_baselines.num_environments=8 \
#habitat_baselines.evaluate=False

### Multi GPU
set -x
python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=6 \
--master_port=25678 \
--use_env \
habitat-baselines/habitat_baselines/run.py \
--config-name=objectnav/saved_config/vlm_rl_all_input.yaml \
habitat_baselines.num_environments=8 \
habitat_baselines.evaluate=False
