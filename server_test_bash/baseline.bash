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

### single GPU
#python habitat-baselines/habitat_baselines/run.py \
#--config-name=objectnav/saved_config/baseline_one_episode.yaml \
#habitat_baselines.num_environments=4 \
#habitat_baselines.evaluate=True

### Multi GPU
set -x
python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=3 \
--use_env \
habitat-baselines/habitat_baselines/run.py \
--config-name=objectnav/saved_config/baseline_one_episode.yaml habitat_baselines.evaluate=False
