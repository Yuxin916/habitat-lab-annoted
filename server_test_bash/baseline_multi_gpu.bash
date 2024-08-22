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

### Start of commands to be run
### make this the working directory for all time
cd /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/habitat-lab-annoted
### conda env
source /home/i2r/stucaiy/miniconda3/etc/profile.d/conda.sh
conda activate habitat-sim
### CUDA related
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
### source lab and baseline to use habitat
export PYTHONPATH=$PYTHONPATH:habitat-lab
export PYTHONPATH=$PYTHONPATH:habitat-baselines
### hf cache ralative path
export HF_HOME="../spatial_bot_test"
### logging
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet 
### export CUDA_VISIBLE_DEVICES=1,2


### single GPU 
python habitat-baselines/habitat_baselines/run.py \
--config-name=objectnav/saved_config/baseline_one_episode.yaml \
habitat_baselines.num_environments=4 \
habitat_baselines.evaluate=True

### Multi GPU 
###set -x
###python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 \
###--master_port=25678 \
###--use_env \
###habitat-baselines/habitat_baselines/run.py \
###--config-name=objectnav/saved_config/baseline_one_episode.yaml habitat_baselines.evaluate=False \
