#!/bin/bash
# Job script example
#PBS -l select=1:ncpus=20:ngpus=7:mem=128gb:host=dgx02
#PBS -N vlm_inference
#PBS -l software=nvidia-smi
#PBS -l walltime=00:10:00
#PBS -m abe
#PBS -M stucaiy@i2r.a-star.edu.sg
#PBS -o /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/vlm_infer_test.out
#PBS -e /home/i2r/stucaiy/scratch/yuxin_projects/icra2024/vlm_infer_test.error


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

### GPU 
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

### python file
python server_test_script/spatial_quick_start.py
