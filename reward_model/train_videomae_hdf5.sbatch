#!/bin/bash

#SBATCH --job-name=videomae_hdf5_4gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=videomae_hdf5_logs/videomae_hdf5_4gpu.out

nvidia-smi
module purge
module load slurm
source /home/qshou/miniconda3/bin/activate RoboTwin_zlc

torchrun --standalone --nproc_per_node=4 videomae_hdf5.py
