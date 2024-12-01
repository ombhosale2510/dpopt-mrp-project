#!/bin/bash
#SBATCH --time=0-00:05:00
#SBATCH --account=def-wzhang25
#SBATCH --mem-per-cpu=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

free -h

echo free -h
