#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=debias
#SBATCH --time=100:00:00
#SBATCH --partition=gpu

module load cuda/10.1

srun train.sh


