#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=debias
#SBATCH --time=0:01:00

module load cuda/10.1

srun train.sh

