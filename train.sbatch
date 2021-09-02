#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_example
#SBATCH --time=5:00:00

module load cuda/10.1

srun train.sh

