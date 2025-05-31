#!/bin/bash
#SBATCH --job-name=bert-base-chinese
#SBATCH --output=logs/finetune-bert-chinese.log
#SBATCH --error=logs/finetune-bert-chines.err
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 


# Run training
CUDA_VISIBLE_DEVICE=2 python bert-base-chinese.py