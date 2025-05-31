#!/bin/bash
#SBATCH --job-name=bert-base-english-ood
#SBATCH --output=logs/finetune-bert-english-ood.log
#SBATCH --error=logs/finetune-bert-english-ood.err
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 


# Run training
CUDA_VISIBLE_DEVICE=2 python bert-base-english-ood.py