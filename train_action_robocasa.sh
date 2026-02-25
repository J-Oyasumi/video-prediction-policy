#!/bin/bash
#SBATCH -J vpp_train_action
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --partition=dgx-b200

ml load cuda/12.8

eval "$(conda shell.bash hook)"
conda activate vpp

accelerate launch train_action_robocasa.py 