#!/bin/bash
#SBATCH -J vpp_train_svd
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --partition=dgx-b200


accelerate launch --main_process_port 29506 step1_train_svd.py --config video_conf/train_robocasa_svd.yaml