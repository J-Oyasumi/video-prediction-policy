#!/bin/bash
#SBATCH -J vpp_evaluation
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=b200-mig90

eval "$(conda shell.bash hook)"
conda activate vpp

python evaluate_robocasa.py \
  --video_model_path /home/disk2/gyj/hyc_ckpt/svd_2camera/checkpoint-100000 \
  --action_model_path /home/disk2/gyj/hyccode/Video-Prediction-Policy/checkpoint/alllayer1 \
  --clip_model_path /home/disk2/gyj/hyc_ckpt/llm/clip-vit-base-patch32 \
  --json_path data/annotation_all/train_all.json \
  --output_dir results/robocasa_evaluation