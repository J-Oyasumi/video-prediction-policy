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
  --video_model_path outputs/svd/train_2026-02-23T06-50-24/checkpoint-33000 \
  --action_model_path logs/text_robocasa/runs/2026-02-24/10-50-41/robocasa/checkpoints/0034506_0.149465.pt \
  --clip_model_path weights/clip-vit-base-patch32 \
  --json_path outputs/eval/OpenCabinet/data.json \
  --output_dir outputs/eval/OpenCabinet