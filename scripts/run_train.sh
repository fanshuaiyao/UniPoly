#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

nohup python -u scripts/train.py \
  --modalities smiles fp text graph \
  --tasks lipo \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth \
  --batch_size 16 \
  --epochs 100 \
  --freeze_encoder \
  > ./logs/train_.log 2>&1 &
