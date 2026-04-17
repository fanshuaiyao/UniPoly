#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

nohup python -u scripts/train.py \
  --modalities smiles fp text graph \
  --tasks clintox \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth \
  --batch_size 16 \
  --epochs 1 \
  --patience 5 \
  --freeze_encoder \
  > ./logs/train_test.log 2>&1 &
