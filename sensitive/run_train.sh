#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

nohup python -u sensitive/diff_tau.py \
  --modalities smiles fp text graph \
  --tasks lipo \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth \
  --batch_size 16 \
  --tau 0.1 \
  --epochs 100 \
  --freeze_encoder \
  > ./logs/diff_tau.log 2>&1 &
