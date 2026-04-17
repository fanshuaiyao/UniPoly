#!/bin/bash
set -e

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

nohup python -u cot/train.py \
  --modalities text graph \
  --tasks lipo \
  > ./logs/cot_train_.log 2>&1 &
