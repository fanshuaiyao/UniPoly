#!/bin/bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/pretrain.py \
  --dataset_name smi_test \
  --modalities smiles fp text graph \
  --epochs 5 \
  --batch_size 2 \
  --lr 5e-5 \
  > ./logs/pretrain_sftg_unfreeze.log 2>&1 &

