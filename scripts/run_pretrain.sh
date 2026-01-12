#!/bin/bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

nohup python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles fp text graph \
  --epochs 20 \
  --batch_size 16 \
  --lr 5e-5 \
  > ./logs/pretrain_sftg_unfreeze.log 2>&1 &
