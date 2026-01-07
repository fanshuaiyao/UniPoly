#!/bin/bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
# Define the command to run the training script with desired parameters
nohup python scripts/pretrain.py \
    --dataset_name smi_all \
    --modalities smiles fp text\
    --epochs 1 \
    --batch_size 128 \
    > ./logs/pretrain.log 2>&1 &

