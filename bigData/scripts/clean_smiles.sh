#!/bin/bash

# ===== 基础路径 =====
BASE_DIR=/home/fsy23/UniPoly/bigData
SCRIPT=${BASE_DIR}/scripts/clean_smiles_csv.py

# ===== 输入输出 =====
INPUT_CSV=${BASE_DIR}/raw/300k.csv
OUTPUT_CSV=${BASE_DIR}/clean/cleaned.csv
INVALID_CSV=${BASE_DIR}/clean/invalid.csv

# ===== 参数 =====
SMILES_COL=smiles_x
CANONICAL_COL=canonical_smiles     # 若想新增列，改成 canonical_smiles
CHUNKSIZE=50000

# ===== 运行 =====
python ${SCRIPT} \
  --input_csv ${INPUT_CSV} \
  --output_csv ${OUTPUT_CSV} \
  --invalid_csv ${INVALID_CSV} \
  --smiles_col ${SMILES_COL} \
  --canonical_col ${CANONICAL_COL} \
  --chunksize ${CHUNKSIZE}
  > clean/clean_smiles.log 2>&1 &