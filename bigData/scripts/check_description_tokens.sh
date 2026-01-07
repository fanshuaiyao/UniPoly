#!/bin/bash

BASE_DIR=/home/fsy23/UniPoly/bigData
SCRIPT=${BASE_DIR}/scripts/check_description_tokens.py

INPUT_CSV=${BASE_DIR}/clean/res_desc_final.csv
TEXT_COL=merged_description
TOKENIZER=/home/fsy23/UniPoly/pretrained_models/T5
CHUNKSIZE=50000

mkdir -p "${BASE_DIR}/logs"

python "${SCRIPT}" \
  --input_csv "${INPUT_CSV}" \
  --text_col "${TEXT_COL}" \
  --tokenizer_name "${TOKENIZER}" \
  --chunksize "${CHUNKSIZE}" \
  > "${BASE_DIR}/logs/check_description_tokens.log" 2>&1
