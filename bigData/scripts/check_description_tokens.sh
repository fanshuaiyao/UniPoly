#!/bin/bash

BASE_DIR=/home/fsy23/UniPoly/bigData
SCRIPT=${BASE_DIR}/scripts/check_description_tokens.py

INPUT_CSV=${BASE_DIR}/clean/cleaned.csv
TEXT_COL=Description
TOKENIZER=/home/fsy23/UniPoly/pretrained_models/T5
CHUNKSIZE=50000

python ${SCRIPT} \
  --input_csv ${INPUT_CSV} \
  --text_col ${TEXT_COL} \
  --tokenizer_name ${TOKENIZER} \
  --chunksize ${CHUNKSIZE}
