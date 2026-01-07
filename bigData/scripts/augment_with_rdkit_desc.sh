#!/bin/bash
set -euo pipefail

ROOT="/home/fsy23/UniPoly"
WORK="/home/fsy23/UniPoly/bigData"
LOGDIR="$WORK/logs"
mkdir -p "$LOGDIR"

cd "$ROOT"
export PYTHONPATH="$ROOT"

INPUT="$WORK/clean/cleaned.csv"
OUTPUT="$WORK/clean/res_desc_final.csv"
LOG="$LOGDIR/augment_rdkit_desc.log"

if [ ! -f "$INPUT" ]; then
  echo "[ERROR] input not found: $INPUT"
  exit 1
fi

echo "[START] $(date)"
echo "[INFO] input=$INPUT"
echo "[INFO] output=$OUTPUT"
echo "[INFO] log=$LOG"

nohup python -u "$WORK/scripts/augment_with_rdkit_desc.py" \
  --input_csv "$INPUT" \
  --output_csv "$OUTPUT" \
  --smiles_col "canonical_smiles" \
  --desc_col "Description" \
  --rdkit_col "rdkit_description" \
  --merged_col "merged_description" \
  --chunksize 50000 \
  > "$LOG" 2>&1 &0

PID=$!
echo "[OK] started pid=$PID"
echo "[OK] tail -f $LOG"
