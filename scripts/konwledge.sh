export PYTHONPATH=$(pwd)

python scripts/build_konwledge.py \
  --input_csv /home/fsy23/UniPoly/moleculenet/classification/clintox.csv \
  --smiles_col smiles \
  --save_every 1000
