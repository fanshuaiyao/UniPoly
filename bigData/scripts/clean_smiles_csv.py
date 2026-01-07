import argparse
import os

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a large SMILES CSV with chunked processing.")
    parser.add_argument("--input_csv", required=True, help="Input CSV path.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path for valid rows.")
    parser.add_argument("--invalid_csv", required=True, help="Output CSV path for invalid rows.")
    parser.add_argument("--smiles_col", default="smiles_x", help="Column name containing SMILES.")
    parser.add_argument(
        "--canonical_col",
        default="smiles_x",
        help="Column name to store canonical SMILES; default replaces smiles_col.",
    )
    parser.add_argument("--chunksize", type=int, default=50000, help="Chunk size for pandas.read_csv.")
    return parser.parse_args()


def canonicalize_smiles(smiles: object) -> tuple[bool, str | None]:
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)) or pd.isna(smiles):
        return False, None
    try:
        smi = str(smiles).strip()
    except Exception:
        return False, None
    if not smi:
        return False, None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return True, canonical


def main() -> int:
    args = parse_args()

    RDLogger.DisableLog("rdApp.*")

    if args.chunksize <= 0:
        raise SystemExit("--chunksize must be > 0")

    if not os.path.exists(args.input_csv):
        raise SystemExit(f"Input not found: {args.input_csv}")

    # Start fresh outputs, then append chunk-by-chunk.
    for path in (args.output_csv, args.invalid_csv):
        if os.path.exists(path):
            os.remove(path)

    total_rows = 0
    valid_rows = 0
    invalid_rows = 0
    wrote_valid_header = False
    wrote_invalid_header = False

    for chunk_idx, chunk in enumerate(pd.read_csv(args.input_csv, chunksize=args.chunksize), start=1):
        if args.smiles_col not in chunk.columns:
            raise SystemExit(f"Column not found: {args.smiles_col}")

        total_rows += len(chunk)

        is_valid: list[bool] = []
        canonical_smiles: list[str | None] = []
        for smi in chunk[args.smiles_col].tolist():
            ok, canon = canonicalize_smiles(smi)
            is_valid.append(ok)
            canonical_smiles.append(canon)

        valid_df = chunk.loc[is_valid].copy()
        invalid_df = chunk.loc[[not v for v in is_valid]].copy()

        if len(valid_df) > 0:
            if args.canonical_col == args.smiles_col:
                valid_df[args.smiles_col] = [c for c, ok in zip(canonical_smiles, is_valid) if ok]
            else:
                valid_df[args.canonical_col] = [c for c, ok in zip(canonical_smiles, is_valid) if ok]

            valid_df.to_csv(
                args.output_csv,
                index=False,
                mode="a",
                header=not wrote_valid_header,
            )
            wrote_valid_header = True

        if len(invalid_df) > 0:
            invalid_df.to_csv(
                args.invalid_csv,
                index=False,
                mode="a",
                header=not wrote_invalid_header,
            )
            wrote_invalid_header = True

        valid_rows += len(valid_df)
        invalid_rows += len(invalid_df)
        print(
            f"chunk={chunk_idx} rows={len(chunk)} total={total_rows} "
            f"valid={valid_rows} invalid={invalid_rows}"
        )

    print(f"Done. total={total_rows} valid={valid_rows} invalid={invalid_rows}")
    print(f"valid_csv={args.output_csv}")
    print(f"invalid_csv={args.invalid_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

