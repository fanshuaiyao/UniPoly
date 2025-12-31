import csv
from collections import defaultdict

import pandas as pd
from rdkit import Chem


def canonicalize_smiles(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def main():
    input_csv = "moleculenet/classification/BBBP.csv"
    output_csv = "moleculenet/classification/BBBP_canonical_duplicates.csv"
    invalid_csv = "moleculenet/classification/BBBP_invalid_smiles.csv"

    df = pd.read_csv(input_csv)
    if "smiles" not in df.columns:
        raise ValueError("Missing smiles column in BBBP.csv")

    canonical_to_originals = defaultdict(list)
    invalid_smiles = []

    for smi in df["smiles"].astype(str):
        can = canonicalize_smiles(smi)
        if can is None:
            invalid_smiles.append(smi)
            continue
        canonical_to_originals[can].append(smi)

    rows = []
    for can_smi, originals in canonical_to_originals.items():
        if len(originals) > 1:
            rows.append(
                {
                    "canonical_smiles": can_smi,
                    "count": len(originals),
                    "original_smiles": "|".join(originals),
                }
            )

    rows.sort(key=lambda r: (-r["count"], r["canonical_smiles"]))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["canonical_smiles", "count", "original_smiles"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Duplicates written to: {output_csv}")
    print(f"Duplicate canonical SMILES: {len(rows)}")
    if invalid_smiles:
        with open(invalid_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["invalid_smiles"])
            for smi in invalid_smiles:
                writer.writerow([smi])
        print(f"Invalid SMILES written to: {invalid_csv}")
        print(f"Invalid SMILES skipped: {len(invalid_smiles)}")


if __name__ == "__main__":
    main()
