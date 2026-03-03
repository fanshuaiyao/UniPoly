"""
Generate SVG (vector) molecular structure images from SMILES strings using RDKit.

Usage:
    python smiles_to_svg.py "CCO"                          # single SMILES
    python smiles_to_svg.py "CCO" "c1ccccc1" -o output_dir # multiple SMILES
    python smiles_to_svg.py -f smiles.txt -o output_dir    # from file (one SMILES per line)
    python smiles_to_svg.py -c input.csv -o output_dir     # from CSV (reads 'smiles' column)
"""

import argparse
import os
import sys

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def smiles_to_svg(smiles, width=400, height=300):
    """Convert a SMILES string to SVG text."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"WARNING: Invalid SMILES: {smiles}")
        return None

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def sanitize_filename(smiles):
    """Create a safe filename from a SMILES string."""
    table = str.maketrans({
        "/": "_slash_",
        "\\": "_bslash_",
        ":": "_colon_",
        "*": "_star_",
        "?": "_q_",
        '"': "_dq_",
        "<": "_lt_",
        ">": "_gt_",
        "|": "_pipe_",
        "#": "_hash_",
        "(": "_lp_",
        ")": "_rp_",
        "[": "_lb_",
        "]": "_rb_",
        "=": "_eq_",
        "+": "_plus_",
        "@": "_at_",
    })
    name = smiles.translate(table)
    # Truncate if too long
    if len(name) > 100:
        name = name[:100]
    return name


def main():
    parser = argparse.ArgumentParser(description="Generate SVG molecular images from SMILES")
    parser.add_argument("smiles", nargs="*", help="SMILES strings")
    parser.add_argument("-f", "--file", help="Text file with one SMILES per line")
    parser.add_argument("-c", "--csv", help="CSV file with a 'smiles' column")
    parser.add_argument("-o", "--output", default="mol_svg", help="Output directory (default: mol_svg)")
    parser.add_argument("-W", "--width", type=int, default=400, help="SVG width in pixels (default: 400)")
    parser.add_argument("-H", "--height", type=int, default=300, help="SVG height in pixels (default: 300)")
    args = parser.parse_args()

    smiles_list = list(args.smiles)

    # Read from text file
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    smiles_list.append(s)

    # Read from CSV
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        col = None
        for c in ["smiles", "SMILES", "Smiles", "Drug", "drug"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            print(f"ERROR: No SMILES column found in {args.csv}. Available columns: {list(df.columns)}")
            sys.exit(1)
        smiles_list.extend(df[col].dropna().tolist())

    if not smiles_list:
        parser.print_help()
        print("\nERROR: No SMILES provided. Pass them as arguments, or use -f / -c.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    success = 0
    for i, smi in enumerate(smiles_list):
        svg_text = smiles_to_svg(smi, args.width, args.height)
        if svg_text is None:
            continue
        filename = f"{i}_{sanitize_filename(smi)}.svg"
        filepath = os.path.join(args.output, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_text)
        success += 1
        print(f"[{success}] {smi} -> {filepath}")

    print(f"\nDone. {success}/{len(smiles_list)} SVG files saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
