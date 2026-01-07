import argparse
import csv
import os

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Crippen, Descriptors, Fragments, rdMolDescriptors, rdmolops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a large CSV with RDKit-generated descriptions (chunked processing)."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV path.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--smiles_col",
        default="canonical_smiles",
        help='Preferred SMILES column; falls back to "smiles" if missing.',
    )
    parser.add_argument("--desc_col", default="Description", help='Original description column name.')
    parser.add_argument("--rdkit_col", default="rdkit_description", help="New RDKit description column name.")
    parser.add_argument("--merged_col", default="merged_description", help="New merged description column name.")
    parser.add_argument("--chunksize", type=int, default=50000, help="Chunk size for pandas.read_csv.")
    parser.add_argument("--separator", default="", help="Separator between descriptions.")
    return parser.parse_args()


def _as_str_or_empty(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        # Non-scalar objects (e.g., lists) can raise; treat them as present.
        pass
    try:
        return str(value)
    except Exception:
        return ""


# 提取RDKit的分子描述----get_rdkit_description
def get_rdkit_description(smiles):
    smi = smiles.strip()
    if not smi:
        raise ValueError("empty smiles")

    # 1. 分子对象创建与合法性检查
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("invalid smiles")

    # 2. 规范化（canonicalize）并重新构建 mol，确保特征针对标准结构
    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    mol = Chem.MolFromSmiles(canonical_smi)
    if mol is None:
        raise ValueError("invalid smiles")

    # 3. 基础物理常数提取
    Formula = rdMolDescriptors.CalcMolFormula(mol)
    MolWt = round(Descriptors.MolWt(mol), 2)
    try:
        MolLogP = round(Crippen.MolLogP(mol), 2)
    except Exception:
        MolLogP = round(Descriptors.MolLogP(mol), 2)
    try:
        TPSA = round(rdMolDescriptors.CalcTPSA(mol), 2)
    except Exception:
        TPSA = round(Descriptors.TPSA(mol), 2)

    # 4. 结构特征与原子统计
    NumHDonors = rdMolDescriptors.CalcNumHBD(mol)
    NumHAcceptors = rdMolDescriptors.CalcNumHBA(mol)
    RingCount = mol.GetRingInfo().NumRings()
    NumRotatableBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    stereo_centers = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    HeavyAtoms = mol.GetNumHeavyAtoms()
    NumAromaticRings = rdMolDescriptors.CalcNumAromaticRings(mol)
    FractionCSP3 = round(Descriptors.FractionCSP3(mol), 3)
    BalabanJ = round(Descriptors.BalabanJ(mol), 3)

    # 5. 关键官能团扫描（使用 RDKit 内置的片段库）
    fg_list = []
    if Fragments.fr_benzene(mol) > 0:
        fg_list.append("aromatic benzene rings")
    if Fragments.fr_amide(mol) > 0:
        fg_list.append("amide groups")
    if Fragments.fr_Ar_OH(mol) > 0:
        fg_list.append("phenolic hydroxyls")
    if Fragments.fr_NH2(mol) > 0:
        fg_list.append("primary amine groups")
    if Fragments.fr_halogen(mol) > 0:
        fg_list.append("halogen substituents")
    if Fragments.fr_ester(mol) > 0:
        fg_list.append("ester linkages")
    if Fragments.fr_ether(mol) > 0:
        fg_list.append("ether groups")
    fg_list = sorted(set(fg_list))[:8]

    # 6. 输出：2~3 句短文本（半结构化 + 少量自然语言）
    sentence1 = f"RDKit summary: A molecule with formula {Formula} and molecular weight {MolWt} g/mol."
    sentence2 = (
        f"Key properties: logP={MolLogP}; TPSA={TPSA} Å²; HBD={NumHDonors}; HBA={NumHAcceptors}; "
        f"rotBonds={NumRotatableBonds}; rings={RingCount} (aromatic={NumAromaticRings}); "
        f"heavyAtoms={HeavyAtoms}; fracCSP3={FractionCSP3}; stereoCenters={stereo_centers}; "
        f"balabanJ={BalabanJ}."
    )
    if fg_list:
        sentence3 = f"Substructures: {', '.join(fg_list)}."
        return f"{sentence1} {sentence2} {sentence3}"
    return f"{sentence1} {sentence2}"


def main() -> int:
    args = parse_args()

    RDLogger.DisableLog("rdApp.*")

    if args.chunksize <= 0:
        raise SystemExit("--chunksize must be > 0")

    if not os.path.exists(args.input_csv):
        raise SystemExit(f"Input not found: {args.input_csv}")

    # Start fresh output, then append chunk-by-chunk (mode='a').
    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    total_rows = 0
    rdkit_ok_rows = 0
    rdkit_fail_rows = 0
    wrote_header = False

    header_row: int | None = None
    with open(args.input_csv, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 50:
                break
            normalized = {cell.strip().lstrip("\ufeff") for cell in row if isinstance(cell, str)}
            if "CID" in normalized and args.desc_col in normalized:
                header_row = i
                break
    if header_row is None:
        raise SystemExit(f'Failed to locate header row within first 50 lines (need "CID" and "{args.desc_col}")')

    reader = pd.read_csv(
        args.input_csv,
        skiprows=range(header_row),
        header=0,
        chunksize=args.chunksize,
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        original_cols = chunk.columns.tolist()

        has_fallback_smiles_col = "smiles" in chunk.columns
        has_preferred_smiles_col = args.smiles_col in chunk.columns
        if not has_preferred_smiles_col and not has_fallback_smiles_col:
            raise SystemExit(f'No usable SMILES column: "{args.smiles_col}" or "smiles"')

        smiles_series = chunk[args.smiles_col] if has_preferred_smiles_col else chunk["smiles"]
        fallback_smiles_series = chunk["smiles"] if has_preferred_smiles_col and has_fallback_smiles_col else None
        desc_series = chunk[args.desc_col] if args.desc_col in chunk.columns else pd.Series("", index=chunk.index)

        rdkit_desc_list: list[str] = []
        merged_desc_list: list[str] = []
        ok_this_chunk = 0
        fail_this_chunk = 0
        empty_smiles_this_chunk = 0

        if fallback_smiles_series is None:
            for smiles_value, desc_value in zip(smiles_series.tolist(), desc_series.tolist()):
                smiles = _as_str_or_empty(smiles_value).strip()
                if not smiles:
                    empty_smiles_this_chunk += 1
                original_desc = _as_str_or_empty(desc_value)
                try:
                    rdkit_desc = get_rdkit_description(smiles)
                    ok_this_chunk += 1
                    merged_desc = original_desc + args.separator + rdkit_desc if rdkit_desc else original_desc
                except Exception:
                    rdkit_desc = ""
                    merged_desc = original_desc
                    fail_this_chunk += 1
                rdkit_desc_list.append(rdkit_desc)
                merged_desc_list.append(merged_desc)
        else:
            for preferred_value, fallback_value, desc_value in zip(
                smiles_series.tolist(),
                fallback_smiles_series.tolist(),
                desc_series.tolist(),
            ):
                preferred_smiles = _as_str_or_empty(preferred_value).strip()
                fallback_smiles = _as_str_or_empty(fallback_value).strip()
                smiles = preferred_smiles or fallback_smiles
                if not smiles:
                    empty_smiles_this_chunk += 1
                original_desc = _as_str_or_empty(desc_value)
                try:
                    rdkit_desc = get_rdkit_description(smiles)
                    ok_this_chunk += 1
                    merged_desc = original_desc + args.separator + rdkit_desc if rdkit_desc else original_desc
                except Exception:
                    rdkit_desc = ""
                    merged_desc = original_desc
                    fail_this_chunk += 1
                rdkit_desc_list.append(rdkit_desc)
                merged_desc_list.append(merged_desc)

        chunk[args.rdkit_col] = rdkit_desc_list
        chunk[args.merged_col] = merged_desc_list
        chunk = chunk[original_cols + [args.rdkit_col, args.merged_col]]

        chunk.to_csv(
            args.output_csv,
            index=False,
            mode="a",
            header=not wrote_header,
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar="\\",
        )
        wrote_header = True

        rdkit_ok_rows += ok_this_chunk
        rdkit_fail_rows += fail_this_chunk
        print(
            f"chunk={chunk_idx} rows={len(chunk)} total={total_rows} "
            f"rdkit_ok={rdkit_ok_rows} rdkit_fail={rdkit_fail_rows} "
            f"empty_smiles_chunk={empty_smiles_this_chunk} rdkit_fail_chunk={fail_this_chunk}"
        )

    print(f"Done. total={total_rows} rdkit_ok={rdkit_ok_rows} rdkit_fail={rdkit_fail_rows}")
    print(f"output_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
