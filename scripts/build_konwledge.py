"""
输入一个包含 smiles 列的 CSV，为其生成三列“知识”并输出到新 CSV（支持断点续跑）。

新增三列：
1) rdkit_desc   : RDKit 生成的理化/结构描述
2) pubchem_desc : PubChem 抓取的专家库知识（需要联网）
3) description  : rdkit_desc 与 pubchem_desc 拼接（用 \\n 连接）

防中断/续跑：
- 默认输出到 <input>_with_knowledge.csv
- 每处理 save_every（默认 1000）条就 flush+fsync 落盘，并写 progress 文件
- 中断后重复运行同样命令，会根据输出文件已写入行数自动跳过并继续追加
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import requests
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors


def _try_import_tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        tqdm = None
    return tqdm

# 提取RDKit的分子描述----get_rdkit_description
def get_rdkit_description(smiles):
    # 1. 分子对象创建与合法性检查
    mol = Chem.MolFromSmiles(smiles) 
    if not mol:  
        return "Error: Invalid SMILES string."  # 返回错误提示并终止

    # 为了确保输出的描述是针对标准结构的，进行规范化处理
    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)  # 生成标准且含手性的 SMILES
    mol = Chem.MolFromSmiles(canonical_smi)  # 重新加载标准分子对象以保证计算精度

    # 2. 基础物理常数提取
    Formula = rdMolDescriptors.CalcMolFormula(mol)  # 计算分子的化学式（如 C27H33N3O2）
    MolWt = round(Descriptors.MolWt(mol), 2)  # 计算精确分子量，保留两位小数
    MolLogP = round(Descriptors.MolLogP(mol), 2)  # 计算脂水分配系数 LogP（反映亲脂性）
    TPSA = round(Descriptors.TPSA(mol), 2)  # 计算总极性表面积 TPSA（反映渗透性）

    # 3. 结构特征与原子统计
    NumHDonors = Descriptors.NumHDonors(mol)  # 统计氢键供体数量
    NumHAcceptors = Descriptors.NumHAcceptors(mol)  # 统计氢键受体数量
    RingCount = mol.GetRingInfo().NumRings()  # 统计分子中环的总数
    NumRotatableBonds = Descriptors.NumRotatableBonds(mol)  # 统计可旋转化学键数量（反映分子柔性）
    stereo_centers = rdMolDescriptors.CalcNumAtomStereoCenters(mol)  # 统计手性中心（立体中心）的数量
    HeavyAtoms = mol.GetNumHeavyAtoms()
    # 获取芳香环数量 (NumAromaticRings)
    NumAromaticRings = Descriptors.NumAromaticRings(mol)
    
    # 获取 sp3 杂化碳原子比例 (FractionCSP3)
    # 这个值反映分子的立体程度，使用 round 保留两位小数
    FractionCSP3 = round(Descriptors.FractionCSP3(mol), 2)
    
    # 获取 Balaban J 指数 (BalabanJ)
    # 这是一个描述分子拓扑复杂性的指数
    BalabanJ = round(Descriptors.BalabanJ(mol), 2)

    # 4. 关键官能团扫描（使用 RDKit 内置的片段库）
    fg_list = []  # 创建一个列表用于存放检测到的官能团
    if Fragments.fr_benzene(mol) > 0: fg_list.append("aromatic benzene rings")  # 检测苯环
    if Fragments.fr_amide(mol) > 0: fg_list.append("amide groups")  # 检测酰胺键
    if Fragments.fr_Ar_OH(mol) > 0: fg_list.append("phenolic hydroxyls")  # 检测酚羟基
    if Fragments.fr_NH2(mol) > 0: fg_list.append("primary amine groups")  # 检测伯胺
    if Fragments.fr_halogen(mol) > 0: fg_list.append("halogen substituents")  # 检测卤素（F, Cl, Br, I）
    if Fragments.fr_ester(mol) > 0: fg_list.append("ester linkages")  # 检测酯基
    if Fragments.fr_ether(mol) > 0: fg_list.append("ether groups")  # 检测醚键
    
    # 构造官能团描述短语
    if fg_list:  # 如果找到了已知官能团
        fg_text = "The structural framework incorporates " + ", ".join(fg_list) + ". "  # 拼接官能团描述词
    else:  # 如果没找到常见官能团
        fg_text = "The structure consists of a complex hydrocarbon-based framework. "  # 使用通用描述

    # 5. 最终描述文本生成
    description = (
    f"{smiles}：This molecule, defined by the chemical formula {Formula}, possesses a molecular weight of {MolWt} g/mol. "
    f"{fg_text}"
    f"Topological analysis indicates the presence of {RingCount} ring system(s), including {NumAromaticRings} aromatic ring(s), "
    f"with a connectivity complexity quantified by a Balaban J index of {BalabanJ}. "
    f"The stereochemical configuration is defined by {stereo_centers} stereogenic center(s)). "
    f"From a physicochemical perspective, the molecule exhibits a LogP of {MolLogP} and a TPSA of {TPSA} Å², key parameters governing its lipophilicity and polar surface interactions. "
    f"It contains {HeavyAtoms} heavy atom(s), reflecting the size of its non-hydrogen atomic framework. "
    f"The molecular flexibility is moderated by {NumRotatableBonds} rotatable bonds and a FractionCSP3 of {FractionCSP3}, "
    f"while its interaction profile is defined by {NumHDonors} hydrogen bond donor(s) and {NumHAcceptors} acceptor(s)."
)
    return description  # 返回最终生成的知识文本


# 提取PubChem的分子描述-----get_pubchem_description
def get_pubchem_description(smiles):
    """
    根据用户提供的 JSON 结构，精准提取 Names and Identifiers 下的 Record Description
    """
    try:
        # 第一步：获取 CID
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        if not compounds: return "CID not found."
        cid = compounds[0].cid
        
        # 第二步：获取 PUG View JSON
        pug_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        response = requests.get(pug_url, timeout=15)
        if response.status_code != 200: return "API Error."
        
        data = response.json()
        final_descriptions = []

        # 第三步：按照目标路径精准导航
        # 路径: Record -> Section
        sections = data.get("Record", {}).get("Section", [])
        
        for sec in sections:
            # 找到一级目录: Names and Identifiers
            if sec.get("TOCHeading") == "Names and Identifiers":
                sub_sections = sec.get("Section", [])
                
                for sub_sec in sub_sections:
                    # 找到二级目录: Record Description
                    if sub_sec.get("TOCHeading") == "Record Description":
                        informations = sub_sec.get("Information", [])
                        
                        for info in informations:
                            # 提取 StringWithMarkup 里的所有文本
                            value = info.get("Value", {})
                            markup_list = value.get("StringWithMarkup", [])
                            
                            for markup in markup_list:
                                text = markup.get("String", "")
                                if text:
                                    final_descriptions.append(text.strip())

        # 第四步：拼接所有找到的段落
        if not final_descriptions:
            return ""
            
        return "".join(final_descriptions)

    except Exception as e:
        return f"Error: {str(e)}"
    
def _combine(rdkit_desc: str, pubchem_desc: str) -> str:
    parts = [s.strip() for s in (rdkit_desc, pubchem_desc) if s and str(s).strip()]
    return "\n".join(parts)


def _default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(input_csv.stem + "_with_knowledge.csv")


def _count_data_rows(csv_path: Path) -> int:
    # 统计除表头外的数据行数，用于断点续跑
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _write_progress(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_knowledge_csv(
    input_csv: Path,
    output_csv: Path,
    smiles_col: str = "smiles",
    save_every: int = 1000,
    resume: bool = True,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    already_done = 0
    mode = "w"
    if resume and output_csv.exists():
        already_done = _count_data_rows(output_csv)
        mode = "a"

    progress_path = output_csv.with_suffix(output_csv.suffix + ".progress.json")
    tqdm = _try_import_tqdm()

    # 简单缓存：同一 SMILES 不重复计算/请求（PubChem 很慢）
    rdkit_cache: Dict[str, str] = {}
    pubchem_cache: Dict[str, str] = {}

    with input_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        if smiles_col not in reader.fieldnames:
            raise ValueError(f"Input CSV missing column: {smiles_col}")

        out_fields = list(reader.fieldnames) + ["rdkit_desc", "pubchem_desc", "description"]

        with output_csv.open(mode, encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            if mode == "w":
                writer.writeheader()

            # 跳过已完成的行，做到“接着跑”
            for _ in range(already_done):
                try:
                    next(reader)
                except StopIteration:
                    print(f"Nothing to do: {output_csv} already has all rows.")
                    return

            processed_since_save = 0
            start_time = time.time()

            iterable = reader
            if tqdm is not None:
                iterable = tqdm(reader, desc=f"Building knowledge for {input_csv.name}", unit="row")

            for idx, row in enumerate(iterable, start=already_done + 1):
                smi = (row.get(smiles_col) or "").strip()

                rdkit_desc = rdkit_cache.get(smi)
                if rdkit_desc is None:
                    rdkit_desc = get_rdkit_description(smi)
                    rdkit_cache[smi] = rdkit_desc

                pubchem_desc = pubchem_cache.get(smi)
                if pubchem_desc is None:
                    pubchem_desc = get_pubchem_description(smi)
                    pubchem_cache[smi] = pubchem_desc

                row["rdkit_desc"] = rdkit_desc
                row["pubchem_desc"] = pubchem_desc
                row["description"] = _combine(rdkit_desc, pubchem_desc)

                writer.writerow(row)
                processed_since_save += 1

                # 处理到一定数量就强制落盘（防中断）
                if save_every > 0 and processed_since_save >= save_every:
                    fout.flush()
                    os.fsync(fout.fileno())
                    processed_since_save = 0
                    _write_progress(
                        progress_path,
                        {
                            "input_csv": str(input_csv),
                            "output_csv": str(output_csv),
                            "smiles_col": smiles_col,
                            "rows_written": _count_data_rows(output_csv),
                            "last_index_1based": idx,
                            "last_smiles": smi,
                            "elapsed_sec": round(time.time() - start_time, 2),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    )

            # 最后再落一次
            fout.flush()
            os.fsync(fout.fileno())
            _write_progress(
                progress_path,
                {
                    "input_csv": str(input_csv),
                    "output_csv": str(output_csv),
                    "smiles_col": smiles_col,
                    "rows_written": _count_data_rows(output_csv),
                    "done": True,
                    "elapsed_sec": round(time.time() - start_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

    print(f"Done. Output: {output_csv}")
    print(f"Progress file: {progress_path}")
    print("中断后兜底：用同样命令再跑一次，会自动跳过已完成部分并继续追加。")


def _set_proxy(proxy: Optional[str]) -> None:
    if not proxy:
        return
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True, help="输入 CSV（必须包含 smiles 列）")
    p.add_argument("--smiles_col", default="smiles", help="SMILES 列名（默认 smiles）")
    p.add_argument("--output_csv", default=None, help="输出 CSV（默认 <input>_with_knowledge.csv）")
    p.add_argument("--save_every", type=int, default=1000, help="每处理多少条强制保存一次（默认 1000）")
    p.add_argument("--no_resume", action="store_true", help="不做断点续跑（覆盖输出文件）")
    p.add_argument("--proxy", default="http://127.0.0.1:7890", help="HTTP(S) proxy（默认 127.0.0.1:7890）")
    p.add_argument("--no_proxy", action="store_true", help="不设置代理环境变量")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_proxy(None if args.no_proxy else args.proxy)
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv) if args.output_csv else _default_output_path(input_csv)
    build_knowledge_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        smiles_col=args.smiles_col,
        save_every=args.save_every,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()


