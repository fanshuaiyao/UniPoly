import os
# proxy_url = "http://127.0.0.1:7890" 
# os.environ["http_proxy"] = proxy_url
# os.environ["https_proxy"] = proxy_url
import sys
import requests
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, AllChem, rdMolDescriptors

# smiles规范化-----get_canonical
def get_canonical(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                can_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                print(f"规范化结果: {can_smi}")
            else:
                print("错误: 无法解析该 SMILES，请检查输入是否正确。")
        except Exception as e:
            print(f"程序运行出错: {e}")


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
#     description = (
#     f"{smiles}：This molecule, defined by the chemical formula {Formula}, possesses a molecular weight of {MolWt} g/mol. "
#     f"{fg_text}"
#     f"Topological analysis indicates the presence of {RingCount} ring system(s), including {NumAromaticRings} aromatic ring(s), "
#     f"with a connectivity complexity quantified by a Balaban J index of {BalabanJ}. "
#     f"The stereochemical configuration is defined by {stereo_centers} stereogenic center(s)). "
#     f"From a physicochemical perspective, the molecule exhibits a LogP of {MolLogP} and a TPSA of {TPSA} Å², key parameters governing its lipophilicity and polar surface interactions. "
#     f"It contains {HeavyAtoms} heavy atom(s), reflecting the size of its non-hydrogen atomic framework. "
#     f"The molecular flexibility is moderated by {NumRotatableBonds} rotatable bonds and a FractionCSP3 of {FractionCSP3}, "
#     f"while its interaction profile is defined by {NumHDonors} hydrogen bond donor(s) and {NumHAcceptors} acceptor(s)."
# )
    description = (
    f"RDKit summary: Formula={Formula}; MolWt={MolWt}; LogP={MolLogP}; TPSA={TPSA}. "
    f"Rings={RingCount} (aromatic={NumAromaticRings}); RotBonds={NumRotatableBonds}; "
    f"StereoCenters={stereo_centers}; HeavyAtoms={HeavyAtoms}; FractionCSP3={FractionCSP3}; BalabanJ={BalabanJ}. "
    f"Functional groups: {', '.join(fg_list) if fg_list else 'none'}. "
    f"H-bond: donors={NumHDonors}; acceptors={NumHAcceptors}."
)
    return description  # 返回最终生成的知识文本


# 纯净的理化性质----get_all_rdkit_properties
def get_all_rdkit_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    # 1. 基础识别
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = round(Descriptors.MolWt(mol), 4)
    exact_mw = round(Descriptors.ExactMolWt(mol), 4)
    
    # 2. 理化性质
    logp = round(Descriptors.MolLogP(mol), 4)
    tpsa = round(Descriptors.TPSA(mol), 4)
    
    # 3. 结构计数
    heavy_atoms = mol.GetNumHeavyAtoms()
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    # 4. 环与拓扑
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    stereo_centers = rdMolDescriptors.CalcNumAtomStereoCenters(mol)

    return {
        "Formula": formula, "MolWt": mw, "ExactMolWt": exact_mw,
        "LogP": logp, "TPSA": tpsa, "Heavy_Atoms": heavy_atoms,
        "H_Bond_Donors": hbd, "H_Bond_Acceptors": hba, "Rotatable_Bonds": rot_bonds,
        "Ring_Count": rings, "Aromatic_Rings": aromatic_rings,
        "Stereocenters": stereo_centers
    }


if __name__ == "__main__":
    # smi = "CC1=C(C(=CC=C1)C)OCC(C)N"
    # smi = "CC(=O)OC1=CC=CC=C1C(=O)O"
    smi = "CCCNC(=O)NS(=O)(=O)c1ccc(Cl)cc1"

    rdkit = get_rdkit_description(smi)
    pubchem = get_pubchem_description(smi)
    res = res = rdkit + "\n" + pubchem
    print(res)