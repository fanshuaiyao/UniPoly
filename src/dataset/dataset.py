import os
import json
import torch
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Dataset
from .geom_data import mol2coords
from .graph_data import mol_to_graph_data_obj_simple
from transformers import AutoTokenizer
from rdkit.Chem import rdFingerprintGenerator
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np
from torch_geometric.data import Data
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


class UniDataset(Dataset):
    def __init__(self, root, dataset, smiles_model_name, text_model_name, task_type="pretrain", transform=None, pre_transform=None):
        # 基础配置
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        # 载入 SMILES -> 文本描述映射
        self.text_dict = json.load(open('./data/smiles_all_dict.json', 'r'))
        # 处理后的 Data 列表
        self.data_list = []
        # 初始化分词器
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.task_type = task_type
        
        # 处理后数据的缓存路径
        processed_dir = os.path.join(self.root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        self.processed_file = os.path.join(processed_dir, f"{self.dataset}.pt")
        
        if os.path.exists(self.processed_file):
            # 存在缓存则直接加载
            self.data_list = torch.load(self.processed_file)
            print(f"数据从缓存加载：loaded {self.processed_file} with {len(self.data_list)} samples")  # Loaded the processed dataset from ...
            
        else:
            # 否则重新处理并保存
            # 先统计最大 token 长度用于 padding
            csv_path = f"{self.root}/raw/{self.dataset}.csv"
            self.max_length_smiles, self.max_length_text = self.get_max_token_length(csv_path, self.smiles_tokenizer, self.text_tokenizer,self.text_dict)
            
            self.new_process()
            torch.save(self.data_list, self.processed_file)
            print(f"processed and saved {self.processed_file} with {len(self.data_list)} samples")  # Processed and saved the dataset to ...

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


    def get_max_token_length(self, csv_path, smiles_tokenizer, text_tokenizer,text_dict):
        # 扫描全量数据，确定最长 SMILES/文本长度
        df = pd.read_csv(csv_path)
        max_smiles_length = 0
        max_text_length = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking token lengths"):
            # 假设 SMILES 在第一列
            smiles = row["smiles"]
            
            # SMILES 分词
            smiles_tokens = self.smiles_tokenizer.encode(smiles)
            max_smiles_length = max(max_smiles_length, len(smiles_tokens))
            
            # 文本分词
            input_text = self.text_dict[smiles] if smiles in self.text_dict else ''
            text_tokens = self.text_tokenizer.encode(input_text)
            max_text_length = max(max_text_length, len(text_tokens))
        
        print(f"Max SMILES token length: {max_smiles_length}")
        print(f"Max text token length: {max_text_length}")
        
        return max_smiles_length, max_text_length


    def old_process(self):
        # 将 CSV 转成 PyG Data 列表
        csv_path = f"{self.root}/raw/{self.dataset}.csv"
        df = pd.read_csv(csv_path)
        # Morgan 指纹生成器
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
            # 假设：第 1 列是 SMILES，第 2 列是标签
            smiles = row["smiles"]
            property = row["property"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            # 将通配符原子替换为氢
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
            
            # 构建 2D 图结构
            data = mol_to_graph_data_obj_simple(mol)
            
            # 标签
            data.y = torch.tensor([property], dtype=torch.float)
            data.smiles = smiles

            # SMILES 分词并 padding
            tokenizer_output = self.smiles_tokenizer(
                smiles,
                return_tensors='pt',
                max_length=self.max_length_smiles+5,
                padding='max_length',
                truncation=True
            )
            data.input_ids_smiles = tokenizer_output.input_ids
            data.attention_mask_smiles = tokenizer_output.attention_mask

            # 文本描述分词
            data.text = self.text_dict[smiles] if smiles in self.text_dict else ''
            tokenizer_output = self.text_tokenizer(
                data.text,
                return_tensors='pt',
                max_length=self.max_length_text+5,
                padding='max_length',
                truncation=True
            )
            data.input_ids_text = tokenizer_output.input_ids
            data.attention_mask_text = tokenizer_output.attention_mask
            # 指纹特征
            data.fp = torch.tensor(mfpgen.GetFingerprint(mol), dtype=torch.float).unsqueeze(0)
            try:
                # 生成 3D 坐标与原子序数
                mol = Chem.MolFromSmiles(smiles)
                data.pos,data.z = mol2coords(mol)
            except Exception as e:
                print(e)
                print(f"Failed to generate 3D coordinates for {smiles}")
                continue
            # 保存样本
            self.data_list.append(data)

        # 输出处理样本数
        print("Dataset processed. Total samples:", len(self.data_list))


    def new_process(self):
        csv_path = f"{self.root}/raw/{self.dataset}.csv"
        df = pd.read_csv(csv_path)

        # === 1) 自动选择 smiles 列
        smiles_col = "smiles"
        if smiles_col not in df.columns:
            raise ValueError(f"CSV 中找不到 SMILES 列：canonical_smiles / smiles 均不存在。现有列：{list(df.columns)}")

        # === 2) 下游任务才需要 label；预训练不需要 ===
        # 如果你还没在 __init__ 里加 task_type / label_col，这里也能跑：
        task_type = getattr(self, "task_type", "pretrain")  # 默认预训练
        label_col = getattr(self, "label_col", None)        # 默认 None

        if task_type == "downstream":
            if not label_col:
                # 兼容你现在的写法：默认用 property
                label_col = "property"
            if label_col not in df.columns:
                raise ValueError(f"downstream 模式需要 label_col={label_col}，但 CSV 没有该列。现有列：{list(df.columns)}")

        # === 3) Morgan 指纹生成器 ===
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

        self.data_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):

            # --- 3.1 读 smiles，并强制转换成干净字符串 ---
            smi_raw = row[smiles_col]
            if pd.isna(smi_raw):
                continue
            smiles = str(smi_raw).strip()
            if not smiles:
                continue

            # --- 3.2 解析 mol（这里就能过滤掉非法 smiles） ---
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # 通配符原子替换为氢
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)

            # --- 3.3 构图 ---
            data = mol_to_graph_data_obj_simple(mol)

            # --- 3.4 标签：只有 downstream 才设置 data.y ---
            if task_type == "downstream":
                y_raw = row[label_col]
                if pd.isna(y_raw):
                    continue
                try:
                    y = float(y_raw)
                except Exception:
                    continue
                data.y = torch.tensor([y], dtype=torch.float)

            # --- 3.5 保存 smiles
            data.smiles = smiles

            # --- 3.6 SMILES tokenizer
            tok_smi = self.smiles_tokenizer(
                smiles,
                return_tensors="pt",
                max_length=min(self.max_length_smiles + 5, 512),  # 你如果要固定512，可以直接写512
                padding="max_length",
                truncation=True,
            )
            data.input_ids_smiles = tok_smi.input_ids
            data.attention_mask_smiles = tok_smi.attention_mask

            # --- 3.7 文本描述
            text = ""
            if isinstance(self.text_dict, dict):
                text = self.text_dict.get(smiles, "")
            data.text = text

            tok_txt = self.text_tokenizer(
                text,
                return_tensors="pt",
                max_length=min(self.max_length_text + 5, 512),  # 同理可固定512
                padding="max_length",
                truncation=True,
            )
            data.input_ids_text = tok_txt.input_ids
            data.attention_mask_text = tok_txt.attention_mask

            # --- 3.8 指纹 ---
            fp = mfpgen.GetFingerprint(mol)
            data.fp = torch.tensor(fp, dtype=torch.float).unsqueeze(0)

            # --- 3D：只有 geom 模态才算 ---
            use_geom = "geom" in getattr(self, "modalities", [])
            if use_geom:
                try:
                    mol3d = Chem.MolFromSmiles(smiles)
                    mol3d = Chem.AddHs(mol3d)     # 可选，但推荐
                    data.pos, data.z = mol2coords(mol3d)
                except Exception:
                    continue   # geom 失败 → 跳样本（合理）

            self.data_list.append(data)

        print("Dataset processed. Total samples:", len(self.data_list))
