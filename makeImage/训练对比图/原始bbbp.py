"""
BBBP 真实化学空间可视化 (修复版)
修复内容: 移除了 TSNE 初始化中的 n_iter 参数，以适配您的环境。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# --- 配置项 ---
# 请确认路径是否正确
CSV_PATH = "/home/fsy23/UniPoly/moleculenet/classification/BBBP.csv"
SAVE_PATH = "/home/fsy23/UniPoly/makeGraph/训练对比图/bbbp_chemical_space_mimic.png"

# 设置绘图风格
plt.style.use('seaborn-v0_8-white')
# 尝试设置中文字体，防止乱码（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def get_morgan_fingerprint(smiles, n_bits=2048, radius=2):
    """
    生成真实的化学特征：Morgan 指纹 (ECFP4)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 使用半径为2，2048位的指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None

def load_and_process_data(csv_path):
    print(f"正在读取数据: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在 {csv_path}")
        return None, None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return None, None
    
    # 确定标签列
    label_col = 'p_np'
    if label_col not in df.columns:
        # 如果找不到 p_np，尝试找最后一列或者名为 label 的列
        if 'label' in df.columns:
            label_col = 'label'
        else:
            label_col = df.columns[-1]
        print(f"未找到 'p_np'，使用列 '{label_col}' 作为标签")

    smiles_list = df['smiles'].tolist()
    labels_raw = df[label_col].values
    
    print("正在计算真实化学指纹 (Morgan Fingerprints)...")
    features = []
    valid_labels = []
    
    for i, smi in enumerate(smiles_list):
        fp = get_morgan_fingerprint(smi)
        if fp is not None:
            features.append(fp)
            valid_labels.append(labels_raw[i])
            
    features = np.array(features)
    labels = np.array(valid_labels)
    
    print(f"有效分子数: {len(features)}")
    return features, labels

def plot_chemical_space_mimic(features, labels, save_path):
    print("正在进行 t-SNE 降维 (这可能需要几秒钟)...")
    
    # --- 修复点：移除了 n_iter 参数 ---
    tsne = TSNE(n_components=2, 
                random_state=42,  # 固定种子
                perplexity=40,    # 困惑度
                init='pca',       # 初始化方式
                learning_rate='auto',
                verbose=1)
    
    try:
        embeddings_2d = tsne.fit_transform(features)
    except Exception as e:
        print(f"t-SNE 降维失败: {e}")
        # 如果 init='pca' 也报错，尝试最简配置
        print("尝试使用最简 t-SNE 配置重试...")
        tsne_simple = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne_simple.fit_transform(features)

    # --- 开始绘图 ---
    plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.gca()
    
    # 颜色：0 (负样本/橙色), 1 (正样本/蓝色)
    colors = {0: '#FF9F43', 1: '#2E86DE'}
    label_names = {0: 'Negative', 1: 'Positive'}

    unique_labels = np.unique(labels)
    
    for label_val in unique_labels:
        mask = labels == label_val
        # 处理可能的非 0/1 标签
        c = colors.get(label_val, plt.cm.tab10(label_val % 10))
        l_name = label_names.get(label_val, str(label_val))
        
        ax.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=c,
            label=l_name,
            alpha=0.7,        # 透明度
            s=25,             # 点大小
            edgecolors='none' # 无边框
        )

    plt.title("BBBP Chemical Space", fontsize=16, fontweight='bold', y=1.02)

    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off') # 完全关闭坐标轴

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"\n可视化完成！图像已保存至: {save_path}")

if __name__ == "__main__":
    X_real, y_real = load_and_process_data(CSV_PATH)
    
    if X_real is not None and len(X_real) > 0:
        plot_chemical_space_mimic(X_real, y_real, SAVE_PATH)
    else:
        print("没有数据可绘图，请检查 CSV 路径或内容。")