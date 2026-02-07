"""
BBBP 未训练化学空间可视化
使用原始 Morgan 指纹进行 t-SNE 降维
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import warnings
import os

warnings.filterwarnings('ignore')

# --- 配置项 ---
CSV_PATH = "/home/fsy23/UniPoly/tsne_visualization/BBBP.csv"  # 请根据实际路径修改
SAVE_PATH = "/home/fsy23/UniPoly/tsne_visualization/bbbp_untrained.png"

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def get_morgan_fingerprint(smiles, n_bits=2048, radius=2):
    """生成 Morgan 指纹 (ECFP4)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None

def load_and_process_data(csv_path):
    """加载数据并计算指纹"""
    print(f"正在读取数据: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在 {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)
    
    label_col = 'p_np'
    if label_col not in df.columns:
        if 'label' in df.columns:
            label_col = 'label'
        else:
            label_col = df.columns[-1]
    
    smiles_list = df['smiles'].tolist()
    labels_raw = df[label_col].values
    
    print("正在计算 Morgan 指纹...")
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

def plot_tsne(features, labels, save_path, title="BBBP Chemical Space (Untrained)"):
    """t-SNE 降维并绘图"""
    print("正在进行 t-SNE 降维...")
    
    tsne = TSNE(n_components=2, 
                random_state=42,
                perplexity=40,
                init='pca',
                learning_rate='auto',
                verbose=1)
    
    embeddings_2d = tsne.fit_transform(features)
    
    # 绘图
    plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.gca()
    
    colors = {0: '#FF9F43', 1: '#2E86DE'}
    label_names = {0: 'Negative (Non-BBB)', 1: 'Positive (BBB)'}
    
    for label_val in np.unique(labels):
        mask = labels == label_val
        c = colors.get(label_val, plt.cm.tab10(label_val % 10))
        l_name = label_names.get(label_val, str(label_val))
        
        ax.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=c,
            label=l_name,
            alpha=0.7,
            s=25,
            edgecolors='none'
        )
    
    plt.title(title, fontsize=16, fontweight='bold', y=1.02)
    plt.legend(loc='upper right', framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    print(f"\n可视化完成！图像已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    X, y = load_and_process_data(CSV_PATH)
    
    if X is not None and len(X) > 0:
        plot_tsne(X, y, SAVE_PATH)
    else:
        print("没有数据可绘图，请检查 CSV 路径或内容。")
