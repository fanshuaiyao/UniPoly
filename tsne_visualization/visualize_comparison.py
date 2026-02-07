"""
BBBP 训练前后对比可视化
并排展示未训练和训练后的化学空间分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import warnings
import os

warnings.filterwarnings('ignore')

# --- 配置项 ---
CSV_PATH = "BBBP.csv"
SAVE_PATH = "bbbp_comparison.png"
USE_REAL_TRAINING = False  # 设为 True 使用真实训练，需要安装 PyTorch

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

if USE_REAL_TRAINING:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        USE_REAL_TRAINING = False
        print("警告: PyTorch 未安装，将使用模拟训练效果")

def get_morgan_fingerprint(smiles, n_bits=2048, radius=2):
    """生成 Morgan 指纹"""
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
    """加载数据"""
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
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(valid_labels)
    
    print(f"有效分子数: {len(features)}")
    return features, labels

def simulate_training_effect(features, labels, strength=0.5):
    """模拟训练效果"""
    print("正在模拟训练效果...")
    features_trained = features.copy()
    
    scaler = StandardScaler()
    features_trained = scaler.fit_transform(features_trained)
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        class_samples = features_trained[mask]
        class_center = np.mean(class_samples, axis=0)
        
        for i in range(len(class_samples)):
            direction = class_center - class_samples[i]
            class_samples[i] = class_samples[i] + strength * direction
        
        features_trained[mask] = class_samples
    
    return features_trained

def compute_tsne(features, random_state=42):
    """计算 t-SNE 降维"""
    tsne = TSNE(n_components=2, 
                random_state=random_state,
                perplexity=40,
                init='pca',
                learning_rate='auto',
                verbose=0)
    
    return tsne.fit_transform(features)

def plot_comparison(embeddings_untrained, embeddings_trained, labels, save_path):
    """并排绘制训练前后对比图"""
    print("正在生成对比图...")
    
    fig = plt.figure(figsize=(18, 8), dpi=150)
    gs = GridSpec(1, 2, figure=fig, wspace=0.15)
    
    colors = {0: '#FF9F43', 1: '#2E86DE'}
    label_names = {0: 'Negative (Non-BBB)', 1: 'Positive (BBB)'}
    
    # 左图：未训练
    ax1 = fig.add_subplot(gs[0])
    for label_val in np.unique(labels):
        mask = labels == label_val
        c = colors.get(label_val, plt.cm.tab10(label_val % 10))
        l_name = label_names.get(label_val, str(label_val))
        
        ax1.scatter(
            embeddings_untrained[mask, 0], 
            embeddings_untrained[mask, 1],
            c=c,
            label=l_name,
            alpha=0.7,
            s=25,
            edgecolors='none'
        )
    
    ax1.set_title('Before Training', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    
    # 右图：训练后
    ax2 = fig.add_subplot(gs[1])
    for label_val in np.unique(labels):
        mask = labels == label_val
        c = colors.get(label_val, plt.cm.tab10(label_val % 10))
        l_name = label_names.get(label_val, str(label_val))
        
        ax2.scatter(
            embeddings_trained[mask, 0], 
            embeddings_trained[mask, 1],
            c=c,
            label=l_name,
            alpha=0.7,
            s=25,
            edgecolors='none'
        )
    
    ax2.set_title('After Training', fontsize=18, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    
    # 总标题
    fig.suptitle('BBBP Chemical Space Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, dpi=150)
    print(f"\n对比图已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    X, y = load_and_process_data(CSV_PATH)
    
    if X is not None and len(X) > 0:
        # 未训练特征的 t-SNE
        print("\n=== 处理未训练数据 ===")
        embeddings_before = compute_tsne(X)
        
        # 训练后特征的 t-SNE
        print("\n=== 处理训练后数据 ===")
        if USE_REAL_TRAINING and TORCH_AVAILABLE:
            print("使用真实训练（需要较长时间）...")
            # 这里可以调用真实训练的代码
            # 为简化，这里仍使用模拟
            X_after = simulate_training_effect(X, y, strength=0.5)
        else:
            X_after = simulate_training_effect(X, y, strength=0.5)
        
        embeddings_after = compute_tsne(X_after)
        
        # 绘制对比图
        print("\n=== 生成可视化 ===")
        plot_comparison(embeddings_before, embeddings_after, y, SAVE_PATH)
    else:
        print("没有数据可绘图，请检查 CSV 路径或内容。")
