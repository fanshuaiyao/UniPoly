"""
BBBP 训练后化学空间可视化 (真实训练版)
使用简单的神经网络进行训练，提取学习到的特征表示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import warnings
import os

warnings.filterwarnings('ignore')

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch 未安装，将无法使用真实训练功能")
    print("安装命令: pip install torch")

# --- 配置项 ---
CSV_PATH = "/home/fsy23/UniPoly/tsne_visualization/BBBP.csv"
SAVE_PATH = "/home/fsy23/UniPoly/tsne_visualization/bbbp_trained_real.png"

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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

class BBBPClassifier(nn.Module):
    """简单的神经网络分类器，包含特征提取层"""
    def __init__(self, input_dim=2048, hidden_dim=256, feature_dim=128):
        super(BBBPClassifier, self).__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """提取中间层特征用于可视化"""
        return self.feature_extractor(x)

def train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
    """训练神经网络模型"""
    print("正在训练神经网络...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = BBBPClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] '
                  f'Train Loss: {train_loss/len(train_loader):.4f} '
                  f'Train Acc: {train_acc:.2f}% '
                  f'Val Acc: {val_acc:.2f}%')
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    return model

def extract_learned_features(model, features):
    """使用训练好的模型提取特征"""
    print("正在提取学习到的特征...")
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(device)
        learned_features = model.extract_features(features_tensor)
        learned_features = learned_features.cpu().numpy()
    
    return learned_features

def plot_tsne(features, labels, save_path, title="BBBP Chemical Space (Trained)"):
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
    if not TORCH_AVAILABLE:
        print("\n请先安装 PyTorch: pip install torch")
        print("或使用模拟版本: visualize_trained_simulated.py")
        exit(1)
    
    X, y = load_and_process_data(CSV_PATH)
    
    if X is not None and len(X) > 0:
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        model = train_model(X_train, y_train, X_val, y_val, epochs=20)
        
        # 提取学习到的特征
        X_learned = extract_learned_features(model, X_scaled)
        
        # 可视化
        plot_tsne(X_learned, y, SAVE_PATH)
    else:
        print("没有数据可绘图，请检查 CSV 路径或内容。")
