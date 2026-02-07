# BBBP t-SNE 可视化工具

本项目提供了多种方式来可视化 BBBP 数据集的化学空间分布，包括训练前后的对比。

## 文件说明

### 1. visualize_untrained.py
展示**未训练**的化学空间分布（使用原始 Morgan 指纹）

### 2. visualize_trained_simulated.py  ⭐ 推荐
**模拟训练效果**，无需真实训练模型即可展示训练后的聚类效果
- 通过将同类样本向类中心靠拢来模拟学习效果
- 快速生成，适合演示和对比

### 3. visualize_trained_real.py
使用**真实神经网络训练**，提取学习到的特征进行可视化
- 需要安装 PyTorch
- 训练时间较长但效果真实

### 4. visualize_comparison.py  ⭐⭐ 强烈推荐
**并排对比**训练前后的化学空间分布
- 一张图同时展示训练前后的效果
- 默认使用模拟训练（快速），可切换为真实训练

## 使用方法

### 准备数据
将 `BBBP.csv` 文件放在与脚本相同的目录下，或修改脚本中的 `CSV_PATH` 变量。

### 安装依赖
```bash
pip install pandas numpy matplotlib scikit-learn rdkit
```

如需使用真实训练（可选）:
```bash
pip install torch
```

### 运行脚本

**快速开始 - 生成对比图**（推荐）:
```bash
python visualize_comparison.py
```

**单独生成各个图**:
```bash
# 未训练分布
python visualize_untrained.py

# 模拟训练后分布
python visualize_trained_simulated.py

# 真实训练后分布（需要 PyTorch）
python visualize_trained_real.py
```

## 输出文件
- `bbbp_untrained.png` - 未训练的化学空间
- `bbbp_trained_simulated.png` - 模拟训练后的化学空间
- `bbbp_trained_real.png` - 真实训练后的化学空间
- `bbbp_comparison.png` - 训练前后对比图

## 自定义参数

### 调整模拟训练强度
在 `visualize_trained_simulated.py` 或 `visualize_comparison.py` 中修改:
```python
X_trained = simulate_training_effect(X, y, strength=0.5)  # 0-1 之间
```

### 调整 t-SNE 参数
```python
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=40,        # 调整困惑度 (5-50)
    learning_rate='auto'
)
```

## 配色方案
- 橙色 (#FF9F43): Negative (Non-BBB)
- 蓝色 (#2E86DE): Positive (BBB)

## 注意事项
- 确保 CSV 文件包含 `smiles` 列和标签列 (`p_np` 或 `label`)
- t-SNE 降维可能需要几秒到几分钟，取决于数据量
- 真实训练版本需要 5-10 分钟完成训练
