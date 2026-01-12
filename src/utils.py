import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from src.dataset.dataloader import make_custom_collate


def scale_targets(dataset, task):
    """
    对数据集的目标值（性质标签）进行标准化处理
    
    功能说明:
    - 对于某些任务（er, iv），先进行log10变换以处理偏态分布
    - 使用StandardScaler对所有目标值进行标准化（均值为0，标准差为1）
    - 标准化后的数据更适合神经网络训练
    
    参数:
        dataset: 数据集对象，包含data.y属性（性质标签）
        task: 任务名称字符串，如'tg', 'er', 'de'等
        
    返回:
        scaler: 训练好的StandardScaler对象，用于后续的反标准化
    """
    scaler = StandardScaler()
    y_values = np.array([data.y.item() for data in dataset]).reshape(-1, 1)

    # 对于弹性模量(er)和固有粘度(iv)任务，先进行log10变换
    # 因为这些性质通常呈对数正态分布
    if task in ['er', 'iv']:
        y_values = np.log10(y_values)
        for data in dataset:
            data.y = torch.tensor(np.log10(data.y.item()), dtype=torch.float)

    # 拟合标准化器（计算均值和标准差）
    scaler.fit(y_values)
    print("Scaling y values with mean:", scaler.mean_[0], "and std:", scaler.scale_[0])

    # 对所有数据进行标准化
    for data in dataset:
        data.y = torch.tensor(scaler.transform([[data.y.item()]]), dtype=torch.float)

    return scaler

def get_data_loader(dataset, indices=None, batch_size=32, shuffle=False, modalities=None):
    """
    创建PyTorch数据加载器（DataLoader）
    
    功能说明:
    - 根据提供的索引从数据集中提取子集
    - 创建DataLoader对象，用于批量加载数据
    - 使用自定义的collate_fn处理多模态数据的批量化
    
    参数:
        dataset: 完整的数据集对象
        indices: 可选，要使用的数据索引列表。如果为None，则使用全部数据
        batch_size: 批次大小，默认32
        shuffle: 是否打乱数据顺序，默认False（训练时设为True，验证/测试时设为False）
        
    返回:
        loader: PyTorch DataLoader对象，用于迭代加载批次数据
    """
    if indices is None:
        indices = range(len(dataset))
    # 根据索引提取数据子集
    subset_dataset = [dataset[i] for i in indices]
    
    # 创建DataLoader，使用自定义的collate_fn处理多模态数据
    collate_fn = make_custom_collate(modalities) if modalities is not None else custom_collate
    loader = DataLoader(
        subset_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn,  # 自定义批量化函数，处理图数据等复杂结构
        shuffle=shuffle,
        drop_last=True
    )
    
    print(f"Created dataloader with {len(subset_dataset)} samples")
    return loader

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """
    训练模型一个epoch（一个完整的训练周期）
    
    功能说明:
    - 将模型设置为训练模式（启用dropout等）
    - 遍历所有训练批次，进行前向传播、损失计算、反向传播和参数更新
    - 记录每个批次的损失和预测值
    - 计算整个epoch的平均损失和R²分数
    
    参数:
        model: 要训练的模型对象
        train_loader: 训练数据加载器
        criterion: 损失函数（如MSELoss）
        optimizer: 优化器（如Adam）
        scheduler: 学习率调度器（如CosineAnnealingLR）
        device: 计算设备（'cuda'或'cpu'）
        
    返回:
        avg_train_loss: 平均训练损失
        train_r2: 训练集的R²分数（决定系数，衡量模型拟合程度）
    """
    model.train()  # 设置为训练模式
    train_losses = []
    train_preds = []
    train_targets = []

    # 遍历所有训练批次
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)  # 将数据移动到指定设备（GPU/CPU）
        optimizer.zero_grad()  # 清零梯度

        # 前向传播：模型预测
        outputs,_ = model(batch)
        # 计算损失
        loss = criterion(outputs, batch.y)
        # 反向传播：计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 更新学习率
        scheduler.step()
        
        # 记录损失和预测值（用于后续计算指标）
        train_losses.append(loss.item())
        train_preds.extend(outputs.detach().cpu().numpy())
        train_targets.extend(batch.y.detach().cpu().numpy())

    # 计算平均损失和R²分数
    avg_train_loss = sum(train_losses) / len(train_losses)
    train_r2 = r2_score(train_targets, train_preds)
    
    return avg_train_loss, train_r2

def evaluate(model, data_loader, criterion, device):
    """
    评估模型在给定数据集上的性能
    
    功能说明:
    - 将模型设置为评估模式（禁用dropout等）
    - 不计算梯度（torch.no_grad），节省内存和计算
    - 遍历所有批次，进行前向传播和损失计算
    - 计算平均损失和R²分数
    
    参数:
        model: 要评估的模型对象
        data_loader: 数据加载器（可以是验证集或测试集）
        criterion: 损失函数
        device: 计算设备（'cuda'或'cpu'）
        
    返回:
        avg_loss: 平均损失值
        r2: R²分数（决定系数，越接近1表示拟合越好）
        targets: 真实标签列表
        preds: 预测值列表
    """
    model.eval()  # 设置为评估模式
    losses = []
    preds = []
    targets = []

    # 评估模式下不计算梯度，节省内存和计算资源
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device)
            # 前向传播：模型预测
            outputs,_ = model(batch)
            # 计算损失
            loss = criterion(outputs, batch.y)
            losses.append(loss.item())
            # 收集预测值和真实值
            preds.extend(outputs.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    # 计算平均损失和R²分数
    avg_loss = sum(losses) / len(losses)
    r2 = r2_score(targets, preds)
    
    return avg_loss, r2, targets, preds

def test_model(model, test_loader, scaler, device):
    """
    在测试集上评估模型，并返回反标准化后的评估指标
    
    功能说明:
    - 在测试集上进行模型预测
    - 将标准化后的预测值和真实值反标准化回原始尺度
    - 计算R²、MAE、RMSE等评估指标（在原始尺度上）
    
    参数:
        model: 训练好的模型对象
        test_loader: 测试集数据加载器
        scaler: StandardScaler对象，用于反标准化
        device: 计算设备（'cuda'或'cpu'）
        
    返回:
        dict: 包含以下键的字典
            - 'test_r2': R²分数（决定系数，越接近1越好）
            - 'test_mae': 平均绝对误差（Mean Absolute Error）
            - 'test_rmse': 均方根误差（Root Mean Squared Error）
    """
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            outputs,_ = model(batch)
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(batch.y.cpu().numpy())

    y_true = np.array(test_targets)
    y_pred = np.array(test_preds)
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    test_r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    test_mae = metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_true_unscaled, y_pred_unscaled))

    return {'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse}

def train_and_evaluate(model, scaler, train_loader, val_loader, test_loader, device, num_epochs=100, patience=5):
    """
    完整的模型训练和评估流程
    
    功能说明:
    - 初始化损失函数、优化器和学习率调度器
    - 进行多个epoch的训练，每个epoch包括训练和验证
    - 实现早停机制：如果验证集R²连续patience个epoch没有提升，则停止训练
    - 保存验证集R²最高的模型状态
    - 使用最佳模型在测试集上评估，返回评估指标
    
    参数:
        model: 要训练的模型对象
        scaler: StandardScaler对象，用于测试时的反标准化
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        test_loader: 测试集数据加载器
        device: 计算设备（'cuda'或'cpu'）
        num_epochs: 最大训练轮数，默认100
        patience: 早停的耐心值，默认5（连续5个epoch无提升则停止）
        
    返回:
        dict: 测试集评估指标字典，包含'test_r2', 'test_mae', 'test_rmse'
    """
    # 定义损失函数：均方误差损失（适合回归任务）
    criterion = nn.MSELoss()
    # 定义优化器：Adam优化器，学习率1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 定义学习率调度器：余弦退火，周期为10个epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 记录最佳验证集R²和对应的模型状态
    best_val_r2 = -float('inf')
    best_model_state = None
    epochs_no_improve = 0  # 连续无提升的epoch数

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段：在训练集上训练一个epoch
        avg_train_loss, train_r2 = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)

        # 验证阶段：在验证集上评估模型
        avg_val_loss, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        
        # 早停机制：如果验证集R²提升，保存最佳模型
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_state = model.state_dict().copy()  # 保存模型状态
            epochs_no_improve = 0
            print(f"Epoch {epoch+1}: Validation R2 improved to {val_r2:.4f}.")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: No improvement in Validation R2 for {epochs_no_improve} epoch(s).")

        # 如果连续patience个epoch没有提升，提前停止训练
        if epochs_no_improve >= patience:
            print(f"Early stopping after {patience} epochs with no improvement.")
            break

        # 打印当前epoch的训练和验证指标
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}, Training R2: {train_r2:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation R2: {val_r2:.4f}")
        print("-" * 50)

    # 加载最佳模型状态（验证集R²最高的模型）
    model.load_state_dict(best_model_state)

    # 测试阶段：在测试集上评估最佳模型
    test_metrics = test_model(model, test_loader, scaler, device)
    return test_metrics


def compute_contrastive_loss(embeddings, temperature=0.07):
    """
    计算对比损失（用于预训练阶段）
    
    功能说明:
    - 对比学习的目标：让同一分子的不同模态表示在嵌入空间中靠近，
      让不同分子的表示在嵌入空间中远离
    - 计算所有模态对之间的对比损失
    - 使用温度参数缩放相似度分数，控制分布的尖锐程度
    
    参数:
        embeddings: 多模态嵌入张量，形状为 [batch_size, num_modalities, embedding_dim]
                   例如：[32, 5, 256] 表示32个样本，5种模态，256维嵌入
        temperature: 温度参数，默认0.07。较小的温度值会使分布更尖锐
        
    返回:
        total_loss: 平均对比损失值（所有模态对的平均）
    
    工作原理:
    1. 对于每对不同的模态（如SMILES和Graph），计算它们之间的相似度矩阵
    2. 正样本：同一分子的不同模态（对角线元素）
    3. 负样本：不同分子的表示（非对角线元素）
    4. 使用交叉熵损失，让正样本的相似度最大化，负样本的相似度最小化
    """
    batch_size, num_modalities, embedding_dim = embeddings.shape
    device = embeddings.device
    total_loss = 0.0
    count = 0

    # 遍历所有模态对（如SMILES-Graph, SMILES-Text等）
    for i in range(num_modalities):
        for j in range(num_modalities):
            if i != j:  # 只计算不同模态之间的损失
                embedding_i = embeddings[:, i, :]  # [batch_size, embedding_dim] 第i种模态
                embedding_j = embeddings[:, j, :]  # [batch_size, embedding_dim] 第j种模态

                # L2归一化：将嵌入向量归一化到单位球面上
                embedding_i = nn.functional.normalize(embedding_i, p=2, dim=1)
                embedding_j = nn.functional.normalize(embedding_j, p=2, dim=1)

                # 计算相似度矩阵：[batch_size, batch_size]
                # 矩阵中(i,j)位置的元素表示第i个样本的模态i与第j个样本的模态j的相似度
                logits = torch.matmul(embedding_i, embedding_j.T) / temperature
                
                # 标签：对角线元素是正样本（同一分子的不同模态）
                labels = torch.arange(batch_size).to(device)
                
                # 计算交叉熵损失：让对角线元素（正样本）的相似度最大化
                loss_i = nn.functional.cross_entropy(logits, labels)
                total_loss += loss_i
                count += 1

    # 返回所有模态对的平均损失
    total_loss = total_loss / count
    return total_loss