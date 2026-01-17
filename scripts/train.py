

import os
# 这里的端口必须和你本地转发到服务器的端口一致
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# 如果之前设置了镜像站，也可以在这里一起写死
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset import UniDataset
from src.modules import UniEncoderAttention
from src.utils import scale_targets, train_and_evaluate, get_data_loader


def parse_arguments():
    """
    解析命令行参数
    
    功能说明:
    - 定义所有可用的命令行参数
    - 设置默认值和帮助信息
    - 返回解析后的参数对象
    
    返回:
        args: 包含所有命令行参数的对象
    """
    parser = argparse.ArgumentParser(description="Train UniEncoderAttention Model")
    
    # 任务列表：要训练的性质预测任务
    parser.add_argument(
        '--tasks',
        nargs='+',  # 允许输入多个值
        default=['tg', 'er', 'de', 'td', 'tm', 'iv', 'bc'],
        help="List of tasks to train on. Example: --tasks tg er de"
    )
    
    # 模型名称：用于保存和标识模型
    parser.add_argument(
        '--model_name',
        type=str,
        default='UniEncoderAttention',
        help="Name of the model."
    )
    
    # 模态列表：要使用的多模态输入类型
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['smiles', 'fp', 'text', 'graph'],
        help="List of model modalities. Example: --modalities smiles text"
    )
    
    # 冻结编码器：如果设置，则冻结预训练编码器的权重，只训练融合层和预测层
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',  # 布尔标志，出现即True
        help="Freeze encoders weights if set."
    )
    
    # 预训练模型路径：可选，用于加载之前训练好的模型继续训练
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default=None,
        help="Path to the pretrained model."
    )
    
    # 训练轮数：最大训练epoch数
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of training epochs."
    )
    
    # 早停耐心值：验证集R²连续多少个epoch不提升就停止训练
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help="Early stopping patience."
    )
    
    # 结果保存路径：评估指标保存的CSV文件路径
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results/results.csv',
        help="Directory to save results CSV."
    )
    
    # 模型保存路径：训练好的模型权重保存的目录
    parser.add_argument(
        '--models_dir',
        type=str,
        default='./saved_models',
        help="Directory to save trained models."
    )
    
    # 批次大小：每个训练批次的样本数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="Batch size for training."
    )
    
    return parser.parse_args()


def main():
    """
    主训练函数
    
    功能说明:
    - 解析命令行参数
    - 初始化设备和预训练模型配置
    - 加载数据集
    - 对每个任务进行训练、评估和结果保存
    
    训练流程:
    1. 数据准备：加载数据集、标准化、划分训练/验证/测试集
    2. 模型初始化：创建模型、加载预训练权重（可选）
    3. 训练和评估：调用train_and_evaluate进行完整训练流程
    4. 保存结果：保存模型权重和评估指标
    """
    # 解析命令行参数
    args = parse_arguments()
    print("命令行参数解析完成")
    
    # 设置计算设备：优先使用GPU，如果没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    
    # 忽略警告信息，保持输出简洁
    warnings.filterwarnings("ignore")
    
    # 预训练模型配置字典：定义各模态使用的预训练模型路径
    pre_trained_model_dict = {
        'smiles_model_name': "./pretrained_models/smiles450k",  # SMILES编码器：RoBERTa
        'text_model_name': "./pretrained_models/T5",  # 文本编码器：T5
        'gnn_model_name': "./pretrained_models/Mole-BERT.pth",  # 图编码器：GIN (Mole-BERT)
        'geom_model_name': "./pretrained_models/schnet_qm9_heat_capacity_model.pth"  # 几何编码器：SchNet
    }
    
    # 获取输出路径
    result_output_dir = args.results_dir  # 结果CSV文件保存路径
    model_output_dir = args.models_dir  # 模型权重保存路径
    
    # 获取任务列表
    task_list = args.tasks
    
    # 构建数据集名称列表：格式为 'smi_' + 任务名（如 'smi_tg'）
    dataset_name_list = ['smi_' + task for task in task_list]
    
    # 为每个任务加载数据集
    dataset_list = [
        UniDataset(
            root='./data',  # 数据根目录
            dataset=dataset_name,  # 数据集名称（如 'smi_tg'）
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            text_model_name=pre_trained_model_dict['text_model_name'],
            task_type="downstream"
        )
        for dataset_name in dataset_name_list
    ]
    
    # 获取训练配置参数
    model_modality_list = args.modalities  # 要使用的模态列表
    freeze_encoder = args.freeze_encoder  # 是否冻结编码器
    pretrained_model_path = args.pretrained_model_path  # 预训练模型路径（可选）
    epochs = args.epochs  # 训练轮数
    patience = args.patience  # 早停耐心值
    
    # 用于存储所有任务的结果
    results = []
    
    # 对每个任务进行训练
    for task in task_list:
        print(f"\nStarting task: {task}")
        
        # 获取当前任务的数据集
        dataset = dataset_list[task_list.index(task)]
        
        # 数据标准化：对目标值进行标准化处理（某些任务会先进行log10变换）
        scaler = scale_targets(dataset, task)

        # 数据集划分：将数据集划分为训练集、验证集和测试集
        # 第一次划分：80%训练+验证，20%测试
        train_val_indices, test_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.2,  # 测试集占20%
            random_state=42  # 固定随机种子，保证可复现
        )
        # 第二次划分：从80%中再划分，最终为 72%训练，8%验证，20%测试
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=0.1,  # 验证集占训练+验证集的10%，即总数据的8%
            random_state=42
        )

        # 创建数据加载器：用于批量加载数据
        train_loader = get_data_loader(
            dataset, 
            indices=train_indices, 
            batch_size=args.batch_size, 
            shuffle=True,  # 训练集需要打乱
            modalities=model_modality_list
        )
        val_loader = get_data_loader(
            dataset, 
            indices=val_indices, 
            batch_size=args.batch_size, 
            shuffle=False,  # 验证集不需要打乱
            modalities=model_modality_list

        )
        test_loader = get_data_loader(
            dataset, 
            indices=test_indices, 
            batch_size=args.batch_size, 
            shuffle=False,  # 测试集不需要打乱
            modalities=model_modality_list
        )

        # 初始化模型：创建多模态融合模型
        model = UniEncoderAttention(
            joint_embedding_dim=256,  # 统一嵌入空间维度
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            text_model_name=pre_trained_model_dict['text_model_name'],
            gnn_model_name=pre_trained_model_dict['gnn_model_name'],
            geom_model_name=pre_trained_model_dict['geom_model_name'],
            modality_list=model_modality_list,  # 要使用的模态列表
            freeze_encoder=freeze_encoder  # 是否冻结编码器权重
        )
        
        # 如果提供了预训练模型路径，加载预训练权重
        if pretrained_model_path:
            state = torch.load(pretrained_model_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained model from {pretrained_model_path}")
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
            # model.load_state_dict(torch.load(pretrained_model_path))
            # print(f"Loaded pretrained model from {pretrained_model_path}")
        
        # 将模型移动到指定设备（GPU/CPU）
        model.to(device)
        print("Using GPU for model training." if torch.cuda.is_available() else "Using CPU for model training.")

        # 训练和评估：执行完整的训练流程（包括训练、验证、测试）
        metrics = train_and_evaluate(
            model, scaler, train_loader, val_loader, test_loader,
            device, num_epochs=epochs, patience=patience
        )
        
        # 打印测试集评估结果
        print(
            f"Test R2: {metrics['test_r2']:.4f}, "
            f"MAE: {metrics['test_mae']:.4f}, RMSE: {metrics['test_rmse']:.4f}"
        )

        # 保存最佳模型：保存验证集R²最高的模型权重
        attention_weights = model.attention_visual_weights.detach().cpu().numpy()  # 保存注意力权重用于可视化
        os.makedirs(os.path.join(model_output_dir, task), exist_ok=True)  # 创建保存目录
        torch.save(
            model.state_dict(), 
            os.path.join(model_output_dir, f'{task}/{args.model_name}_best.pth')
        )
        print(f"Model saved with R2: {metrics['test_r2']:.4f}")

        # 保存结果：将评估指标保存到字典中
        result = {
            'task': task,  # 任务名称
            'model_name': args.model_name,  # 模型名称
            'model_modality_list': model_modality_list,  # 使用的模态列表
            'avg_test_r2': float(f"{metrics['test_r2']:.4g}"),  # 测试集R²分数
            'std_test_r2': 0.0,  # R²标准差（单次运行为0）
            'avg_test_mae': float(f"{metrics['test_mae']:.4g}"),  # 测试集平均绝对误差
            'std_test_mae': 0.0,  # MAE标准差
            'avg_test_rmse': float(f"{metrics['test_rmse']:.4g}"),  # 测试集均方根误差
            'std_test_rmse': 0.0,  # RMSE标准差
            'attention_weights': attention_weights  # 注意力权重（用于分析模态重要性）
        }
    
        results.append(result)

        # 保存结果到CSV文件：追加模式，如果文件不存在则创建并写入表头
        os.makedirs(os.path.dirname(result_output_dir), exist_ok=True)  # 创建结果目录
        # results_df = pd.DataFrame(results)
        # results_df.to_csv(
        #     result_output_dir,
        #     mode='a',  # 追加模式
        #     header=not os.path.exists(result_output_dir),  # 如果文件不存在则写入表头
        #     index=False  # 不保存行索引
        # )

        pd.DataFrame([result]).to_csv(
            result_output_dir,
            mode='a',
            header=not os.path.exists(result_output_dir),
            index=False
        )
        print(f"Results have been appended to '{result_output_dir}'.")
        

if __name__ == "__main__":
    # 当脚本直接运行时，执行主函数
    main()
