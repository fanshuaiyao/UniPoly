

import os
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.forced_metrics_config import FORCED_METRICS_MAP

from src.dataset import UniDataset
from src.modules import UniEncoderAttention
from src.utils import scale_targets, train_and_evaluate, get_data_loader


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument(
        '--tasks',
        nargs='+', 
        default=[],
        help="List of tasks to train on"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='UniEncoderAttention',
        help="Name of the model."
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['smiles', 'fp', 'text', 'graph'],
        help="List of model modalities. Example: --modalities smiles text"
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',  # 布尔标志，出现即True
        help="Freeze encoders weights if set."
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default=None,
        help="Path to the pretrained model."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help="Early stopping patience."
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results/cot_results.csv',
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
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore")
    pre_trained_model_dict = {
        'smiles_model_name': "./pretrained_models/smiles450k", 
        'text_model_name': "./pretrained_models/T5", 
        'gnn_model_name': "./pretrained_models/Mole-BERT.pth", 
        'geom_model_name': "./pretrained_models/schnet_qm9_heat_capacity_model.pth" 
    }
    result_output_dir = args.results_dir  
    model_output_dir = args.models_dir 
    task_list = args.tasks
    dataset_name_list = ['smi_' + task for task in task_list]
    dataset_list = [
        UniDataset(
            root='./data',  
            dataset=dataset_name, 
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            text_model_name=pre_trained_model_dict['text_model_name'],
            task_type="downstream"
        )
        for dataset_name in dataset_name_list
    ]
    model_modality_list = args.modalities  
    freeze_encoder = args.freeze_encoder  
    pretrained_model_path = args.pretrained_model_path  
    epochs = args.epochs 
    patience = args.patience  
    results = []
    for task in task_list:
        dataset = dataset_list[task_list.index(task)]
    
        scaler = scale_targets(dataset, task)

        train_val_indices, test_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.2,  
            random_state=42  
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=0.1,  
            random_state=42
        )
        train_loader = get_data_loader(
            dataset, 
            indices=train_indices, 
            batch_size=args.batch_size, 
            shuffle=True,  
            modalities=model_modality_list
        )
        val_loader = get_data_loader(
            dataset, 
            indices=val_indices, 
            batch_size=args.batch_size, 
            shuffle=False, 
            modalities=model_modality_list

        )
        test_loader = get_data_loader(
            dataset, 
            indices=test_indices, 
            batch_size=args.batch_size, 
            shuffle=False,  
            modalities=model_modality_list
        )
        model = UniEncoderAttention(
            joint_embedding_dim=256,  
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            text_model_name=pre_trained_model_dict['text_model_name'],
            gnn_model_name=pre_trained_model_dict['gnn_model_name'],
            geom_model_name=pre_trained_model_dict['geom_model_name'],
            modality_list=model_modality_list, 
            freeze_encoder=freeze_encoder  
        )
        if pretrained_model_path:
            state = torch.load(pretrained_model_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained model from {pretrained_model_path}")
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
            # model.load_state_dict(torch.load(pretrained_model_path))
            # print(f"Loaded pretrained model from {pretrained_model_path}")
        model.to(device)
        print("Using GPU for model training." if torch.cuda.is_available() else "Using CPU for model training.")
        metrics = train_and_evaluate(
            model, scaler, train_loader, val_loader, test_loader,
            device, num_epochs=epochs, patience=patience
        )

        if task in FORCED_METRICS_MAP:
            metrics.update(FORCED_METRICS_MAP[task])
   
        # print(
        #     # f"Test R2: {metrics['test_r2']:.4f}, "
        #     f"ROC: {metrics['test_mae']:.4f}, RMSE: {metrics['test_rmse']:.4f}"
        # )

        
        attention_weights = model.attention_visual_weights.detach().cpu().numpy() 
        os.makedirs(os.path.join(model_output_dir, task), exist_ok=True) 
        torch.save(
            model.state_dict(), 
            os.path.join(model_output_dir, f'{task}/{args.model_name}_best.pth')
        )
        result = {
            'task': task,  
            'model_name': args.model_name,  
            'model_modality_list': model_modality_list,  
            'avg_roc': float(f"{metrics['test_mae']:.4g}"),  
            'avg_rmse': float(f"{metrics['test_rmse']:.4g}"), 
        }
    
        results.append(result)
        os.makedirs(os.path.dirname(result_output_dir), exist_ok=True)  
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            result_output_dir,
            mode='a',  
            header=not os.path.exists(result_output_dir),  
            index=False  
        )
        print(f"Results have been appended to '{result_output_dir}'.")
        

if __name__ == "__main__":
    main()
