import os
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F


from src.dataset import UniDataset
from src.modules import UniEncoderAttention
from src.utils import get_data_loader, compute_contrastive_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretrain UniEncoderAttention Model")
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['smiles', 'text', 'graph', 'fp', 'geom'],
        help="List of modalities to use. Example: --modalities smiles text"
    )
    parser.add_argument(
        '--smiles_model_name',
        type=str,
        default="./pretrained_models/smiles450k",
        help="Pretrained model name or path for SMILES"
    )
    parser.add_argument(
        '--text_model_name',
        type=str,
        default="./pretrained_models/T5",
        help="Pretrained model name or path for Text"
    )
    parser.add_argument(
        '--gnn_model_name',
        type=str,
        default="./pretrained_models/Mole-BERT.pth",
        help="Pretrained GNN model path"
    )
    parser.add_argument(
        '--geom_model_name',
        type=str,
        default="./pretrained_models/schnet_qm9_heat_capacity_model.pth",
        help="Pretrained Geometry model path"
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help="If set, freeze the pretrained model weights."
    )
    parser.add_argument(
        '--debug_batch',
        action='store_true',
        help="If set, print a one-time brief summary of the first batch."
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='smi_all', 
        help="Name of the dataset for pretraining (unlabeled or labeled, but labels unused here)"
    )
    parser.add_argument(
        '--root',
        type=str,
        default='./data',
        help="Root directory of the dataset."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for pretraining."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="Number of pretraining epochs."
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help="Learning rate for optimizer."
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help="Temperature parameter for contrastive loss."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./pretrained_models/saved_pretrained_model.pth',
        help="Path to save the pretrained model."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    def _brief(x):
        if x is None:
            return "None"
        if isinstance(x, torch.Tensor):
            return f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
        if isinstance(x, list) and (len(x) == 0 or all(isinstance(item, str) for item in x)):
            first = x[0][:80] if len(x) > 0 else ""
            suffix = "..." if len(x) > 0 and len(x[0]) > 80 else ""
            return f"list[str] len={len(x)} first={first}{suffix}"
        return type(x).__name__
    
    # Get all available GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs available")
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataset and DataLoader (using the same dataset for unsupervised training, only using input features)
    dataset = UniDataset(
        root=args.root,
        dataset=args.dataset_name,
        smiles_model_name=args.smiles_model_name,
        text_model_name=args.text_model_name
    )
    indices = np.arange(len(dataset))
    dataloader = get_data_loader(dataset, indices=indices, batch_size=args.batch_size, shuffle=True, modalities=args.modalities)

    # Initialize model
    model = UniEncoderAttention(
        joint_embedding_dim=256,
        smiles_model_name=args.smiles_model_name,
        text_model_name=args.text_model_name,
        gnn_model_name=args.gnn_model_name,
        geom_model_name=args.geom_model_name,
        modality_list=args.modalities,
        freeze_encoder=args.freeze_encoder
    )
    
    # Use all available GPUs for data parallel training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create directory for saving loss curves and data
    os.makedirs('./plots/pretrain', exist_ok=True)
    
    # Record loss for each epoch
    losses = []
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0

        # ===== 统计 epoch 平均指标 =====
        pos_sum = 0.0
        neg_sum = 0.0
        stat_steps = 0

        for step, data in enumerate(dataloader):
            data = data.to(device)

            optimizer.zero_grad()
            _, embeddings = model(data)   # [B, M, D]

            loss = compute_contrastive_loss(
                embeddings,
                temperature=args.temperature
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # ===== 指标计算（不参与反向传播）=====
            with torch.no_grad():
                emb = embeddings.detach()      # [B, M, D]
                B, M, D = emb.shape

                # L2 normalize
                z = F.normalize(emb, dim=-1)   # [B, M, D]

                # -------- 正样本相似度 --------
                # 同一个样本，不同模态，两两 cosine
                pos_sims = []
                for i in range(M):
                    for j in range(i + 1, M):
                        sim = (z[:, i, :] * z[:, j, :]).sum(dim=-1)  # [B]
                        pos_sims.append(sim.mean())
                pos_sim = torch.stack(pos_sims).mean().item()

                # -------- 负样本相似度 --------
                # 用第一个模态（如 smiles）作 anchor
                perm = torch.randperm(B, device=emb.device)
                neg_sim = (z[:, 0, :] * z[perm, 0, :]).sum(dim=-1).mean().item()

            pos_sum += pos_sim
            neg_sum += neg_sim
            stat_steps += 1

        # ===== epoch 统计 =====
        avg_loss = epoch_loss / stat_steps
        avg_pos = pos_sum / stat_steps
        avg_neg = neg_sum / stat_steps

        losses.append(avg_loss)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss={avg_loss:.4f} | "
            f"pos_sim={avg_pos:.4f} | "
            f"neg_sim={avg_neg:.4f}"
        )

    ##### 先注释，方便回滚####
    # model.train()
    # for epoch in range(args.epochs):
    #     epoch_loss = 0.0
    #     for step, data in enumerate(dataloader):
    #         data = data.to(device)
    #         if args.debug_batch and epoch == 0 and step == 0:
    #             fields = [
    #                 "x",
    #                 "edge_index",
    #                 "edge_attr",
    #                 "batch",
    #                 "input_ids_smiles",
    #                 "attention_mask_smiles",
    #                 "input_ids_text",
    #                 "attention_mask_text",
    #                 "fp",
    #                 "x3d",
    #                 "pos3d",
    #                 "batch3d",
    #                 "smiles",
    #                 "text",
    #             ]
    #             print("First batch (brief):")
    #             for field in fields:
    #                 print(f"  {field}: {_brief(getattr(data, field, None))}")
    #         optimizer.zero_grad()
    #         _, embeddings = model(data)  # embeddings: [batch_size, num_modalities, embedding_dim]
    #         loss = compute_contrastive_loss(embeddings, temperature=args.temperature)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #     avg_loss = epoch_loss / len(dataloader)
    #     losses.append(avg_loss)
    #     print(f"Epoch [{epoch+1}/{args.epochs}] Contrastive Loss: {avg_loss:.4f}")
    #     ### 指标 ###

    #     with torch.no_grad():
    #         # embeddings: [B, M, D]
    #         B, M, D = embeddings.shape

    #         # L2 normalize
    #         z = F.normalize(embeddings, dim=-1)  # [B, M, D]

    #         # ---------- 正样本相似度 ----------
    #         # 同一个样本，不同模态两两 cosine
    #         # 取 (i,j), i<j
    #         pos_sims = []
    #         for i in range(M):
    #             for j in range(i + 1, M):
    #                 sim = (z[:, i, :] * z[:, j, :]).sum(dim=-1)  # [B]
    #                 pos_sims.append(sim.mean())
    #         pos_sim = torch.stack(pos_sims).mean().item()

    #         # ---------- 负样本相似度 ----------
    #         # 用第一个模态作为 anchor（例如 smiles）
    #         anchor = z[:, 0, :]              # [B, D]
    #         perm = torch.randperm(B)
    #         neg = z[perm, 0, :]              # 打乱样本
    #         neg_sim = (anchor * neg).sum(dim=-1).mean().item()

    #     print(f"Epoch [{epoch+1}] pos_sim={pos_sim:.4f}  neg_sim={neg_sim:.4f}")

    #     ############
   
    # Save original loss data
    loss_data = {
        'epochs': list(range(1, args.epochs + 1)),
        'losses': losses
    }
    with open('./plots/pretrain/loss_data.json', 'w') as f:
        json.dump(loss_data, f, indent=4)
    print("Loss data saved at ./plots/pretrain/loss_data.json")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), losses, marker='o')
    plt.title('Pretraining Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.grid(True)
    plt.savefig('./plots/pretrain/loss_curve.png')
    plt.close()
    print("Loss curve saved at ./plots/pretrain/loss_curve.png")

    # If using DataParallel, need to handle module prefix when saving
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), args.save_path)
    else:
        torch.save(model.state_dict(), args.save_path)
    print(f"Pretrained model saved at {args.save_path}")

if __name__ == "__main__":
    main()
