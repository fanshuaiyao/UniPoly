import torch
from torch_geometric.data import Batch

# def custom_collate(data_list):
#     # Initialize lists to hold batched data
    
#     x_list = []
#     edge_index_list = []
#     edge_attr_list = []
#     smiles_list = []
#     text_list = []
#     input_ids_smiles_list = []
#     attention_mask_smiles_list = []
#     input_ids_text_list = []
#     attention_mask_text_list = []
#     fp_list = []
#     batch2d_list = []
#     x3d_list = []
#     pos3d_list = []
#     batch3d_list = []
#     y_list = []
    
#     num_nodes2d_cum = 0
#     num_nodes3d_cum = 0
    
#     for batch_idx, data in enumerate(data_list):
#         #smiles
#         smiles_list.append(data.smiles)
#         input_ids_smiles_list.append(data.input_ids_smiles)
#         attention_mask_smiles_list.append(data.attention_mask_smiles)
        
#         #text
#         text_list.append(data.text)
#         input_ids_text_list.append(data.input_ids_text)
#         attention_mask_text_list.append(data.attention_mask_text)
        
#         # 2D graph data
#         num_nodes2d = data.x.size(0)
#         x_list.append(data.x)
#         edge_index_list.append(data.edge_index + num_nodes2d_cum)
#         if data.edge_attr is not None:
#             edge_attr_list.append(data.edge_attr)
#         batch2d_list.append(torch.full((num_nodes2d,), batch_idx, dtype=torch.long))
#         num_nodes2d_cum += num_nodes2d
        
#         # 3D graph data
#         num_nodes3d = data.pos[1].size(0)
#         x3d_list.append(data.z[1])
#         pos3d_list.append(data.pos[1])
#         batch3d_list.append(torch.full((num_nodes3d,), batch_idx, dtype=torch.long))
#         num_nodes3d_cum += num_nodes3d
        
#         #fingerprint
#         fp_list.append(data.fp)
        
#         # Labels
#         y_list.append(data.y)
    
#     # Create batched data
#     batch = Batch()
#     batch.x = torch.cat(x_list, dim=0)
#     batch.edge_index = torch.cat(edge_index_list, dim=1)
#     if edge_attr_list:
#         batch.edge_attr = torch.cat(edge_attr_list, dim=0)
#     batch.batch = torch.cat(batch2d_list, dim=0)
#     batch.smiles = smiles_list
#     batch.input_ids_smiles = torch.cat(input_ids_smiles_list, dim=0)
#     batch.attention_mask_smiles = torch.cat(attention_mask_smiles_list, dim=0)
#     batch.text = text_list
#     batch.input_ids_text = torch.cat(input_ids_text_list, dim=0)
#     batch.attention_mask_text = torch.cat(attention_mask_text_list, dim=0)
#     batch.fp = torch.cat(fp_list, dim=0)
#     batch.x3d = torch.cat(x3d_list, dim=0)
#     batch.pos3d = torch.cat(pos3d_list, dim=0)
#     batch.batch3d = torch.cat(batch3d_list, dim=0)
    
#     batch.y = torch.cat(y_list, dim=0)
    
#     return batch



# 感知模态
def make_custom_collate(modalities, need_y: bool = True):
    use_3d = ("geom" in modalities) or ("3d" in modalities)

    def custom_collate(data_list):
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        smiles_list = []
        text_list = []
        input_ids_smiles_list = []
        attention_mask_smiles_list = []
        input_ids_text_list = []
        attention_mask_text_list = []
        fp_list = []
        batch2d_list = []

        # 3D（只有 use_3d 才会用到）
        x3d_list = []
        pos3d_list = []
        batch3d_list = []

        y_list = []

        num_nodes2d_cum = 0

        for batch_idx, data in enumerate(data_list):
            # smiles/text
            smiles_list.append(getattr(data, "smiles", ""))
            input_ids_smiles_list.append(data.input_ids_smiles)
            attention_mask_smiles_list.append(data.attention_mask_smiles)

            text_list.append(getattr(data, "text", ""))
            input_ids_text_list.append(data.input_ids_text)
            attention_mask_text_list.append(data.attention_mask_text)

            # 2D graph
            num_nodes2d = data.x.size(0)
            x_list.append(data.x)
            edge_index_list.append(data.edge_index + num_nodes2d_cum)
            if getattr(data, "edge_attr", None) is not None:
                edge_attr_list.append(data.edge_attr)
            batch2d_list.append(torch.full((num_nodes2d,), batch_idx, dtype=torch.long))
            num_nodes2d_cum += num_nodes2d

            # 3D graph（只有需要才处理）
            if use_3d:
                # 兼容你现在的 data.pos[1] / data.z[1] 结构
                if not hasattr(data, "pos") or data.pos is None or not hasattr(data, "z") or data.z is None:
                    # 缺 3D 的样本直接跳过（或者你也可以 raise）
                    continue

                num_nodes3d = data.pos[1].size(0)
                x3d_list.append(data.z[1])
                pos3d_list.append(data.pos[1])
                batch3d_list.append(torch.full((num_nodes3d,), batch_idx, dtype=torch.long))

            # fp
            fp_list.append(data.fp)

            # y（只有需要 label 才处理）
            if need_y and hasattr(data, "y") and data.y is not None:
                y_list.append(data.y)

        batch = Batch()
        batch.x = torch.cat(x_list, dim=0)
        batch.edge_index = torch.cat(edge_index_list, dim=1)
        if edge_attr_list:
            batch.edge_attr = torch.cat(edge_attr_list, dim=0)
        batch.batch = torch.cat(batch2d_list, dim=0)

        batch.smiles = smiles_list
        batch.input_ids_smiles = torch.cat(input_ids_smiles_list, dim=0)
        batch.attention_mask_smiles = torch.cat(attention_mask_smiles_list, dim=0)

        batch.text = text_list
        batch.input_ids_text = torch.cat(input_ids_text_list, dim=0)
        batch.attention_mask_text = torch.cat(attention_mask_text_list, dim=0)

        batch.fp = torch.cat(fp_list, dim=0)

        if use_3d:
            batch.x3d = torch.cat(x3d_list, dim=0)
            batch.pos3d = torch.cat(pos3d_list, dim=0)
            batch.batch3d = torch.cat(batch3d_list, dim=0)

        if need_y and y_list:
            batch.y = torch.cat(y_list, dim=0)

        return batch

    return custom_collate
