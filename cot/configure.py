import argparse
import deepspeed

# train
parser = argparse.ArgumentParser()
parser.add_argument('--world_size', type=int, default=4)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--llm_epochs', type=int, default=200)
parser.add_argument('--llm_lr', type=float, default=1e-5)
parser.add_argument('--lambda_contrast', type=float, default=1)
parser.add_argument('--decay', type=float, default=0.00005)
parser.add_argument('--eval_every_n_epochs', type=int, default=1)
parser.add_argument('--early_stop_epochs_llm', type=int, default=15)
parser.add_argument('--output_save_path', type=str, default='../checkpoints/finetune/')
parser.add_argument('--loss_temperature', type=float, default=0.1)
parser.add_argument('--save_repr', action='store_true', help="Set to True if --save_repr is specified; defaults to False")
parser.add_argument('--generate_cot', action='store_true', help="Set to True if --generate_cot is specified; defaults to False")

# llama model
parser.add_argument('--model_name', type=str, default='../llama3-8b-instruct')
parser.add_argument('--smiles_max_length', type=int, default=400)
parser.add_argument('--cot_max_length', type=int, default=800)
parser.add_argument('--final_prompt_max_length', type=int, default=400)

# lora
parser.add_argument('--use_lora', action='store_true')
parser.add_argument('--no_use_lora', action='store_false', dest='use_lora')
parser.add_argument('--lora_rank', type=int, default=4)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.05)

# gnn model
parser.add_argument('--atom_feature_size', type=int, default=9)
parser.add_argument('--bond_feature_size', type=int, default=3)
parser.add_argument('--attentionfp_input_size', type=int, default=300)
parser.add_argument('--attentionfp_hidden_size', type=int, default=300)
parser.add_argument('--attentionfp_output_size', type=int, default=300)
parser.add_argument('--atom_layers', type=int, default=4)
parser.add_argument('--mol_layers', type=int, default=2)
parser.add_argument('--gnn_dropout_ratio', type=float, default=0.2)

parser.add_argument('--cross_attn_num_heads', type=int, default=2)
parser.add_argument('--gnn_weight', type=float, default=0)
parser.add_argument('--llm_weight', type=float, default=1)

# prediction head
parser.add_argument('--mlp_input_dim', type=int, default=4096)
parser.add_argument('--mlp_hidden_dim', type=int, default=256)
parser.add_argument('--mlp_output_dim', type=int, default=1)

# dataset
parser.add_argument('--root', type=str, default='../data', help="root")
parser.add_argument('--valid_rate', type=float, default=0.1, help="valid_rate")
parser.add_argument('--test_rate', type=float, default=0.1, help="test_rate")
parser.add_argument('--split_type', type=str, default='random', help="split_type")
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--split_seed', type=int, default=7, help="split_seed")
parser.add_argument('--use_multimodal', action='store_true')
parser.add_argument('--no_use_multimodal', action='store_false', dest='use_multimodal')
parser.add_argument('--use_cot', action='store_true', help="Set use_cot to True")
parser.add_argument('--no_use_cot', action='store_false', dest='use_cot', help="Set use_cot to False")
parser.add_argument('--encoding', type=str, default='UTF-8', help="encoding")

parser.add_argument('--datasets', nargs='+', default=['Tox21'], help="List of datasets to use")
parser.add_argument('--num_tasks', type=int, default=1, help="num_tasks")
parser.add_argument('--num_labels', type=int, default=12, help="num_labels")

# deepspeed
parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()

args.finetune_model_save_path = args.output_save_path + args.datasets[0] + '/llama_predictor'
args.finetune_MLP_save_path = args.output_save_path + 'predictor'
args.embedding_save_path = args.output_save_path + 'llm_embedding'

args.dataset_task_type = {
 
    'Caco-2': 'regression',
    'HIA': 'classification',
    'Pgp': 'classification',
    'Bioav': 'classification',
    'Lipo': 'regression',
    'AqSol': 'regression',

   
    'BBB': 'classification',
    'PPBR': 'regression',
    'VDss': 'regression',

 
    'CYP2C9-Inhibition': 'classification',
    'CYP2D6-Inhibition': 'classification',
    'CYP3A4-Inhibition': 'classification',
    'CYP2C9-Substrate': 'classification',
    'CYP2D6-Substrate': 'classification',
    'CYP3A4-Substrate': 'classification',

 
    'Half-life': 'regression',
    'CL-Hepa': 'regression',
    'CL-Micro': 'regression',

   
    'LD50': 'regression',
    'hERG': 'classification',
    'Ames': 'classification',
    'DILI': 'classification',
}

args.best_valid_initial = {

    'Caco-2': 10000,
    'HIA': 0,
    'Pgp': 0,
    'Bioav': 0,
    'Lipo': 10000,
    'AqSol': 10000,

   
    'BBB': 0,
    'PPBR': 10000,
    'VDss': 10000,

    'CYP2C9-Inhibition': 0,
    'CYP2D6-Inhibition': 0,
    'CYP3A4-Inhibition': 0,
    'CYP2C9-Substrate': 0,
    'CYP2D6-Substrate': 0,
    'CYP3A4-Substrate': 0,


    'Half-life': 10000,
    'CL-Hepa': 10000,
    'CL-Micro': 10000,

   
    'LD50': 10000,
    'hERG': 0,
    'Ames': 0,
    'DILI': 0,
}

args.best_valid_test = {
    'Caco-2': 0,
    'HIA': 0,
    'Pgp': 0,
    'Bioav': 0,
    'Lipo': 0,
    'AqSol': 0,
    'BBB': 0,
    'PPBR': 0,
    'VDss': 0,
    'CYP2C9-Inhibition': 0,
    'CYP2D6-Inhibition': 0,
    'CYP3A4-Inhibition': 0,
    'CYP2C9-Substrate': 0,
    'CYP2D6-Substrate': 0,
    'CYP3A4-Substrate': 0,
    'Half-life': 0,
    'CL-Hepa': 0,
    'CL-Micro': 0,
    'LD50': 0,
    'hERG': 0,
    'Ames': 0,
    'DILI': 0,
}

args.datasets_property_prompt = {
 
    'Caco-2': 'Caco-2 Permeability',
    'HIA': 'Human Intestinal Absorption',
    'Pgp': 'P-glycoprotein Inhibition/Interaction',
    'Bioav': 'Bioavailability',
    'Lipo': 'Lipophilicity',
    'AqSol': 'Aqueous Solubility',

    'BBB': 'Blood-Brain Barrier Penetration',
    'PPBR': 'Plasma Protein Binding Rate',
    'VDss': 'Volume of Distribution at Steady State',

    'CYP2C9-Inhibition': 'CYP2C9 Inhibition',
    'CYP2D6-Inhibition': 'CYP2D6 Inhibition',
    'CYP3A4-Inhibition': 'CYP3A4 Inhibition',
    'CYP2C9-Substrate': 'CYP2C9 Substrate',
    'CYP2D6-Substrate': 'CYP2D6 Substrate',
    'CYP3A4-Substrate': 'CYP3A4 Substrate',

 
    'Half-life': 'Half-life',
    'CL-Hepa': 'Hepatic Clearance',
    'CL-Micro': 'Microsomal Clearance',

    'LD50': 'Median Lethal Dose',
    'hERG': 'hERG Toxicity',
    'Ames': 'Ames Mutagenicity',
    'DILI': 'Drug-Induced Liver Injury',
}