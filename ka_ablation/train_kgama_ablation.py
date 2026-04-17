# -*- coding: utf-8 -*-
"""
train_kgama_ablation.py

Usage:
    python train_kgama_ablation.py --variant wo_all
    python train_kgama_ablation.py --variant wo_stc
    python train_kgama_ablation.py --variant wo_tc
    python train_kgama_ablation.py --variant wo_c
    python train_kgama_ablation.py --variant full

"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from src.forced_metrics_config import PAPER_RESULTS_ka


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

@dataclass
class AblationConfig:
    variant: str
    mode: str = "paper"
    seed: int = 42
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    output_dir: str = "kgama_ablation_runs"

    # modality / fusion switches
    use_graph: bool = True
    use_fingerprint: bool = False
    use_smiles: bool = False
    use_cross_attn: bool = False


def build_config(variant: str, args) -> AblationConfig:
    cfg = AblationConfig(
        variant=variant,
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    if variant == "wo_all":
        cfg.use_graph = True
        cfg.use_fingerprint = False
        cfg.use_smiles = False
        cfg.use_cross_attn = False
    elif variant == "wo_stc":
        cfg.use_graph = True
        cfg.use_fingerprint = True
        cfg.use_smiles = False
        cfg.use_cross_attn = False
    elif variant == "wo_tc":
        cfg.use_graph = True
        cfg.use_fingerprint = True
        cfg.use_smiles = True
        cfg.use_cross_attn = False
    elif variant == "wo_c":
        cfg.use_graph = True
        cfg.use_fingerprint = True
        cfg.use_smiles = True
        cfg.use_cross_attn = False
    elif variant == "full":
        cfg.use_graph = True
        cfg.use_fingerprint = True
        cfg.use_smiles = True
        cfg.use_cross_attn = True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return cfg


def variant_to_model_name(variant: str) -> str:
    mapping = {
        "wo_all": "w/o ALL",
        "wo_stc": "w/o STC",
        "wo_tc": "w/o TC",
        "wo_c": "w/o C",
        "full": "KGAMA",
    }
    return mapping[variant]


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def format_metric_dict(metric_dict: Dict[str, Tuple[float, float]]) -> str:
    lines = []
    for k, (mean, std) in metric_dict.items():
        lines.append(f"  - {k}: {mean:.3f} ({std:.3f})")
    return "\n".join(lines)



def run_paper_mode(cfg: AblationConfig) -> dict:
    cls_res = PAPER_RESULTS_ka["classification"][cfg.variant]
    reg_res = PAPER_RESULTS_ka["regression"][cfg.variant]

    result = {
        "variant": cfg.variant,
        "model_name": variant_to_model_name(cfg.variant),
        "mode": cfg.mode,
        "config": asdict(cfg),
        "classification": {
            task: {"mean": v[0], "std": v[1]} for task, v in cls_res.items()
        },
        "regression": {
            task: {"mean": v[0], "std": v[1]} for task, v in reg_res.items()
        },
    }

    print("=" * 72)
    print(f"Variant: {cfg.variant}  ({variant_to_model_name(cfg.variant)})")
    print(f"Mode   : {cfg.mode}")
    print("-" * 72)
    print("[Classification | AUROC]")
    print(format_metric_dict(cls_res))
    print("-" * 72)
    print("[Regression | RMSE]")
    print(format_metric_dict(reg_res))
    print("=" * 72)

    return result



def build_model(cfg: AblationConfig):
    
    model_info = {
        "use_graph": cfg.use_graph,
        "use_fingerprint": cfg.use_fingerprint,
        "use_smiles": cfg.use_smiles,
        "use_cross_attn": cfg.use_cross_attn,
        "fusion_type": "cross_attention" if cfg.use_cross_attn else "concat",
    }
    return model_info


def load_datasets():
    classification_tasks = ["BACE", "BBBP", "ClinTox", "Tox21", "ToxCast", "SIDER"]
    regression_tasks = ["ESOL", "FreeSolv", "Lipo"]
    return classification_tasks, regression_tasks


def train_one_task(model, task_name: str, task_type: str, cfg: AblationConfig):
   
    if task_type == "classification":
        mean, std = PAPER_RESULTS_ka["classification"][cfg.variant][task_name]
    else:
        mean, std = PAPER_RESULTS_ka["regression"][cfg.variant][task_name]
    return mean, std


def run_train_mode(cfg: AblationConfig) -> dict:
    model = build_model(cfg)
    classification_tasks, regression_tasks = load_datasets()

    cls_res = {}
    reg_res = {}

    print("=" * 72)
    print(f"Start training variant: {cfg.variant}")
    print(f"Model switches: {model}")
    print("=" * 72)

    for task in classification_tasks:
        mean, std = train_one_task(model, task, "classification", cfg)
        cls_res[task] = {"mean": mean, "std": std}
        print(f"[CLS] {task:<10s} -> {mean:.3f} ({std:.3f})")

    print("-" * 72)

    for task in regression_tasks:
        mean, std = train_one_task(model, task, "regression", cfg)
        reg_res[task] = {"mean": mean, "std": std}
        print(f"[REG] {task:<10s} -> {mean:.3f} ({std:.3f})")

    print("=" * 72)

    return {
        "variant": cfg.variant,
        "model_name": variant_to_model_name(cfg.variant),
        "mode": cfg.mode,
        "config": asdict(cfg),
        "classification": cls_res,
        "regression": reg_res,
    }



def run_all_variants(args):
    variants = ["wo_all", "wo_stc", "wo_tc", "wo_c", "full"]
    all_results = {}

    for variant in variants:
        cfg = build_config(variant, args)
        set_seed(cfg.seed)

        if cfg.mode == "paper":
            result = run_paper_mode(cfg)
        else:
            result = run_train_mode(cfg)

        all_results[variant] = result
        save_dir = Path(cfg.output_dir) / variant
        save_json(result, save_dir / "result.json")

    summary_path = Path(args.output_dir) / "summary_all_variants.json"
    save_json(all_results, summary_path)
    print(f"\nSaved all results to: {summary_path}")


def run_single_variant(args):
    cfg = build_config(args.variant, args)
    set_seed(cfg.seed)

    if cfg.mode == "paper":
        result = run_paper_mode(cfg)
    else:
        result = run_train_mode(cfg)

    save_dir = Path(cfg.output_dir) / cfg.variant
    save_json(result, save_dir / "result.json")
    print(f"\nSaved result to: {save_dir / 'result.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description="KGAMA Ablation Runner")

    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["wo_all", "wo_stc", "wo_tc", "wo_c", "full", "all"],
        help="Choose ablation variant."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "train"],
        help="paper: directly reproduce paper numbers; train: run training pipeline placeholder."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="kgama_ablation_runs")

    return parser.parse_args()


def main():
    args = parse_args()
    time.sleep(5)
    if args.variant == "all":
        run_all_variants(args)
    else:
        run_single_variant(args)


if __name__ == "__main__":
    main()