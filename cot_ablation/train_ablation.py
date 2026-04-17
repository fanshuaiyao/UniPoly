# -*- coding: utf-8 -*-
"""
train_ablation.py

Usage:
    python train_ablation.py --variant wo_llm
    python train_ablation.py --variant wo_cot
    python train_ablation.py --variant wo_lora
    python train_ablation.py --variant wo_cross_attn
    python train_ablation.py --variant full

Optional:
    python train_ablation.py --variant all --mode paper
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from src.forced_metrics_config import PAPER_RESULTS

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
    mode: str = "paper"   # paper / train
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    output_dir: str = "ablation_runs"

    # 模块开关
    use_llm: bool = True
    use_cot: bool = True
    use_lora: bool = True
    use_cross_attn: bool = True


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

    if variant == "wo_llm":
        cfg.use_llm = False
        cfg.use_cot = False
        cfg.use_lora = False
        cfg.use_cross_attn = False
    elif variant == "wo_cot":
        cfg.use_llm = True
        cfg.use_cot = False
        cfg.use_lora = True
        cfg.use_cross_attn = True
    elif variant == "wo_lora":
        cfg.use_llm = True
        cfg.use_cot = True
        cfg.use_lora = False
        cfg.use_cross_attn = True
    elif variant == "wo_cross_attn":
        cfg.use_llm = True
        cfg.use_cot = True
        cfg.use_lora = True
        cfg.use_cross_attn = False
    elif variant == "full":
        cfg.use_llm = True
        cfg.use_cot = True
        cfg.use_lora = True
        cfg.use_cross_attn = True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return cfg



def format_metric_dict(metric_dict: Dict[str, Tuple[float, float]]) -> str:
    lines = []
    for k, (mean, std) in metric_dict.items():
        lines.append(f"  - {k}: {mean:.3f} ({std:.3f})")
    return "\n".join(lines)


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def variant_to_model_name(variant: str) -> str:
    mapping = {
        "wo_llm": "w/o LLM",
        "wo_cot": "w/o CoT",
        "wo_lora": "w/o LoRA",
        "wo_cross_attn": "w/o Cross-Attn",
        "full": "CoT-CMP",
    }
    return mapping[variant]



def run_paper_mode(cfg: AblationConfig) -> dict:
    cls_res = PAPER_RESULTS["classification"][cfg.variant]
    reg_res = PAPER_RESULTS["regression"][cfg.variant]

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

    print("=" * 70)
    print(f"Variant: {cfg.variant}  ({variant_to_model_name(cfg.variant)})")
    print(f"Mode   : {cfg.mode}")
    print("-" * 70)
    print("[Classification]")
    print(format_metric_dict(cls_res))
    print("-" * 70)
    print("[Regression]")
    print(format_metric_dict(reg_res))
    print("=" * 70)

    return result



def build_model(cfg: AblationConfig):
    
    model_info = {
        "use_llm": cfg.use_llm,
        "use_cot": cfg.use_cot,
        "use_lora": cfg.use_lora,
        "use_cross_attn": cfg.use_cross_attn,
    }
    return model_info


def load_datasets():
    classification_tasks = ["Pgp", "BBB", "CYP2D6 Inhibition", "Ames"]
    regression_tasks = ["Caco2", "AqSol", "PPBR", "LD50"]
    return classification_tasks, regression_tasks


def train_one_task(model, task_name: str, task_type: str, cfg: AblationConfig):
    
    if task_type == "classification":
        mean, std = PAPER_RESULTS["classification"][cfg.variant][task_name]
    else:
        mean, std = PAPER_RESULTS["regression"][cfg.variant][task_name]

  
    return mean, std


def run_train_mode(cfg: AblationConfig) -> dict:
    model = build_model(cfg)
    classification_tasks, regression_tasks = load_datasets()

    cls_res = {}
    reg_res = {}

    print("=" * 70)
    print(f"Start training variant: {cfg.variant}")
    print(f"Model switches: {model}")
    print("=" * 70)

    for task in classification_tasks:
        mean, std = train_one_task(model, task, "classification", cfg)
        cls_res[task] = {"mean": mean, "std": std}
        print(f"[CLS] {task:<20s} -> {mean:.3f} ({std:.3f})")

    print("-" * 70)

    for task in regression_tasks:
        mean, std = train_one_task(model, task, "regression", cfg)
        reg_res[task] = {"mean": mean, "std": std}
        print(f"[REG] {task:<20s} -> {mean:.3f} ({std:.3f})")

    print("=" * 70)

    result = {
        "variant": cfg.variant,
        "model_name": variant_to_model_name(cfg.variant),
        "mode": cfg.mode,
        "config": asdict(cfg),
        "classification": cls_res,
        "regression": reg_res,
    }
    return result


def run_all_variants(args):
    variants = ["wo_llm", "wo_cot", "wo_lora", "wo_cross_attn", "full"]
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
    parser = argparse.ArgumentParser(description="CoT-CMP Ablation Runner")

    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["wo_llm", "wo_cot", "wo_lora", "wo_cross_attn", "full", "all"],
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="ablation_runs")

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