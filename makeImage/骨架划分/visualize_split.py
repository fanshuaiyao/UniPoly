"""
Visualize random split vs scaffold split on BBB_Martins dataset.

Uses scaffold-level Tanimoto distance for t-SNE + bar chart of scaffold overlap.

Usage:
    pip install rdkit matplotlib scikit-learn pandas
    python visualize_split.py --data_dir path/to/bbb_martins
"""

import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RDLogger.DisableLog('rdApp.*')


def get_morgan_fp(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return fp


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def random_split(n_total, n_train, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(n_total)
    return indices[:n_train], indices[n_train:]


def scaffold_split(df, n_train):
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(df["Drug"]):
        s = get_scaffold(smiles)
        if s is not None:
            scaffolds[s].append(i)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    train_idx, test_idx = [], []
    for indices in scaffold_sets:
        if len(train_idx) + len(indices) <= n_train:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)
    return np.array(train_idx), np.array(test_idx)


def compute_tanimoto_distance_matrix(fps):
    """Compute pairwise Tanimoto distance matrix."""
    n = len(fps)
    dist = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        for j, s in enumerate(sims):
            dist[i][i+1+j] = 1.0 - s
            dist[i+1+j][i] = 1.0 - s
    return dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.data_dir, "train_val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    n_train = len(train_df)
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Total samples: {len(df)} (Train: {n_train}, Test: {len(test_df)})")

    # Compute scaffold fingerprints (use scaffold SMILES for fingerprint, not original molecule)
    print("Computing scaffold fingerprints...")
    scaffold_list = []
    scaffold_fps = []
    for smiles in df["Drug"]:
        scaf = get_scaffold(smiles)
        scaffold_list.append(scaf)
        if scaf is not None:
            mol = Chem.MolFromSmiles(scaf)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                scaffold_fps.append(fp)
            else:
                scaffold_fps.append(AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles('C'), 2, nBits=1024))
        else:
            scaffold_fps.append(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles('C'), 2, nBits=1024))

    # Tanimoto distance matrix -> t-SNE
    print("Computing Tanimoto distance matrix...")
    dist_matrix = compute_tanimoto_distance_matrix(scaffold_fps)

    print("Running t-SNE on distance matrix...")
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000,
                metric="precomputed", init="random")
    coords = tsne.fit_transform(dist_matrix)

    # Splits
    rand_train, rand_test = random_split(len(df), n_train, seed=args.seed)
    scaf_train, scaf_test = scaffold_split(df, n_train)

    def get_scaffold_sets(indices):
        return set(scaffold_list[i] for i in indices if scaffold_list[i] is not None)

    # Stats
    rand_train_scaf = get_scaffold_sets(rand_train)
    rand_test_scaf = get_scaffold_sets(rand_test)
    scaf_train_scaf = get_scaffold_sets(scaf_train)
    scaf_test_scaf = get_scaffold_sets(scaf_test)

    rand_shared = len(rand_train_scaf & rand_test_scaf)
    rand_train_only = len(rand_train_scaf - rand_test_scaf)
    rand_test_only = len(rand_test_scaf - rand_train_scaf)

    scaf_shared = len(scaf_train_scaf & scaf_test_scaf)
    scaf_train_only = len(scaf_train_scaf - scaf_test_scaf)
    scaf_test_only = len(scaf_test_scaf - scaf_train_scaf)

    # --- Plot: 2x2 layout ---
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.3)

    color_train = "#4CAF50"
    color_test = "#F44336"
    alpha = 0.6
    s = 10

    # ---- Top left: Random Split scatter ----
    ax1 = fig.add_subplot(gs[0, 0])
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[rand_train] = True
    test_mask = ~train_mask
    ax1.scatter(coords[train_mask, 0], coords[train_mask, 1],
                c=color_train, label=f"Train ({train_mask.sum()})",
                alpha=alpha, s=s, edgecolors="none")
    ax1.scatter(coords[test_mask, 0], coords[test_mask, 1],
                c=color_test, label=f"Test ({test_mask.sum()})",
                alpha=alpha, s=s, edgecolors="none")
    ax1.set_title("Random Split", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.legend(fontsize=11)

    # ---- Top right: Scaffold Split scatter ----
    ax2 = fig.add_subplot(gs[0, 1])
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[scaf_train] = True
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[scaf_test] = True
    ax2.scatter(coords[train_mask, 0], coords[train_mask, 1],
                c=color_train, label=f"Train ({train_mask.sum()})",
                alpha=alpha, s=s, edgecolors="none")
    ax2.scatter(coords[test_mask, 0], coords[test_mask, 1],
                c=color_test, label=f"Test ({test_mask.sum()})",
                alpha=alpha, s=s, edgecolors="none")
    ax2.set_title("Scaffold Split", fontsize=14, fontweight="bold")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(fontsize=11)

    # ---- Bottom left: Random split scaffold overlap bar ----
    ax3 = fig.add_subplot(gs[1, 0])
    categories = ["Train Only", "Shared", "Test Only"]
    rand_vals = [rand_train_only, rand_shared, rand_test_only]
    bar_colors = [color_train, "#FFC107", color_test]
    bars = ax3.bar(categories, rand_vals, color=bar_colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, rand_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax3.set_ylabel("Number of Scaffolds", fontsize=12)
    ax3.set_title("Random Split — Scaffold Overlap", fontsize=13)
    ax3.set_ylim(0, max(rand_vals) * 1.2)

    # ---- Bottom right: Scaffold split scaffold overlap bar ----
    ax4 = fig.add_subplot(gs[1, 1])
    scaf_vals = [scaf_train_only, scaf_shared, scaf_test_only]
    bars = ax4.bar(categories, scaf_vals, color=bar_colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, scaf_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax4.set_ylabel("Number of Scaffolds", fontsize=12)
    ax4.set_title("Scaffold Split — Scaffold Overlap", fontsize=13)
    ax4.set_ylim(0, max(scaf_vals) * 1.2)

    # Use dataset folder name as identifier
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))

    plt.suptitle(f"{dataset_name}: Random Split vs Scaffold Split",
                 fontsize=16, fontweight="bold", y=0.98)

    save_dir = "/root/UniPoly/makeImage/骨架划分"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_split_visualization.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nVisualization saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
