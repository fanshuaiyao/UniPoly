"""
Visualize random split vs scaffold split using hexbin heatmap.

Each hex cell shows the proportion of Test samples: Test / (Train + Test).
Random split should show uniform color, scaffold split should show distinct regions.

Usage:
    python visualize_split_hexbin.py --data_dir /root/UniPoly/ADMETdata/bbb_martins
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
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

RDLogger.DisableLog('rdApp.*')


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
    n = len(fps)
    dist = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        for j, s in enumerate(sims):
            dist[i][i+1+j] = 1.0 - s
            dist[i+1+j][i] = 1.0 - s
    return dist


def hexbin_test_ratio(ax, coords, labels, gridsize=25):
    """
    Draw a hexbin heatmap where color = Test ratio in each hex cell.
    labels: array of 0 (Train) and 1 (Test).
    Returns the hexbin object for colorbar.
    """
    x, y = coords[:, 0], coords[:, 1]

    # Total count per hex
    hb_total = ax.hexbin(x, y, C=np.ones(len(x)), reduce_C_function=np.sum,
                         gridsize=gridsize, mincnt=1, alpha=0)
    offsets = hb_total.get_offsets()
    total_counts = hb_total.get_array()

    # Test count per hex
    hb_test = ax.hexbin(x, y, C=labels, reduce_C_function=np.sum,
                        gridsize=gridsize, mincnt=1, alpha=0)
    test_counts = hb_test.get_array()

    # Clear invisible hexbins
    ax.cla()

    # Compute ratio
    ratio = np.divide(test_counts, total_counts, where=total_counts > 0)

    # Draw final hexbin with ratio as color
    cmap = plt.cm.RdYlGn_r  # Green=Train(0), Red=Test(1)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    hb = ax.hexbin(x, y, C=labels, reduce_C_function=lambda vals: np.sum(vals) / len(vals),
                   gridsize=gridsize, mincnt=1, cmap=cmap, norm=norm,
                   edgecolors='white', linewidths=0.3)
    return hb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gridsize", type=int, default=25)
    args = parser.parse_args()

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))

    train_df = pd.read_csv(os.path.join(args.data_dir, "train_val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    n_train = len(train_df)
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {len(df)} (Train: {n_train}, Test: {len(test_df)})")

    # Scaffold fingerprints
    print("Computing scaffold fingerprints...")
    scaffold_fps = []
    for smiles in df["Drug"]:
        scaf = get_scaffold(smiles)
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

    # Tanimoto distance -> t-SNE
    print("Computing Tanimoto distance matrix...")
    dist_matrix = compute_tanimoto_distance_matrix(scaffold_fps)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000,
                metric="precomputed", init="random")
    coords = tsne.fit_transform(dist_matrix)

    # Splits
    rand_train, rand_test = random_split(len(df), n_train, seed=args.seed)
    scaf_train, scaf_test = scaffold_split(df, n_train)

    # Labels: 0=Train, 1=Test
    rand_labels = np.zeros(len(df))
    rand_labels[rand_test] = 1.0

    scaf_labels = np.zeros(len(df))
    scaf_labels[scaf_test] = 1.0

    # --- Plot ---
    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)

    # Random split hexbin
    ax1 = fig.add_subplot(gs[0, 0])
    hb1 = hexbin_test_ratio(ax1, coords, rand_labels, gridsize=args.gridsize)
    ax1.set_title(f"Random Split", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE 1", fontsize=12)
    ax1.set_ylabel("t-SNE 2", fontsize=12)

    # Scaffold split hexbin
    ax2 = fig.add_subplot(gs[0, 1])
    hb2 = hexbin_test_ratio(ax2, coords, scaf_labels, gridsize=args.gridsize)
    ax2.set_title(f"Scaffold Split", fontsize=14, fontweight="bold")
    ax2.set_xlabel("t-SNE 1", fontsize=12)
    ax2.set_ylabel("t-SNE 2", fontsize=12)

    # Shared colorbar
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(hb2, cax=cax)
    # cb.set_label("Test Ratio (0=All Train, 1=All Test)", fontsize=11)

    # plt.suptitle(f"Random Split vs Scaffold Split",
    #              fontsize=16, fontweight="bold", y=1.02)

    save_dir = "/home/fsy23/UniPoly/makeImage/骨架划分"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_hexbin3.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nVisualization saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
