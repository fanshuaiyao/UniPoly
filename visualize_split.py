"""
Visualize random split vs scaffold split on BBB_Martins dataset.

Uses the original train_val.csv as Train and test.csv as Test directly.

Usage:
    pip install rdkit matplotlib scikit-learn pandas
    python visualize_split.py --data_dir path/to/bbb_martins
"""

import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def get_morgan_fp(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


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
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return train_idx, test_idx


def scaffold_split(df, n_train):
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(df["Drug"]):
        s = get_scaffold(smiles)
        if s is not None:
            scaffolds[s].append(i)

    # Sort scaffolds by size (largest first)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train_idx, test_idx = [], []
    for indices in scaffold_sets:
        if len(train_idx) + len(indices) <= n_train:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)

    return np.array(train_idx), np.array(test_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to bbb_martins directory containing train_val.csv and test.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "train_val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    n_train = len(train_df)
    n_test = len(test_df)
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Total samples: {len(df)} (Train: {n_train}, Test: {n_test})")

    # Compute Morgan fingerprints
    print("Computing Morgan fingerprints...")
    fps = []
    valid_count = 0
    for smiles in df["Drug"]:
        fp = get_morgan_fp(smiles)
        if fp is not None:
            fps.append(fp)
            valid_count += 1
        else:
            fps.append(np.zeros(1024))
    fps = np.array(fps)
    print(f"Valid molecules: {valid_count}/{len(df)}")

    # PCA to 2D
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(fps)

    # Two split methods, keeping the same train/test ratio as the original files
    rand_train, rand_test = random_split(len(df), n_train, seed=args.seed)
    scaf_train, scaf_test = scaffold_split(df, n_train)

    # Count scaffolds per split
    scaffolds_all = [get_scaffold(s) for s in df["Drug"]]

    def count_unique_scaffolds(indices):
        return len(set(scaffolds_all[i] for i in indices if scaffolds_all[i] is not None))

    def count_shared_scaffolds(idx_a, idx_b):
        sa = set(scaffolds_all[i] for i in idx_a if scaffolds_all[i] is not None)
        sb = set(scaffolds_all[i] for i in idx_b if scaffolds_all[i] is not None)
        return len(sa & sb)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {"Train": "#4CAF50", "Test": "#F44336"}
    alpha = 0.5
    s = 12

    # Random split
    ax = axes[0]
    labels_rand = np.empty(len(df), dtype=object)
    labels_rand[rand_train] = "Train"
    labels_rand[rand_test] = "Test"
    for label in ["Train", "Test"]:
        mask = labels_rand == label
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[label], label=f"{label} ({mask.sum()})",
                   alpha=alpha, s=s, edgecolors="none")
    shared_rand = count_shared_scaffolds(rand_train, rand_test)
    ax.set_title(f"Random Split\n"
                 f"Train scaffolds: {count_unique_scaffolds(rand_train)} | "
                 f"Test scaffolds: {count_unique_scaffolds(rand_test)}\n"
                 f"Shared scaffolds: {shared_rand}",
                 fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=11)

    # Scaffold split
    ax = axes[1]
    labels_scaf = np.empty(len(df), dtype=object)
    labels_scaf[scaf_train] = "Train"
    labels_scaf[scaf_test] = "Test"
    for label in ["Train", "Test"]:
        mask = labels_scaf == label
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[label], label=f"{label} ({mask.sum()})",
                   alpha=alpha, s=s, edgecolors="none")
    shared_scaf = count_shared_scaffolds(scaf_train, scaf_test)
    ax.set_title(f"Scaffold Split\n"
                 f"Train scaffolds: {count_unique_scaffolds(scaf_train)} | "
                 f"Test scaffolds: {count_unique_scaffolds(scaf_test)}\n"
                 f"Shared scaffolds: {shared_scaf}",
                 fontsize=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=11)

    plt.suptitle("BBB_Martins: Random Split vs Scaffold Split", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_visualization1.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nVisualization saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
