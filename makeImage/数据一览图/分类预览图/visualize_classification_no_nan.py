"""
Visualize classification label distribution using bar chart.

Shows the count of class 0 and class 1 in the labeled subset of the dataset.
NaN values are ignored.

Usage:
    python visualize_classification_no_nan.py --csv_file path/to/data.csv --label_column label_name
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Visualize classification label distribution (ignore NaN)"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the CSV file"
    )
    parser.add_argument(
        "--label_column",
        type=str,
        required=True,
        help="Name of the column containing the labels (0 or 1)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output image filename (default: {csv_filename}_distribution.png)"
    )
    args = parser.parse_args()

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    df = pd.read_csv(args.csv_file)

    if args.label_column not in df.columns:
        raise ValueError(
            f"Column '{args.label_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    labeled_df = df[df[args.label_column].notna()]
    count_0 = (labeled_df[args.label_column] == 0).sum()
    count_1 = (labeled_df[args.label_column] == 1).sum()
    total = count_0 + count_1

    if total == 0:
        raise ValueError(
            f"No labeled class 0/1 samples found in column '{args.label_column}'."
        )

    categories = ["Class 0", "Class 1"]
    values = [count_0, count_1]
    colors = ["#2E7D32", "#6D011F"]

    csv_filename = os.path.basename(args.csv_file)
    title_name = os.path.splitext(csv_filename)[0]

    print("Label distribution (NaN ignored):")
    print(f"  Class 0: {count_0} ({count_0 / total * 100:.1f}%)")
    print(f"  Class 1: {count_1} ({count_1 / total * 100:.1f}%)")
    print(f"  Total: {total}")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    bars = ax.bar(categories, values, color=colors, edgecolor='black', width=0.6, alpha=0.8)

    label_offset = max(values) * 0.01 if max(values) > 0 else 0.01
    for bar, val in zip(bars, values):
        height = bar.get_height()
        percent = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + label_offset,
            f"{percent:.2f}%",
            ha='center',
            va='bottom',
            fontsize=11
        )

    stats_text = f"Total = {total}\nClass 0 = {count_0}\nClass 1 = {count_1}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
    )

    ax.set_xlabel('Class', fontsize=12, labelpad=10)
    ax.set_ylabel('Count', fontsize=12, labelpad=10)
    ax.set_title(f'{title_name} Dataset Distribution', fontsize=14, pad=15)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

    if args.output_name:
        output_path = os.path.join(os.getcwd(), args.output_name)
    else:
        output_path = os.path.join(os.getcwd(), f"{title_name}_distribution1.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
