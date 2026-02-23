"""
Visualize classification label distribution using bar chart.

Shows the count of class 0 and class 1 in the dataset.

Usage:
    python visualize_classification.py --csv_file path/to/data.csv --label_column label_name
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Visualize classification label distribution"
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
        help="Output image filename (default: {label_column}_distribution.png)"
    )
    args = parser.parse_args()

    # 2. 设置风格（参考回归任务）
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Read CSV file
    df = pd.read_csv(args.csv_file)

    # Check if label column exists
    if args.label_column not in df.columns:
        raise ValueError(f"Column '{args.label_column}' not found in CSV. "
                         f"Available columns: {list(df.columns)}")

    # Count labels (including NaN)
    count_0 = ((df[args.label_column] == 0).sum())
    count_1 = ((df[args.label_column] == 1).sum())
    count_nan = df[args.label_column].isna().sum()

    # Build categories, values, colors based on what exists
    categories = []
    values = []
    colors = []

    # Color scheme (参考回归任务风格，使用深色调)
    color_0 = "#2E7D32"  # 深绿色
    color_1 = "#6D011F"  # 深红色（回归任务使用的颜色）
    color_nan = "#757575"  # 灰色

    # Add Class 0
    if count_0 > 0 or (count_0 == 0 and count_1 > 0):
        categories.append("Class 0")
        values.append(count_0)
        colors.append(color_0)

    # Add Class 1
    if count_1 > 0 or (count_1 == 0 and count_0 > 0):
        categories.append("Class 1")
        values.append(count_1)
        colors.append(color_1)

    # Add NaN only if there are any
    if count_nan > 0:
        categories.append("NaN")
        values.append(count_nan)
        colors.append(color_nan)

    total_non_nan = count_0 + count_1
    total = total_non_nan + count_nan

    # Extract filename (without path and extension) for title and output
    csv_filename = os.path.basename(args.csv_file)
    title_name = csv_filename.split('.')[0]

    print(f"Label distribution:")
    if count_0 > 0 or (count_0 == 0 and count_1 > 0):
        print(f"  Class 0: {count_0} ({count_0/total_non_nan*100:.1f}%)" if total_non_nan > 0 else f"  Class 0: {count_0}")
    if count_1 > 0 or (count_1 == 0 and count_0 > 0):
        print(f"  Class 1: {count_1} ({count_1/total_non_nan*100:.1f}%)" if total_non_nan > 0 else f"  Class 1: {count_1}")
    if count_nan > 0:
        print(f"  NaN: {count_nan}")
    print(f"  Total: {total}")

    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # 4. 绘制柱状图
    bars = ax.bar(categories, values, color=colors, edgecolor='black', width=0.6, alpha=0.8)

    # 添加百分比标签（柱子顶部，字体11与信息框统一）
    for i, (bar, val, category) in enumerate(zip(bars, values, categories)):
        height = bar.get_height()
        percent = val / total * 100 if total > 0 else 0
        # 使用2位小数精度，使总和更接近100%
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(values) * 0.01,
                f"{percent:.1f}%", ha='center', va='bottom', fontsize=11)

    # 5. 添加统计信息框（左上角，文字左对齐）
    # Build stats text based on what exists
    stats_text = f"Total = {total}\n"
    if count_0 > 0 or (count_0 == 0 and count_1 > 0):
        stats_text += f"Class 0 = {count_0}\n"
    if count_1 > 0 or (count_1 == 0 and count_0 > 0):
        stats_text += f"Class 1 = {count_1}"
    if count_nan > 0:
        if "Class 0" in stats_text or "Class 1" in stats_text:
            stats_text += f"\nNaN = {count_nan}"
        else:
            stats_text += f"NaN = {count_nan}"

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.05, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    # 6. 设置标签（参考回归任务风格）
    ax.set_xlabel('Class', fontsize=12, labelpad=10)
    ax.set_ylabel('Count', fontsize=12, labelpad=10)
    ax.set_title(f'{title_name} Dataset Distribution', fontsize=14, pad=15)
    ax.set_ylim(0, max(values) * 1.2)

    # Set output filename
    if args.output_name:
        output_path = os.path.join(os.getcwd(), args.output_name)
    else:
        output_path = os.path.join(os.getcwd(), f"{title_name}_distribution.png")

    # 7. 保存（参考回归任务：dpi=300）
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
