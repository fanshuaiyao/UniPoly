import argparse
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check max token length of Description column in a CSV."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV path.")
    parser.add_argument(
        "--text_col",
        default="Description",
        help="Column name containing text (default: Description).",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="GT4SD/multitask-text-and-chemistry-t5-base-augm",
        help="HuggingFace tokenizer name or path.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Chunk size for pandas.read_csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    max_len = 0
    all_lengths = []

    for chunk in pd.read_csv(args.input_csv, chunksize=args.chunksize):
        texts = chunk[args.text_col].dropna().astype(str).tolist()

        for text in tqdm(texts, desc="Counting tokens", leave=False):
            tokens = tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
            length = len(tokens)
            all_lengths.append(length)
            max_len = max(max_len, length)

    lengths = np.array(all_lengths)

    print("\n========== Description Token Statistics ==========")
    print(f"Total samples: {len(lengths)}")
    print(f"Max token length: {lengths.max()}")
    print(f"Mean token length: {lengths.mean():.2f}")
    print(f"Median token length: {np.median(lengths):.2f}")
    print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(lengths, 99):.2f}")
    print("==================================================")


if __name__ == "__main__":
    main()
