import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Add task metadata to teacher parquet files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing parquet files produced by offline_teacher_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to write patched parquet files. Defaults to in-place updates.",
    )
    parser.add_argument(
        "--sst2-count",
        type=int,
        default=5000,
        help="Number of SST-2 examples stored at the start of the parquet files.",
    )
    parser.add_argument(
        "--mmlu-count",
        type=int,
        default=1000,
        help="Number of MMLU examples following SST-2.",
    )
    parser.add_argument(
        "--gsm8k-count",
        type=int,
        default=1000,
        help="Number of GSM8K examples following MMLU.",
    )
    return parser.parse_args()


def compute_task_labels(length: int, sst2: int, mmlu: int, gsm8k: int):
    labels = []
    for idx in range(length):
        if idx < sst2:
            labels.append("sst2")
        elif idx < sst2 + mmlu:
            labels.append("mmlu")
        elif idx < sst2 + mmlu + gsm8k:
            labels.append("gsm8k")
        else:
            labels.append("unknown")
    return labels


def add_task_column(parquet_path: Path, output_dir: Path | None, counts) -> None:
    df = pd.read_parquet(parquet_path)
    total = len(df)
    labels = compute_task_labels(total, *counts)
    df["task_name"] = labels

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / parquet_path.name
    else:
        out_path = parquet_path
    df.to_parquet(out_path, index=False)
    print(f"Patched {parquet_path} -> {out_path} ({total} rows)")


def main():
    args = parse_args()
    counts = (args.sst2_count, args.mmlu_count, args.gsm8k_count)
    parquet_files = sorted(p for p in args.input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.input_dir}")

    for parquet_path in parquet_files:
        add_task_column(parquet_path, args.output_dir, counts)


if __name__ == "__main__":
    main()

