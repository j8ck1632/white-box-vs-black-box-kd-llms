import argparse
from pathlib import Path

import pandas as pd


def summarize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    per_task_cols = [col for col in df.columns if col.startswith("validation_accuracy_")]
    summary = (
        df.groupby(["distill_type", "seed"])
        .agg(
            final_epoch=("epoch", "max"),
            final_train_loss=("train_loss", "last"),
            final_val_loss=("validation_loss", "last"),
            final_val_acc=("validation_accuracy", "last"),
            **{f"{col}_final": (col, "last") for col in per_task_cols},
        )
        .reset_index()
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize KD experiment results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("D:/wbvb_results/results_summary.csv"),
        help="Path to results_summary.csv",
    )
    args = parser.parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"{args.csv} not found")
    summary = summarize(args.csv)
    print(summary)
    out_path = args.csv.with_name("results_summary_by_seed.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nWrote per-seed summary to {out_path}")


if __name__ == "__main__":
    main()


