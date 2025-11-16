import pandas as pd
from pathlib import Path


def summarize_results(csv_path: Path):
    df = pd.read_csv(csv_path)
    relevant = df[
        ["distill_type", "seed", "epoch", "train_loss", "validation_loss", "validation_accuracy"]
    ].copy()
    grouped = (
        relevant.groupby(["distill_type", "seed"])
        .agg(
            final_epoch=("epoch", "max"),
            final_train_loss=("train_loss", "last"),
            final_val_loss=("validation_loss", "last"),
            final_val_acc=("validation_accuracy", "last"),
        )
        .reset_index()
    )
    return grouped


def main():
    csv_path = Path("Whitebox vs Blackbox/results/results_summary.csv")
    if not csv_path.exists():
        print(f"No results found at {csv_path}")
        return
    summary = summarize_results(csv_path)
    print(summary)


if __name__ == "__main__":
    main()

