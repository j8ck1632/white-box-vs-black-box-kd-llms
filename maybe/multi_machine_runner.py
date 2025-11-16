"""
Lightweight single-machine training loop for the white-box vs black-box study.

Usage examples:

    # Machine A runs all black-box seeds
    python multi_machine_runner.py --distill-type black_box --seeds 0 1 2 3

    # Machine B runs white-box (hidden) seeds 0 and 1
    python multi_machine_runner.py --distill-type hidden_state --seeds 0 1

This script intentionally avoids Ray so that three independent machines can
split the workload without cluster orchestration.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from distillation_core import OfflineDistillationDataset, compute_loss, list_parquet_files
from distillation_student import DistillationStudent


def _select_parquet_file(user_path: str | None) -> str:
    if user_path:
        return user_path
    files = list_parquet_files(config.OFFLINE_DATA_PATH)
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found under {config.OFFLINE_DATA_PATH}. "
            "Run offline_teacher_data.py first."
        )
    return files[0]


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_single_seed(
    distill_type: str,
    seed: int,
    parquet_file: str,
    output_dir: Path,
    learning_rate: float,
    num_epochs: int,
    max_batches: int | None = None,
) -> List[dict]:
    _set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = OfflineDistillationDataset(parquet_file, tokenizer, config.MAX_SEQ_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = DistillationStudent(config.STUDENT_MODEL_NAME).to(device)
    student_model.train()

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    amp_enabled = device.type == "cuda"
    scaler = None
    if amp_enabled:
        param_dtype = next(student_model.parameters()).dtype
        use_grad_scaler = param_dtype != torch.float16
        if use_grad_scaler:
            scaler = torch.amp.GradScaler("cuda", enabled=True)

    metrics: List[dict] = []
    batches_seen = 0
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batches_seen += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            teacher_data = {}
            if "teacher_logits" in batch:
                teacher_data["teacher_logits"] = batch["teacher_logits"].to(device)
            if "teacher_hidden_state" in batch:
                teacher_data["teacher_hidden_state"] = batch["teacher_hidden_state"].to(device)
            if "teacher_attention_map" in batch:
                teacher_data["teacher_attention_map"] = batch["teacher_attention_map"].to(device)

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if amp_enabled else contextlib.nullcontext()
            )
            with autocast_ctx:
                student_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_hidden_states=distill_type in ["hidden_state", "combined"],
                    return_attention=distill_type in ["attention", "combined"],
                    output_attentions=distill_type in ["attention", "combined"],
                )
                losses = compute_loss(student_outputs, labels, teacher_data, distill_type)
                total_loss = losses["total_loss"]

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

            if max_batches and batches_seen >= max_batches:
                print(f"Reached max_batches={max_batches}; stopping early.")
                break

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        epoch_time = time.time() - epoch_start
        metrics.append(
            {
                "epoch": epoch,
                "avg_loss": avg_epoch_loss,
                "distill_type": distill_type,
                "seed": seed,
                "epoch_time_sec": epoch_time,
                "batches": num_batches,
            }
        )
        print(
            f"[{distill_type}][seed={seed}] Epoch {epoch} "
            f"loss={avg_epoch_loss:.4f} elapsed={epoch_time:.1f}s"
        )
        if max_batches and batches_seen >= max_batches:
            break

    total_time = time.time() - total_start
    run_summary = {
        "distill_type": distill_type,
        "seed": seed,
        "epochs_completed": len(metrics),
        "total_time_sec": total_time,
        "parquet_file": parquet_file,
        "learning_rate": learning_rate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"seed_{seed}_metrics.json", "w") as f:
        json.dump(
            {
                "summary": run_summary,
                "epochs": metrics,
            },
            f,
            indent=2,
        )
    return metrics


def parse_args(cli_args: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-machine KD runner (no Ray)")
    parser.add_argument(
        "--distill-type",
        choices=["black_box", "hidden_state", "attention", "combined"],
        required=True,
        help="Which experimental group to run on this machine.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="List of random seeds to run sequentially on this machine.",
    )
    parser.add_argument(
        "--parquet-file",
        type=str,
        default=None,
        help="Optional explicit Parquet file path. Defaults to the first file found.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(config.OUTPUT_PATH, "multi_machine"),
        help="Where to drop JSON metrics per seed.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Override learning rate for experimentation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of epochs to train per seed.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on total batches for quick smoke tests.",
    )
    return parser.parse_args(cli_args)


def main(cli_args: List[str] | None = None):
    args = parse_args(cli_args)
    parquet_file = _select_parquet_file(args.parquet_file)
    output_dir = Path(args.output_dir) / args.distill_type
    print(f"Using Parquet file: {parquet_file}")
    print(f"Writing per-seed metrics to: {output_dir}")

    for seed in args.seeds:
        print(f"\n=== Running distill_type={args.distill_type} seed={seed} ===")
        _train_single_seed(
            distill_type=args.distill_type,
            seed=seed,
            parquet_file=parquet_file,
            output_dir=output_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            max_batches=args.max_batches,
        )


if __name__ == "__main__":
    main()


