"""
Local training script for white-box vs black-box knowledge distillation.

This version removes the Ray dependency entirely and runs trials sequentially
on the current machine. Results are logged to a pandas DataFrame and saved to
`config.OUTPUT_PATH/results_summary.csv`.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import re

import config
os.environ.setdefault("OMP_NUM_THREADS", str(config.MAX_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(config.MAX_CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(config.MAX_CPU_THREADS))

from distillation_core import (
    OfflineDistillationDataset,
    compute_loss,
    list_parquet_files,
)
from distillation_student import DistillationStudent


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(config.MAX_CPU_THREADS, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataset(tokenizer: AutoTokenizer) -> OfflineDistillationDataset:
    parquet_files = list_parquet_files(config.OFFLINE_DATA_PATH)
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found under {config.OFFLINE_DATA_PATH}. "
            "Please generate offline teacher data before training."
        )
    first_file = parquet_files[0]
    print(f"Loading offline dataset from {first_file}")
    return OfflineDistillationDataset(first_file, tokenizer, config.MAX_SEQ_LENGTH)


def _split_dataset(dataset, seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    val_size = max(1, int(len(dataset) * config.VALIDATION_SPLIT))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError(
            f"Dataset too small ({len(dataset)} samples) for validation "
            f"split {config.VALIDATION_SPLIT}"
        )
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def _create_dataloader(subset, shuffle: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _prepare_teacher_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    teacher_data: Dict[str, torch.Tensor] = {}
    if "teacher_logits" in batch and batch["teacher_logits"] is not None:
        teacher_data["teacher_logits"] = batch["teacher_logits"].to(device)
    if "teacher_hidden_state" in batch and batch["teacher_hidden_state"] is not None:
        teacher_data["teacher_hidden_state"] = batch["teacher_hidden_state"].to(device)
    if "teacher_attention_map" in batch and batch["teacher_attention_map"] is not None:
        teacher_data["teacher_attention_map"] = batch["teacher_attention_map"].to(device)
    return teacher_data


_non_digit = re.compile(r"[^0-9\-]")


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.replace("\n", " ").replace("Answer:", "").replace("answer:", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _normalize_numeric(text: str) -> str:
    stripped = _non_digit.sub("", text)
    return stripped.lstrip("0") or stripped


def _answers_match(pred: str, target: str) -> bool:
    if not target:
        return False
    if pred == target:
        return True
    # Numeric fallback (e.g., GSM8K)
    pred_num = _normalize_numeric(pred)
    target_num = _normalize_numeric(target)
    if pred_num and target_num and pred_num == target_num:
        return True
    # Match first token/word for single-word answers
    pred_first = pred.split(" ")[0]
    target_first = target.split(" ")[0]
    if pred_first == target_first:
        return True
    return False


def _compute_batch_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: AutoTokenizer,
    task_names: List[str],
    task_totals: Dict[str, int],
    task_correct: Dict[str, int],
) -> Tuple[int, int]:
    preds = logits.argmax(dim=-1)
    total = 0
    correct = 0
    for pred_seq, label_seq, task_name in zip(preds, labels, task_names):
        mask = label_seq != -100
        if mask.sum() == 0:
            continue
        pred_tokens = pred_seq[mask]
        label_tokens = label_seq[mask]
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        target_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
        pred_norm = _normalize_text(pred_text)
        target_norm = _normalize_text(target_text)
        if _answers_match(pred_norm, target_norm):
            correct += 1
            task_correct[task_name] = task_correct.get(task_name, 0) + 1
        task_totals[task_name] = task_totals.get(task_name, 0) + 1
        total += 1
    return correct, total


def evaluate_model(
    model: DistillationStudent,
    data_loader: DataLoader,
    device: torch.device,
    distill_type: str,
    tokenizer: AutoTokenizer,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    correct = 0
    evaluated = 0
    task_totals: Dict[str, int] = {}
    task_correct: Dict[str, int] = {}
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_data = _prepare_teacher_batch(batch, device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_states=distill_type in ["hidden_state", "combined"],
                return_attention=distill_type in ["attention", "combined"],
                output_attentions=distill_type in ["attention", "combined"],
            )
            losses = compute_loss(outputs, labels, teacher_data, distill_type)
            total_loss += losses["task_loss"].item()
            batch_correct, batch_total = _compute_batch_accuracy(
                outputs["logits"],
                labels,
                tokenizer,
                batch.get("task_name", ["unknown"] * labels.size(0)),
                task_totals,
                task_correct,
            )
            correct += batch_correct
            evaluated += batch_total
            num_batches += 1
    model.train()
    if num_batches == 0:
        return 0.0, 0.0, {}
    accuracy = (correct / evaluated) if evaluated else 0.0
    per_task = {
        task: task_correct.get(task, 0) / total if total else 0.0
        for task, total in task_totals.items()
    }
    return total_loss / num_batches, accuracy, per_task


def run_trial(distill_type: str, seed: int, learning_rate: float, batch_size: int) -> List[Dict[str, float]]:
    print(f"\n=== Trial distill_type={distill_type}, seed={seed} ===")
    set_random_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = _load_dataset(tokenizer)
    train_subset, val_subset = _split_dataset(dataset, seed)
    effective_batch_size = batch_size
    if distill_type in ["hidden_state", "attention", "combined"]:
        effective_batch_size = getattr(config, "WHITEBOX_BATCH_SIZE", batch_size)
    train_loader = _create_dataloader(train_subset, shuffle=True, batch_size=effective_batch_size)
    val_loader = _create_dataloader(val_subset, shuffle=False, batch_size=effective_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = DistillationStudent(config.STUDENT_MODEL_NAME).to(device)
    print(f"Using device: {device}")
    student_model.train()
    param_dtype = next(student_model.parameters()).dtype

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    amp_enabled = (
        device.type == "cuda"
        and getattr(config, "USE_AUTOMATIC_MIXED_PRECISION", False)
        and param_dtype == torch.float32
    )
    scaler = GradScaler("cuda", enabled=amp_enabled)
    autocast_kwargs = {
        "device_type": "cuda",
        "dtype": torch.float16,
        "enabled": amp_enabled,
    }

    history: List[Dict[str, float]] = []
    num_epochs = config.NUM_EPOCHS

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batches = 0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_data = _prepare_teacher_batch(batch, device)

            cm = autocast(**autocast_kwargs) if amp_enabled else contextlib.nullcontext()
            with cm:
                outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_hidden_states=distill_type in ["hidden_state", "combined"],
                    return_attention=distill_type in ["attention", "combined"],
                    output_attentions=distill_type in ["attention", "combined"],
                )
                losses = compute_loss(outputs, labels, teacher_data, distill_type)
                total_loss = losses["total_loss"]

            optimizer.zero_grad()
            if scaler is not None and scaler.is_enabled():
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
            batches += 1

        avg_train_loss = epoch_loss / batches if batches else 0.0
        val_loss, val_accuracy, task_breakdown = evaluate_model(
            student_model, val_loader, device, distill_type, tokenizer
        )
        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} "
            f"time={duration:.1f}s"
        )

        record = {
            "timestamp": time.time(),
            "distill_type": distill_type,
            "seed": seed,
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "learning_rate": learning_rate,
            "num_train_batches": batches,
            "epoch_time_sec": duration,
        }
        for task, acc in task_breakdown.items():
            record[f"validation_accuracy_{task}"] = acc
        history.append(record)

    return history


def save_results(records: List[Dict[str, float]]) -> None:
    if not records:
        print("No records to save.")
        return
    df = pd.DataFrame(records)
    output_dir = Path(config.OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "results_summary.csv"
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(records)} record(s) to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Local KD training (no Ray).")
    parser.add_argument(
        "--distill-types",
        type=str,
        default="black_box,hidden_state,attention,combined",
        help="Comma-separated list of distillation types to run.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Comma-separated list of integer seeds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Override learning rate (default uses config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size per step (default from config).",
    )
    parser.add_argument(
        "--single-trial",
        action="store_true",
        help="Run only the first distill_type/seed pair for a smoke test.",
    )
    args = parser.parse_args()

    distill_types = [t.strip() for t in args.distill_types.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not distill_types:
        raise ValueError("At least one distillation type must be specified.")
    if not seeds:
        raise ValueError("At least one seed must be specified.")

    combos: List[Tuple[str, int]] = [(dt, seed) for dt in distill_types for seed in seeds]
    if args.single_trial:
        combos = combos[:1]
        print("Running in single-trial mode.")

    all_records: List[Dict[str, float]] = []
    for distill_type, seed in combos:
        trial_records = run_trial(distill_type, seed, args.learning_rate, args.batch_size)
        all_records.extend(trial_records)

    save_results(all_records)
    print("Training run complete.")


if __name__ == "__main__":
    main()

