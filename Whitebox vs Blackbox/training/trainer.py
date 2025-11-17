from __future__ import annotations

import contextlib
import time
from typing import Dict, List

import torch
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

import config
from distillation_core import compute_loss
from distillation_student import DistillationStudent

from .data import (
    build_tokenizer,
    create_dataloader,
    load_offline_dataset,
    split_dataset,
)
from .evaluation import evaluate_model
from .utils import prepare_teacher_batch, set_random_seed


def _configure_amp(device: torch.device, model: torch.nn.Module):
    param_dtype = next(model.parameters()).dtype
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
    return amp_enabled, scaler, autocast_kwargs


def _effective_batch_size(distill_type: str, batch_size: int) -> int:
    if distill_type in ["hidden_state", "attention", "combined"]:
        return getattr(config, "WHITEBOX_BATCH_SIZE", batch_size)
    return batch_size


def run_trial(
    distill_type: str,
    seed: int,
    learning_rate: float,
    batch_size: int,
) -> List[Dict[str, float]]:
    print(f"\n=== Trial distill_type={distill_type}, seed={seed} ===")
    set_random_seed(seed)

    tokenizer = build_tokenizer()
    dataset = load_offline_dataset(tokenizer)
    train_subset, val_subset = split_dataset(dataset, seed)
    effective_batch_size = _effective_batch_size(distill_type, batch_size)
    train_loader = create_dataloader(train_subset, shuffle=True, batch_size=effective_batch_size)
    val_loader = create_dataloader(val_subset, shuffle=False, batch_size=effective_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = DistillationStudent(config.STUDENT_MODEL_NAME).to(device)
    student_model.train()

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    amp_enabled, scaler, autocast_kwargs = _configure_amp(device, student_model)

    history: List[Dict[str, float]] = []
    num_epochs = config.NUM_EPOCHS
    grad_accum_steps = max(1, getattr(config, "GRADIENT_ACCUMULATION_STEPS", 1))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batches = 0
        start_time = time.time()
        total_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            train_loader,
            total=total_batches,
            desc=f"{distill_type} | epoch {epoch + 1}/{num_epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_data = prepare_teacher_batch(batch, device)

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

            loss_scale = total_loss / grad_accum_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_scale).backward()
                should_step = (step % grad_accum_steps == 0) or (step == total_batches)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss_scale.backward()
                should_step = (step % grad_accum_steps == 0) or (step == total_batches)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            step_loss = total_loss.item()
            epoch_loss += step_loss
            batches += 1
            progress.set_postfix({"loss": f"{step_loss:.4f}"}, refresh=False)

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

        record: Dict[str, float] = {
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

