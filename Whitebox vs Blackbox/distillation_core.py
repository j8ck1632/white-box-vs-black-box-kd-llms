"""
Core components shared by both Ray-based and single-machine training flows.

This module intentionally avoids any Ray imports so it can be reused on
standalone machines that only need PyTorch + Transformers.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config


def _ensure_stride(max_length: int, stride: int) -> int:
    stride = max(1, stride)
    return math.ceil(max_length / stride)


class OfflineDistillationDataset(Dataset):
    """
    Dataset for loading pre-computed teacher outputs without Ray.

    The implementation mirrors the original Ray-based dataset but is kept in a
    separate module so it can be imported by lightweight training scripts.
    """

    def __init__(self, parquet_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_vocab_size = getattr(config, "TEACHER_VOCAB_SIZE", 32000)
        self.teacher_num_heads = getattr(config, "TEACHER_NUM_HEADS", 32)
        hidden_stride = getattr(config, "HIDDEN_STRIDE", 1)
        attention_stride = getattr(config, "ATTENTION_STRIDE", 1)
        self.max_hidden_seq_len = _ensure_stride(self.max_length, hidden_stride)
        self.max_attention_seq_len = _ensure_stride(self.max_length, attention_stride)

        df = pd.read_parquet(parquet_path)
        self.prompts = df["prompt"].tolist()
        self.answers = df["answer"].tolist()
        self.task_names = df["task_name"].tolist() if "task_name" in df.columns else ["unknown"] * len(df)

        self.teacher_topk_indices = df["teacher_topk_indices"].tolist() if "teacher_topk_indices" in df.columns else None
        self.teacher_topk_values = df["teacher_topk_values"].tolist() if "teacher_topk_values" in df.columns else None
        self.teacher_logits_dense = df["teacher_logits"].tolist() if "teacher_logits" in df.columns else None
        self.teacher_hidden_state = df["teacher_hidden_state"].tolist() if "teacher_hidden_state" in df.columns else None
        self.teacher_attention_map = df["teacher_attention_map"].tolist() if "teacher_attention_map" in df.columns else None

        self.student_input_ids = df["student_input_ids"].tolist() if "student_input_ids" in df.columns else None
        self.student_attention_mask = df["student_attention_mask"].tolist() if "student_attention_mask" in df.columns else None

    @staticmethod
    def _to_py(value):
        if hasattr(value, "as_py"):
            value = value.as_py()
        if isinstance(value, np.ndarray):
            return [OfflineDistillationDataset._to_py(elem) for elem in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [OfflineDistillationDataset._to_py(elem) for elem in value]
        return value

    @staticmethod
    def _to_numpy(value, dtype):
        data = OfflineDistillationDataset._to_py(value)
        try:
            return np.array(data, dtype=dtype)
        except (ValueError, TypeError):
            return np.stack([np.array(elem, dtype=dtype) for elem in data])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        answer = self.answers[idx]

        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length"
        )
        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length"
        )
        answer_ids = answer_tokens["input_ids"].squeeze(0)
        answer_mask = answer_tokens["attention_mask"].squeeze(0)
        answer_ids = answer_ids.masked_fill(answer_mask == 0, -100)

        item = {
            "input_ids": prompt_tokens["input_ids"].squeeze(0),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "labels": answer_ids,
            "task_name": self.task_names[idx],
        }

        if self.teacher_topk_indices is not None and self.teacher_topk_values is not None:
            item["teacher_logits"] = self._reconstruct_logits(idx)
        elif self.teacher_logits_dense is not None:
            logits_array = self._to_numpy(self.teacher_logits_dense[idx], np.float32)
            logits_tensor = torch.tensor(logits_array, dtype=torch.float32)
            if logits_tensor.size(0) != self.max_length:
                logits_tensor = self._pad_or_truncate_logits(logits_tensor)
            item["teacher_logits"] = logits_tensor

        if self.teacher_hidden_state is not None:
            hidden_array = self._to_numpy(self.teacher_hidden_state[idx], np.float16)
            hidden_tensor = torch.tensor(hidden_array, dtype=torch.float32)
            item["teacher_hidden_state"] = self._pad_hidden(hidden_tensor)

        if self.teacher_attention_map is not None:
            attn_array = self._to_numpy(self.teacher_attention_map[idx], np.float16)
            attn_tensor = torch.tensor(attn_array, dtype=torch.float32)
            item["teacher_attention_map"] = self._pad_attention(attn_tensor)

        if self.student_input_ids is not None and self.student_attention_mask is not None:
            stored_student_ids = self.student_input_ids[idx]
            stored_student_mask = self.student_attention_mask[idx]

            current = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                padding="max_length"
            )
            current_ids = current["input_ids"].squeeze(0).cpu().numpy()
            current_mask = current["attention_mask"].squeeze(0).cpu().numpy()

            stored_ids_array = np.array(stored_student_ids)
            stored_mask_array = np.array(stored_student_mask)

            stored_len = int(stored_mask_array.sum())
            current_len = int(current_mask.sum())

            if stored_len > 0 and current_len > 0:
                stored_actual = stored_ids_array[:stored_len]
                current_actual = current_ids[:current_len]
                if len(stored_actual) != len(current_actual) or not np.array_equal(stored_actual, current_actual):
                    if not hasattr(self, "_tokenization_warned"):
                        print(
                            f"WARNING: Tokenization mismatch detected for example {idx}. "
                            f"This may indicate tokenizer version differences."
                        )
                        self._tokenization_warned = True

        return item

    def _reconstruct_logits(self, idx: int) -> torch.Tensor:
        indices = self._to_numpy(self.teacher_topk_indices[idx], np.int64)
        values = self._to_numpy(self.teacher_topk_values[idx], np.float32)
        if indices.shape != values.shape:
            raise ValueError(f"Top-k shapes mismatch at index {idx}: {indices.shape} vs {values.shape}")
        seq_len, _ = indices.shape
        logits = torch.full((self.max_length, self.teacher_vocab_size), fill_value=-1e9, dtype=torch.float32)
        target_len = min(self.max_length, seq_len)
        index_tensor = torch.from_numpy(indices[:target_len])
        value_tensor = torch.from_numpy(values[:target_len])
        logits[:target_len].scatter_(1, index_tensor, value_tensor)
        return logits

    def _pad_or_truncate_logits(self, tensor: torch.Tensor) -> torch.Tensor:
        seq_len = tensor.size(0)
        if seq_len == self.max_length:
            return tensor
        if seq_len > self.max_length:
            return tensor[:self.max_length]
        pad = torch.full(
            (self.max_length - seq_len, tensor.size(1)),
            fill_value=-1e9,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=0)

    def _pad_hidden(self, tensor: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_dim = tensor.shape
        target_len = self.max_hidden_seq_len
        if seq_len >= target_len:
            return tensor[:target_len]
        pad = torch.zeros((target_len - seq_len, hidden_dim), dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)

    def _pad_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        heads, seq_len, _ = tensor.shape
        target_len = self.max_attention_seq_len
        output = torch.zeros(
            (self.teacher_num_heads, target_len, target_len),
            dtype=tensor.dtype,
        )
        copy_heads = min(heads, self.teacher_num_heads)
        copy_len = min(seq_len, target_len)
        output[:copy_heads, :copy_len, :copy_len] = tensor[:copy_heads, :copy_len, :copy_len]
        return output


def compute_loss(
    student_outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    teacher_data: Dict[str, Optional[torch.Tensor]],
    distill_type: str,
) -> Dict[str, torch.Tensor]:
    student_logits = student_outputs["logits"]
    student_logits_float = student_logits.float()
    losses: Dict[str, torch.Tensor] = {}
    logits_dtype = student_logits.dtype

    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    losses["task_loss"] = task_loss

    if "teacher_logits" in teacher_data and teacher_data["teacher_logits"] is not None:
        teacher_logits = teacher_data["teacher_logits"].float()
        temperature = 3.0
        student_logits_soft = F.log_softmax(student_logits_float / temperature, dim=-1)
        teacher_logits_soft = F.softmax(teacher_logits / temperature, dim=-1)
        kd_loss = F.kl_div(
            student_logits_soft,
            teacher_logits_soft,
            reduction="batchmean",
        ) * (temperature ** 2)
        losses["kd_loss"] = kd_loss.to(student_logits.device, dtype=logits_dtype)
    else:
        losses["kd_loss"] = torch.tensor(0.0, device=student_logits.device)

    gamma_1 = config.GAMMA_1 if distill_type in ["hidden_state", "combined"] else 0.0
    if gamma_1 > 0 and "projected_hidden_state" in student_outputs and "teacher_hidden_state" in teacher_data:
        student_hidden = student_outputs["projected_hidden_state"].float()
        teacher_hidden = teacher_data["teacher_hidden_state"].float()
        seq_len = min(student_hidden.size(1), teacher_hidden.size(1))
        align_hidden_loss = F.mse_loss(
            student_hidden[:, :seq_len, :],
            teacher_hidden[:, :seq_len, :],
        )
        losses["align_hidden_loss"] = align_hidden_loss.to(student_logits.device, dtype=logits_dtype)
    else:
        losses["align_hidden_loss"] = torch.tensor(0.0, device=student_logits.device)

    gamma_2 = config.GAMMA_2 if distill_type in ["attention", "combined"] else 0.0
    if gamma_2 > 0 and "attention_map" in student_outputs and "teacher_attention_map" in teacher_data:
        student_attn = student_outputs["attention_map"].float()
        teacher_attn = teacher_data["teacher_attention_map"].float()
        batch_size = min(student_attn.size(0), teacher_attn.size(0))
        seq_len = min(student_attn.size(-1), teacher_attn.size(-1))
        align_attn_loss = F.mse_loss(
            student_attn[:batch_size, :, :seq_len, :seq_len],
            teacher_attn[:batch_size, :, :seq_len, :seq_len],
        )
        losses["align_attn_loss"] = align_attn_loss.to(student_logits.device, dtype=logits_dtype)
    else:
        losses["align_attn_loss"] = torch.tensor(0.0, device=student_logits.device)

    total_loss = (
        config.ALPHA * losses["task_loss"]
        + config.BETA * losses["kd_loss"]
        + gamma_1 * losses["align_hidden_loss"]
        + gamma_2 * losses["align_attn_loss"]
    )
    losses["total_loss"] = total_loss
    return losses


def list_parquet_files(base_path: str) -> List[str]:
    """Return all parquet files under base_path (non-recursive aware)."""
    matches: List[str] = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".parquet"):
                matches.append(os.path.join(root, file))
    return matches


