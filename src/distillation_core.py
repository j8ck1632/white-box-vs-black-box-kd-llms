"""
Core components shared by both Ray-based and single-machine training flows.

This module intentionally avoids any Ray imports so it can be reused on
standalone machines that only need PyTorch + Transformers.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import pyarrow as pa

import config


def _ensure_stride(max_length: int, stride: int) -> int:
    stride = max(1, stride)
    return math.ceil(max_length / stride)


class OfflineDistillationDataset(Dataset):
    """
    Dataset for loading pre-computed teacher outputs.
    Uses PyArrow for direct random access to Parquet files to avoid memory spikes.
    """

    def __init__(self, parquet_path: Union[str, List[str]], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_vocab_size = getattr(config, "TEACHER_VOCAB_SIZE", 32000)
        self.teacher_num_heads = getattr(config, "TEACHER_NUM_HEADS", 32)
        self.hidden_stride = getattr(config, "HIDDEN_STRIDE", 1)
        self.attention_stride = getattr(config, "ATTENTION_STRIDE", 1)
        self.max_hidden_seq_len = _ensure_stride(self.max_length, self.hidden_stride)
        self.max_attention_seq_len = _ensure_stride(self.max_length, self.attention_stride)

        if isinstance(parquet_path, str):
            self.parquet_paths = [parquet_path]
        else:
            self.parquet_paths = parquet_path

        # 1. Index the dataset (Map global_idx -> (file_idx, row_group_idx, row_in_group))
        print(f"Indexing {len(self.parquet_paths)} Parquet files...")
        self.index_map: List[Tuple[int, int, int]] = []
        
        # We also need schema info to know which columns to load
        self.column_names = set()
        
        for file_idx, path in enumerate(self.parquet_paths):
            pf = pq.ParquetFile(path)
            if file_idx == 0:
                self.column_names = set(pf.schema.names)
            
            for rg_idx in range(pf.num_row_groups):
                rg_meta = pf.metadata.row_group(rg_idx)
                num_rows = rg_meta.num_rows
                for r in range(num_rows):
                    self.index_map.append((file_idx, rg_idx, r))
        
        print(f"Indexed {len(self.index_map)} examples.")

        # Lazy file handles (per process)
        self.file_handles = {}

    def _get_row(self, idx):
        file_idx, rg_idx, row_in_rg = self.index_map[idx]
        path = self.parquet_paths[file_idx]
        
        # Manage file handles per-process (safe for DataLoader workers)
        pid = os.getpid()
        key = (pid, file_idx)
        
        if key not in self.file_handles:
            self.file_handles[key] = pq.ParquetFile(path)
        
        pf = self.file_handles[key]
        
        # Read the specific Row Group
        # This loads the whole row group into memory, but it's small (batch_size ~ 4-8)
        table = pf.read_row_group(rg_idx)
        
        # Convert just the specific row to dictionary
        # slice(offset, length)
        row_table = table.slice(row_in_rg, length=1)
        row = row_table.to_pydict()
        
        # Unwrap lists (to_pydict returns lists of values, so [value])
        return {k: v[0] for k, v in row.items()}

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
            # Handle ragged arrays if necessary
            return np.stack([np.array(elem, dtype=dtype) for elem in data])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        row = self._get_row(idx)

        # 1. Get Inputs
        input_ids_list = row["input_ids"]
        attn_mask_list = row["attention_mask"]
        
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask_list, dtype=torch.long)
        
        # Ensure length matches MAX_SEQ_LENGTH
        if input_ids.size(0) < self.max_length:
            pad_len = self.max_length - input_ids.size(0)
            input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # 2. Create Labels (Mask Prompt)
        labels = input_ids.clone()
        prompt_len = row.get("prompt_length", 0)
        
        if prompt_len > 0:
            labels[:prompt_len] = -100
        
        labels[attention_mask == 0] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_name": row.get("task_name", "unknown"),
        }

        # 3. Reconstruct Teacher Logits
        if "teacher_topk_indices" in self.column_names and "teacher_topk_values" in self.column_names:
            if row.get("teacher_topk_indices") is not None:
                item["teacher_logits"] = self._reconstruct_logits(row["teacher_topk_indices"], row["teacher_topk_values"])
        elif "teacher_logits" in self.column_names:
            if row.get("teacher_logits") is not None:
                logits_array = self._to_numpy(row["teacher_logits"], np.float32)
                logits_tensor = torch.tensor(logits_array, dtype=torch.float32)
                if logits_tensor.size(0) != self.max_length:
                    logits_tensor = self._pad_or_truncate_logits(logits_tensor)
                item["teacher_logits"] = logits_tensor

        # 4. Reconstruct Teacher Hidden States
        if "teacher_hidden_state" in self.column_names and row.get("teacher_hidden_state") is not None:
            hidden_array = self._to_numpy(row["teacher_hidden_state"], np.float16)
            hidden_tensor = torch.tensor(hidden_array, dtype=torch.float32)
            item["teacher_hidden_state"] = self._pad_hidden(hidden_tensor)

        # 5. Reconstruct Teacher Attention
        if "teacher_attention_map" in self.column_names and row.get("teacher_attention_map") is not None:
            attn_array = self._to_numpy(row["teacher_attention_map"], np.float16)
            attn_tensor = torch.tensor(attn_array, dtype=torch.float32)
            item["teacher_attention_map"] = self._pad_attention(attn_tensor)

        return item

    def _reconstruct_logits(self, indices_data, values_data) -> torch.Tensor:
        indices = self._to_numpy(indices_data, np.int64)
        values = self._to_numpy(values_data, np.float32)
        
        seq_len = indices.shape[0]
        logits = torch.full((self.max_length, self.teacher_vocab_size), fill_value=-100.0, dtype=torch.float32)
        
        target_len = min(self.max_length, seq_len)
        if target_len > 0:
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
            fill_value=-100.0,
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
        if copy_len > 0:
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

    # 1. Task Loss (Standard Cross Entropy on Answer tokens only)
    # labels have -100 for prompt and padding, so this is correct.
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    losses["task_loss"] = task_loss

    # 2. KD Loss (Logits)
    # We align the student's logits to the teacher's logits.
    if "teacher_logits" in teacher_data and teacher_data["teacher_logits"] is not None:
        teacher_logits = teacher_data["teacher_logits"].float()
        temperature = 2.0 # Standard temp
        
        # We should only compute KD loss on non-padding tokens? 
        # Or even better, on the Answer tokens only? 
        # Usually KD is applied to the whole sequence or just the target. 
        # Let's apply to non-padding tokens.
        
        # Create mask from labels (where labels != -100 implies valid answer token)
        # OR use attention_mask (valid sequence).
        # Let's align on valid sequence (Prompt + Answer).
        # Note: teacher_logits are reconstructed sparse logits. -100.0 filler.
        # Softmax handles -100.0 as near zero.
        
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

    # 3. Hidden State Alignment
    gamma_1 = config.GAMMA_1 if distill_type in ["hidden_state", "combined"] else 0.0
    if gamma_1 > 0 and "projected_hidden_state" in student_outputs and "teacher_hidden_state" in teacher_data:
        student_hidden = student_outputs["projected_hidden_state"].float()
        teacher_hidden = teacher_data["teacher_hidden_state"].float()
        
        # Downsample student to match teacher stride
        stride = getattr(config, "HIDDEN_STRIDE", 1)
        if stride > 1:
            student_hidden = student_hidden[:, ::stride, :]
            
        # Truncate to matching length (should match due to padding, but safety first)
        seq_len = min(student_hidden.size(1), teacher_hidden.size(1))
        
        align_hidden_loss = F.mse_loss(
            student_hidden[:, :seq_len, :],
            teacher_hidden[:, :seq_len, :],
        )
        losses["align_hidden_loss"] = align_hidden_loss.to(student_logits.device, dtype=logits_dtype)
    else:
        losses["align_hidden_loss"] = torch.tensor(0.0, device=student_logits.device)

    # 4. Attention Alignment
    gamma_2 = config.GAMMA_2 if distill_type in ["attention", "combined"] else 0.0
    if gamma_2 > 0 and "attention_map" in student_outputs and "teacher_attention_map" in teacher_data:
        student_attn = student_outputs["attention_map"].float()
        teacher_attn = teacher_data["teacher_attention_map"].float()
        
        # Downsample student to match teacher stride
        stride = getattr(config, "ATTENTION_STRIDE", 1)
        if stride > 1:
            student_attn = student_attn[:, :, ::stride, ::stride]
            
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
