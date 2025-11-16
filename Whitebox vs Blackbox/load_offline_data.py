"""
Utility script to load and inspect pre-computed offline teacher data.

This script demonstrates how to load the parquet files containing teacher model outputs
(logits, hidden states, attention maps) that were pre-computed by offline_teacher_data.py.
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any
import config


def _to_py(value):
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        python_list = value.tolist()
        return [_to_py(elem) for elem in python_list]
    if isinstance(value, (list, tuple)):
        return [_to_py(elem) for elem in value]
    return value


def _to_numpy(value, dtype):
    data = _to_py(value)
    try:
        return np.array(data, dtype=dtype)
    except (ValueError, TypeError):
        return np.stack([np.array(elem, dtype=dtype) for elem in data])


def _topk_to_dense(indices, values, vocab_size: int) -> np.ndarray:
    """Reconstruct dense logits from top-k representation."""
    indices = np.array(_to_py(indices), dtype=np.int64)
    values = np.array(_to_py(values), dtype=np.float32)
    if indices.shape != values.shape:
        raise ValueError(f"Mismatch between indices {indices.shape} and values {values.shape}")
    seq_len, _ = indices.shape
    logits = np.full((seq_len, vocab_size), -1e9, dtype=np.float32)
    row_idx = np.arange(seq_len)[:, None]
    logits[row_idx, indices] = values
    return logits


def load_all_offline_data(data_path: str = None) -> pd.DataFrame:
    """
    Load all parquet files from the offline teacher data directory.
    
    Args:
        data_path: Path to offline teacher data directory (defaults to config.OFFLINE_DATA_PATH)
        
    Returns:
        Combined DataFrame with all examples
    """
    if data_path is None:
        data_path = config.OFFLINE_DATA_PATH
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory not found: {data_path}")
    
    # Find all Parquet files
    parquet_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {data_path}")
    
    print(f"Found {len(parquet_files)} Parquet file(s)")
    
    # Load and combine all parquet files
    dfs = []
    for parquet_file in parquet_files:
        print(f"Loading {os.path.basename(parquet_file)}...")
        df = pd.read_parquet(parquet_file)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal examples loaded: {len(combined_df):,}")
    print(f"Columns: {list(combined_df.columns)}")
    
    return combined_df


def inspect_example(df: pd.DataFrame, idx: int = 0):
    """
    Inspect a single example from the dataset.
    
    Args:
        df: DataFrame containing offline teacher data
        idx: Index of example to inspect (default: 0)
    """
    if idx >= len(df):
        raise IndexError(f"Index {idx} out of range (dataset has {len(df)} examples)")
    
    row = df.iloc[idx]
    
    print(f"\n{'='*60}")
    print(f"Example {idx}:")
    print(f"{'='*60}\n")
    
    # Basic info
    print(f"Prompt: {row['prompt'][:200]}..." if len(str(row['prompt'])) > 200 else f"Prompt: {row['prompt']}")
    print(f"\nAnswer: {row['answer']}")
    
    # Teacher outputs
    if "teacher_topk_values" in row and row["teacher_topk_values"] is not None:
        vocab_size = getattr(config, "TEACHER_VOCAB_SIZE", 32000)
        logits = _topk_to_dense(row["teacher_topk_indices"], row["teacher_topk_values"], vocab_size)
        print(f"\nTeacher Logits (reconstructed from top-{config.TOP_K_LOGITS}):")
        print(f"  Shape: {logits.shape}")
        print(f"  Dtype: {logits.dtype}")
        print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"  Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
    
    if "teacher_hidden_state" in row and row["teacher_hidden_state"] is not None:
        hidden = np.array(_to_py(row["teacher_hidden_state"]), dtype=np.float16)
        print(f"\nTeacher Hidden State:")
        print(f"  Shape: {hidden.shape}")
        print(f"  Dtype: {hidden.dtype}")
        print(f"  Range: [{hidden.min():.4f}, {hidden.max():.4f}]")
        print(f"  Mean: {hidden.mean():.4f}, Std: {hidden.std():.4f}")
    
    if "teacher_attention_map" in row and row["teacher_attention_map"] is not None:
        attn = np.array(_to_py(row["teacher_attention_map"]), dtype=np.float16)
        print(f"\nTeacher Attention Map:")
        print(f"  Shape: {attn.shape}")
        print(f"  Dtype: {attn.dtype}")
        print(f"  Range: [{attn.min():.4f}, {attn.max():.4f}]")
        print(f"  Mean: {attn.mean():.4f}, Std: {attn.std():.4f}")
        print(f"  Attention heads: {attn.shape[0]}")
        print(f"  Sequence length: {attn.shape[1]}")
    
    if "student_input_ids" in row and row["student_input_ids"] is not None:
        student_ids = row["student_input_ids"]
        print(f"\nStudent Input IDs:")
        print(f"  Length: {len(student_ids)}")
        print(f"  First 10: {student_ids[:10]}")


def get_dataset_stats(df: pd.DataFrame):
    """
    Get statistics about the dataset.
    
    Args:
        df: DataFrame containing offline teacher data
    """
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"{'='*60}\n")
    
    print(f"Total examples: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Check data types
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Check for missing data
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Sample sizes for teacher outputs
    if "teacher_topk_values" in df.columns and len(df) > 0:
        sample = df.iloc[0]
        if sample["teacher_topk_values"] is not None:
            logits_shape = np.array(sample["teacher_topk_values"]).shape
            print(f"\nStored top-k logits shape per example: {logits_shape}")
    
    if "teacher_hidden_state" in df.columns and len(df) > 0:
        sample = df.iloc[0]
        if sample["teacher_hidden_state"] is not None:
            hidden_shape = np.array(sample["teacher_hidden_state"]).shape
            print(f"Teacher Hidden State shape per example: {hidden_shape}")
    
    if "teacher_attention_map" in df.columns and len(df) > 0:
        sample = df.iloc[0]
        if sample["teacher_attention_map"] is not None:
            attn_shape = np.array(sample["teacher_attention_map"]).shape
            print(f"Teacher Attention Map shape per example: {attn_shape}")


def convert_to_tensors(df: pd.DataFrame, indices: List[int] = None):
    """
    Convert teacher outputs from lists to PyTorch tensors.
    
    This is useful for training or further processing.
    
    Args:
        df: DataFrame containing offline teacher data
        indices: List of indices to convert (None = all)
        
    Returns:
        Dictionary with tensors for each example
    """
    
    if indices is None:
        indices = list(range(len(df)))
    
    results = []
    for idx in indices:
        row = df.iloc[idx]
        
        item = {
            "prompt": row["prompt"],
            "answer": row["answer"],
        }
        
        # Convert teacher logits from top-k representation
        if "teacher_topk_values" in row and row["teacher_topk_values"] is not None:
            vocab_size = getattr(config, "TEACHER_VOCAB_SIZE", 32000)
            logits_array = _topk_to_dense(row["teacher_topk_indices"], row["teacher_topk_values"], vocab_size)
            item["teacher_logits"] = torch.tensor(logits_array, dtype=torch.float32)
        
        # Convert teacher hidden state
        if "teacher_hidden_state" in row and row["teacher_hidden_state"] is not None:
            hidden_array = _to_numpy(row["teacher_hidden_state"], np.float16)
            item["teacher_hidden_state"] = torch.tensor(hidden_array, dtype=torch.float16)
        
        # Convert teacher attention map
        if "teacher_attention_map" in row and row["teacher_attention_map"] is not None:
            attn_array = _to_numpy(row["teacher_attention_map"], np.float16)
            item["teacher_attention_map"] = torch.tensor(attn_array, dtype=torch.float16)
        
        # Student tokenization info
        if "student_input_ids" in row and row["student_input_ids"] is not None:
            ids_array = _to_numpy(row["student_input_ids"], np.int64)
            item["student_input_ids"] = torch.tensor(ids_array, dtype=torch.long)
        
        if "student_attention_mask" in row and row["student_attention_mask"] is not None:
            mask_array = _to_numpy(row["student_attention_mask"], np.int64)
            item["student_attention_mask"] = torch.tensor(mask_array, dtype=torch.long)
        
        results.append(item)
    
    return results


def main():
    """Main function to demonstrate loading and inspecting offline teacher data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and inspect offline teacher data")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to offline teacher data directory (defaults to config.OFFLINE_DATA_PATH)"
    )
    parser.add_argument(
        "--inspect",
        type=int,
        default=0,
        help="Index of example to inspect in detail (default: 0)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics"
    )
    args = parser.parse_args()
    
    print("Loading offline teacher data...")
    print(f"Path: {args.path or config.OFFLINE_DATA_PATH}\n")
    
    # Load all data
    df = load_all_offline_data(args.path)
    
    # Show statistics
    if args.stats or args.inspect == 0:  # Show stats by default
        get_dataset_stats(df)
    
    # Inspect a specific example
    if args.inspect is not None:
        inspect_example(df, args.inspect)
    
    # Example: Convert to tensors (first 3 examples)
    print(f"\n{'='*60}")
    print("Example: Converting first 3 examples to PyTorch tensors...")
    print(f"{'='*60}\n")
    tensor_data = convert_to_tensors(df, indices=[0, 1, 2])
    
    print(f"Converted {len(tensor_data)} examples to tensors")
    print(f"\nKeys in tensor data: {list(tensor_data[0].keys())}")
    for key, value in tensor_data[0].items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: tensor shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__}")


if __name__ == "__main__":
    main()
