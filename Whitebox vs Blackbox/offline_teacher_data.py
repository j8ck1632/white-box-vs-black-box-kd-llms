"""
Offline Teacher Data Pre-computation Script

This script runs the teacher model (Mistral-7B) once on all datasets
and saves the outputs (logits, hidden states, attention maps) to disk.
This enables efficient offline distillation without running the teacher
during student training.
"""

import os
import sys
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import time
from datetime import datetime, timedelta
import signal
import gc  # For explicit garbage collection

# Fix Windows console encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config

TOP_K_LOGITS = getattr(config, "TOP_K_LOGITS", 128)
HIDDEN_STRIDE = max(1, getattr(config, "HIDDEN_STRIDE", 1))
ATTENTION_STRIDE = max(1, getattr(config, "ATTENTION_STRIDE", 1))

TASK_NAME_SST2 = "sst2"
TASK_NAME_MMLU = "mmlu"
TASK_NAME_GSM8K = "gsm8k"


def configure_temp_directories():
    """Pin temporary directories to the high-capacity drive when available."""
    temp_dir = getattr(config, "SYSTEM_TEMP_DIR", None)
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        os.environ["TMP"] = temp_dir
        os.environ["TEMP"] = temp_dir


configure_temp_directories()

# Module-level cache for model and tokenizer so we only pay load cost once
_model_cache = {}
_tokenizer_cache = {}
_student_tokenizer_cache = {}


def _to_py(value):
    """
    Convert pyarrow/numpy containers (possibly nested) to pure Python structures.
    
    Some columns (e.g., attention maps) come back as numpy object arrays with
    nested numpy arrays at each level. We need to recursively walk these
    structures and turn every layer into plain Python lists/scalars so downstream
    numpy conversions see uniform shapes.
    """
    if hasattr(value, "as_py"):
        value = value.as_py()
    
    # Handle numpy arrays (including object-dtype) by converting each element.
    if isinstance(value, np.ndarray):
        python_list = value.tolist()
        return [_to_py(elem) for elem in python_list]
    
    # Recurse into generic containers (lists/tuples) but avoid treating strings
    # or bytes as iterables.
    if isinstance(value, (list, tuple)):
        return [_to_py(elem) for elem in value]
    
    return value


def _to_numpy_array(value, dtype):
    """Convert nested stored values into a numpy array with the desired dtype."""
    python_value = _to_py(value)
    try:
        return np.array(python_value, dtype=dtype)
    except (ValueError, TypeError):
        # Fall back to stacking per-element arrays when direct casting fails.
        return np.stack([np.array(elem, dtype=dtype) for elem in python_value])


def _downsample_sequence(data: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return data
    return data[::stride]


def _downsample_attention(data: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return data
    return data[:, ::stride, ::stride]


def _extract_topk_logits(
    logits_tensor: torch.Tensor,
    top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the top-k logits (values + indices) for each token in the batch.
    """
    vocab = logits_tensor.size(-1)
    effective_k = min(top_k, vocab)
    values, indices = torch.topk(logits_tensor, k=effective_k, dim=-1)
    return values, indices


def get_model_and_tokenizer(model_name: str, hf_token: str = None):
    """
    Get or load model and tokenizer, caching them in the worker process.
    
    This function caches the model and tokenizer at the module level,
    so they are loaded only once per run and reused across batches.
    """
    cache_key = model_name
    
    # Return cached model and tokenizer if available
    if cache_key in _model_cache and cache_key in _tokenizer_cache:
        return _model_cache[cache_key], _tokenizer_cache[cache_key]
    
    # Get token if not provided
    if not hf_token:
        hf_token = getattr(config, 'HUGGING_FACE_TOKEN', None) or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    # Build kwargs with token if available
    tokenizer_kwargs = {"token": hf_token} if hf_token else {}
    model_kwargs = {
        "dtype": torch.float16,  # Use dtype instead of torch_dtype (deprecated)
        "device_map": "auto",
        "low_cpu_mem_usage": True,  # Reduce memory pressure during loading
        "attn_implementation": "eager",  # Use eager attention to support output_attentions=True
        # SDPA (default) is faster but doesn't support attention map output
        **({"token": hf_token} if hf_token else {})
    }
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (this may take a while on first call)
    # Using eager attention implementation to support output_attentions=True
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.eval()
    
    # Cache model and tokenizer
    _tokenizer_cache[cache_key] = tokenizer
    _model_cache[cache_key] = model
    
    return model, tokenizer


def get_student_tokenizer(student_model_name: str, hf_token: str = None):
    """
    Get or load student tokenizer, caching it in the worker process.
    
    This is needed to ensure teacher outputs align with student tokenization.
    """
    cache_key = student_model_name
    
    # Return cached student tokenizer if available
    if cache_key in _student_tokenizer_cache:
        return _student_tokenizer_cache[cache_key]
    
    # Get token if not provided
    if not hf_token:
        hf_token = getattr(config, 'HUGGING_FACE_TOKEN', None) or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    # Build kwargs with token if available
    tokenizer_kwargs = {"token": hf_token} if hf_token else {}
    
    # Load student tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, **tokenizer_kwargs)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    
    # Cache student tokenizer
    _student_tokenizer_cache[cache_key] = student_tokenizer
    
    return student_tokenizer


def validate_tensor_data(data: np.ndarray, name: str) -> bool:
    """
    Validate tensor data for NaN, Inf, and shape consistency.
    
    Args:
        data: Numpy array to validate
        name: Name of the data for error messages
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    if data is None:
        raise ValueError(f"{name} is None")
    
    # Check for NaN
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        raise ValueError(f"{name} contains {nan_count} NaN values")
    
    # Check for Inf
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        raise ValueError(f"{name} contains {inf_count} Inf values")
    
    # Check shape
    if data.size == 0:
        raise ValueError(f"{name} is empty")
    
    return True


def process_batch_with_teacher(batch: Dict[str, List[Any]], model_name: str, student_model_name: str, hf_token: str = None) -> Dict[str, List]:
    """
    Process a batch of prompts through the teacher model.
    
    This function runs on a Ray remote worker with GPU access.
    
    The model and tokenizer are cached at the module level, so they are loaded
    only once per run and reused across batches.
    
    CRITICAL: This function tokenizes with BOTH student and teacher tokenizers to ensure
    alignment. Teacher outputs are computed on teacher-tokenized sequences, but we also
    store student tokenization info for proper alignment during training.
    
    Args:
        batch: Dictionary containing 'prompt' and 'answer' lists
        model_name: HuggingFace model identifier for the teacher
        student_model_name: HuggingFace model identifier for the student
        hf_token: HuggingFace authentication token (required for gated models)
        
    Returns:
        Dictionary containing enriched batch data with:
        - prompt: Original prompts
        - answer: Original answers
        - task_name: Source benchmark identifier for each example
        - teacher_topk_indices / teacher_topk_values: Compressed logits
        - teacher_hidden_state: Teacher final layer hidden states (float16)
        - teacher_attention_map: Teacher final layer attention maps (float16)
        - student_input_ids: Student tokenization for alignment (stored for verification)
        - student_attention_mask: Student attention mask for alignment
    """
    # Get cached or load teacher model and tokenizer
    model, teacher_tokenizer = get_model_and_tokenizer(model_name, hf_token)
    
    # Get cached or load student tokenizer (for alignment)
    student_tokenizer = get_student_tokenizer(student_model_name, hf_token)
    
    # Extract and validate prompts and answers
    prompts = batch.get("prompt", [])
    answers = batch.get("answer", [])
    raw_task_names = batch.get("task_name")
    if raw_task_names is None:
        raw_task_names = ["unknown"] * len(prompts)
    
    # Ensure all prompts are strings and filter out any invalid entries
    valid_prompts = []
    valid_answers = []
    valid_task_names = []
    for prompt, answer, task_name in zip(prompts, answers, raw_task_names):
        # Convert prompt to string if it's not already
        if prompt is None:
            continue
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        # Convert answer to string if it's not already
        answer_str = str(answer) if not isinstance(answer, str) else answer
        
        # Skip empty prompts
        if not prompt_str or len(prompt_str.strip()) == 0:
            continue
            
        normalized_task = "unknown"
        if isinstance(task_name, str):
            normalized_task = task_name.strip() or "unknown"
        elif task_name is not None:
            normalized_task = str(task_name).strip() or "unknown"

        valid_prompts.append(prompt_str)
        valid_answers.append(answer_str)
        valid_task_names.append(normalized_task)
    
    # Skip if no valid prompts
    if not valid_prompts:
        return {
            "prompt": [],
            "answer": [],
            "task_name": [],
            "teacher_topk_indices": [],
            "teacher_topk_values": [],
            "teacher_hidden_state": [],
            "teacher_attention_map": [],
            "student_input_ids": [],
            "student_attention_mask": [],
        }
    
    # Tokenize with STUDENT tokenizer first (this is what will be used during training)
    student_tokenized = student_tokenizer(
        valid_prompts,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    student_input_ids = student_tokenized["input_ids"].cpu()
    student_attention_mask = student_tokenized["attention_mask"].cpu()
    
    # Tokenize with TEACHER tokenizer for teacher model forward pass
    teacher_tokenized = teacher_tokenizer(
        valid_prompts,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    
    input_ids = teacher_tokenized["input_ids"].to(model.device)
    attention_mask = teacher_tokenized["attention_mask"].to(model.device)
    
    # Forward pass with all outputs
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
    
    # Extract outputs - keep only top-k logits to reduce storage
    topk_values, topk_indices = _extract_topk_logits(outputs.logits, TOP_K_LOGITS)
    topk_values = topk_values.detach().cpu().numpy().astype(np.float16)
    topk_indices = topk_indices.detach().cpu().numpy().astype(np.int32)
    
    # Get final layer hidden states (last element of hidden_states tuple)
    # Use float16 for hidden states to save space (less critical for precision)
    teacher_hidden_state = outputs.hidden_states[-1].cpu().numpy().astype(np.float16)
    
    # Get tensor batch size and sequence length
    tensor_batch_size, seq_len = input_ids.shape
    
    # Get final layer attention maps (last element of attentions tuple)
    # CRITICAL: Fail loudly if attentions are not available instead of silently creating dummy data
    if outputs.attentions is not None and len(outputs.attentions) > 0:
        teacher_attention_map = outputs.attentions[-1].cpu().numpy().astype(np.float16)
    else:
        # Fail loudly - attention maps are required for white-box distillation
        raise RuntimeError(
            f"CRITICAL: Teacher model did not return attention maps even though output_attentions=True. "
            f"This indicates a problem with the model configuration or attention implementation. "
            f"Model: {model_name}, Batch size: {tensor_batch_size}"
        )
    
    # Validate all outputs before processing
    try:
        validate_tensor_data(topk_values, "teacher_logits_topk_values")
        validate_tensor_data(teacher_hidden_state, "teacher_hidden_state")
        validate_tensor_data(teacher_attention_map, "teacher_attention_map")
    except ValueError as e:
        raise ValueError(f"Data validation failed for batch: {e}")
    
    # Verify batch size consistency
    assert tensor_batch_size == len(valid_prompts), f"Batch size mismatch: {tensor_batch_size} != {len(valid_prompts)}"
    
    # Verify attention map shape
    expected_attn_shape = (tensor_batch_size, model.config.num_attention_heads, seq_len, seq_len)
    if teacher_attention_map.shape != expected_attn_shape:
        raise ValueError(
            f"Attention map shape mismatch: expected {expected_attn_shape}, got {teacher_attention_map.shape}"
        )
    
    # Filter out padding positions using attention_mask to save space
    # This dramatically reduces storage since most sequences are shorter than 512 tokens
    result = {
        "prompt": valid_prompts,
        "answer": valid_answers,
        "task_name": valid_task_names,
    }
    
    # Process each example individually to filter padding
    for i in range(tensor_batch_size):
        # Get actual sequence length (non-padding tokens) for teacher tokenization
        actual_seq_len = int(attention_mask[i].sum().item())
        
        # Validate sequence length is reasonable
        if actual_seq_len == 0:
            raise ValueError(f"Example {i} has zero-length sequence after tokenization")
        if actual_seq_len > config.MAX_SEQ_LENGTH:
            raise ValueError(f"Example {i} exceeds max sequence length: {actual_seq_len} > {config.MAX_SEQ_LENGTH}")
        
        # Only save non-padding positions
        # Logits: store compressed top-k representation
        logits_values_slice = topk_values[i, :actual_seq_len, :]
        logits_indices_slice = topk_indices[i, :actual_seq_len, :]
        validate_tensor_data(logits_values_slice, f"teacher_logits_topk_values[{i}]")
        result.setdefault("teacher_topk_indices", []).append(
            logits_indices_slice.astype(np.int32).tolist()
        )
        result.setdefault("teacher_topk_values", []).append(
            logits_values_slice.astype(np.float16).tolist()
        )
        
        # Hidden states: [seq_len, hidden_dim] -> [actual_seq_len, hidden_dim]
        hidden_slice = teacher_hidden_state[i, :actual_seq_len, :]
        hidden_slice = _downsample_sequence(hidden_slice, HIDDEN_STRIDE)
        validate_tensor_data(hidden_slice, f"teacher_hidden_state[{i}]")
        result.setdefault("teacher_hidden_state", []).append(hidden_slice.tolist())
        
        # Attention maps: [num_heads, seq_len, seq_len] -> [num_heads, actual_seq_len, actual_seq_len]
        attn_slice = teacher_attention_map[i, :, :actual_seq_len, :actual_seq_len]
        attn_slice = _downsample_attention(attn_slice, ATTENTION_STRIDE)
        validate_tensor_data(attn_slice, f"teacher_attention_map[{i}]")
        result.setdefault("teacher_attention_map", []).append(attn_slice.tolist())
        
        # Store student tokenization for alignment verification
        student_seq_len = int(student_attention_mask[i].sum().item())
        result.setdefault("student_input_ids", []).append(
            student_input_ids[i, :student_seq_len].tolist()
        )
        result.setdefault("student_attention_mask", []).append(
            student_attention_mask[i, :student_seq_len].tolist()
        )
    
    # Explicit memory cleanup to help prevent memory accumulation
    # This is especially important on Windows with limited paging file
    del topk_values, topk_indices, teacher_hidden_state, teacher_attention_map
    del student_input_ids, student_attention_mask, input_ids, attention_mask
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return result


def load_and_preprocess_datasets() -> List[Dict[str, str]]:
    """
    Load and preprocess all evaluation datasets into a unified format.
    
    Returns:
        List of dictionaries with 'prompt' and 'answer' keys
    """
    all_data = []
    
    # Load SST-2 (Sentiment Analysis)
    print("Loading SST-2 dataset...")
    sst2_dataset = load_dataset(config.SST2_DATASET, config.SST2_CONFIG, split="train")
    # Take a subset (5k examples) for faster processing
    SST2_SUBSET_SIZE = 5000
    print(f"Using {SST2_SUBSET_SIZE:,} examples from SST-2 (out of {len(sst2_dataset):,} total)")
    
    for example in sst2_dataset.select(range(min(SST2_SUBSET_SIZE, len(sst2_dataset)))):
        # Ensure prompt is a string
        prompt = str(example.get("sentence", "")) if example.get("sentence") is not None else ""
        # Convert label (0/1) to text answer
        answer = "positive" if example.get("label") == 1 else "negative"
        # Only add if we have valid data
        if prompt:
            all_data.append({"prompt": prompt, "answer": answer, "task_name": TASK_NAME_SST2})
    
    # Load MMLU (subset - using a few-shot format)
    print("Loading MMLU dataset...")
    try:
        # MMLU doesn't have a "train" split, use "auxiliary_train" or "validation"
        mmlu_dataset = load_dataset(config.MMLU_DATASET, "all", split="auxiliary_train")
        # Take a subset (first 1000 examples)
        for example in mmlu_dataset.select(range(min(1000, len(mmlu_dataset)))):
            # Ensure question is a string
            question = str(example['question']) if example.get('question') is not None else ""
            # Format choices as a string (choices might be a list)
            choices = example.get('choices', [])
            if isinstance(choices, list):
                choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            else:
                choices_str = str(choices)
            
            prompt = f"Question: {question}\nChoices:\n{choices_str}\nAnswer:"
            # Convert answer to string (MMLU answers are typically integers 0-3)
            answer = str(example.get("answer", ""))
            # Only add if we have valid data
            if question and answer:
                all_data.append({"prompt": prompt, "answer": answer, "task_name": TASK_NAME_MMLU})
    except Exception as e:
        print(f"Warning: Could not load MMLU dataset: {e}")
    
    # Load GSM8K (Math Word Problems)
    print("Loading GSM8K dataset...")
    try:
        gsm8k_dataset = load_dataset(config.GSM8K_DATASET, "main", split="train")
        # Take a subset (first 1000 examples)
        for example in gsm8k_dataset.select(range(min(1000, len(gsm8k_dataset)))):
            # Ensure prompt and answer are strings
            prompt = str(example.get("question", "")) if example.get("question") is not None else ""
            answer = str(example.get("answer", "")) if example.get("answer") is not None else ""
            # Only add if we have valid data
            if prompt and answer:
                all_data.append({"prompt": prompt, "answer": answer, "task_name": TASK_NAME_GSM8K})
    except Exception as e:
        print(f"Warning: Could not load GSM8K dataset: {e}")
    
    print(f"Total examples loaded: {len(all_data)}")
    return all_data


def handle_interrupt(sig, frame):
    """Handle Ctrl+C or termination signals gracefully."""
    print("\nInterrupt received. Cleaning up resources...")
    torch.cuda.empty_cache()
    gc.collect()
    sys.exit(0)


def verify_saved_data(data_path: str, num_samples: int = 10):
    """
    Verify that saved data can be loaded correctly and has valid structure.
    
    Args:
        data_path: Path to the saved Parquet data
        num_samples: Number of random samples to verify
    """
    print(f"\n{'='*60}")
    print("Verifying saved data...")
    print(f"{'='*60}\n")
    
    # Find all Parquet files
    parquet_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {data_path}")
    
    print(f"Found {len(parquet_files)} Parquet file(s)")
    
    # Load first file for verification
    df = pd.read_parquet(parquet_files[0])
    
    print(f"Loaded dataset with {len(df)} examples")
    print(f"Columns: {list(df.columns)}")
    
    # Verify required columns exist
    required_columns = [
        "prompt",
        "answer",
        "task_name",
        "teacher_topk_indices",
        "teacher_topk_values",
        "teacher_hidden_state",
        "teacher_attention_map",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Sample random examples for verification
    sample_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    
    validation_errors = []
    
    for idx in sample_indices:
        row = df.iloc[idx]
        
        # Verify compressed logits
        try:
            indices = _to_numpy_array(row["teacher_topk_indices"], np.int32)
            values = _to_numpy_array(row["teacher_topk_values"], np.float16)
            validate_tensor_data(values, f"topk_values[example_{idx}]")
            if indices.shape != values.shape:
                validation_errors.append(
                    f"Example {idx}: top-k indices shape {indices.shape} != values shape {values.shape}"
                )
        except Exception as e:
            validation_errors.append(f"Example {idx}: top-k logits validation failed: {e}")
        
        # Verify hidden states
        try:
            hidden = _to_numpy_array(row["teacher_hidden_state"], np.float16)
            validate_tensor_data(hidden, f"hidden_state[example_{idx}]")
            if len(hidden.shape) != 2:
                validation_errors.append(f"Example {idx}: hidden_state shape should be 2D, got {hidden.shape}")
        except Exception as e:
            validation_errors.append(f"Example {idx}: hidden_state validation failed: {e}")
        
        # Verify attention maps
        try:
            attn = _to_numpy_array(row["teacher_attention_map"], np.float16)
            validate_tensor_data(attn, f"attention_map[example_{idx}]")
            if len(attn.shape) != 3:
                validation_errors.append(f"Example {idx}: attention_map shape should be 3D, got {attn.shape}")
        except Exception as e:
            validation_errors.append(f"Example {idx}: attention_map validation failed: {e}")
        
        # Verify prompt and answer are strings
        if not isinstance(row["prompt"], str):
            validation_errors.append(f"Example {idx}: prompt should be string, got {type(row['prompt'])}")
        if not isinstance(row["answer"], str):
            validation_errors.append(f"Example {idx}: answer should be string, got {type(row['answer'])}")
        if "task_name" not in row or not isinstance(row["task_name"], str):
            validation_errors.append(f"Example {idx}: task_name should be string, got {type(row.get('task_name'))}")
    
    if validation_errors:
        print("VALIDATION ERRORS FOUND:")
        for error in validation_errors:
            print(f"  - {error}")
        raise ValueError(f"Data validation failed with {len(validation_errors)} error(s)")
    
    print(f"✓ Verified {len(sample_indices)} random examples - all valid")
    print(f"✓ Data verification passed!")
    print(f"{'='*60}\n")


def _flatten_result_rows(result_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Convert batched result dict into a list of row dictionaries."""
    rows = []
    num_items = len(result_dict.get("prompt", []))
    task_names = result_dict.get("task_name", [])
    for i in range(num_items):
        rows.append(
            {
                "prompt": result_dict["prompt"][i],
                "answer": result_dict["answer"][i],
                "task_name": task_names[i] if i < len(task_names) else "unknown",
                "teacher_topk_indices": result_dict["teacher_topk_indices"][i],
                "teacher_topk_values": result_dict["teacher_topk_values"][i],
                "teacher_hidden_state": result_dict["teacher_hidden_state"][i],
                "teacher_attention_map": result_dict["teacher_attention_map"][i],
                "student_input_ids": result_dict["student_input_ids"][i],
                "student_attention_mask": result_dict["student_attention_mask"][i],
            }
        )
    return rows


def _batch_iterator(data: List[Dict[str, str]], batch_size: int):
    """Yield successive batches from the preprocessed dataset list."""
    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        yield {
            "prompt": [item["prompt"] for item in batch_data],
            "answer": [item["answer"] for item in batch_data],
            "task_name": [item.get("task_name", "unknown") for item in batch_data],
        }


def _append_to_parquet(rows: List[Dict[str, Any]], writer: pq.ParquetWriter, output_file: str):
    """Write rows to Parquet, creating the writer lazily on first call."""
    if not rows:
        return writer, 0
    table = pa.Table.from_pylist(rows)
    if writer is None:
        compression = getattr(config, "PARQUET_COMPRESSION", None)
        compression_level = getattr(config, "PARQUET_COMPRESSION_LEVEL", None)
        writer = pq.ParquetWriter(
            output_file,
            table.schema,
            compression=compression,
            compression_level=compression_level,
            use_dictionary=True,
        )
    writer.write_table(table)
    return writer, len(rows)


def main():
    """
    Main function to pre-compute teacher model outputs on a single machine.
    """
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        signal.signal(signal.SIGTERM, handle_interrupt)
    except AttributeError:
        # SIGTERM is not available on some platforms (e.g., Windows)
        pass
    
    hf_token = getattr(config, 'HUGGING_FACE_TOKEN', None) or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("ERROR: No Hugging Face token found!")
        print("The model 'mistralai/Mistral-7B-v0.1' requires authentication.")
        print("\nPlease set HUGGING_FACE_TOKEN in config.py")
        raise ValueError("Hugging Face token is required but not found.")
    
    print("Hugging Face token found. Proceeding with model download...")
    
    print("Loading and preprocessing datasets...")
    all_data = load_and_preprocess_datasets()
    total_examples = len(all_data)
    batch_size = int(os.getenv("OFFLINE_BATCH_SIZE", "2"))
    total_batches = (total_examples + batch_size - 1) // batch_size
    
    print(f"\n{'='*60}")
    print("STREAMING CONFIGURATION")
    print(f"{'='*60}")
    print(f"- Total examples: {total_examples:,}")
    print(f"- Batch size: {batch_size}")
    print(f"- Top-k logits stored: {TOP_K_LOGITS}")
    print(f"- Hidden/attention stride: {HIDDEN_STRIDE}/{ATTENTION_STRIDE}")
    print(f"- Output path: {config.OFFLINE_DATA_PATH}")
    print(f"{'='*60}\n")
    
    # Load models once (cached)
    model, teacher_tokenizer = get_model_and_tokenizer(config.TEACHER_MODEL_NAME, hf_token)
    student_tokenizer = get_student_tokenizer(config.STUDENT_MODEL_NAME, hf_token)
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.2f} GB)")
    
    # Prepare output directory/file
    output_dir = config.OFFLINE_DATA_PATH
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "offline_teacher_data.parquet")
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file removed: {output_file}")
    
    writer = None
    rows_written = 0
    start_time = time.time()
    
    try:
        for batch_index, batch_dict in enumerate(_batch_iterator(all_data, batch_size), start=1):
            try:
                result = process_batch_with_teacher(
                    batch_dict,
                    config.TEACHER_MODEL_NAME,
                    config.STUDENT_MODEL_NAME,
                    hf_token
                )
            except Exception as batch_error:
                print(f"\nERROR processing batch {batch_index}: {batch_error}")
                raise
            
            rows = _flatten_result_rows(result)
            writer, written = _append_to_parquet(rows, writer, output_file)
            rows_written += written
            
            if batch_index % 10 == 0 or batch_index == 1:
                elapsed = time.time() - start_time
                rate = rows_written / elapsed if elapsed > 0 else 0
                eta = (total_examples - rows_written) / rate if rate > 0 else float('inf')
                print(
                    f"Batch {batch_index}/{total_batches} | "
                    f"Rows written: {rows_written:,} | "
                    f"Rate: {rate:.2f} ex/s | "
                    f"Elapsed: {str(timedelta(seconds=int(elapsed)))} | "
                    f"ETA: {str(timedelta(seconds=int(eta)))}"
                )
            
            # Proactively free memory between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        if writer:
            writer.close()
        torch.cuda.empty_cache()
        gc.collect()
    
    elapsed_time = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    print(f"\n{'='*60}")
    print("Streaming job complete")
    print(f"  Rows written: {rows_written:,}")
    print(f"  Output file: {output_file}")
    print(f"  Total time: {elapsed_str}")
    print(f"{'='*60}\n")
    
    verify_saved_data(config.OFFLINE_DATA_PATH, num_samples=20)


if __name__ == "__main__":
    main()

