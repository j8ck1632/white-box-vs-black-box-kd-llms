"""
Offline Teacher Data Pre-computation Script

This script runs the teacher model (Llama-3-8B) once on all datasets
and saves the outputs (logits, hidden states, attention maps) to disk.
This enables efficient offline distillation without running the teacher
during student training.
"""

import os
import ray
import torch
import pyarrow.parquet as pq
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd

import config


@ray.remote(num_gpus=1)
def process_batch_with_teacher(batch: Dict[str, List[Any]], model_name: str) -> Dict[str, List]:
    """
    Process a batch of prompts through the teacher model.
    
    This function runs on a Ray remote worker with GPU access.
    
    Args:
        batch: Dictionary containing 'prompt' and 'answer' lists
        model_name: HuggingFace model identifier for the teacher
        
    Returns:
        Dictionary containing enriched batch data with:
        - prompt: Original prompts
        - answer: Original answers
        - teacher_logits: Teacher model logits
        - teacher_hidden_state: Teacher final layer hidden states
        - teacher_attention_map: Teacher final layer attention maps
    """
    # Load tokenizer and model on this GPU worker
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    prompts = batch["prompt"]
    answers = batch["answer"]
    
    # Tokenize all prompts
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)
    
    # Forward pass with all outputs
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
    
    # Extract outputs
    teacher_logits = outputs.logits.cpu().numpy()
    
    # Get final layer hidden states (last element of hidden_states tuple)
    teacher_hidden_state = outputs.hidden_states[-1].cpu().numpy()
    
    # Get final layer attention maps (last element of attentions tuple)
    teacher_attention_map = outputs.attentions[-1].cpu().numpy()
    
    # Convert numpy arrays to lists for Ray Data compatibility
    # Ray Data will serialize these properly when writing to Parquet
    batch_size = len(prompts)
    
    # Convert numpy arrays to lists (each example is a numpy array)
    result = {
        "prompt": prompts,
        "answer": answers,
        "teacher_logits": [teacher_logits[i].tolist() for i in range(batch_size)],
        "teacher_hidden_state": [teacher_hidden_state[i].tolist() for i in range(batch_size)],
        "teacher_attention_map": [teacher_attention_map[i].tolist() for i in range(batch_size)],
    }
    
    # Clean up model from GPU memory
    del model
    torch.cuda.empty_cache()
    
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
    
    for example in sst2_dataset:
        prompt = example["sentence"]
        # Convert label (0/1) to text answer
        answer = "positive" if example["label"] == 1 else "negative"
        all_data.append({"prompt": prompt, "answer": answer})
    
    # Load MMLU (subset - using a few-shot format)
    print("Loading MMLU dataset...")
    try:
        mmlu_dataset = load_dataset(config.MMLU_DATASET, "all", split="train")
        # Take a subset (first 1000 examples)
        for example in mmlu_dataset.select(range(min(1000, len(mmlu_dataset)))):
            prompt = f"Question: {example['question']}\nChoices: {example['choices']}\nAnswer:"
            answer = example["answer"]
            all_data.append({"prompt": prompt, "answer": answer})
    except Exception as e:
        print(f"Warning: Could not load MMLU dataset: {e}")
    
    # Load GSM8K (Math Word Problems)
    print("Loading GSM8K dataset...")
    try:
        gsm8k_dataset = load_dataset(config.GSM8K_DATASET, "main", split="train")
        # Take a subset (first 1000 examples)
        for example in gsm8k_dataset.select(range(min(1000, len(gsm8k_dataset)))):
            prompt = example["question"]
            answer = example["answer"]
            all_data.append({"prompt": prompt, "answer": answer})
    except Exception as e:
        print(f"Warning: Could not load GSM8K dataset: {e}")
    
    print(f"Total examples loaded: {len(all_data)}")
    return all_data


def main():
    """
    Main function to pre-compute teacher model outputs.
    """
    # Initialize Ray
    print("Initializing Ray...")
    if not ray.is_initialized():
        ray.init()
    
    # Load and preprocess datasets
    print("Loading and preprocessing datasets...")
    all_data = load_and_preprocess_datasets()
    
    # Convert to Ray Dataset
    print("Creating Ray Dataset...")
    ray_dataset = ray.data.from_items(all_data)
    
    # Process batches through teacher model
    print("Processing batches through teacher model...")
    batch_size = 4  # Process 4 examples per batch
    
    enriched_dataset = ray_dataset.map_batches(
        process_batch_with_teacher,
        fn_kwargs={
            "model_name": config.TEACHER_MODEL_NAME
        },
        batch_size=batch_size,
        num_gpus=1
    )
    
    # Save to Parquet format
    print(f"Saving enriched dataset to {config.OFFLINE_DATA_PATH}...")
    os.makedirs(config.OFFLINE_DATA_PATH, exist_ok=True)
    
    enriched_dataset.write_parquet(config.OFFLINE_DATA_PATH)
    
    print(f"Offline teacher data saved successfully to {config.OFFLINE_DATA_PATH}")
    print(f"Total examples processed: {len(all_data)}")


if __name__ == "__main__":
    main()

