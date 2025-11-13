"""
Main Ray Tune Training Script

This script implements the core training function for knowledge distillation
experiments and configures Ray Tune to run all 28 trials (4 groups x 7 seeds).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any
import ray
from ray import train, tune
from ray.train import Checkpoint
import pyarrow.parquet as pq
import pandas as pd
from transformers import AutoTokenizer

from distillation_student import DistillationStudent
import config


class OfflineDistillationDataset(Dataset):
    """
    Dataset class for loading pre-computed teacher outputs.
    """
    
    def __init__(self, parquet_path: str, tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            parquet_path: Path to the Parquet file with pre-computed teacher data
            tokenizer: Tokenizer for the student model
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data from Parquet
        print(f"Loading dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        self.prompts = df["prompt"].tolist()
        self.answers = df["answer"].tolist()
        
        # Load teacher outputs (these are stored as numpy arrays)
        # Note: In a real implementation, you'd need to handle numpy array serialization
        # For now, we'll assume they're stored and can be loaded
        self.teacher_logits = df["teacher_logits"].tolist() if "teacher_logits" in df.columns else None
        self.teacher_hidden_state = df["teacher_hidden_state"].tolist() if "teacher_hidden_state" in df.columns else None
        self.teacher_attention_map = df["teacher_attention_map"].tolist() if "teacher_attention_map" in df.columns else None
        
        print(f"Loaded {len(self.prompts)} examples")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        answer = self.answers[idx]
        
        # Tokenize prompt and answer
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length"
        )
        
        # Create labels from answer (simplified - in practice you'd want proper answer encoding)
        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            padding="max_length"
        )
        
        item = {
            "input_ids": prompt_tokens["input_ids"].squeeze(0),
            "attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "labels": answer_tokens["input_ids"].squeeze(0),
        }
        
        # Add teacher outputs if available
        # Convert from lists back to numpy arrays, then to tensors
        if self.teacher_logits is not None:
            logits_array = np.array(self.teacher_logits[idx], dtype=np.float32)
            item["teacher_logits"] = torch.tensor(logits_array, dtype=torch.float32)
        
        if self.teacher_hidden_state is not None:
            hidden_array = np.array(self.teacher_hidden_state[idx], dtype=np.float32)
            item["teacher_hidden_state"] = torch.tensor(hidden_array, dtype=torch.float32)
        
        if self.teacher_attention_map is not None:
            attn_array = np.array(self.teacher_attention_map[idx], dtype=np.float32)
            item["teacher_attention_map"] = torch.tensor(attn_array, dtype=torch.float32)
        
        return item


def compute_loss(
    student_outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    teacher_data: Dict[str, torch.Tensor],
    distill_type: str
) -> Dict[str, torch.Tensor]:
    """
    Compute the total loss based on the distillation type.
    
    Loss formula: L_total = α·L_task + β·L_KD + γ₁·L_align_hidden + γ₂·L_align_attn
    
    Args:
        student_outputs: Dictionary from student model forward pass
        labels: Ground truth labels
        teacher_data: Dictionary containing teacher outputs
        distill_type: One of "black_box", "hidden_state", "attention", "combined"
        
    Returns:
        Dictionary with individual loss components and total loss
    """
    losses = {}
    
    # L_task: Task loss (Cross-Entropy on ground truth)
    student_logits = student_outputs["logits"]
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    losses["task_loss"] = task_loss
    
    # L_KD: Knowledge Distillation loss (KL divergence on logits)
    if "teacher_logits" in teacher_data and teacher_data["teacher_logits"] is not None:
        teacher_logits = teacher_data["teacher_logits"]
        
        # Softmax with temperature
        temperature = 3.0
        student_logits_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_logits_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        kd_loss = F.kl_div(
            student_logits_soft,
            teacher_logits_soft,
            reduction="batchmean"
        ) * (temperature ** 2)
        losses["kd_loss"] = kd_loss
    else:
        losses["kd_loss"] = torch.tensor(0.0, device=student_logits.device)
    
    # L_align_hidden: Hidden state alignment loss (MSE)
    gamma_1 = config.GAMMA_1 if distill_type in ["hidden_state", "combined"] else 0.0
    if gamma_1 > 0 and "projected_hidden_state" in student_outputs and "teacher_hidden_state" in teacher_data:
        student_hidden = student_outputs["projected_hidden_state"]
        teacher_hidden = teacher_data["teacher_hidden_state"]
        
        # Ensure same sequence length
        seq_len = min(student_hidden.size(1), teacher_hidden.size(1))
        student_hidden = student_hidden[:, :seq_len, :]
        teacher_hidden = teacher_hidden[:, :seq_len, :]
        
        align_hidden_loss = F.mse_loss(student_hidden, teacher_hidden)
        losses["align_hidden_loss"] = align_hidden_loss
    else:
        losses["align_hidden_loss"] = torch.tensor(0.0, device=student_logits.device)
    
    # L_align_attn: Attention map alignment loss (MSE)
    gamma_2 = config.GAMMA_2 if distill_type in ["attention", "combined"] else 0.0
    if gamma_2 > 0 and "attention_map" in student_outputs and "teacher_attention_map" in teacher_data:
        student_attn = student_outputs["attention_map"]
        teacher_attn = teacher_data["teacher_attention_map"]
        
        # Ensure same dimensions
        batch_size = min(student_attn.size(0), teacher_attn.size(0))
        seq_len = min(student_attn.size(-1), teacher_attn.size(-1))
        
        student_attn = student_attn[:batch_size, :, :seq_len, :seq_len]
        teacher_attn = teacher_attn[:batch_size, :, :seq_len, :seq_len]
        
        align_attn_loss = F.mse_loss(student_attn, teacher_attn)
        losses["align_attn_loss"] = align_attn_loss
    else:
        losses["align_attn_loss"] = torch.tensor(0.0, device=student_logits.device)
    
    # Total loss
    total_loss = (
        config.ALPHA * losses["task_loss"] +
        config.BETA * losses["kd_loss"] +
        gamma_1 * losses["align_hidden_loss"] +
        gamma_2 * losses["align_attn_loss"]
    )
    losses["total_loss"] = total_loss
    
    return losses


def train_student(config_dict: Dict[str, Any]):
    """
    Main training function executed by Ray Tune.
    
    This function:
    1. Loads the DistillationStudent model
    2. Loads the pre-computed offline dataset
    3. Implements the training loop with appropriate loss function
    4. Reports metrics back to Ray Tune
    
    Args:
        config_dict: Configuration dictionary from Ray Tune containing:
            - distill_type: One of "black_box", "hidden_state", "attention", "combined"
            - seed: Random seed for reproducibility
            - learning_rate: Learning rate (optional, uses config default if not provided)
    """
    # Set random seed for reproducibility
    seed = config_dict.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Get distillation type
    distill_type = config_dict["distill_type"]
    learning_rate = config_dict.get("learning_rate", config.LEARNING_RATE)
    
    print(f"Starting training with distill_type={distill_type}, seed={seed}, lr={learning_rate}")
    
    # Load student model
    print("Loading student model...")
    student_model = DistillationStudent(config.STUDENT_MODEL_NAME)
    student_model.train()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load offline dataset
    print(f"Loading offline dataset from {config.OFFLINE_DATA_PATH}...")
    
    # Find all Parquet files in the directory
    parquet_files = []
    for root, dirs, files in os.walk(config.OFFLINE_DATA_PATH):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {config.OFFLINE_DATA_PATH}")
    
    # Use the first Parquet file (or combine multiple if needed)
    dataset = OfflineDistillationDataset(parquet_files[0], tokenizer, config.MAX_SEQ_LENGTH)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Training loop
    num_epochs = config.NUM_EPOCHS
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Prepare teacher data
            teacher_data = {}
            if "teacher_logits" in batch:
                teacher_data["teacher_logits"] = batch["teacher_logits"].to(device) if batch["teacher_logits"] is not None else None
            if "teacher_hidden_state" in batch:
                teacher_data["teacher_hidden_state"] = batch["teacher_hidden_state"].to(device) if batch["teacher_hidden_state"] is not None else None
            if "teacher_attention_map" in batch:
                teacher_data["teacher_attention_map"] = batch["teacher_attention_map"].to(device) if batch["teacher_attention_map"] is not None else None
            
            # Forward pass
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_states=distill_type in ["hidden_state", "combined"],
                return_attention=distill_type in ["attention", "combined"],
                output_attentions=distill_type in ["attention", "combined"]
            )
            
            # Compute loss
            losses = compute_loss(student_outputs, labels, teacher_data, distill_type)
            total_loss = losses["total_loss"]
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Log metrics every N batches
            if batch_idx % 10 == 0:
                train.report({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_loss": total_loss.item(),
                    "task_loss": losses["task_loss"].item(),
                    "kd_loss": losses["kd_loss"].item(),
                    "align_hidden_loss": losses["align_hidden_loss"].item(),
                    "align_attn_loss": losses["align_attn_loss"].item(),
                    "distill_type": distill_type,
                    "seed": seed
                })
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Report epoch-level metrics
        train.report({
            "epoch": epoch,
            "epoch_loss": avg_epoch_loss,
            "distill_type": distill_type,
            "seed": seed,
            "validation_loss": avg_epoch_loss,  # Simplified - in practice you'd have a validation set
            "validation_accuracy": 0.0  # Placeholder - compute from validation set
        })
        
        print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")


def main():
    """
    Main function to configure and launch Ray Tune.
    """
    # Initialize Ray
    print("Initializing Ray...")
    if not ray.is_initialized():
        ray.init()
    
    # Define search space (4 experimental groups x 7 seeds = 28 trials)
    search_space = {
        "distill_type": tune.grid_search([
            "black_box",
            "hidden_state",
            "attention",
            "combined"
        ]),
        "seed": tune.grid_search(list(range(7))),  # 7 random seeds
        "learning_rate": config.LEARNING_RATE,
    }
    
    # Configure trainable with resources
    trainable_with_resources = tune.with_resources(
        train_student,
        {"cpu": config.RAY_NUM_CPUS_PER_TRIAL, "gpu": config.RAY_NUM_GPUS_PER_TRIAL}
    )
    
    # Create Tuner
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="validation_accuracy",  # Metric to optimize
            mode="max",  # We want to maximize it
        ),
        run_config=train.RunConfig(
            name="knowledge_distillation_experiment",
            storage_path=config.OUTPUT_PATH,
            stop={"epoch": config.NUM_EPOCHS},
        )
    )
    
    # Launch all 28 trials
    print("Launching Ray Tune with 28 trials...")
    results = tuner.fit()
    
    # Analyze results
    print("\n=== Experiment Results ===")
    df = results.get_dataframe()
    
    # Group by distill_type and compute mean accuracy
    if "validation_accuracy" in df.columns:
        summary = df.groupby("config/distill_type")["validation_accuracy"].mean()
        print("\nMean validation accuracy by distillation type:")
        print(summary)
    
    # Save results to CSV
    output_csv = os.path.join(config.OUTPUT_PATH, "results_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()

