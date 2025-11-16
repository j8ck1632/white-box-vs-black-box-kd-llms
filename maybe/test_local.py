"""
Local test script for knowledge distillation - runs a single trial without Ray Tune.
Use this to test your setup before running the full experiment.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer

from distillation_student import DistillationStudent
import config

# Override config for testing
TEST_BATCH_SIZE = 4  # Smaller batch size
TEST_NUM_EPOCHS = 1  # Just 1 epoch for testing
TEST_MAX_EXAMPLES = 100  # Only use first 100 examples


class SimpleTestDataset(Dataset):
    """Simple test dataset with dummy data"""
    
    def __init__(self, num_examples=100, tokenizer=None, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_examples = num_examples
        
        # Create dummy prompts and answers
        self.prompts = [f"This is test prompt number {i}." for i in range(num_examples)]
        self.answers = [f"Test answer {i}" for i in range(num_examples)]
        
        # Create dummy teacher data with more reasonable values
        vocab_size = tokenizer.vocab_size if tokenizer else 32000
        # Generate teacher logits with smaller variance to avoid numerical instability
        # Use smaller std (0.5 instead of 1.0) and ensure they're reasonable
        self.teacher_logits = [
            (np.random.randn(max_length, vocab_size) * 0.5).astype(np.float32)
            for _ in range(num_examples)
        ]
        # Generate teacher hidden states with smaller variance
        self.teacher_hidden_state = [
            (np.random.randn(max_length, 4096) * 0.1).astype(np.float32)
            for _ in range(num_examples)
        ]
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        answer = self.answers[idx]
        
        # Tokenize prompt and answer together
        full_text = f"{prompt} {answer}"
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length"
        )
        
        # Create labels: -100 for prompt tokens (ignore), actual tokens for answer
        input_ids = tokens["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Find where prompt ends (approximate - set first part to -100)
        prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        labels[:prompt_length] = -100  # Ignore prompt tokens in loss
        
        return {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "teacher_logits": torch.tensor(self.teacher_logits[idx], dtype=torch.float32),
            "teacher_hidden_state": torch.tensor(self.teacher_hidden_state[idx], dtype=torch.float32),
        }


def compute_loss(student_outputs, labels, teacher_data, distill_type):
    """Simplified loss computation"""
    student_logits = student_outputs["logits"]
    
    # Task loss
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # KD loss - ensure teacher_logits match student_logits dimensions
    teacher_logits = teacher_data["teacher_logits"]
    
    # Align sequence lengths
    batch_size, seq_len, vocab_size = student_logits.shape
    if teacher_logits.shape[1] != seq_len:
        # Truncate or pad teacher_logits to match
        if teacher_logits.shape[1] > seq_len:
            teacher_logits = teacher_logits[:, :seq_len, :]
        else:
            # Pad with zeros if needed
            pad_size = seq_len - teacher_logits.shape[1]
            padding = torch.zeros(batch_size, pad_size, teacher_logits.shape[2], 
                                device=teacher_logits.device, dtype=teacher_logits.dtype)
            teacher_logits = torch.cat([teacher_logits, padding], dim=1)
    
    # Ensure vocab size matches
    if teacher_logits.shape[2] != vocab_size:
        # Truncate vocab dimension if needed
        teacher_logits = teacher_logits[:, :, :vocab_size]
    
    temperature = 3.0
    # Clamp logits to prevent extreme values that cause NaN
    student_logits_clamped = torch.clamp(student_logits / temperature, min=-50.0, max=50.0)
    teacher_logits_clamped = torch.clamp(teacher_logits / temperature, min=-50.0, max=50.0)
    
    student_logits_soft = F.log_softmax(student_logits_clamped, dim=-1)
    teacher_logits_soft = F.softmax(teacher_logits_clamped, dim=-1)
    
    # Add small epsilon to teacher_logits_soft to prevent log(0) in KL divergence
    teacher_logits_soft = teacher_logits_soft + 1e-8
    teacher_logits_soft = teacher_logits_soft / teacher_logits_soft.sum(dim=-1, keepdim=True)  # Renormalize
    
    kd_loss = F.kl_div(student_logits_soft, teacher_logits_soft, reduction="batchmean", log_target=False) * (temperature ** 2)
    
    # Clamp KD loss to prevent NaN
    kd_loss = torch.clamp(kd_loss, min=0.0, max=1000.0)
    
    # Hidden state alignment (if needed)
    align_hidden_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
    if distill_type in ["hidden_state", "combined"] and "projected_hidden_state" in student_outputs:
        student_hidden = student_outputs["projected_hidden_state"]
        teacher_hidden = teacher_data["teacher_hidden_state"]
        
        # Ensure same dtype
        if teacher_hidden.dtype != student_hidden.dtype:
            teacher_hidden = teacher_hidden.to(dtype=student_hidden.dtype)
        
        seq_len = min(student_hidden.size(1), teacher_hidden.size(1))
        align_hidden_loss = F.mse_loss(
            student_hidden[:, :seq_len, :],
            teacher_hidden[:, :seq_len, :]
        )
    
    # Attention alignment (if needed)
    align_attn_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
    if distill_type in ["attention", "combined"] and "attention_map" in student_outputs:
        # For testing, we'll skip attention alignment as it requires matching dimensions
        pass
    
    # Clamp individual losses to prevent NaN
    task_loss = torch.clamp(task_loss, min=0.0, max=1000.0)
    align_hidden_loss = torch.clamp(align_hidden_loss, min=0.0, max=1000.0)
    align_attn_loss = torch.clamp(align_attn_loss, min=0.0, max=1000.0)
    
    # For testing with dummy data, reduce KD loss weight to prevent instability
    # Real teacher data would be more aligned with student, so this is just for testing
    test_beta = config.BETA * 0.1  # Reduce KD weight for dummy data
    
    total_loss = config.ALPHA * task_loss + test_beta * kd_loss
    if distill_type in ["hidden_state", "combined"]:
        total_loss += config.GAMMA_1 * align_hidden_loss
    if distill_type in ["attention", "combined"]:
        total_loss += config.GAMMA_2 * align_attn_loss
    
    # Final check: if total_loss is NaN or Inf, replace with a large but finite value
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = torch.tensor(1000.0, device=total_loss.device, dtype=total_loss.dtype)
    
    return {
        "task_loss": task_loss,
        "kd_loss": kd_loss,
        "align_hidden_loss": align_hidden_loss,
        "align_attn_loss": align_attn_loss,
        "total_loss": total_loss
    }


def test_training():
    """Test training with minimal configuration"""
    print("=" * 60)
    print("LOCAL TEST MODE - Single Trial")
    print("=" * 60)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: Running on CPU - this will be very slow!")
        print("Consider using a GPU if available.")
    
    # Load model
    print("\nLoading student model...")
    try:
        # Pass device to model initialization for better control
        student_model = DistillationStudent(config.STUDENT_MODEL_NAME, device=str(device))
        
        # If model wasn't moved by device_map, move it manually
        # Check if model is already on the correct device
        model_device = next(student_model.student_model.parameters()).device
        if model_device != device:
            student_model = student_model.to(device)
        
        # Fix dtype mismatch: ensure hidden_state_projector matches model dtype
        # On CPU, use float32 for better compatibility
        if device.type == "cpu":
            model_dtype = torch.float32
            # Convert model to float32 if it's in float16
            if next(student_model.student_model.parameters()).dtype == torch.float16:
                student_model.student_model = student_model.student_model.to(dtype=torch.float32)
        else:
            model_dtype = next(student_model.student_model.parameters()).dtype
        
        # Ensure projector is on the same device and dtype as the model
        student_model.hidden_state_projector = student_model.hidden_state_projector.to(
            device=device, dtype=model_dtype
        )
        
        # Set attention implementation to 'eager' for attention output support
        if hasattr(student_model.student_model, 'set_attn_implementation'):
            try:
                student_model.student_model.set_attn_implementation('eager')
            except:
                pass  # If not supported, continue anyway
        
        student_model.train()
        print(f"✓ Model loaded successfully (dtype: {model_dtype})")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return
    
    # Create test dataset
    print(f"\nCreating test dataset ({TEST_MAX_EXAMPLES} examples)...")
    try:
        dataset = SimpleTestDataset(
            num_examples=TEST_MAX_EXAMPLES,
            tokenizer=tokenizer,
            max_length=128  # Shorter sequences for testing
        )
        print(f"✓ Dataset created with {len(dataset)} examples")
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return
    
    train_loader = DataLoader(
        dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # No multiprocessing for testing
        pin_memory=(device.type == "cuda")
    )
    
    # Setup optimizer with lower learning rate for testing stability
    test_lr = config.LEARNING_RATE * 0.1  # Use 10% of normal LR for testing
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=test_lr,
        weight_decay=0.01
    )
    print(f"Using learning rate: {test_lr} (reduced for testing)")
    
    # Test different distillation types
    distill_types = ["black_box", "hidden_state", "attention", "combined"]
    
    for distill_type in distill_types:
        print(f"\n{'=' * 60}")
        print(f"Testing distillation type: {distill_type}")
        print(f"{'=' * 60}")
        
        # Reset model (optional - for cleaner test)
        student_model.train()
        
        try:
            for epoch in range(TEST_NUM_EPOCHS):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    # Move to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    teacher_data = {
                        "teacher_logits": batch["teacher_logits"].to(device=device, dtype=model_dtype),
                        "teacher_hidden_state": batch["teacher_hidden_state"].to(device=device, dtype=model_dtype),
                    }
                    
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
                    
                    # Check for NaN before backward pass
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"  WARNING: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Clip gradients more aggressively to prevent explosion
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=0.5)
                    
                    # Check for NaN gradients
                    has_nan_grad = False
                    for param in student_model.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        print(f"  WARNING: NaN gradients detected at batch {batch_idx}, skipping update...")
                        optimizer.zero_grad()  # Clear gradients
                        continue
                    
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                    
                    if batch_idx % 5 == 0:
                        loss_val = total_loss.item()
                        task_val = losses['task_loss'].item()
                        kd_val = losses['kd_loss'].item()
                        print(f"  Batch {batch_idx}/{len(train_loader)}: "
                              f"loss={loss_val:.4f}, "
                              f"task={task_val:.4f}, "
                              f"kd={kd_val:.4f}")
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            print(f"✓ {distill_type} distillation type test passed")
            
        except Exception as e:
            print(f"✗ Error testing {distill_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)
    print("\nIf all tests passed, you can now run the full experiment with:")
    print("  python train_student.py")
    print("\nOr test with Ray Tune (single trial):")
    print("  python train_student.py --test  # (if you add test mode)")


if __name__ == "__main__":
    test_training()


