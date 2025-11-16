# Experiment Outline: White-Box vs Black-Box Knowledge Distillation in LLMs

## Research Question

**What is the comparative value of different knowledge distillation signals when distilling from a large teacher model to a small student model?**

Specifically, we compare:
- **(a) Black-Box**: Final logits only
- **(b) White-Box (Hidden)**: Final logits + final layer hidden states
- **(c) White-Box (Attention)**: Final logits + final layer attention maps
- **(d) White-Box (Combined)**: All signals (logits + hidden states + attention maps)

## Hypothesis

For simple Natural Language Understanding (NLU) tasks (e.g., sentiment analysis), all methods will perform similarly. However, for complex **reasoning** and **math** tasks, the white-box methods will significantly outperform the black-box baseline by successfully transferring more of the teacher's internal "thought process."

## Models

### Teacher Model
- **Model**: `mistralai/Mistral-7B-v0.1`
- **Parameters**: 7 billion
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Purpose**: Large, pre-trained model that provides knowledge to distill

### Student Model
- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Parameters**: 1.1 billion (~6.4x smaller)
- **Hidden Dimension**: 2048
- **Attention Heads**: 32
- **Architecture Addition**: Trainable hidden state projector (2048 → 4096 dimensions) to align student and teacher hidden states

## Datasets

The experiment uses a combined dataset from three sources:

| Dataset | Source | Examples | Task Type | Purpose |
|---------|--------|----------|-----------|---------|
| **SST-2** | GLUE benchmark | 5,000 | Sentiment Analysis (NLU) | Simple classification task |
| **MMLU** | cais/mmlu | 1,000 | Multi-task Language Understanding (Reasoning) | Complex reasoning across domains |
| **GSM8K** | gsm8k | 1,000 | Grade-school Math Word Problems | Mathematical reasoning |
| **Total** | - | **7,000** | Mixed | Diverse evaluation across task types |

**Note**: SST-2 is limited to 5,000 examples (from ~67,349 total) for computational efficiency while maintaining statistical validity.

## Experimental Groups

The experiment consists of **4 experimental groups**, each testing a different distillation approach:

| Group | Distillation Type | Signals Used | Loss Components | Purpose |
|-------|------------------|-------------|-----------------|---------|
| **1** | Black-Box | Logits only | L_task + L_KD | Baseline: standard distillation method |
| **2** | White-Box (Hidden) | Logits + Hidden States | L_task + L_KD + L_align_hidden | Test hidden state alignment value |
| **3** | White-Box (Attention) | Logits + Attention Maps | L_task + L_KD + L_align_attn | Test attention map alignment value |
| **4** | White-Box (Combined) | All signals | All loss components | Test combined white-box signals |

## Loss Function

The total loss is computed as:

```
L_total = α·L_task + β·L_KD + γ₁·L_align_hidden + γ₂·L_align_attn
```

### Loss Components

1. **L_task** (α = 1.0)
   - Cross-entropy loss on ground truth labels
   - Ensures student learns the actual task
   - Always active

2. **L_KD** (β = 0.5)
   - KL divergence between teacher and student logits
   - Black-box knowledge distillation signal
   - Always active

3. **L_align_hidden** (γ₁ = 0.1)
   - Mean Squared Error (MSE) between teacher and student hidden states
   - White-box signal: aligns internal representations
   - Active only in groups 2 and 4
   - Uses a trainable projector to map student (2048) → teacher (4096) dimensions

4. **L_align_attn** (γ₂ = 0.1)
   - Mean Squared Error (MSE) between teacher and student attention maps
   - White-box signal: aligns attention patterns
   - Active only in groups 3 and 4
   - Both models have 32 attention heads

### Loss Activation by Group

| Group | L_task | L_KD | L_align_hidden | L_align_attn |
|-------|--------|------|----------------|--------------|
| Black-Box | ✓ | ✓ | ✗ | ✗ |
| White-Box (Hidden) | ✓ | ✓ | ✓ | ✗ |
| White-Box (Attention) | ✓ | ✓ | ✗ | ✓ |
| White-Box (Combined) | ✓ | ✓ | ✓ | ✓ |

## Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate** | 1e-4 | AdamW optimizer learning rate |
| **Batch Size** | 8 | Examples per batch |
| **Epochs** | 3 | Number of training epochs |
| **Max Sequence Length** | 512 | Maximum tokens per example |
| **Gradient Clipping** | 1.0 | Max gradient norm |
| **Weight Decay** | 0.01 | L2 regularization |

### Training Process

1. **Offline Teacher Data Generation** (Pre-computation)
   - Run teacher model (Mistral-7B) on all 7,000 examples
   - Extract and save: logits, hidden states, attention maps
   - Uses eager attention implementation to capture attention maps
   - Saves to Parquet format: `./offline_teacher_data/` (or `D:\offline_teacher_data`)
   - Estimated time: ~1 hour on single GPU

2. **Offline Data Validation & Serialization Regression Tests**
   - Run `python -c "import config; import offline_teacher_data as otd; otd.verify_saved_data(config.OFFLINE_DATA_PATH, num_samples=20)"`.
   - Confirms teacher logits/hidden/attention tensors round-trip from Parquet and match expected shapes.
   - Execute `python -m pytest -s tests/test_offline_teacher_data.py` to ensure `_to_py` / `_to_numpy_array` conversions keep working before training reruns.
   - Archive the validated Parquet snapshot with a timestamp (e.g., `offline_teacher_data_2025-11-14.parquet`).

3. **Student Training** (Per Trial)
   - Load pre-computed teacher outputs
   - Train student model (TinyLlama) using distillation loss
   - Loss components activated based on experimental group
   - Report metrics after each epoch

## Experimental Design

### Trial Structure

- **Total Trials**: 28
- **Experimental Groups**: 4
- **Random Seeds**: 7 (seeds 0-6)
- **Design**: 4 groups × 7 seeds = 28 trials

### Statistical Robustness

- **7 random seeds per group** ensures statistical significance
- Allows for mean, standard deviation, and confidence interval calculations
- Enables proper comparison between groups

### Resource Requirements

- **GPUs per trial**: 1
- **CPUs per trial**: 4
- **Total GPUs needed**: 28 (for full parallel execution)
- **Concurrent trials**: Auto-detected based on available GPUs

## Evaluation Metrics

### Primary Metrics

- **Validation Loss**: Total loss on validation set
- **Validation Accuracy**: Task-specific accuracy
  - SST-2: Classification accuracy
  - MMLU: Multiple-choice accuracy
  - GSM8K: Exact match accuracy

### Secondary Metrics

- **Task Loss**: L_task component
- **KD Loss**: L_KD component
- **Hidden Alignment Loss**: L_align_hidden component (if applicable)
- **Attention Alignment Loss**: L_align_attn component (if applicable)

### Evaluation Process

- Metrics computed after each epoch
- Final evaluation on combined validation set
- Results saved to: `./results/results_summary.csv`

## Implementation Details

### Key Components

1. **`offline_teacher_data.py`**
   - Pre-computes teacher outputs for all examples
   - Streams batches sequentially on a single GPU (no Ray dependency)
   - Saves compressed logits (top-k) plus hidden states/attention maps to Parquet

2. **`distillation_student.py`**
   - Wraps TinyLlama student model
   - Adds trainable hidden state projector
   - Extracts logits, hidden states, attention maps

3. **`train_student.py`**
   - Main training function for Ray Tune
   - Implements loss computation based on distillation type
   - Handles data loading and training loop

4. **`config.py`**
   - Centralized configuration
   - Model names, hyperparameters, loss weights
   - Dataset paths and Ray configuration

### Technical Details

- **Attention Implementation**: Eager (required for attention map extraction)
- **Model Precision**: float16 (for memory efficiency)
- **Device**: GPU (CUDA) with automatic device mapping
- **Framework**: PyTorch + Hugging Face Transformers

### Quality Assurance & Methodology Updates

1. **Deterministic Offline Snapshot**
   - `config.OFFLINE_DATA_PATH` targets the high-capacity drive when available to prevent C: exhaustion.
   - Every offline run now records the exact Parquet location to keep student training reproducible.

2. **Data Validation Gate**
   - `verify_saved_data()` is mandatory before launching any training trials.
   - The script samples random rows, reconstructs logits/hidden/attention tensors, and fails fast on ragged structures.

3. **Serialization Regression Tests**
   - Unit tests in `tests/test_offline_teacher_data.py` cover `_to_py` / `_to_numpy_array`.
   - Run via `python -m pytest -s tests/test_offline_teacher_data.py` (the `-s` flag avoids a known capture bug on Windows PyTorch installs).

4. **Snapshot Archiving**
   - After validation, copy `offline_teacher_data.parquet` to a timestamped filename and log it in the experiment tracker.
   - Student trials reference archived snapshots so future reruns never recompute the teacher unless inputs change.
- **Distributed Training**: Ray Tune for parallel trial execution

## Expected Outcomes

### Research Questions to Answer

1. **Do white-box signals improve over black-box distillation?**
   - Compare groups 2, 3, 4 vs group 1

2. **Which white-box signal is more valuable?**
   - Compare group 2 (hidden) vs group 3 (attention)

3. **Does combining signals help?**
   - Compare group 4 (combined) vs groups 2 and 3

4. **Is the benefit task-dependent?**
   - Analyze performance separately on SST-2 (NLU), MMLU (reasoning), GSM8K (math)

### Success Criteria

- **Statistical Significance**: 7 seeds provide robust statistics
- **Task Coverage**: Three diverse task types (NLU, reasoning, math)
- **Reproducibility**: All hyperparameters and seeds documented
- **Publishability**: Results suitable for top-tier NLP/ML conference

## Output Files

### Results Structure

```
./results/
├── knowledge_distillation_experiment/
│   ├── trial_<id>_<distill_type>_seed_<seed>/
│   │   ├── checkpoint_<epoch>/
│   │   └── result.json
│   └── ...
└── results_summary.csv
```

### Analysis

Results can be analyzed with:

```python
import pandas as pd

df = pd.read_csv("./results/results_summary.csv")
summary = df.groupby("config/distill_type")["validation_accuracy"].agg(['mean', 'std'])
print(summary)
```

## Timeline

1. **Pre-computation**: ~1 hour (offline teacher data generation)
2. **Training**: ~28 trials × ~30 minutes/trial = ~14 hours (with 28 GPUs in parallel)
3. **Analysis**: Variable (post-processing and statistical analysis)

**Total**: ~15-16 hours for complete experiment (with sufficient GPU resources)

## Notes

- Teacher model requires Hugging Face authentication (gated model)
- All teacher outputs are pre-computed to avoid running teacher during training
- Hidden state projector is trainable and learns optimal alignment
- Attention maps extracted from final layer only
- Sequence length limited to 512 tokens for computational efficiency


