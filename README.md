# White-Box vs Black-Box Knowledge Distillation in LLMs

This repository contains the complete implementation for the research experiment comparing white-box and black-box knowledge distillation methods in causal language models.

## Overview

This experiment systematically compares different knowledge distillation signals:
- **Black-Box**: Using only final logits
- **White-Box (Hidden)**: Using logits + hidden states
- **White-Box (Attention)**: Using logits + attention maps
- **White-Box (Combined)**: Using all signals

The experiment runs 28 trials (4 groups × 7 seeds) on a Ray cluster with 28 GPUs.

## Project Structure

```
.
├── requirements.txt              # Python dependencies
├── config.py                     # Configuration parameters
├── distillation_student.py       # DistillationStudent model class
├── offline_teacher_data.py       # Pre-compute teacher outputs
├── train_student.py              # Main Ray Tune training script
└── README.md                     # This file
```

## Setup

### 1. Environment Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ray Cluster Setup

Initialize your Ray cluster. On a multi-node cluster:

```bash
# On head node
ray start --head

# On worker nodes
ray start --address=<head-node-ip>:10001
```

Or if using a Ray cluster manager (e.g., Ray on Kubernetes), configure according to your setup.

### 3. Configuration Setup

Copy the example configuration file and add your Hugging Face token:

```bash
cp config.py.example config.py
```

Then edit `config.py` and replace `YOUR_TOKEN_HERE` with your actual Hugging Face token:

```python
HUGGING_FACE_TOKEN = "your_actual_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

**Note**: `config.py` is ignored by git (it's in `.gitignore`) to protect your token. Only `config.py.example` is tracked in the repository.

### 4. Model Access

The required HuggingFace models:
- `mistralai/Mistral-7B-v0.1` (teacher) - **Requires authentication** (gated model)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (student) - Public, no authentication required

Make sure you have access to the teacher model and your token is set in `config.py`.

## Usage

### Step 1: Pre-compute Teacher Data (Offline)

Run the offline teacher data generation script. This runs the teacher model once on all datasets and saves the outputs:

```bash
python offline_teacher_data.py
```

This will:
- Load datasets (SST-2, MMLU, GSM8K)
- Run the teacher model on all prompts using Ray Data
- Save enriched dataset with teacher outputs to `./offline_teacher_data/` in Parquet format

**Expected output**: A Parquet directory containing:
- `prompt`: Original prompts
- `answer`: Original answers
- `teacher_logits`: Teacher model logits
- `teacher_hidden_state`: Teacher final layer hidden states
- `teacher_attention_map`: Teacher final layer attention maps

### Step 2: Train Student Models

Run the main training script to launch all 28 trials:

```bash
python train_student.py
```

This will:
- Load the pre-computed teacher data
- Launch 28 Ray Tune trials (4 distillation types × 7 seeds)
- Train student models with different distillation signals
- Save results to `./results/`

**Expected runtime**: Several hours to days depending on dataset size and cluster configuration.

## Configuration

Edit `config.py` to adjust experiment parameters:

- **Loss weights**: `ALPHA`, `BETA`, `GAMMA_1`, `GAMMA_2`
- **Training hyperparameters**: `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`
- **Model names**: `TEACHER_MODEL_NAME`, `STUDENT_MODEL_NAME`
- **Paths**: `OFFLINE_DATA_PATH`, `OUTPUT_PATH`
- **Ray resources**: `RAY_NUM_GPUS_PER_TRIAL`, `RAY_NUM_CPUS_PER_TRIAL`

## Experiment Design

### Experimental Groups

| Group | Distillation Type | Signals Used |
|-------|------------------|--------------|
| 1 | Black-Box | L_task + L_KD (logits only) |
| 2 | White-Box (Hidden) | L_task + L_KD + L_align_hidden |
| 3 | White-Box (Attention) | L_task + L_KD + L_align_attn |
| 4 | White-Box (Combined) | All signals |

### Loss Function

The total loss is computed as:

```
L_total = α·L_task + β·L_KD + γ₁·L_align_hidden + γ₂·L_align_attn
```

Where:
- `L_task`: Cross-entropy on ground truth
- `L_KD`: KL divergence on logits (black-box)
- `L_align_hidden`: MSE on hidden states (white-box)
- `L_align_attn`: MSE on attention maps (white-box)

### Evaluation Datasets

- **SST-2**: Sentiment analysis (NLU task)
- **MMLU**: Multi-task language understanding (Reasoning task)
- **GSM8K**: Grade-school math word problems (Math task)

## Results

After training completes, results will be saved to:
- `./results/results_summary.csv`: Summary of all trials
- `./results/`: Ray Tune checkpoints and detailed metrics

To analyze results:

```python
import pandas as pd

df = pd.read_csv("./results/results_summary.csv")
summary = df.groupby("config/distill_type")["validation_accuracy"].mean()
print(summary)
```

## Key Components

### DistillationStudent

The `DistillationStudent` class (`distillation_student.py`) wraps TinyLlama and adds:
- A trainable hidden state projector (2048 → 4096 dimensions)
- Methods to extract logits, hidden states, and attention maps

### Offline Teacher Data Script

The `offline_teacher_data.py` script:
- Uses Ray Data for parallel processing
- Runs teacher model with `@ray.remote(num_gpus=1)`
- Saves all teacher outputs to disk for efficient offline distillation

### Training Script

The `train_student.py` script:
- Implements the `train_student` function for Ray Tune
- Configures search space for 28 trials
- Implements loss calculation based on `distill_type`
- Reports metrics back to Ray Tune

## Troubleshooting

### Out of Memory

If you encounter GPU memory issues:
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `MAX_SEQ_LENGTH` in `config.py`
- Use gradient accumulation

### Ray Cluster Issues

- Ensure Ray is properly initialized: `ray status`
- Check GPU availability: `ray list nodes`
- Verify worker nodes can access shared storage

### Dataset Loading Issues

- Ensure you have internet access for HuggingFace datasets
- Some datasets may require authentication or approval
- Check dataset names in `config.py` match current HuggingFace names

## Citation

If you use this code in your research, please cite:

```
@misc{whitebox_blackbox_kd_llms,
  title={White-Box vs Black-Box Knowledge Distillation in Causal Language Models},
  author={Jack Large, Madelyn Sarbin},
  year={2025}
}
```

## License

Apache 2.0
