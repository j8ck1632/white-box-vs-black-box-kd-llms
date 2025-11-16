"""
Configuration parameters for the knowledge distillation experiment.
"""

import os

# Model names
TEACHER_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
STUDENT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Dataset names
SST2_DATASET = "glue"
SST2_CONFIG = "sst2"
MMLU_DATASET = "cais/mmlu"
GSM8K_DATASET = "gsm8k"

# Loss function weights
ALPHA = 1.0  # Weight for task loss (L_task)
BETA = 0.5   # Weight for KD loss (L_KD)
GAMMA_1 = 0.1  # Weight for hidden state alignment loss (L_align_hidden)
GAMMA_2 = 0.1  # Weight for attention alignment loss (L_align_attn)

# Training hyperparameters (large Colab Pro+ defaults)
LEARNING_RATE = 1e-4
BATCH_SIZE = 80
WHITEBOX_BATCH_SIZE = 20
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 256
VALIDATION_SPLIT = 0.1
MAX_CPU_THREADS = 4
USE_AUTOMATIC_MIXED_PRECISION = True
DATALOADER_NUM_WORKERS = 6  # Increase on beefier hosts to keep the GPU fed

# Teacher model dimensions
TEACHER_HIDDEN_DIM = 4096
TEACHER_NUM_HEADS = 32

# Student model dimensions (from TinyLlama config)
STUDENT_HIDDEN_DIM = 2048
STUDENT_NUM_HEADS = 32

# Paths
# Prefer writing large artifacts to D: if available to avoid C: exhaustion.
_D_DRIVE_AVAILABLE = os.path.exists("D:\\")
_DEFAULT_OFFLINE_DATA_PATH = "D:\\offline_teacher_data" if _D_DRIVE_AVAILABLE else "./offline_teacher_data"
OFFLINE_DATA_PATH = os.environ.get("WBVB_OFFLINE_DATA_PATH", _DEFAULT_OFFLINE_DATA_PATH)
_DEFAULT_OUTPUT_PATH = "D:\\wbvb_results" if _D_DRIVE_AVAILABLE else "./results"
OUTPUT_PATH = os.environ.get("WBVB_OUTPUT_PATH", _DEFAULT_OUTPUT_PATH)

# Offline data compression knobs
TOP_K_LOGITS = 128          # Number of logits to keep per token
HIDDEN_STRIDE = 2           # Down-sample hidden states along sequence
ATTENTION_STRIDE = 2        # Down-sample attention maps along sequence dims
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3

# Teacher vocab size (used to reconstruct logits from top-k representation)
TEACHER_VOCAB_SIZE = 32000

# Hugging Face token (required for gated models)
# WARNING: Replace with your own token before running; this placeholder is safe to commit.
HUGGING_FACE_TOKEN = "hf_your_actual_token_here"

