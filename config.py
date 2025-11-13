"""
Configuration parameters for the knowledge distillation experiment.
"""

# Model names
TEACHER_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
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

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512

# Teacher model dimensions
TEACHER_HIDDEN_DIM = 4096
TEACHER_NUM_HEADS = 32

# Student model dimensions (from TinyLlama config)
STUDENT_HIDDEN_DIM = 2048
STUDENT_NUM_HEADS = 32

# Paths
OFFLINE_DATA_PATH = "./offline_teacher_data"
OUTPUT_PATH = "./results"

# Ray configuration
RAY_NUM_GPUS_PER_TRIAL = 1
RAY_NUM_CPUS_PER_TRIAL = 4

