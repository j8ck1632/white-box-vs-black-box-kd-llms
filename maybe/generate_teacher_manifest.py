"""
Utility script to spit out a single JSON file with all critical teacher metadata.

Share the generated JSON with partner machines so they know exactly which
teacher snapshot, datasets, and compression knobs were used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import config
from distillation_core import list_parquet_files


DATASETS = [
    {"name": "SST-2", "source": "GLUE", "examples": 5000, "task": "Sentiment / NLU"},
    {"name": "MMLU", "source": "cais/mmlu", "examples": 1000, "task": "Multi-domain reasoning"},
    {"name": "GSM8K", "source": "gsm8k", "examples": 1000, "task": "Math word problems"},
]


def _file_metadata(path: str) -> dict:
    stats = os.stat(path)
    return {
        "path": os.path.relpath(path, Path.cwd()),
        "size_bytes": stats.st_size,
    }


def build_manifest(offline_path: str) -> dict:
    parquet_files = list_parquet_files(offline_path)
    offline_section = {
        "base_path": os.path.abspath(offline_path),
        "files": [_file_metadata(p) for p in parquet_files],
        "compression": {
            "top_k_logits": config.TOP_K_LOGITS,
            "hidden_stride": config.HIDDEN_STRIDE,
            "attention_stride": config.ATTENTION_STRIDE,
            "codec": config.PARQUET_COMPRESSION,
            "compression_level": config.PARQUET_COMPRESSION_LEVEL,
        },
    }
    teacher_section = {
        "model_name": config.TEACHER_MODEL_NAME,
        "hidden_dim": config.TEACHER_HIDDEN_DIM,
        "attention_heads": config.TEACHER_NUM_HEADS,
        "vocab_size": config.TEACHER_VOCAB_SIZE,
    }
    student_section = {
        "model_name": config.STUDENT_MODEL_NAME,
        "hidden_dim": config.STUDENT_HIDDEN_DIM,
        "attention_heads": config.STUDENT_NUM_HEADS,
        "projector_dim": f"{config.STUDENT_HIDDEN_DIM}->{config.TEACHER_HIDDEN_DIM}",
    }
    manifest = {
        "teacher": teacher_section,
        "student": student_section,
        "datasets": DATASETS,
        "offline_data": offline_section,
        "loss_weights": {
            "alpha": config.ALPHA,
            "beta": config.BETA,
            "gamma_hidden": config.GAMMA_1,
            "gamma_attention": config.GAMMA_2,
        },
        "training_defaults": {
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.NUM_EPOCHS,
            "max_seq_length": config.MAX_SEQ_LENGTH,
        },
    }
    return manifest


def parse_args(cli_args=None):
    parser = argparse.ArgumentParser(description="Generate teacher manifest JSON.")
    parser.add_argument(
        "--offline-path",
        type=str,
        default=config.OFFLINE_DATA_PATH,
        help="Directory where offline teacher Parquet files live.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="teacher_manifest.json",
        help="Where to write the manifest JSON.",
    )
    return parser.parse_args(cli_args)


def main(cli_args=None):
    args = parse_args(cli_args)
    manifest = build_manifest(args.offline_path)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {output_path} (contains {len(manifest['offline_data']['files'])} parquet references)")


if __name__ == "__main__":
    main()


