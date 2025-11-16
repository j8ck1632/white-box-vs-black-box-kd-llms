"""
Bootstrap script for the **local (non-Ray)** distillation runner.

It will:
1. Create (or reuse) a local virtual environment inside `.wbvb-env`.
2. Install the requirements listed in `requirements.txt`.
3. Ask the user which offline teacher dataset directory to use.
4. Launch the sequential `train_student.py` runner with that dataset wired in.

Usage:
    python bootstrap.py

Notes:
- The training script now executes experiments sequentially on the current
  machine. Use the `single` mode if you only want a smoke test.
- Make sure the offline dataset directory contains one or more `.parquet`
  files produced by `offline_teacher_data.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict
import json

from transformers import AutoTokenizer
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - bootstrap installs requirements soon after
    tqdm = None

import config
from distillation_core import OfflineDistillationDataset, list_parquet_files


BASE_DIR = Path(__file__).resolve().parent
ENV_DIR = BASE_DIR / ".wbvb-env"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
TRAIN_SCRIPT = BASE_DIR / "train_student.py"
PARQUET_SUFFIX = ".parquet"
TORCH_CUDA_INDEX = os.environ.get(
    "WBVB_TORCH_CUDA_INDEX",
    "https://download.pytorch.org/whl/cu121",
)
ALLOW_CPU_FALLBACK = os.environ.get("WBVB_ALLOW_CPU", "0") == "1"
PRESETS = {
    "machine_a": {
        "distill_types": ["black_box"],
        "seeds": [0, 1, 2, 3],
    },
    "machine_b": {
        "distill_types": ["hidden_state"],
        "seeds": [0, 1, 2, 3],
    },
    "machine_c": {
        "distill_types": ["attention", "combined"],
        "seeds": [0, 1, 2],
    },
}


def _run(cmd, **kwargs):
    """Run a subprocess command with logging."""
    print(f"[bootstrap] Running: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def ensure_virtualenv() -> Path:
    """Create the virtual environment if needed and return the Python path."""
    if not ENV_DIR.exists():
        print(f"[bootstrap] Creating virtual environment at {ENV_DIR}")
        _run([sys.executable, "-m", "venv", str(ENV_DIR)])
    else:
        print(f"[bootstrap] Reusing existing virtual environment at {ENV_DIR}")

    if os.name == "nt":
        python_path = ENV_DIR / "Scripts" / "python.exe"
    else:
        python_path = ENV_DIR / "bin" / "python"

    if not python_path.exists():
        raise FileNotFoundError(f"Python executable not found inside venv: {python_path}")

    return python_path


def install_requirements(python_path: Path) -> None:
    """Install project dependencies into the virtual environment."""
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQUIREMENTS_FILE}")

    print("[bootstrap] Installing dependencies (this may take a few minutes)...")
    _run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def _check_cuda(python_path: Path) -> bool:
    cmd = [
        str(python_path),
        "-c",
        "import torch; import json; "
        "print(json.dumps({'available': torch.cuda.is_available(), 'count': torch.cuda.device_count()}))",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout.strip())
    available = bool(data["available"])
    if available:
        print(f"[bootstrap] CUDA detected ({data['count']} device(s))")
    else:
        print("[bootstrap] CUDA not available in the current virtual environment.")
    return available


def ensure_cuda_ready(python_path: Path) -> None:
    """Ensure PyTorch was installed with CUDA support, or attempt to add it."""
    if _check_cuda(python_path):
        return

    if ALLOW_CPU_FALLBACK:
        print("[bootstrap] WBVB_ALLOW_CPU=1 set; continuing without GPU support.")
        return

    print("[bootstrap] Attempting to install CUDA-enabled PyTorch (cu121 wheel)...")
    _run(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            TORCH_CUDA_INDEX,
        ]
    )

    if not _check_cuda(python_path):
        raise RuntimeError(
            "CUDA remains unavailable after installing the cu121 wheel. "
            "Install the correct GPU build manually or set WBVB_ALLOW_CPU=1 to bypass this check."
        )


def prompt_for_dataset() -> Path:
    """Prompt the user for the offline dataset directory."""
    default_path = BASE_DIR / "offline_teacher_data"
    prompt = (
        "\nEnter the path to your offline teacher dataset directory "
        f"(default: {default_path}): "
    )
    user_input = input(prompt).strip()
    dataset_path = Path(user_input) if user_input else default_path
    dataset_path = dataset_path.expanduser().resolve()

    if dataset_path.is_file() and dataset_path.suffix == PARQUET_SUFFIX:
        dataset_path = dataset_path.parent

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    parquet_files = list(dataset_path.glob(f"*{PARQUET_SUFFIX}"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No {PARQUET_SUFFIX} files found in {dataset_path}. "
            "Ensure you've copied the offline teacher dataset."
        )

    print(f"[bootstrap] Using dataset directory: {dataset_path}")
    return dataset_path


def confirm_teacher_data(dataset_path: Path) -> None:
    """Verify that the offline teacher data directory is usable."""
    print("\n[bootstrap] Verifying teacher data integrity...")
    parquet_files = list_parquet_files(str(dataset_path))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path}")

    sample_file = parquet_files[0]
    print(f"[bootstrap] Found {len(parquet_files)} parquet file(s). Inspecting {sample_file}...")

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = OfflineDistillationDataset(
        sample_file,
        tokenizer,
        getattr(config, "MAX_SEQ_LENGTH", 512),
    )

    if len(dataset) == 0:
        raise ValueError(f"{sample_file} contains zero rows.")

    sample = dataset[0]
    required_keys = ["input_ids", "attention_mask", "labels"]
    missing = [k for k in required_keys if k not in sample]
    if missing:
        raise ValueError(
            f"Sample record missing required keys: {missing}. "
            "Ensure the teacher data was produced with the latest pipeline."
        )

    # Optional tensors (teacher logits/hidden/attention) are checked if present.
    tensor_shapes = {}
    for key, value in sample.items():
        if hasattr(value, "shape"):
            tensor_shapes[key] = tuple(value.shape)
        else:
            tensor_shapes[key] = f"type={type(value).__name__}"
    print("[bootstrap] Sample tensor shapes:")
    for key, shape in tensor_shapes.items():
        print(f"  - {key}: {shape}")

    print("[bootstrap] Teacher data looks good!")


def pick_run_configuration():
    """Let the user choose which trials to execute."""
    print("\nConfigure which trial(s) to run (sequential local execution).")
    mode = input(
        "  Run mode [single/full/custom/preset] (default single): "
    ).strip().lower() or "single"

    if mode == "preset":
        preset_name = input(f"  Choose preset {list(PRESETS.keys())}: ").strip().lower()
        preset = PRESETS.get(preset_name)
        if not preset:
            raise ValueError(f"Preset '{preset_name}' not found.")
        return mode, preset["distill_types"], preset["seeds"]

    if mode not in {"single", "full", "custom"}:
        print("  Unknown mode, defaulting to single.")
        mode = "single"

    if mode == "single":
        distill_types = ["black_box"]
        seeds = [42]
    elif mode == "full":
        distill_types = ["black_box", "hidden_state", "attention", "combined"]
        seeds = list(range(7))
    else:
        distill_types_raw = input(
            "  Enter distill types (comma-separated, e.g. black_box,hidden_state): "
        ).strip()
        seeds_raw = input(
            "  Enter seeds (comma-separated integers, e.g. 0,1,4): "
        ).strip()
        distill_types = [t.strip() for t in distill_types_raw.split(",") if t.strip()] or ["black_box"]
        seeds = [int(s.strip()) for s in seeds_raw.split(",") if s.strip()] or [42]

    return mode, distill_types, seeds


def _run_with_progress(cmd, total: int, env: Dict[str, str]) -> None:
    if total <= 1 or tqdm is None:
        _run(cmd, env=env)
        return

    progress = tqdm(total=total, desc="[bootstrap] Trials", unit="trial")
    process = subprocess.Popen(cmd, env=env)
    try:
        while True:
            ret = process.poll()
            if ret is not None:
                progress.n = total
                progress.refresh()
                break
            # We can't easily get per-trial completion without modifying
            # train_student.py, so treat the whole run as one long trial.
            progress.n = min(progress.n + 0.01, total - 0.01)
            progress.refresh()
            time.sleep(1)
    finally:
        progress.close()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def launch_training(python_path: Path, dataset_path: Path, mode: str, distill_types, seeds) -> None:
    """Start the local training script with the requested settings."""
    env = os.environ.copy()
    env["WBVB_OFFLINE_DATA_PATH"] = str(dataset_path)
    d_drive = getattr(config, "_D_DRIVE_AVAILABLE", False)
    if d_drive and not env.get("HF_HOME"):
        hf_cache = Path("D:/hf_cache")
        hf_cache.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(hf_cache)
        env.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))

    cmd = [
        str(python_path),
        str(TRAIN_SCRIPT),
        "--distill-types",
        ",".join(distill_types),
        "--seeds",
        ",".join(str(s) for s in seeds),
    ]
    if mode == "single":
        cmd.append("--single-trial")

    total_trials = len(distill_types) * len(seeds) if not (mode == "single") else 1
    print("\n[bootstrap] Launching sequential training...\n")
    _run_with_progress(cmd, total_trials, env)


def main():
    python_path = ensure_virtualenv()
    install_requirements(python_path)
    ensure_cuda_ready(python_path)
    dataset_path = prompt_for_dataset()
    confirm_teacher_data(dataset_path)
    mode, distill_types, seeds = pick_run_configuration()
    launch_training(python_path, dataset_path, mode, distill_types, seeds)


if __name__ == "__main__":
    main()


