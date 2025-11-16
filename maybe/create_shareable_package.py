"""
Build a distributable archive containing everything collaborators need to run
the experiments without Ray.

Usage:
    python create_shareable_package.py --output shareable_experiment.zip
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).parent.resolve()
DEFAULT_OUTPUT = ROOT / "shareable_experiment.zip"
PAYLOAD_DIR = ROOT / "shareable_payload"

FILE_DEPENDENCIES = [
    "config.py.example",
    "config.py",
    "distillation_student.py",
    "distillation_core.py",
    "multi_machine_runner.py",
    "generate_teacher_manifest.py",
    "tests_dashboard.py",
    "MULTI_MACHINE_PLAN.md",
    "README.md",
]

DIR_DEPENDENCIES = [
    "shareable_bundle",
]


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_payload():
    if PAYLOAD_DIR.exists():
        shutil.rmtree(PAYLOAD_DIR)
    PAYLOAD_DIR.mkdir()

    for rel_path in FILE_DEPENDENCIES:
        src = ROOT / rel_path
        if not src.exists():
            print(f"[WARN] Missing expected file: {rel_path} (skipping)")
            continue
        dst = PAYLOAD_DIR / rel_path
        safe_copy(src, dst)

    for rel_dir in DIR_DEPENDENCIES:
        src_dir = ROOT / rel_dir
        if not src_dir.exists():
            print(f"[WARN] Missing expected directory: {rel_dir} (skipping)")
            continue
        dst_dir = PAYLOAD_DIR / rel_dir
        shutil.copytree(src_dir, dst_dir)


def make_archive(output_path: Path):
    base_name = output_path.with_suffix("")
    shutil.make_archive(str(base_name), "zip", PAYLOAD_DIR)
    final_zip = base_name.with_suffix(".zip")
    if final_zip != output_path:
        shutil.move(final_zip, output_path)
    print(f"Created archive at {output_path}")
    print("Reminder: include offline_teacher_data/ separately (too large for auto-pack).")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a shareable experiment archive.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Destination zip path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()
    prepare_payload()
    make_archive(output_path)
    shutil.rmtree(PAYLOAD_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()


