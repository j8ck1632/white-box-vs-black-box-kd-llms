"""
One-stop entry point for sharing the experiment with partner machines.

Usage:
    python shareable_bundle/experiment_hub.py <command> [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
import generate_teacher_manifest
import multi_machine_runner
import tests_dashboard


PLAN_PATH = Path("MULTI_MACHINE_PLAN.md")


def run_manifest(sub_args: list[str]):
    parser = argparse.ArgumentParser(
        prog="experiment_hub.py manifest",
        description="Export teacher metadata manifest JSON.",
    )
    parser.add_argument(
        "--offline-path",
        type=str,
        default=config.OFFLINE_DATA_PATH,
        help=f"Location of offline Parquet data (default: {config.OFFLINE_DATA_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="teacher_manifest.json",
        help="Destination JSON file (default: teacher_manifest.json).",
    )
    args = parser.parse_args(sub_args)
    manifest_args = [
        "--offline-path",
        args.offline_path,
        "--output",
        args.output,
    ]
    generate_teacher_manifest.main(manifest_args)


def run_training(sub_args: list[str]):
    if not sub_args:
        print("Please pass arguments after 'run'. Example:")
        print("  python shareable_bundle/experiment_hub.py run --distill-type black_box --seeds 0 1 2")
        return
    multi_machine_runner.main(sub_args)


def run_dashboard():
    print("Launching Tkinter dashboard... close the window to return to the terminal.")
    tests_dashboard.main()


def run_plan(sub_args: list[str]):
    parser = argparse.ArgumentParser(
        prog="experiment_hub.py plan",
        description="Print the multi-machine deployment plan.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(PLAN_PATH),
        help="Custom plan path if you maintain forks (default: MULTI_MACHINE_PLAN.md).",
    )
    args = parser.parse_args(sub_args)
    plan_path = Path(args.path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    print(plan_path.read_text())


def build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment_hub.py",
        description="Shareable entry point for the white-box vs black-box experiments.",
    )
    parser.add_argument(
        "command",
        choices=["manifest", "run", "dashboard", "plan"],
        help="Which helper to invoke.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected subcommand.",
    )
    return parser


def main(cli_args: list[str] | None = None):
    if cli_args is None:
        cli_args = sys.argv[1:]
    if not cli_args:
        build_top_parser().print_help()
        return
    command = cli_args[0]
    sub_args = cli_args[1:]
    if command == "manifest":
        run_manifest(sub_args)
    elif command == "run":
        run_training(sub_args)
    elif command == "dashboard":
        run_dashboard()
    elif command == "plan":
        run_plan(sub_args)
    else:
        build_top_parser().print_help()


if __name__ == "__main__":
    main()


