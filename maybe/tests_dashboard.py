"""
Tkinter dashboard that explains the available experiments and how to launch them
on separate machines (no Ray required).
"""

import tkinter as tk
from tkinter import ttk

import config

TESTS = [
    {
        "name": "Black-Box",
        "distill_type": "black_box",
        "description": "Baseline KD using logits only.",
        "recommended_machine": "Machine A (8 GB GPU ok)",
    },
    {
        "name": "White-Box Hidden",
        "distill_type": "hidden_state",
        "description": "Adds hidden-state alignment loss.",
        "recommended_machine": "Machine B (needs more VRAM for projector)",
    },
    {
        "name": "White-Box Attention",
        "distill_type": "attention",
        "description": "Aligns final-layer attention maps.",
        "recommended_machine": "Machine C (attention tensors are heavy).",
    },
    {
        "name": "White-Box Combined",
        "distill_type": "combined",
        "description": "Runs both hidden + attention losses.",
        "recommended_machine": "Whichever box has most VRAM.",
    },
]


def format_instructions(test: dict) -> str:
    command = (
        f"python multi_machine_runner.py --distill-type {test['distill_type']} --seeds 0 1 2"
    )
    return (
        f"{test['name']} ({test['distill_type']})\n"
        f"{test['description']}\n\n"
        f"Suggested host: {test['recommended_machine']}\n"
        f"Offline data: {config.OFFLINE_DATA_PATH}\n"
        f"Results saved under: {config.OUTPUT_PATH}/multi_machine/{test['distill_type']}\n\n"
        f"Launch command:\n{command}\n"
        "\nTip: Adjust the --seeds list depending on how many runs that "
        "machine should own."
    )


def show_instructions(text_widget: tk.Text, test: dict):
    text_widget.configure(state=tk.NORMAL)
    text_widget.delete("1.0", tk.END)
    text_widget.insert(tk.END, format_instructions(test))
    text_widget.configure(state=tk.DISABLED)


def build_gui():
    root = tk.Tk()
    root.title("White-Box vs Black-Box Experiments")
    root.geometry("760x420")

    header = ttk.Label(
        root,
        text="Pick a test to see which machine should run it and the CLI command.",
        font=("Segoe UI", 11, "bold"),
    )
    header.pack(pady=10)

    button_frame = ttk.Frame(root)
    button_frame.pack(pady=5)

    text_widget = tk.Text(root, height=15, wrap=tk.WORD, state=tk.DISABLED)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    for test in TESTS:
        btn = ttk.Button(
            button_frame,
            text=test["name"],
            command=lambda t=test: show_instructions(text_widget, t),
            width=18,
        )
        btn.pack(side=tk.LEFT, padx=5, pady=5)

    footer = ttk.Label(
        root,
        text="Need teacher metadata? Run: python generate_teacher_manifest.py",
    )
    footer.pack(pady=(0, 10))

    root.mainloop()


def main():
    build_gui()


if __name__ == "__main__":
    main()


