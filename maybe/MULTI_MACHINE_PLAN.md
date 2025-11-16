# Three-Machine Plan (No Ray Required)

This runbook explains how to split the experiment across three independent
machines while sharing only lightweight artifacts.

## 1. Prepare Shared Artifacts

1. **Teacher Manifest**  
   ```bash
   python generate_teacher_manifest.py --output shared/teacher_manifest.json
   ```  
   Send the JSON plus the Parquet files in `offline_teacher_data/` to every
   machine. The manifest captures model IDs, compression knobs, and loss weights.

2. **Student Code + Runner**  
   Copy the repository (or at minimum `distillation_student.py`,
   `distillation_core.py`, `multi_machine_runner.py`, and `config.py`) to each
   box. No Ray install is required.

3. **Validation Sanity Checks** (optional but recommended)  
   ```bash
   python -m pytest -s tests/test_offline_teacher_data.py
   python load_offline_data.py --verify 20
   ```

## 2. Assign Workloads

| Machine | Role | CLI |
|---------|------|-----|
| A | Black-box baseline | `python multi_machine_runner.py --distill-type black_box --seeds 0 1 2` |
| B | White-box (Hidden) | `python multi_machine_runner.py --distill-type hidden_state --seeds 0 1 2` |
| C | White-box (Attention + Combined) | `python multi_machine_runner.py --distill-type attention --seeds 0 1 2` and `python multi_machine_runner.py --distill-type combined --seeds 0 1` |

Feel free to rebalance the `--seeds` list depending on GPU availability. Each
command runs sequentially on that host and writes metrics to
`results/multi_machine/<distill_type>/seed_X_metrics.json`.

## 3. Tracking Progress

- Launch the Tkinter dashboard for quick reference:  
  ```bash
  python tests_dashboard.py
  ```
- Completed JSON metrics can be merged by simply concatenating the files into a
  pandas DataFrame using `results/multi_machine/`.

## 4. Tips

- Use `--max-batches 20` on a dry run before committing multiple seeds.
- If a machine only has CPU, expect training to slow down but still work.
- Keep the manifest and config file in sync; rerun the manifest script whenever
  offline data snapshots change.


