# Shareable Experiment Folder

Copy this directory to each collaborator’s machine along with the
`offline_teacher_data/` Parquet snapshot. Everything here is driven through a
single entry point:

```bash
python shareable_bundle/experiment_hub.py <subcommand> [options]
```

## Available Subcommands

- `manifest` – run `generate_teacher_manifest.py` (exports teacher metadata +
  compression knobs into JSON). Example:
  ```
  python shareable_bundle/experiment_hub.py manifest --output shared/teacher_manifest.json
  ```
- `run` – wrap `multi_machine_runner.py` so a box can take ownership of one
  distillation type. All arguments after `run` are passed through, e.g.:
  ```
  python shareable_bundle/experiment_hub.py run --distill-type black_box --seeds 0 1 2
  ```
- `dashboard` – open the Tkinter GUI (`tests_dashboard.py`) that explains which
  machine runs which test and provides copy/paste commands.
- `plan` – print the text of `MULTI_MACHINE_PLAN.md` so remote teammates can
  read the schedule directly in their terminal.

## Typical Flow

1. On the source machine, build the shareable zip (already contains the latest
   `teacher_manifest.json`):
   ```
   python create_shareable_package.py --output shareable_experiment.zip
   ```
2. Ship `shareable_experiment.zip` plus the `offline_teacher_data/` directory.
3. Each collaborator unzips, reads the plan, and launches their assigned run via
   the `run` subcommand (no Ray required).


