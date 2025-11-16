## Whitebox vs Blackbox KD – Handoff Notes

### Current Status (2025-11-16)
- Mixed-task offline teacher data (SST-2 + MMLU + GSM8K) lives in `/content/drive/MyDrive/wbvb/output/offline_teacher_data.parquet`. `load_offline_data.py --stats --inspect 0` is a quick sanity check.
- `train_student.py` supports **DeepSpeed ZeRO‑3 + CPU offload** (`--use-deepspeed --deepspeed-config ds_config_zero3.json`). Config now defaults to `BATCH_SIZE=80`, `WHITEBOX_BATCH_SIZE=20`, `DATALOADER_NUM_WORKERS=6` for Colab Pro+ (80 GB GPU + 150 GB host RAM).
- `ds_config_zero3.json` uses `DeepSpeedCPUAdam` so ZeRO-offload works out of the box. `requirements.txt` includes DeepSpeed; install `mpi4py` too (`pip install mpi4py`) or DeepSpeed will complain about missing MPI bindings.

### Recommended Workflow (Colab Pro+ A100)
1. **Mount Drive & clone repo**
   ```bash
   from google.colab import drive
   drive.mount('/content/drive')

   %cd /content
   !git clone https://github.com/j8ck1632/white-box-vs-black-box-kd-llms.git
   %cd "white-box-vs-black-box-kd-llms/Whitebox vs Blackbox"
   ```
2. **Install deps**
   ```bash
   !pip install -r requirements.txt
   !pip install deepspeed mpi4py
   ```
3. **Point to Drive paths**
   ```python
   import os, pathlib
   os.environ["WBVB_OFFLINE_DATA_PATH"] = "/content/drive/MyDrive/wbvb/output"
   os.environ["WBVB_OUTPUT_PATH"] = "/content/drive/MyDrive/wbvb/results"
   pathlib.Path(os.environ["WBVB_OUTPUT_PATH"]).mkdir(parents=True, exist_ok=True)
   ```
4. **(Optional) regenerate teacher data**
   ```bash
   %cd "/content/white-box-vs-black-box-kd-llms/Whitebox vs Blackbox"
   !python offline_teacher_data.py
   ```
5. **Launch training**
   ```bash
   %cd "/content/white-box-vs-black-box-kd-llms/Whitebox vs Blackbox"
   !PYTHONUNBUFFERED=1 python -u train_student.py \
       --use-deepspeed \
       --deepspeed-config ds_config_zero3.json \
       --distill-types black_box,hidden_state,attention,combined \
       --seeds 0,1,2,3,4,5,6
   ```
6. **Summarize results**
   ```bash
   %cd "/content/white-box-vs-black-box-kd-llms"
   !python scripts/summarize_results.py --csv /content/drive/MyDrive/wbvb/results/results_summary.csv
   ```

### Local CPU/GPU runs (no DeepSpeed)
- `config.py` still works with modest defaults (`BATCH_SIZE=8`, `WHITEBOX_BATCH_SIZE=2`, `DATALOADER_NUM_WORKERS=0`).
- Run `python train_student.py --distill-types ... --seeds ...`; AMP + GradScaler enable automatically on CUDA.

### Notes / Caveats
- DeepSpeed will fail if `mpi4py` isn’t installed—install it once per runtime.
- The DS config’s `train_batch_size` is overwritten inside `run_trial` to match whatever effective batch is being used—no need to edit it manually.
- `results_summary.csv` grows large quickly; archive or filter before presenting stats.
- Console `val_acc` is aggregate; per-task metrics live in `validation_accuracy_<task>` columns.

Ping me if you need more context; otherwise everything’s wired for the next agent. Good luck!