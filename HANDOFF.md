## Whitebox vs Blackbox KD – Handoff Notes

### Current Status (2025-11-15)
- Full GPU rerun completed for `attention` and `combined` trials with batch size 4, but **AMP is disabled**, so all GPU runs produced `train_loss=NaN` and `val_acc=0`. Only the earlier CPU black_box seeds (0–3) have valid metrics.
- Offline teacher parquet data now includes a `task_name` column (`sst2`, `mmlu`, `gsm8k`). Datasets load this metadata and the training loop reports per-task accuracies.
- `results_summary.csv` lives at `D:\wbvb_results\results_summary.csv`. The latest rows (from GPU runs) contain NaNs.
- `scripts\summarize_results.py` aggregates per-seed metrics into `results_summary_by_seed.csv` (also in `D:\wbvb_results`).

### Action Items for Next Agent
1. **Restore AMP with GradScaler**  
   - Set `config.USE_AUTOMATIC_MIXED_PRECISION = True`.  
   - We already cast teacher tensors to the student dtype, so the original dtype mismatch is resolved.  
   - Ensure `WHITEBOX_BATCH_SIZE` (e.g., 4) is set for `hidden_state/attention/combined` runs to avoid OOM.

2. **Force eager attention when capturing maps**  
   - After constructing `DistillationStudent`, call `self.student_model.set_attn_implementation("eager")` when `return_attention` or `output_attentions` will be `True`. This removes the runtime warning and guarantees attention tensors exist.

3. **Re-run GPU trials**  
   - Use bootstrap `custom` mode to run `black_box,hidden_state,attention,combined` with seeds `0-6`.  
   - For attention/combined, keep batch size ≤4.  
   - Verify console logs show finite losses.

4. **Verify CSV output**  
   - After the re-run, inspect `results_summary_by_seed.csv` and ensure `final_train_loss`, `final_val_loss`, and `validation_accuracy_*` columns are finite for all seeds.  
   - Remove or archive the NaN rows from earlier runs to avoid confusion.

5. **Analysis Prep**  
   - Once clean data exists, extend `scripts/summarize_results.py` or build a notebook to compute per-task averages, standard deviations, and plots comparing distillation types.

### Notes / Caveats
- Live console output still prints `val_acc=0.0000` because that line is the overall accuracy; real per-task numbers appear only in the CSV columns (`validation_accuracy_sst2`, etc.).  
- Datasets and results directories default to `D:\...`; ensure the D: drive remains mounted.  
- The HuggingFace cache warning can be ignored—the script sets `HF_HOME` automatically when running via bootstrap.

Ping me if you need more context on the data preprocessing or the TinyLlama adapter architecture. Good luck!




