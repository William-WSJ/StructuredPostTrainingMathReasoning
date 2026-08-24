# Evaluation

This folder contains the scripts used to score outputs and generate the paper figures.

## Files

- `evaluate_results.py`: relaxed answer matching used by the RL reward and by post-training evaluation
- `sft_metric.py`: computes accuracy, token cost, hit-wall rate, and sequence-decoding compliance for a list of output files
- `plot_results_clean.py`: aggregates result JSON files and generates the clean summary figures
- `plot_results_figures.py`: generates the paper-style figures from the same aggregated results
- `plot_raw_output_diagnostics.py`: analyzes raw generation outputs by token bins

## 1. Relaxed Scoring

Use `evaluate_results.py` to score one or more prediction files.

Example:

```bash
python3 evaluation/evaluate_results.py \
  --files path/to/run1.json path/to/run2.json \
  --save_json outputs/relaxed_scores.json
```

Each input file should be a JSONL or JSON file containing at least:

- `ground_truth`
- `model_output`
- `problem` when the task needs option matching

The script returns strict accuracy, relaxed accuracy, and rescue reasons.

## 2. SFT / RL Metrics

Use `sft_metric.py` to compute the paper-style metrics for generated outputs.

Example:

```bash
python3 evaluation/sft_metric.py \
  --files outputs/qwen3_sft.json outputs/qwen3_rl.json \
  --save_json outputs/summary.json
```

Important metrics:

- `Acc`
- `T_Avg`
- `TPC`
- `HW`
- `SD`

The script inspects the output format and checks whether the response follows the required sequence of reasoning markers.

## 3. Clean Summary Plots

Use `plot_results_clean.py` when you already have a metrics directory with result JSON files.

Example:

```bash
python3 evaluation/plot_results_clean.py \
  --root /path/to/metrics \
  --outdir /path/to/figures_results
```

Expected layout:

- `/path/to/metrics/*.json` for post-training result files
- `/path/to/metrics/LLM/*.json` for frontier comparison files

The script writes CSV summaries and trade-off plots into the output directory.

## 4. Raw Output Diagnostics

Use `plot_raw_output_diagnostics.py` for token-bin analysis of raw generations.

Example:

```bash
python3 evaluation/plot_raw_output_diagnostics.py \
  --root /path/to/raw_outputs \
  --outdir /path/to/figures_raw_outputs
```

## Do I Need To Publish Result Files?

No, not strictly.

What should be public is:

- the evaluation scripts
- the exact metric definitions
- the dataset files used for evaluation
- the model checkpoints or enough instructions to reproduce them

Publishing the final result JSON files is optional. They are useful as reference outputs, but the repository can still be complete without them if the scripts and inputs are available.

If you want the repository to feel fully reproducible to other readers, it helps to include at least one example result file for SFT, RL, and frontier comparison.
