# StructuredPostTrainingMathReasoning

Official implementation for **Structured Post-training for Token-Efficient Mathematical Reasoning in Small-Scale Language Models**.

## What Is Here

This repository is organized around the paper's two stages:

- **SFT**: structured supervised fine-tuning with 7-category reasoning data
- **RL**: GRPO-based length-budgeted reinforcement learning

The actual model code and data are already placed in this repository under `sft/`, `rl/`, and `datasets/`.

## Main Files

### SFT

- `sft/train_data/train_split.json`
- `sft/train_data/train_split_7cat_reasoning.json`
- `sft/bash/run_qwen25_sft.sh`
- `sft/bash/run_qwen3_sft.sh`

### RL

- `rl/train_data/thinkprune_mini_20.json`
- `rl/code/grpo.py`
- `rl/code/grpo_3_stage1.py`
- `rl/code/grpo_3_stage2.py`
- `rl/bash/run_qwen3_rl.sh`

### Evaluation Sets

- `datasets/aime2025-I.jsonl`
- `datasets/aime2025-II.jsonl`
- `datasets/math-500.jsonl`
- `datasets/TAL-SCQ5K-EN_test.jsonl`
- `datasets/mathodyssey-en_test_final.json`
- `datasets/Omni_Math_test_final.json`
- `datasets/OE_TO_maths_en_COMP_test_final.json`


## Environment

The scripts assume the project dependencies are installed in the conda environment `lf`.

Activate it manually if you run commands yourself:

```bash
conda activate lf
```

If you need to recreate the environment, install the Python packages with:

```bash
pip install -r requirements.txt
```

The provided bash scripts also activate this environment automatically. To use another environment, run for example:

```bash
CONDA_ENV_NAME=my_env bash sft/bash/run_qwen3_sft.sh
```

To skip conda activation entirely:

```bash
CONDA_ENV_NAME= bash rl/bash/run_qwen3_rl.sh
```

## SFT Scripts

### 1) Qwen2.5-Math-7B

Run:

```bash
bash sft/bash/run_qwen25_sft.sh
```

What it does:

- uses `sft/yaml/qwen25_math_7b_lora_a800_7cat.yaml`
- copies the 7cat SFT data into the LLaMAFactory data folder
- runs `llamafactory-cli train <yaml>`

### 2) Qwen3-8B

Run:

```bash
bash sft/bash/run_qwen3_sft.sh
```

What it does:

- uses `sft/yaml/qwen3_8b_lora_a800_7cat.yaml`
- copies the same 7cat SFT data into the LLaMAFactory data folder
- runs `llamafactory-cli train <yaml>`

### SFT Environment Variables

- `LLAMAFACTORY_DIR`: LLaMAFactory root, default `/root/autodl-tmp/llamafactory-0.9.4`
- `LLAMAFACTORY_CMD`: training command, default `llamafactory-cli`
- `CONFIG_PATH`: override the YAML path if needed

## RL Script

### Qwen3 Example

Run:

```bash
bash rl/bash/run_qwen3_rl.sh
```

What it does:

- runs `rl/code/grpo.py`
- loads `rl/train_data/thinkprune_mini_20.json`
- uses the ThinkPrune-style 0/1 reward
- clips outputs to `CURRENT_LIMIT` before calling `relaxed_hit`

### RL Environment Variables

- `MODEL_NAME`: stage checkpoint path, default `/root/autodl-tmp/Qwen3-8B-L6200-review_stage2`
- `DATA_PATH`: RL training file, default `rl/train_data/thinkprune_mini_20.json`
- `RELAXED_HIT_PY`: path to `evaluate_results.py`, default `/root/autodl-tmp/all_results/evaluate_results.py`
- `CURRENT_LIMIT`: token budget, default `4800`
- `GEN_MAX_NEW_TOKENS`: generation cap, default `5300`
- `OUTPUT_DIR`: output folder
- `DEBUG_PATH`: reward debug log path

## How To Read The Bash Files

The bash files are intentionally simple.

- They set a few default paths.
- They call the Python trainer with the same arguments used in the paper.
- You can override any path from the command line by exporting environment variables first.

Example:

```bash
MODEL_NAME=/path/to/checkpoint CURRENT_LIMIT=5000 bash rl/bash/run_qwen3_rl.sh
```

## Data Format

The SFT JSON uses the LLaMAFactory alpaca-style format:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

The RL JSON uses the prompt/problem/answer style expected by the GRPO scripts.

## Notes

- Validation splits are used in the paper for checkpoint selection and length-budget estimation.
- The repository scripts use the reported budgets directly.
- The paper's final evaluation sets are included under `datasets/`.

## Result Files

You do **not** need to publish every intermediate result file. What should be public is the code, scripts, datasets, and enough instructions for others to rerun evaluation.

If you want the repository to be easier to inspect, you can also release example result JSON files produced by SFT, RL, and frontier-LLM runs, but those files are optional.

## Citation

```bibtex
@article{structured-post-training-math-reasoning,
  title = {Structured Post-training for Token-Efficient Mathematical Reasoning in Small-Scale Language Models},
  author = {Wang, Sijin and Liu, Suyu and Liu, Ning and Xu, Yanyan and Su, Kaile},
  year = {2026}
}
```
