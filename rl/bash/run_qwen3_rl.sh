#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lf}"
if [[ -n "$CONDA_ENV_NAME" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
  else
    echo "conda not found. Install conda or run with CONDA_ENV_NAME= to skip activation."
    exit 1
  fi
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_DIR="$ROOT_DIR/rl/code"
MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/Qwen3-8B-L6200-review_stage2}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/rl/train_data/thinkprune_mini_20.json}"
RELAXED_HIT_PY="${RELAXED_HIT_PY:-/root/autodl-tmp/all_results/evaluate_results.py}"
CURRENT_LIMIT="${CURRENT_LIMIT:-4800}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-5300}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/grpo_qwen3_L4800_stage3_review}"
DEBUG_PATH="${DEBUG_PATH:-$CODE_DIR/grpo_qwen_rewards_stage3.jsonl}"

cd "$CODE_DIR"
python3 -u grpo.py \
  --use_prefill \
  --model_name "$MODEL_NAME" \
  --data_path "$DATA_PATH" \
  --relaxed_hit_py "$RELAXED_HIT_PY" \
  --current_limit "$CURRENT_LIMIT" \
  --gen_max_new_tokens "$GEN_MAX_NEW_TOKENS" \
  --num_generations 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --temperature 0.6 \
  --top_p 0.9 \
  --repetition_penalty 1.2 \
  --num_iterations 1 \
  --max_steps 40 \
  --logging_steps 1 \
  --save_steps 4 \
  --output_dir "$OUTPUT_DIR" \
  --debug_group_rewards_path "$DEBUG_PATH" \
  --debug_group_rewards_max_calls 0
