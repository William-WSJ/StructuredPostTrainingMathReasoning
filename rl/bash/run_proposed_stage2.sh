#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_DIR="$ROOT_DIR/rl/code"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/rl/train_data/thinkprune_mini_20.json}"
RELAXED_HIT_PY="${RELAXED_HIT_PY:-/root/autodl-tmp/all_results/evaluate_results.py}"

MODEL_KEY="${1:-qwen3}"

case "$MODEL_KEY" in
  qwen3)
    MODEL_NAME="${QWEN3_MODEL:-/root/autodl-tmp/Qwen3-8B-L6100-innovation_stage2}"
    PROMPT_ARGS=()
    CURRENT_LIMIT="${CURRENT_LIMIT:-5000}"
    GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-5500}"
    OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/grpo_3_qwen3_L5000_stage3_innovation}"
    DEBUG_PATH="${DEBUG_PATH:-$CODE_DIR/grpo_3_qwen_rewards_stage3_innovation.jsonl}"
    ;;
  glmz1)
    MODEL_NAME="${GLMZ1_MODEL:-/root/autodl-tmp/GLM-Z1-9B-L5400-innovation_stage2}"
    PROMPT_ARGS=(--prompt_style glmz1)
    CURRENT_LIMIT="${CURRENT_LIMIT:-4400}"
    GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-4900}"
    OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/grpo_3_glmz1_L4400_stage3_innovation}"
    DEBUG_PATH="${DEBUG_PATH:-$CODE_DIR/grpo_3_glmz1_rewards_stage3_innovation.jsonl}"
    ;;
  r1)
    MODEL_NAME="${R1_MODEL:-/root/autodl-tmp/R1-Distill-Qwen-7B-L5100-innovation_stage2}"
    PROMPT_ARGS=(--prompt_style r1distill_qwen --append_r1_eos)
    CURRENT_LIMIT="${CURRENT_LIMIT:-4100}"
    GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-4600}"
    OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/grpo_3_r1distill_L4100_stage3_innovation}"
    DEBUG_PATH="${DEBUG_PATH:-$CODE_DIR/grpo_3_r1distill_rewards_stage3_innovation.jsonl}"
    ;;
  *)
    echo "Usage: bash rl/bash/run_proposed_stage2.sh {qwen3|glmz1|r1}"
    exit 1
    ;;
esac

cd "$CODE_DIR"
python3 -u grpo_3_stage2.py \
  --use_prefill \
  "${PROMPT_ARGS[@]}" \
  --model_name "$MODEL_NAME" \
  --data_path "$DATA_PATH" \
  --relaxed_hit_py "$RELAXED_HIT_PY" \
  --current_limit "$CURRENT_LIMIT" \
  --gen_max_new_tokens "$GEN_MAX_NEW_TOKENS" \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --temperature "${TEMPERATURE:-0.6}" \
  --top_p "${TOP_P:-0.9}" \
  --repetition_penalty "${REPETITION_PENALTY:-1.2}" \
  --num_iterations "${NUM_ITERATIONS:-1}" \
  --max_steps "${MAX_STEPS:-40}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --save_steps "${SAVE_STEPS:-4}" \
  --early_stop \
  --output_dir "$OUTPUT_DIR" \
  --debug_group_rewards_path "$DEBUG_PATH" \
  --debug_group_rewards_max_calls "${DEBUG_GROUP_REWARDS_MAX_CALLS:-0}"
