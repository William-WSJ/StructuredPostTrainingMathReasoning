#!/usr/bin/env bash
set -euo pipefail

# 默认使用你合并后的最终模型
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen3-8B-L5000-innovation_stage3}"

# 与 sweep 脚本保持一致的生成参数
TRAINING_PROMPT_SOURCE="${TRAINING_PROMPT_SOURCE:-/root/autodl-tmp/PruneRL/datasets/thinkprune_mini_20.json}"
PREFILL_PREFIX="${PREFILL_PREFIX:-**Theorems**：}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
DO_SAMPLE="${DO_SAMPLE:-0}"
BATCH_SIZE="${BATCH_SIZE:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
PYTHON_EXE="${PYTHON_EXE:-python3}"

# 默认不跑 math500（按你之前需求），需要时设 INCLUDE_MATH500=1
INCLUDE_MATH500="${INCLUDE_MATH500:-0}"

CODE_DIR="/root/autodl-tmp/PruneRL/qwen3_3"

SCRIPTS=(
  "eval_aime2025_i_qwen3_merged.sh"
  "eval_aime2025_ii_qwen3_merged.sh"
  "eval_mathodyssey_qwen3_merged.sh"
  "eval_math500_qwen3_merged.sh"
  "eval_oe_to_maths_qwen3_merged.sh"
  "eval_omni_qwen3_merged.sh"
  "eval_tal_qwen3_merged.sh"
)

if [[ "$INCLUDE_MATH500" == "1" ]]; then
  SCRIPTS+=("eval_math500_qwen3_merged.sh")
fi

echo "[Run-All] MODEL_PATH=$MODEL_PATH"
echo "[Run-All] INCLUDE_MATH500=$INCLUDE_MATH500"
echo "[Run-All] NUM_TASKS=${#SCRIPTS[@]}"

for script in "${SCRIPTS[@]}"; do
  echo "============================================================"
  echo "[Run-All] Running $script"

  MODEL_PATH="$MODEL_PATH" \
  TRAINING_PROMPT_SOURCE="$TRAINING_PROMPT_SOURCE" \
  PREFILL_PREFIX="$PREFILL_PREFIX" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  TEMPERATURE="$TEMPERATURE" \
  TOP_P="$TOP_P" \
  REPETITION_PENALTY="$REPETITION_PENALTY" \
  DO_SAMPLE="$DO_SAMPLE" \
  BATCH_SIZE="$BATCH_SIZE" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
  DTYPE="$DTYPE" \
  PYTHON_EXE="$PYTHON_EXE" \
  bash "$CODE_DIR/$script"
done

echo "============================================================"
echo "[Run-All] Done. Outputs are under /root/autodl-tmp/PruneRL/qwen3_3"
