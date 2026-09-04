#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen3-8B-prune_4000}"
TRAINING_PROMPT_SOURCE="${TRAINING_PROMPT_SOURCE:-/root/autodl-tmp/PruneRL/datasets/thinkprune_700_vanguard_relaxed.json}"
PREFILL_PREFIX="${PREFILL_PREFIX:-**Theorems**：}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
DO_SAMPLE="${DO_SAMPLE:-0}"
BATCH_SIZE="${BATCH_SIZE:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
PYTHON_EXE="${PYTHON_EXE:-python3}"

EXTRA_ARGS=("${@:1}")
if [[ "$DO_SAMPLE" == "1" ]]; then
  EXTRA_ARGS=("--do_sample" "${EXTRA_ARGS[@]}")
fi

echo "[Eval] MODEL_PATH=$MODEL_PATH"
echo "[Eval] SCRIPT=eval_mathodyssey_qwen3_merged.py"

"$PYTHON_EXE" /root/autodl-tmp/PruneRL/qwen3_2/eval_mathodyssey_qwen3_merged.py   --model_path "$MODEL_PATH"   --training_prompt_source "$TRAINING_PROMPT_SOURCE"   --prefill_prefix "$PREFILL_PREFIX"   --max_new_tokens "$MAX_NEW_TOKENS"   --batch_size "$BATCH_SIZE"   --temperature "$TEMPERATURE"   --top_p "$TOP_P"   --repetition_penalty "$REPETITION_PENALTY"   --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"   --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"   --dtype "$DTYPE"   "${EXTRA_ARGS[@]}"
