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
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-/root/autodl-tmp/llamafactory-0.9.4}"
LLAMAFACTORY_CMD="${LLAMAFACTORY_CMD:-llamafactory-cli}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/sft/yaml/qwen3_8b_lora_a800_7cat.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

mkdir -p "$LLAMAFACTORY_DIR/data"
cp "$ROOT_DIR/sft/train_data/train_split.json" "$LLAMAFACTORY_DIR/data/train_split.json"
cp "$ROOT_DIR/sft/train_data/train_split_7cat_reasoning.json" "$LLAMAFACTORY_DIR/data/train_split_7cat_reasoning.json"

cd "$LLAMAFACTORY_DIR"
"$LLAMAFACTORY_CMD" train "$CONFIG_PATH"
