#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-/root/autodl-tmp/llamafactory-0.9.4}"
LLAMAFACTORY_CMD="${LLAMAFACTORY_CMD:-llamafactory-cli}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash sft/bash/run_sft.sh <llamafactory_yaml>"
  echo "Example: bash sft/bash/run_sft.sh sft/yaml/qwen3_8b_lora_a800_7cat.yaml"
  exit 1
fi

CONFIG_PATH="$1"
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$ROOT_DIR/$CONFIG_PATH"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  echo "Put the LLaMAFactory YAML config under sft/yaml/ or pass an absolute path."
  exit 1
fi

if [[ ! -d "$LLAMAFACTORY_DIR" ]]; then
  echo "LLaMAFactory directory not found: $LLAMAFACTORY_DIR"
  echo "Set LLAMAFACTORY_DIR=/path/to/llamafactory before running."
  exit 1
fi

mkdir -p "$LLAMAFACTORY_DIR/data"
cp "$ROOT_DIR/sft/train_data/train_split.json" "$LLAMAFACTORY_DIR/data/train_split.json"
cp "$ROOT_DIR/sft/train_data/train_split_7cat_reasoning.json" "$LLAMAFACTORY_DIR/data/train_split_7cat_reasoning.json"

cd "$LLAMAFACTORY_DIR"
"$LLAMAFACTORY_CMD" train "$CONFIG_PATH"
