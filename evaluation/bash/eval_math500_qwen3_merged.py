#!/usr/bin/env python3
from eval_single_dataset_qwen3_merged import main

if __name__ == "__main__":
    main(
        default_data_path="/root/autodl-tmp/math-500.jsonl",
        default_output_path="/root/autodl-tmp/PruneRL/qwen3_2/math500_results_4000L_merged.jsonl",
    )
