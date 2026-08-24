import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SFT_MODEL_ORDER = ["DeepSeek", "Qwen2.5", "Qwen3", "GLM", "R1"]
RL_MODEL_ORDER = ["Qwen3", "GLM", "R1"]
FRONTIER_MODELS = ["Qwen3.7-Max", "GLM-5.1", "DeepSeek-V4-Flash"]
GROUP_ORDER = ["ALL", "CORE", "BASE_MATH"]
MAX_TOKENS = {"DeepSeek": 4096, "Qwen2.5": 4096, "Qwen3": 8192, "GLM": 8192, "R1": 8192}
MODEL_COLORS = {"DeepSeek": "#4C78A8", "Qwen2.5": "#59A14F", "Qwen3": "#F28E2B", "GLM": "#E15759", "R1": "#B07AA1"}
FRONTIER_COLORS = {"Qwen3.7-Max": "#F28E2B", "GLM-5.1": "#E15759", "DeepSeek-V4-Flash": "#B07AA1"}
VARIANT_MARKERS = {"Base": "o", "Review": "s", "Innovation": "^"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"\bInfinity\b", "null", text)
    return json.loads(text)


def parse_experiment_filename(path):
    name = os.path.basename(path).replace(".json", "").lower()
    parts = name.split("-")
    if len(parts) < 3:
        return None
    model = {"deepseek": "DeepSeek", "qwen2_5": "Qwen2.5", "qwen3": "Qwen3", "glmz1": "GLM", "r1": "R1"}.get(parts[0])
    phase = {"sft": "SFT", "rl": "RL"}.get(parts[1])
    variant = {"base": "Base", "base_1": "Base", "review": "Review", "review_1": "Review", "innovation": "Innovation"}.get(parts[2])
    if not model or not phase or not variant:
        return None
    return model, phase, variant


def normalize_dataset(record):
    dataset = str(record.get("dataset", "")).upper().strip()
    file_path = os.path.basename(str(record.get("file_path", ""))).lower()
    if dataset == "UNKNOWN" and ("math-500" in file_path or "math500" in file_path):
        return "MATH500"
    if "AIME" in dataset:
        return "AIME"
    if dataset in {"MATH-500", "MATH_500"}:
        return "MATH500"
    if "MATHODYSSEY" in dataset or "MATH-ODYSSEY" in dataset:
        return "MATHODYSSEY"
    return dataset


def groups_for_record(record):
    ds = normalize_dataset(record)
    groups = ["ALL"]
    if ds in {"MATH500", "TAL"}:
        groups.append("BASE_MATH")
    if ds in {"AIME", "OMNI", "OE_TO_MATHS", "MATHODYSSEY"}:
        groups.append("CORE")
    return groups


def empty_stats():
    return {"count": 0, "correct": 0, "total_tokens": 0.0, "hw": 0.0, "sd": 0.0}


def add_record(stats, record):
    count = int(record.get("count", 0) or 0)
    correct = int(record.get("correct_count", 0) or 0)
    avg = float(record.get("total_stats", {}).get("avg", 0.0) or 0.0)
    stats["count"] += count
    stats["correct"] += correct
    stats["total_tokens"] += count * avg
    stats["hw"] += count * float(record.get("hit_wall_pct", 0.0) or 0.0)
    stats["sd"] += count * float(record.get("seq_dec_pct", 0.0) or 0.0)


def summarize(stats, max_tokens=8192):
    n = stats["count"]
    c = stats["correct"]
    if n == 0:
        return None
    return {
        "count": n,
        "acc": c / n,
        "t_avg": stats["total_tokens"] / n,
        "tpc": stats["total_tokens"] / c if c else max_tokens,
        "hw": stats["hw"] / n,
        "sd": stats["sd"] / n,
    }


def load_post_training(root):
    buckets = defaultdict(empty_stats)
    for path in glob.glob(os.path.join(root, "*.json")):
        meta = parse_experiment_filename(path)
        if meta is None:
            continue
        model, phase, variant = meta
        for record in load_json(path):
            for group in groups_for_record(record):
                add_record(buckets[(model, phase, variant, group)], record)
    rows = []
    for (model, phase, variant, group), stats in buckets.items():
        row = summarize(stats, MAX_TOKENS.get(model, 8192))
        if row:
            row.update({"model": model, "phase": phase, "variant": variant, "group": group})
            rows.append(row)
    return pd.DataFrame(rows)


def load_frontier(root):
    names = {"deepseek-v4-flash": "DeepSeek-V4-Flash", "glm-5_1": "GLM-5.1", "qwen3_7-max": "Qwen3.7-Max"}
    buckets = defaultdict(empty_stats)
    for path in glob.glob(os.path.join(root, "*.json")):
        model = names.get(os.path.basename(path).replace(".json", ""), os.path.basename(path).replace(".json", ""))
        for record in load_json(path):
            for group in groups_for_record(record):
                add_record(buckets[(model, group)], record)
    rows = []
    for (model, group), stats in buckets.items():
        row = summarize(stats)
        if row:
            row.update({"model": model, "group": group})
            rows.append(row)
    return pd.DataFrame(rows)


def setup():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(outdir, name):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + ".png")
    pdf = os.path.join(outdir, name + ".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return [png, pdf]


def variant_label(phase, variant):
    if phase == "SFT":
        return {"Base": "Base", "Review": "LTR", "Innovation": "Proposed"}[variant]
    return {"Base": "SFT-Init", "Review": "ThinkPrune", "Innovation": "Proposed"}[variant]


def plot_stage_tradeoff(df, outdir, phase, group="CORE"):
    models = SFT_MODEL_ORDER if phase == "SFT" else RL_MODEL_ORDER
    variants = ["Base", "Review", "Innovation"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    sub = df[(df.phase == phase) & (df.group == group)].copy()
    for model in models:
        msub = sub[sub.model == model]
        if msub.empty:
            continue
        ordered = msub.set_index("variant").reindex(variants).dropna()
        if len(ordered) > 1:
            ax.plot(ordered["tpc"], ordered["acc"] * 100, color=MODEL_COLORS[model], alpha=0.42, linewidth=1.5)
        for variant in variants:
            row = msub[msub.variant == variant]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.scatter(row.tpc, row.acc * 100, marker=VARIANT_MARKERS[variant], s=82,
                       color=MODEL_COLORS[model], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("TPC: total tokens per correct answer")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    model_handles = [Line2D([0], [0], marker='o', linestyle='', color=MODEL_COLORS[m], label=m, markersize=7) for m in models]
    variant_handles = [Line2D([0], [0], marker=VARIANT_MARKERS[v], linestyle='', color='#444444', label=variant_label(phase, v), markersize=7) for v in variants]
    leg1 = ax.legend(handles=model_handles, title="Model", frameon=False, loc="upper left",
                     bbox_to_anchor=(1.02, 1.00), borderaxespad=0.0)
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, title="Variant", frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 0.52), borderaxespad=0.0)
    fig.subplots_adjust(right=0.76)
    return savefig(outdir, f"fig_{phase.lower()}_core_tradeoff_clean")


def plot_frontier_tradeoff(df, frontier_df, outdir, group="CORE"):
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    small = df[(df.phase == "RL") & (df.group == group) & (df.variant == "Innovation") & (df.model.isin(RL_MODEL_ORDER))]
    for _, row in small.iterrows():
        ax.scatter(row.tpc, row.acc * 100, marker='o', s=80 + row.hw * 600,
                   color=MODEL_COLORS[row.model], edgecolor='black', linewidth=0.8)
    front = frontier_df[frontier_df.group == group]
    for _, row in front.iterrows():
        ax.scatter(row.tpc, row.acc * 100, marker='s', s=80 + row.hw * 600,
                   color=FRONTIER_COLORS.get(row.model, '#777777'), edgecolor='#555555', linewidth=0.8)
    ax.set_xlabel("TPC: total tokens per correct answer")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    type_handles = [Line2D([0], [0], marker='o', linestyle='', color='#444444', label='Ours-RL', markersize=7),
                    Line2D([0], [0], marker='s', linestyle='', color='#444444', label='Frontier LLM', markersize=7)]
    model_handles = [Line2D([0], [0], marker='o', linestyle='', color=MODEL_COLORS[m], label=m, markersize=7) for m in RL_MODEL_ORDER]
    frontier_handles = [Line2D([0], [0], marker='s', linestyle='', color=FRONTIER_COLORS[m], label=m, markersize=7) for m in FRONTIER_MODELS]
    leg1 = ax.legend(handles=type_handles, title="Type", frameon=False, loc="upper left",
                     bbox_to_anchor=(1.02, 1.00), borderaxespad=0.0)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles + frontier_handles, title="Model family", frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 0.66), borderaxespad=0.0)
    fig.subplots_adjust(right=0.74)
    return savefig(outdir, "fig_frontier_core_tradeoff_clean")


def fmt_pct(v):
    return f"{v * 100:.1f}"


def fmt_int(v):
    return f"{int(round(v))}"


def compact_rows(df, phase, group="CORE"):
    models = SFT_MODEL_ORDER if phase == "SFT" else RL_MODEL_ORDER
    baseline = "Review"
    rows = []
    for model in models:
        sub = df[(df.phase == phase) & (df.group == group) & (df.model == model)].set_index("variant")
        if not {"Base", baseline, "Innovation"}.issubset(set(sub.index)):
            continue
        base = sub.loc["Base"]
        ref = sub.loc[baseline]
        prop = sub.loc["Innovation"]
        rows.append({
            "model": model,
            "base_acc": base.acc * 100,
            "ref_acc": ref.acc * 100,
            "prop_acc": prop.acc * 100,
            "dacc_ref": (prop.acc - ref.acc) * 100,
            "dacc_base": (prop.acc - base.acc) * 100,
            "base_tpc": base.tpc,
            "ref_tpc": ref.tpc,
            "prop_tpc": prop.tpc,
            "dtpc_ref": prop.tpc - ref.tpc,
            "dtpc_base": prop.tpc - base.tpc,
            "prop_hw": prop.hw * 100,
            "prop_sd": prop.sd * 100,
        })
    return pd.DataFrame(rows)


def latex_compact_table(rows, phase):
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp/metrics")
    parser.add_argument("--outdir", default="/root/autodl-tmp/metrics/figures_results")
    args = parser.parse_args()
    setup()
    df = load_post_training(args.root)
    frontier_df = load_frontier(os.path.join(args.root, "LLM"))
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "post_training_summary_clean.csv"), index=False)
    frontier_df.to_csv(os.path.join(args.outdir, "frontier_summary_clean.csv"), index=False)
    outputs = []
    outputs.extend(plot_stage_tradeoff(df, args.outdir, "SFT", "CORE"))
    outputs.extend(plot_stage_tradeoff(df, args.outdir, "RL", "CORE"))
    outputs.extend(plot_frontier_tradeoff(df, frontier_df, args.outdir, "CORE"))
    Path = __import__('pathlib').Path
    print("Generated files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
