import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["DeepSeek", "Qwen2.5", "Qwen3", "GLM", "R1"]
SFT_MODEL_ORDER = ["DeepSeek", "Qwen2.5", "Qwen3", "GLM", "R1"]
RL_MODEL_ORDER = ["Qwen3", "GLM", "R1"]
GROUP_ORDER = ["ALL", "CORE", "BASE_MATH"]
MAX_TOKENS = {
    "DeepSeek": 4096,
    "Qwen2.5": 4096,
    "Qwen3": 8192,
    "GLM": 8192,
    "R1": 8192,
}

MODEL_COLORS = {
    "DeepSeek": "#4C78A8",
    "Qwen2.5": "#59A14F",
    "Qwen3": "#F28E2B",
    "GLM": "#E15759",
    "R1": "#B07AA1",
    "DeepSeek-V4-Flash": "#B07AA1",
    "GLM-5.1": "#E15759",
    "Qwen3.7-Max": "#F28E2B",
}

VARIANT_MARKERS = {
    "Base": "o",
    "Review": "s",
    "Innovation": "^",
}


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

    model = {
        "deepseek": "DeepSeek",
        "qwen2_5": "Qwen2.5",
        "qwen3": "Qwen3",
        "glmz1": "GLM",
        "r1": "R1",
    }.get(parts[0])
    phase = {"sft": "SFT", "rl": "RL"}.get(parts[1])
    variant = {
        "base": "Base",
        "base_1": "Base",
        "review": "Review",
        "review_1": "Review",
        "innovation": "Innovation",
    }.get(parts[2])

    if not model or not phase or not variant:
        return None
    return model, phase, variant


def normalize_dataset(record):
    dataset = str(record.get("dataset", "")).upper().strip()
    file_path = os.path.basename(str(record.get("file_path", ""))).lower()

    if dataset == "UNKNOWN" and "math-500" in file_path:
        return "MATH500"
    if "AIME" in dataset:
        return "AIME"
    if dataset in {"MATH-500", "MATH_500"}:
        return "MATH500"
    if "MATHODYSSEY" in dataset or "MATH-ODYSSEY" in dataset:
        return "MATHODYSSEY"
    return dataset


def groups_for_record(record):
    dataset = normalize_dataset(record)
    groups = ["ALL"]
    if dataset in {"MATH500", "TAL"}:
        groups.append("BASE_MATH")
    if dataset in {"AIME", "OMNI", "OE_TO_MATHS", "MATHODYSSEY"}:
        groups.append("CORE")
    return groups


def empty_stats():
    return {
        "count": 0,
        "correct": 0,
        "total_tokens": 0.0,
        "correct_tokens": 0.0,
        "incorrect_tokens": 0.0,
        "hw": 0.0,
        "sd": 0.0,
        "cov_key_points": 0.0,
        "cov_boxed": 0.0,
    }


def add_record(stats, record):
    count = int(record.get("count", 0) or 0)
    correct = int(record.get("correct_count", 0) or 0)
    avg = float(record.get("total_stats", {}).get("avg", 0.0) or 0.0)
    ok_avg = float(record.get("correct", {}).get("avg", 0.0) or 0.0)
    bad_avg = float(record.get("incorrect", {}).get("avg", 0.0) or 0.0)

    stats["count"] += count
    stats["correct"] += correct
    stats["total_tokens"] += count * avg
    stats["correct_tokens"] += correct * ok_avg
    stats["incorrect_tokens"] += (count - correct) * bad_avg
    stats["hw"] += count * float(record.get("hit_wall_pct", 0.0) or 0.0)
    stats["sd"] += count * float(record.get("seq_dec_pct", 0.0) or 0.0)
    stats["cov_key_points"] += count * float(record.get("cov_key_points", 0.0) or 0.0)
    stats["cov_boxed"] += count * float(record.get("cov_boxed", 0.0) or 0.0)


def summarize(stats, max_tokens=8192):
    count = stats["count"]
    correct = stats["correct"]
    if count == 0:
        return None

    acc = correct / count
    t_avg = stats["total_tokens"] / count
    tpc = stats["total_tokens"] / correct if correct > 0 else max_tokens
    return {
        "count": count,
        "correct": correct,
        "acc": acc,
        "t_avg": t_avg,
        "tpc": tpc,
        "hw": stats["hw"] / count,
        "sd": stats["sd"] / count,
        "c_kp": stats["cov_key_points"] / count,
        "c_box": stats["cov_boxed"] / count,
    }


def load_post_training(root):
    files = [
        p
        for p in glob.glob(os.path.join(root, "*.json"))
        if parse_experiment_filename(p) is not None
    ]
    buckets = defaultdict(empty_stats)

    for path in files:
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
        if row is None:
            continue
        row.update({"model": model, "phase": phase, "variant": variant, "group": group})
        rows.append(row)
    return pd.DataFrame(rows)


def load_frontier(root):
    model_names = {
        "deepseek-v4-flash": "DeepSeek-V4-Flash",
        "glm-5_1": "GLM-5.1",
        "qwen3_7-max": "Qwen3.7-Max",
    }
    buckets = defaultdict(empty_stats)

    for path in glob.glob(os.path.join(root, "*.json")):
        raw_name = os.path.basename(path).replace(".json", "")
        model = model_names.get(raw_name, raw_name)
        for record in load_json(path):
            for group in groups_for_record(record):
                add_record(buckets[(model, group)], record)

    rows = []
    for (model, group), stats in buckets.items():
        row = summarize(stats)
        if row is None:
            continue
        row.update({"model": model, "group": group, "type": "Frontier LLM"})
        rows.append(row)
    return pd.DataFrame(rows)


def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )


def savefig(outdir, name):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{name}.png")
    pdf = os.path.join(outdir, f"{name}.pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return png, pdf


def label_variant(phase, variant):
    if phase == "SFT":
        return {"Base": "Base", "Review": "LTR", "Innovation": "Proposed"}[variant]
    return {"Base": "SFT-Init", "Review": "ThinkPrune", "Innovation": "Proposed"}[variant]


def plot_accuracy_tpc_pareto(df, outdir, group):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, phase, models in zip(axes, ["SFT", "RL"], [SFT_MODEL_ORDER, RL_MODEL_ORDER]):
        sub = df[(df["phase"] == phase) & (df["group"] == group)].copy()
        sub["model"] = pd.Categorical(sub["model"], models, ordered=True)
        sub = sub.sort_values(["model", "variant"])

        for model in models:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            color = MODEL_COLORS[model]
            for _, row in msub.iterrows():
                ax.scatter(
                    row["tpc"],
                    row["acc"] * 100,
                    marker=VARIANT_MARKERS[row["variant"]],
                    s=90 if row["variant"] == "Innovation" else 65,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.annotate(
                    f"{model}\n{label_variant(phase, row['variant'])}",
                    (row["tpc"], row["acc"] * 100),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )

            ordered = msub.set_index("variant").reindex(["Base", "Review", "Innovation"]).dropna()
            if len(ordered) >= 2:
                ax.plot(
                    ordered["tpc"],
                    ordered["acc"] * 100,
                    color=color,
                    alpha=0.45,
                    linewidth=1.4,
                )

        ax.set_title(f"{phase} stage ({group})")
        ax.set_xlabel("TPC: total tokens per correct answer")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    return savefig(outdir, f"fig_accuracy_tpc_pareto_{group.lower()}")


def plot_rl_core_acc_tpc(df, outdir):
    sub = df[
        (df["phase"] == "RL")
        & (df["group"] == "CORE")
        & (df["model"].isin(RL_MODEL_ORDER))
        & (df["variant"].isin(["Review", "Innovation"]))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(RL_MODEL_ORDER))
    width = 0.34

    for i, variant in enumerate(["Review", "Innovation"]):
        vals = [
            sub[(sub["model"] == m) & (sub["variant"] == variant)]["acc"].iloc[0] * 100
            for m in RL_MODEL_ORDER
        ]
        axes[0].bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            label=label_variant("RL", variant),
            color="#9DA7B1" if variant == "Review" else "#D55E00",
        )

    for i, variant in enumerate(["Review", "Innovation"]):
        vals = [
            sub[(sub["model"] == m) & (sub["variant"] == variant)]["tpc"].iloc[0]
            for m in RL_MODEL_ORDER
        ]
        axes[1].bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            label=label_variant("RL", variant),
            color="#9DA7B1" if variant == "Review" else "#D55E00",
        )

    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("TPC")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(RL_MODEL_ORDER)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.legend(frameon=False)

    axes[0].set_title("CORE accuracy")
    axes[1].set_title("CORE token cost")
    return savefig(outdir, "fig_rl_core_acc_tpc")


def plot_delta_heatmaps(df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    configs = [
        ("SFT", "acc", "Delta Acc: Proposed - LTR", axes[0, 0]),
        ("SFT", "tpc", "Delta TPC: Proposed - LTR", axes[0, 1]),
        ("RL", "acc", "Delta Acc: Proposed - ThinkPrune", axes[1, 0]),
        ("RL", "tpc", "Delta TPC: Proposed - ThinkPrune", axes[1, 1]),
    ]

    for phase, metric, title, ax in configs:
        models = SFT_MODEL_ORDER if phase == "SFT" else RL_MODEL_ORDER
        values = np.full((len(models), len(GROUP_ORDER)), np.nan)

        for i, model in enumerate(models):
            for j, group in enumerate(GROUP_ORDER):
                prop = df[
                    (df["phase"] == phase)
                    & (df["model"] == model)
                    & (df["group"] == group)
                    & (df["variant"] == "Innovation")
                ]
                base = df[
                    (df["phase"] == phase)
                    & (df["model"] == model)
                    & (df["group"] == group)
                    & (df["variant"] == "Review")
                ]
                if prop.empty or base.empty:
                    continue
                delta = prop[metric].iloc[0] - base[metric].iloc[0]
                values[i, j] = delta * 100 if metric == "acc" else delta

        vmax = np.nanmax(np.abs(values))
        im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(GROUP_ORDER)))
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isnan(values[i, j]):
                    continue
                text = f"{values[i, j]:+.1f}" if metric == "acc" else f"{values[i, j]:+.0f}"
                color = "white" if vmax and abs(values[i, j]) >= 0.55 * vmax else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.set_ylabel("percentage points" if metric == "acc" else "tokens", rotation=90)

    return savefig(outdir, "fig_delta_heatmaps")


def plot_delta_heatmaps_vs_base(df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    configs = [
        ("SFT", "acc", "Delta Acc: Proposed - Base", axes[0, 0]),
        ("SFT", "tpc", "Delta TPC: Proposed - Base", axes[0, 1]),
        ("RL", "acc", "Delta Acc: Proposed - SFT-Init", axes[1, 0]),
        ("RL", "tpc", "Delta TPC: Proposed - SFT-Init", axes[1, 1]),
    ]

    for phase, metric, title, ax in configs:
        models = SFT_MODEL_ORDER if phase == "SFT" else RL_MODEL_ORDER
        values = np.full((len(models), len(GROUP_ORDER)), np.nan)

        for i, model in enumerate(models):
            for j, group in enumerate(GROUP_ORDER):
                prop = df[
                    (df["phase"] == phase)
                    & (df["model"] == model)
                    & (df["group"] == group)
                    & (df["variant"] == "Innovation")
                ]
                base = df[
                    (df["phase"] == phase)
                    & (df["model"] == model)
                    & (df["group"] == group)
                    & (df["variant"] == "Base")
                ]
                if prop.empty or base.empty:
                    continue
                delta = prop[metric].iloc[0] - base[metric].iloc[0]
                values[i, j] = delta * 100 if metric == "acc" else delta

        vmax = np.nanmax(np.abs(values))
        im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(GROUP_ORDER)))
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isnan(values[i, j]):
                    continue
                text = f"{values[i, j]:+.1f}" if metric == "acc" else f"{values[i, j]:+.0f}"
                color = "white" if vmax and abs(values[i, j]) >= 0.55 * vmax else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.set_ylabel("percentage points" if metric == "acc" else "tokens", rotation=90)

    return savefig(outdir, "fig_delta_heatmaps_vs_base")


def plot_frontier_core_tradeoff(df, frontier_df, outdir):
    ours_map = {
        "Qwen3": "Qwen3-8B (Proposed)",
        "GLM": "GLM-Z1-9B (Proposed)",
        "R1": "R1-Distill-Qwen-7B (Proposed)",
    }
    small = df[
        (df["phase"] == "RL")
        & (df["group"] == "CORE")
        & (df["variant"] == "Innovation")
        & (df["model"].isin(["Qwen3", "GLM", "R1"]))
    ].copy()
    small["type"] = "Ours-RL"
    small["label"] = small["model"].map(ours_map)

    front = frontier_df[frontier_df["group"] == "CORE"].copy()
    front["label"] = front["model"]

    both = pd.concat([small, front], ignore_index=True, sort=False)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for typ, marker, edge in [("Ours-RL", "o", "black"), ("Frontier LLM", "s", "#555555")]:
        sub = both[both["type"] == typ]
        for _, row in sub.iterrows():
            size = 70 + 8 * row["hw"] * 100
            ax.scatter(
                row["tpc"],
                row["acc"] * 100,
                s=size,
                marker=marker,
                color=MODEL_COLORS.get(row["model"], "#777777"),
                edgecolor=edge,
                linewidth=0.9,
                alpha=0.85,
                label=typ,
            )
            ax.annotate(
                row["label"],
                (row["tpc"], row["acc"] * 100),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="best")
    ax.set_title("CORE trade-off: small post-trained models vs frontier LLMs")
    ax.set_xlabel("TPC: total tokens per correct answer")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    caption = "Bubble size indicates hit-wall rate."
    ax.text(0.01, -0.16, caption, transform=ax.transAxes, fontsize=8)
    return savefig(outdir, "fig_frontier_core_tradeoff")


def export_summary(df, frontier_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    df.sort_values(["phase", "group", "model", "variant"]).to_csv(
        os.path.join(outdir, "post_training_summary.csv"), index=False
    )
    frontier_df.sort_values(["group", "model"]).to_csv(
        os.path.join(outdir, "frontier_summary.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/autodl-tmp/metrics")
    parser.add_argument("--outdir", default="/root/autodl-tmp/metrics/figures_results")
    parser.add_argument("--group", default="ALL", choices=GROUP_ORDER)
    args = parser.parse_args()

    setup_matplotlib()
    df = load_post_training(args.root)
    frontier_df = load_frontier(os.path.join(args.root, "LLM"))
    export_summary(df, frontier_df, args.outdir)

    outputs = []
    outputs.extend(plot_accuracy_tpc_pareto(df, args.outdir, args.group))
    outputs.extend(plot_accuracy_tpc_pareto(df, args.outdir, "CORE"))
    outputs.extend(plot_rl_core_acc_tpc(df, args.outdir))
    outputs.extend(plot_delta_heatmaps(df, args.outdir))
    outputs.extend(plot_delta_heatmaps_vs_base(df, args.outdir))
    outputs.extend(plot_frontier_core_tradeoff(df, frontier_df, args.outdir))

    print("Generated files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
