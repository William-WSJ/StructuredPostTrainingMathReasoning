import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken

try:
    from evaluate_results import relaxed_hit
except Exception:
    relaxed_hit = None

ENCODER = tiktoken.get_encoding('cl100k_base')

GROUP_ORDER = ['ALL', 'CORE', 'BASE_MATH']
RL_MODELS = ['Qwen3', 'GLM', 'R1']
SFT_MODELS = ['DeepSeek', 'Qwen2.5', 'Qwen3', 'GLM', 'R1']
FRONTIER_MODELS = ['Qwen3.7-Max', 'GLM-5.1', 'DeepSeek-V4-Flash']

MAX_TOKENS = {
    'DeepSeek': 4096,
    'Qwen2.5': 4096,
    'Qwen3': 8192,
    'GLM': 8192,
    'R1': 8192,
    'Qwen3.7-Max': 8192,
    'GLM-5.1': 8192,
    'DeepSeek-V4-Flash': 8192,
}

# Vertical reference lines for the effective prompt/budget used by the final compressed setting.
PROMPT_BUDGETS = {
    'Qwen3': 5000,
    'Qwen3.7-Max': 5000,
    'GLM': 4400,
    'GLM-5.1': 4400,
    'R1': 4100,
    'DeepSeek-V4-Flash': 4100,
}

MODEL_COLORS = {
    'DeepSeek': '#4C78A8',
    'Qwen2.5': '#59A14F',
    'Qwen3': '#F28E2B',
    'GLM': '#E15759',
    'R1': '#B07AA1',
    'Qwen3.7-Max': '#F28E2B',
    'GLM-5.1': '#E15759',
    'DeepSeek-V4-Flash': '#4C78A8',
}

VARIANT_LABEL = {
    ('SFT', 'Base'): 'Base',
    ('SFT', 'Review'): 'LTR',
    ('SFT', 'Innovation'): 'Proposed',
    ('RL', 'Base'): 'SFT-Init',
    ('RL', 'Review'): 'ThinkPrune',
    ('RL', 'Innovation'): 'Proposed',
}

DEFAULT_BINS = [0, 512, 1000, 2000, 4000, 6000, 8192, 12000, 20000, 50000, 100000]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'\bInfinity\b', 'null', text)
    return json.loads(text)


def parse_experiment_filename(path):
    name = os.path.basename(path).replace('.json', '').lower()
    parts = name.split('-')
    if len(parts) < 3:
        return None
    model = {
        'deepseek': 'DeepSeek',
        'qwen2_5': 'Qwen2.5',
        'qwen3': 'Qwen3',
        'glmz1': 'GLM',
        'r1': 'R1',
    }.get(parts[0])
    phase = {'sft': 'SFT', 'rl': 'RL'}.get(parts[1])
    variant = {
        'base': 'Base',
        'base_1': 'Base',
        'review': 'Review',
        'review_1': 'Review',
        'innovation': 'Innovation',
    }.get(parts[2])
    if not model or not phase or not variant:
        return None
    return model, phase, variant


def parse_frontier_filename(path):
    name = os.path.basename(path).replace('.json', '')
    return {
        'deepseek-v4-flash': 'DeepSeek-V4-Flash',
        'glm-5_1': 'GLM-5.1',
        'qwen3_7-max': 'Qwen3.7-Max',
    }.get(name, name)


def normalize_dataset_name_from_record(record, raw_file=''):
    ds = str(record.get('dataset', '')).upper().strip()
    fp = os.path.basename(str(record.get('file_path', raw_file))).lower()
    if ds == 'UNKNOWN' and ('math-500' in fp or 'math500' in fp):
        return 'MATH500'
    if 'AIME' in ds or 'aime2025' in fp:
        return 'AIME'
    if ds in {'MATH-500', 'MATH_500'} or 'math-500' in fp or 'math500' in fp:
        return 'MATH500'
    if 'TAL' in ds or 'tal' in fp:
        return 'TAL'
    if 'OMNI' in ds or 'omni' in fp:
        return 'OMNI'
    if 'OE_TO_MATHS' in ds or 'oe_to_maths' in fp:
        return 'OE_TO_MATHS'
    if 'MATHODYSSEY' in ds or 'mathodyssey' in fp:
        return 'MATHODYSSEY'
    return ds or 'UNKNOWN'


def groups_for_dataset(dataset):
    groups = ['ALL']
    if dataset in {'MATH500', 'TAL'}:
        groups.append('BASE_MATH')
    if dataset in {'AIME', 'OMNI', 'OE_TO_MATHS', 'MATHODYSSEY'}:
        groups.append('CORE')
    return groups


def iter_summary_entries(root):
    for path in glob.glob(os.path.join(root, '*.json')):
        meta = parse_experiment_filename(path)
        if meta is None:
            continue
        model, phase, variant = meta
        for rec in load_json(path):
            raw = rec.get('file_path')
            if raw:
                yield {'model': model, 'phase': phase, 'variant': variant, 'summary': path, 'raw': raw}

    llm_dir = os.path.join(root, 'LLM')
    for path in glob.glob(os.path.join(llm_dir, '*.json')):
        model = parse_frontier_filename(path)
        for rec in load_json(path):
            raw = rec.get('file_path')
            if raw:
                yield {'model': model, 'phase': 'Frontier', 'variant': 'Frontier', 'summary': path, 'raw': raw}


def get_correct(item):
    if isinstance(item.get('correct'), bool):
        return item['correct']
    if isinstance(item.get('is_correct'), bool):
        return item['is_correct']
    if isinstance(item.get('hit'), bool):
        return item['hit']
    if relaxed_hit is not None:
        ok, _ = relaxed_hit(item)
        return bool(ok)
    return False


def load_raw_outputs(root):
    rows = []
    seen = set()
    missing = []
    for entry in iter_summary_entries(root):
        raw_path = entry['raw']
        key = (entry['model'], entry['phase'], entry['variant'], raw_path)
        if key in seen:
            continue
        seen.add(key)
        if not os.path.exists(raw_path):
            missing.append(raw_path)
            continue
        dataset = normalize_dataset_name_from_record({}, raw_path)
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                output = str(item.get('model_output', ''))
                token_len = len(ENCODER.encode(output))
                correct = get_correct(item)
                max_tokens = MAX_TOKENS.get(entry['model'], 8192)
                rows.append({
                    'model': entry['model'],
                    'phase': entry['phase'],
                    'variant': entry['variant'],
                    'variant_label': VARIANT_LABEL.get((entry['phase'], entry['variant']), entry['variant']),
                    'dataset': normalize_dataset_name_from_record(item, raw_path) or dataset,
                    'raw_file': raw_path,
                    'line_no': line_no,
                    'token_len': token_len,
                    'correct': int(correct),
                    'max_tokens': max_tokens,
                    'prompt_budget': PROMPT_BUDGETS.get(entry['model'], np.nan),
                    'hit_wall': int(token_len >= max_tokens),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, missing
    df['groups'] = df['dataset'].map(groups_for_dataset)
    return df, missing


def expand_groups(df):
    rows = []
    for row in df.to_dict('records'):
        for group in row.pop('groups'):
            new_row = dict(row)
            new_row['group'] = group
            rows.append(new_row)
    return pd.DataFrame(rows)


def setup_matplotlib():
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 120,
    })


def savefig(outdir, name):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + '.png')
    pdf = os.path.join(outdir, name + '.pdf')
    plt.tight_layout()
    plt.savefig(png, dpi=600, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    return [png, pdf]


def bin_label(left, right):
    def fmt(v):
        if v >= 1000:
            return f'{int(v/1000)}k' if v % 1000 == 0 else f'{v/1000:.1f}k'
        return str(int(v))
    return f'{fmt(left)}-{fmt(right)}'


def add_token_bins(df, bins):
    labels = [bin_label(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    out = df.copy()
    out['token_bin'] = pd.cut(out['token_len'], bins=bins, right=False, labels=labels, include_lowest=True)
    return out, labels


def plot_single_model_token_bin_accuracy(df, outdir, bins, phase, model='Qwen3'):
    variants = ['Base', 'Review', 'Innovation']
    sub = df[
        (df['phase'] == phase)
        & (df['group'] == 'CORE')
        & (df['model'] == model)
        & (df['variant'].isin(variants))
    ].copy()
    if sub.empty:
        return []

    sub, labels = add_token_bins(sub, bins)
    fig, ax = plt.subplots(figsize=(8.3, 3.9))
    x = np.arange(len(labels))
    width = 0.24
    colors = {'Base': '#4C78A8', 'Review': '#9DA7B1', 'Innovation': '#D55E00'}

    for i, variant in enumerate(variants):
        vals = []
        counts = []
        for lab in labels:
            b = sub[(sub['variant'] == variant) & (sub['token_bin'] == lab)]
            vals.append(100 * b['correct'].mean() if len(b) else np.nan)
            counts.append(len(b))
        xpos = x + (i - 1) * width
        ax.bar(
            xpos,
            np.nan_to_num(vals, nan=0.0),
            width=width,
            color=colors[variant],
            label=VARIANT_LABEL[(phase, variant)],
            alpha=0.92,
        )
        for xi, val, cnt in zip(xpos, vals, counts):
            if cnt:
                y = max(val if not np.isnan(val) else 0, 1.5)
                ax.text(xi, y + 1.2, str(cnt), ha='center', va='bottom', fontsize=6, rotation=90)

    budget = PROMPT_BUDGETS.get(model)
    if budget and bins[0] <= budget <= bins[-1]:
        budget_bin = np.searchsorted(bins, budget, side='right') - 1
        if 0 <= budget_bin < len(labels):
            ax.axvline(budget_bin + 0.5, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_xlabel('Generated tokens (cl100k_base bins)')
    ax.set_ylabel('Accuracy within token bin (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, loc='upper left', ncol=3)
    return savefig(outdir, f"raw_{model.lower()}_{phase.lower()}_core_token_bin_accuracy")


def plot_rl_core_token_bin_accuracy(df, outdir, bins):
    return plot_single_model_token_bin_accuracy(df, outdir, bins, phase='RL', model='Qwen3')


def plot_length_cdf_rl_core(df, outdir):
    sub = df[(df['phase'] == 'RL') & (df['group'] == 'CORE') & (df['model'].isin(RL_MODELS))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    styles = {'Base': ('#4C78A8', '-'), 'Review': ('#9DA7B1', '--'), 'Innovation': ('#D55E00', '-')}
    for ax, model in zip(axes, RL_MODELS):
        for variant in ['Base', 'Review', 'Innovation']:
            vals = np.sort(sub[(sub['model'] == model) & (sub['variant'] == variant)]['token_len'].to_numpy())
            if len(vals) == 0:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            color, ls = styles[variant]
            ax.plot(vals, y, label=VARIANT_LABEL[('RL', variant)], color=color, linestyle=ls, linewidth=1.8)
        budget = PROMPT_BUDGETS.get(model)
        if budget:
            ax.axvline(budget, color='black', linestyle=':', linewidth=1, label='prompt budget')
        ax.axvline(MAX_TOKENS[model], color='black', linestyle='-.', linewidth=1, alpha=0.7, label='hit-wall threshold')
        ax.set_title(model)
        ax.set_xlabel('Generated tokens')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel('Cumulative fraction')
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), frameon=False, loc='lower right')
    return savefig(outdir, 'raw_rl_core_length_cdf')




def plot_hitwall_rates(df, outdir):
    sub = df[(df['phase'] == 'RL') & (df['group'] == 'CORE') & (df['model'].isin(RL_MODELS))].copy()
    if sub.empty:
        return []
    records = []
    for (model, variant, label), g in sub.groupby(['model', 'variant', 'variant_label']):
        records.append({
            'model': model,
            'variant': variant,
            'label': label,
            'hw': 100 * g['hit_wall'].mean(),
            'n': len(g),
        })
    hdf = pd.DataFrame(records)
    order = [('Qwen3', 'Base'), ('Qwen3', 'Review'), ('Qwen3', 'Innovation'),
             ('GLM', 'Base'), ('GLM', 'Review'), ('GLM', 'Innovation'),
             ('R1', 'Base'), ('R1', 'Review'), ('R1', 'Innovation')]
    labels, vals, colors = [], [], []
    for model, variant in order:
        row = hdf[(hdf['model'] == model) & (hdf['variant'] == variant)]
        if row.empty:
            continue
        row = row.iloc[0]
        labels.append(f"{model}\n{row['label']}")
        vals.append(row['hw'])
        colors.append(MODEL_COLORS.get(model, '#777777'))
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('Hit-wall rate (%)')
    ax.set_title('CORE hit-wall behavior from raw outputs')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.35)
    return savefig(outdir, 'raw_core_hitwall_rates')



def export_tables(df, outdir, bins):
    os.makedirs(outdir, exist_ok=True)
    df.drop(columns=['groups'], errors='ignore').to_csv(os.path.join(outdir, 'raw_output_samples.csv'), index=False)
    grouped = df.groupby(['phase', 'model', 'variant', 'variant_label', 'group']).agg(
        n=('correct', 'size'),
        acc=('correct', 'mean'),
        t_avg=('token_len', 'mean'),
        t_median=('token_len', 'median'),
        t_p90=('token_len', lambda x: np.percentile(x, 90)),
        t_p95=('token_len', lambda x: np.percentile(x, 95)),
        hit_wall=('hit_wall', 'mean'),
    ).reset_index()
    grouped.to_csv(os.path.join(outdir, 'raw_output_group_summary.csv'), index=False)

    bdf, labels = add_token_bins(df, bins)
    bin_summary = bdf.groupby(['phase', 'model', 'variant', 'variant_label', 'group', 'token_bin'], observed=False).agg(
        n=('correct', 'size'),
        acc=('correct', 'mean'),
        t_avg=('token_len', 'mean'),
        hit_wall=('hit_wall', 'mean'),
    ).reset_index()
    bin_summary.to_csv(os.path.join(outdir, 'raw_output_token_bin_summary.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/autodl-tmp/metrics')
    parser.add_argument('--outdir', default='/root/autodl-tmp/metrics/figures_raw_outputs')
    parser.add_argument('--bins', default=','.join(str(x) for x in DEFAULT_BINS), help='comma-separated token bin edges')
    args = parser.parse_args()

    bins = [int(x) for x in args.bins.split(',') if x.strip()]
    setup_matplotlib()
    raw, missing = load_raw_outputs(args.root)
    if raw.empty:
        raise SystemExit('No raw outputs were loaded.')
    df = expand_groups(raw)
    export_tables(df, args.outdir, bins)

    outputs = []
    outputs.extend(plot_single_model_token_bin_accuracy(df, args.outdir, bins, phase='SFT', model='Qwen3'))
    outputs.extend(plot_single_model_token_bin_accuracy(df, args.outdir, bins, phase='RL', model='Qwen3'))

    if missing:
        with open(os.path.join(args.outdir, 'missing_raw_files.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(set(missing))))
    print(f'Loaded raw samples: {len(raw)}')
    print(f'Expanded group rows: {len(df)}')
    print('Generated files:')
    for p in outputs:
        print(p)
    if missing:
        print(f'Missing raw files: {len(set(missing))}; see missing_raw_files.txt')


if __name__ == '__main__':
    main()
