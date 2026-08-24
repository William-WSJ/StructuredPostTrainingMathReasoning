import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tiktoken

# ==========================================
# 1. 依赖导入
# ==========================================
try:
    from evaluate_results import relaxed_hit
except ImportError as e:
    raise SystemExit(
        "[Error] Cannot import `relaxed_hit` from evaluate_results.py."
    ) from e

# ==========================================
# 2. 统计学函数 (新增 P75)
# ==========================================
def get_full_stats(token_list: List[int]) -> Dict[str, float]:
    if not token_list:
        return {
            "avg": 0.0, "median": 0.0, "std": 0.0, 
            "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, 
            "max": 0.0, "min": 0.0
        }
    arr = np.array(token_list, dtype=np.int64)
    return {
        "avg": float(arr.mean()), 
        "median": float(np.median(arr)), 
        "std": float(arr.std()),
        "p75": float(np.percentile(arr, 75)), # <--- 核心新增：P75
        "p90": float(np.percentile(arr, 90)), 
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)), 
        "max": float(arr.max()), 
        "min": float(arr.min())
    }

def parse_dataset_variant(file_path: str) -> Tuple[str, str]:
    b = os.path.basename(file_path).lower()
    dataset = "Unknown"
    for name in ["aime2025", "math500", "mathodyssey", "oe_to_maths", "omni", "tal"]:
        if name in b:
            dataset = name.upper() if name != "aime2025" else "AIME 2025"
            break
            
    variant = "Unknown"
    if "_base" in b: variant = "Base"
    elif "_5cat" in b: variant = "5cat"
    elif "_7cat" in b: variant = "7cat"
    return dataset, variant

# ==========================================
# 3. 细粒度标签检测与【自适应】解耦逻辑
# ==========================================
def analyze_tags(output: str) -> Dict[str, bool]:
    """独立检测每个标签是否存在 (转小写保证鲁棒性)"""
    out_lower = output.lower()
    return {
        "think": "<think>" in out_lower,
        "think_close": "</think>" in out_lower,
        "theorems": "**theorems**" in out_lower, 
        "key_points": "**key points**" in out_lower,
        "solution": "**solution**" in out_lower,
        "final_answer": "**final answer**" in out_lower,
        "boxed": "\\boxed{" in out_lower
    }

import re

def check_seq_decouple(output: str, is_7cat: bool = True) -> bool:
    """
    自适应 5cat / 7cat 的逻辑链顺序检查
    1. 兼容 **key points**、### Key Points、> Solution 等多种 Markdown 变体
    2. 顺序要求：KP -> Solution -> (Final Answer / Boxed 可选)
    """
    # 1. 定义兼容多种 Markdown 语法的正则表达式
    # (?i) 忽略大小写 | (?:\*{1,3}|#{1,6}|>\s*)? 兼容粗体/标题/引用前缀 | \s* 兼容前后空格
    patterns = {
        "kp":           r'(?i)(?:\*{1,3}|#{1,6}|>\s*)?\s*key\s+points',
        "solution":     r'(?i)(?:\*{1,3}|#{1,6}|>\s*)?\s*solution',
        "final_answer": r'(?i)(?:\*{1,3}|#{1,6}|>\s*)?\s*final\s+answer',
        "boxed":        r'\\boxed\s*\{'
    }

    # 2. 提取各标签首次出现的位置索引（不存在则为 -1）
    indices = {}
    for key, pat in patterns.items():
        match = re.search(pat, output)
        indices[key] = match.start() if match else -1

    # 3. 核心顺序校验逻辑
    if is_7cat:
        # 7cat 必须包含 KP 和 Solution
        if indices["kp"] == -1 or indices["solution"] == -1:
            return False
        # 强制顺序：KP 必须在 Solution 之前
        if not (indices["kp"] < indices["solution"]):
            return False
    else:
        # 5cat 不要求 KP，但必须包含 Solution
        if indices["solution"] == -1:
            return False

    # 4. 处理可选标签：若 Final Answer 或 Boxed 存在，则必须排在 Solution 之后
    if indices["final_answer"] != -1 and not (indices["solution"] < indices["final_answer"]):
        return False
    if indices["boxed"] != -1 and not (indices["solution"] < indices["boxed"]):
        return False

    return True
# def check_seq_decouple(output: str, is_7cat: bool = True) -> bool:
#     """自动兼容 5cat 和 7cat 的业务流顺序检查"""
#     out_lower = output.lower()
    
#     if is_7cat:
#         # --- 7cat 逻辑：包含 Key Points ---
#         tags = ["**key points**", "**solution**", "**final answer**", "\\boxed{"]
#         if not all(tag in out_lower for tag in tags):
#             return False
        
#         idx_keyp = out_lower.find("**key points**")
#         idx_sol = out_lower.find("**solution**")
#         idx_fans = out_lower.find("**final answer**")
#         idx_box = out_lower.find("\\boxed{")
        
#         if not (idx_keyp < idx_sol):
#             return False
#         if not (idx_sol < idx_fans and idx_sol < idx_box):
#             return False
#         return True
#     else:
#         # --- 5cat 逻辑：原版复现，仅含 Solution, Final Answer, boxed ---
#         tags = ["**solution**", "**final answer**", "\\boxed{"]
#         if not all(tag in out_lower for tag in tags):
#             return False
        
#         idx_sol = out_lower.find("**solution**")
#         idx_fans = out_lower.find("**final answer**")
#         idx_box = out_lower.find("\\boxed{")
        
#         if not (idx_sol < idx_fans and idx_sol < idx_box):
#             return False
#         return True

# ==========================================
# 4. 核心解析逻辑
# ==========================================
def analyze_jsonl(file_path: str, encoder, max_tokens: int = 8000) -> Optional[Dict[str, Any]]:
    all_ts, ok_ts, bad_ts = [], [], []
    
    counts = {
        "hit_wall": 0, "seq_dec": 0,
        "think": 0, "think_close": 0,
        "theorems": 0, "key_points": 0,
        "solution": 0, "final_answer": 0, "boxed": 0
    }

    dataset, variant = parse_dataset_variant(file_path)
    # 根据 variant 自动判断是否使用 7cat 逻辑 (如果不是明确的 5cat，默认按 7cat 处理)
    is_7cat_logic = (variant != "5cat")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            output = item.get("model_output", "")
            tok = len(encoder.encode(output))

            all_ts.append(tok)
            ok, _ = relaxed_hit(item)
            if ok: ok_ts.append(tok)
            else: bad_ts.append(tok)
            
            # 检测细粒度标签
            tags = analyze_tags(output)
            for k, v in tags.items():
                if v: counts[k] += 1
                
            # 撞墙率 (只认物理截断)
            if tok >= max_tokens:
                counts["hit_wall"] += 1

            # 绝对顺序解耦 (自适应传递 is_7cat 标志)
            if check_seq_decouple(output, is_7cat=is_7cat_logic):
                counts["seq_dec"] += 1

    if not all_ts:
        return None

    total = len(all_ts)
    correct = len(ok_ts)
    ok_stats = get_full_stats(ok_ts)
    acc = correct / total if total else 0.0

    # 计算分布
    total_stats = get_full_stats(all_ts)
    bad_stats = get_full_stats(bad_ts)
    avg_all = float(total_stats["avg"])
    tpc = (avg_all / acc) if acc > 0 else float("inf")

    res = {
        "file_path": file_path, "dataset": dataset, "variant": variant,
        "count": total, "correct_count": correct, "acc": acc, "tpc": tpc,
        "hit_wall_pct": counts["hit_wall"] / total if total else 0.0,
        "seq_dec_pct": counts["seq_dec"] / total if total else 0.0,
        "total_stats": total_stats,  # <--- 核心新增：输出所有样本的整体分布！
        "incorrect": bad_stats,      # 保留错误样本分布用于对比分析
        "correct": ok_stats,
    }
    for tag_key in ["think", "think_close", "theorems", "key_points", "solution", "final_answer", "boxed"]:
        res[f"cov_{tag_key}"] = counts[tag_key] / total if total else 0.0
        
    return res

# ==========================================
# 5. 主函数与终端输出
# ==========================================
def fmt_p(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 100.0:3.0f}%"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--save_json", type=str, required=True)
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--max_tokens", type=int, default=8000)
    args = parser.parse_args()

    enc = tiktoken.get_encoding(args.encoding)

    # 终端打印加上全量的 P75 和 P95，让你跑的时候肉眼可见长尾耗散的降低
    header = (
        f"{'File Path':<28} | {'Acc':<5} | {'HitWl':>5} | {'SeqD':>4} | "
        f"{'TPC':>5} | {'TotP75':>6} | {'TotP90':>6} | {'TotP95':>6}"
    )
    print("\n" + header)
    print("-" * len(header))

    final_data = []
    
    for file_path in args.files:
        if not os.path.exists(file_path): continue
        res = analyze_jsonl(file_path, enc, args.max_tokens)
        if not res: continue

        display_path = file_path if len(file_path) <= 28 else "..." + file_path[-25:]
        acc_str = f"{res['acc']*100:.1f}%"

        print(
            f"{display_path:<28} | {acc_str:<5} | "
            f"{fmt_p(res['hit_wall_pct']):>5} | {fmt_p(res['seq_dec_pct']):>4} | "
            f"{res['tpc']:>5.0f} | {res['total_stats']['p75']:6.0f} | "
            f"{res['total_stats']['p90']:6.0f} | {res['total_stats']['p95']:6.0f}"
        )
        final_data.append(res)

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()