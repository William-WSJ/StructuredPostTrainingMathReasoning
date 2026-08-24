import argparse
import json
import math
import re
import sympy as sp
from typing import Dict, List, Optional, Tuple, Set

# ==========================================
# 1. 核心解析逻辑 (保持你原有的架构名)
# ==========================================

def _norm(s: str) -> str:
    if not s: return ""
    text = s.lower()
    # 归一化 LaTeX：移除空格、装饰符，统一分式
    text = re.sub(r'\s+', '', text)
    text = text.replace('\\text{', '').replace('}', '').replace('\\left', '').replace('\\right', '')
    text = text.replace('$', '').replace('\\(', '').replace('\\)', '')
    text = text.replace('\\,', '').replace('\\;', '').replace('\\:', '').replace('\\!', '')
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    # 竞赛题增强：移除单位和角度干扰
    text = re.sub(r'\\circ|度|deg|degrees|\^\\circ', '', text)
    return text

def _extract_answer(output: str) -> str:
    # 1. 优先提取 \boxed
    boxed_match = re.findall(r"\\boxed\{(.*)\}", output, re.DOTALL)
    if boxed_match:
        ans = boxed_match[-1]
        depth, res = 0, ""
        for char in ans:
            if char == '{': depth += 1
            elif char == '}':
                if depth == 0: break
                depth -= 1
            res += char
        return res.strip()
    # 2. 其次提取 Final Answer
    m = re.search(r"Final\s+Answer\s*[:：]\s*(.*)", output, re.IGNORECASE | re.DOTALL)
    if m: return m.group(1).split('\n')[0].strip().strip('*')
    # 3. 兜底：最后一行
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    return lines[-1][:100] if lines else ""

# ==========================================
# 2. 竞赛级增强判定逻辑 (内部逻辑增强)
# ==========================================

def _clean_for_sympy(x: str) -> str:
    """递归处理分式、组合数、pi、复数，供 SymPy 解析"""
    if not x: return ""
    x = x.replace("^", "**").replace("i", "I").replace(r"\pi", "pi")
    # 递归处理 \frac{a}{b} -> ((a)/(b))
    while r'\frac' in x:
        new_x = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'((\1)/(\2))', x)
        if new_x == x: break
        x = new_x
    # 递归处理 \binom{n}{k}
    while r'binom' in x:
        new_x = re.sub(r'\\d?binom\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'binomial(\1,\2)', x)
        if new_x == x: break
        x = new_x
    x = x.replace("{", "(").replace("}", ")").replace("\\", "")
    return x

OPT_LINE = re.compile(r"(?:^|\n)\s*([A-E])\s*[\.|\)|:：]\s*(.+?)(?=(?:\n\s*[A-E]\s*[\.|\)|:：])|\Z)", re.S | re.I)

def _option_equiv(problem: str, gt_raw: str, pred_raw: str) -> bool:
    """处理选择题选项映射 (针对 TAL-SCQ5K)"""
    if not problem: return False
    opts = {k.upper(): v.strip() for k, v in OPT_LINE.findall(problem)}
    if not opts: return False
    gt_u, pred_u = gt_raw.strip().upper(), pred_raw.strip().upper()
    if re.fullmatch(r"[A-E]", gt_u):
        if re.search(rf"\b{gt_u}\b", pred_u): return True
        if gt_u in opts and _norm(opts[gt_u]) == _norm(pred_raw): return True
    m = re.search(r"\b([A-E])\b", pred_u)
    if m and m.group(1) in opts:
        if _norm(opts[m.group(1)]) == _norm(gt_raw): return True
    return False

# ==========================================
# 3. 判分核心 (对接原本的逻辑名)
# ==========================================

def relaxed_hit(item: dict) -> Tuple[bool, str]:
    gt_raw = item.get("ground_truth", "")
    out = item.get("model_output", "")
    problem = item.get("problem", "")

    gt = _norm(gt_raw)
    pred_raw = _extract_answer(out)
    pred = _norm(pred_raw)

    if not pred: return False, "no_match"
    # 1. 严格匹配
    if pred and (gt == pred or gt in pred): return True, "strict"
    # 2. 选项映射
    if _option_equiv(problem, gt_raw, pred_raw): return True, "option_equiv"
    # 3. 数值等效
    try:
        g_v = float(sp.sympify(_clean_for_sympy(gt_raw)).evalf())
        p_v = float(sp.sympify(_clean_for_sympy(pred_raw)).evalf())
        if abs(g_v - p_v) < 1e-8: return True, "num_equiv"
    except: pass
    # 4. 集合判定
    if '{' in gt_raw:
        try:
            get_s = lambda s: {_norm(i) for i in re.sub(r'[\{\}\[\]\(\)]', '', s).split(',') if i.strip()}
            if get_s(gt_raw) == get_s(pred_raw): return True, "set_equiv"
        except: pass
    # 5. 符号判定 (sqrtdenest 化简嵌套根号)
    if sp:
        try:
            e1 = sp.sqrtdenest(sp.sympify(_clean_for_sympy(gt_raw)))
            e2 = sp.sqrtdenest(sp.sympify(_clean_for_sympy(pred_raw)))
            if sp.simplify(e1 - e2) == 0: return True, "expr_equiv"
        except: pass
    return False, "no_match"

# ==========================================
# 4. 统计与文件保存 (恢复你原有的 run 和 main)
# ==========================================

def run(path: str) -> dict:
    total = 0
    strict_correct = 0
    rescued = 0
    reason_count: Dict[str, int] = {}
    rescued_examples: List[dict] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            item = json.loads(line)
            gt = _norm(item.get("ground_truth", ""))
            pred = _norm(_extract_answer(item.get("model_output", "")))
            
            # 严格判分
            if pred and (gt == pred or gt in pred):
                strict_correct += 1
                continue
            
            # 宽泛判分
            ok, reason = relaxed_hit(item)
            reason_count[reason] = reason_count.get(reason, 0) + 1
            if ok:
                rescued += 1
                if len(rescued_examples) < 8:
                    rescued_examples.append({
                        "id": item.get("unique_id", item.get("id", "")),
                        "reason": reason,
                        "gt": item.get("ground_truth", ""),
                        "pred": _extract_answer(item.get("model_output", "")),
                    })

    strict_acc = strict_correct / total if total else 0.0
    relaxed_correct = strict_correct + rescued
    relaxed_acc = relaxed_correct / total if total else 0.0

    return {
        "file": path,
        "total": total,
        "strict_correct": strict_correct,
        "strict_acc": strict_acc,
        "rescued": rescued,
        "relaxed_correct": relaxed_correct,
        "relaxed_acc": relaxed_acc,
        "delta": relaxed_acc - strict_acc,
        "rescued_reason_count": reason_count,
        "rescued_examples": rescued_examples,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--save_json", default="")
    args = parser.parse_args()

    results = [run(p) for p in args.files]

    print("name\tstrict\trelaxed\trescue\tdelta")
    for r in results:
        print(
            f"{r['file']}\t"
            f"{r['strict_correct']}/{r['total']} ({r['strict_acc']:.2%})\t"
            f"{r['relaxed_correct']}/{r['total']} ({r['relaxed_acc']:.2%})\t"
            f"{r['rescued']}\t{r['delta']:+.2%}"
        )
        print(f"  reasons: {r['rescued_reason_count']}")

    # ============= 这里是你的文件保存逻辑 =============
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.save_json}")

if __name__ == "__main__":
    main()