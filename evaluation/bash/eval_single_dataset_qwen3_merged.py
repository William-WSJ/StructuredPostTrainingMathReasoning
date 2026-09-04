#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple


def load_relaxed_hit(relaxed_hit_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate_results", relaxed_hit_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {relaxed_hit_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "relaxed_hit"):
        raise RuntimeError(f"`relaxed_hit` not found in: {relaxed_hit_py}")
    return module.relaxed_hit


def extract_training_instruction(training_prompt_source: str) -> str:
    with open(training_prompt_source, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Expected a non-empty JSON list: {training_prompt_source}")
    raw_prompt = str(data[0].get("prompt", ""))
    if "User:" in raw_prompt:
        return raw_prompt.split("User:", 1)[0].strip()
    return raw_prompt.strip() or "Solve the problem."


def format_prefill_prompt(instruction: str, problem_text: str, prefill_prefix: str, append_im_end: bool) -> str:
    if append_im_end and "<|im_end|>" not in instruction:
        instruction = (
            instruction.rstrip()
            + "\n\nAfter you provide the final answer, end the assistant message with <|im_end|>."
        )

    actual_prefill = f"<think>\n{prefill_prefix}"
    return (
        f"<|im_start|>user\n{instruction}\n{problem_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{actual_prefill}"
    )


def iter_dataset(path: str) -> Iterable[Dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as e:
                    raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {e}")
                if not isinstance(item, dict):
                    raise RuntimeError(f"Expected object at {path}:{line_no}, got {type(item)}")
                yield item
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise RuntimeError(f"Expected object at {path}[{idx - 1}], got {type(item)}")
            yield item
        return

    if isinstance(data, dict):
        yield data
        return

    raise RuntimeError(f"Unsupported data format in {path}: {type(data)}")


def _format_tal_options(answer_option_list) -> str:
    if not answer_option_list:
        return ""

    lines: List[str] = []
    for group in answer_option_list:
        if isinstance(group, dict):
            group = [group]
        if not isinstance(group, list):
            continue
        for option in group:
            if not isinstance(option, dict):
                continue
            label = str(option.get("aoVal", "")).strip()
            content = str(option.get("content", "")).strip()
            if label and content:
                lines.append(f"{label}: {content}")

    if not lines:
        return ""

    return "\nOptions:\n" + "\n".join(lines)


def get_problem_and_gt(item: Dict) -> Tuple[str, str, str]:
    for key in ["problem", "question", "prompt", "input"]:
        if key in item and isinstance(item[key], str) and item[key].strip():
            problem = item[key].strip()
            break
    else:
        problem = str(item)

    tal_options = _format_tal_options(item.get("answer_option_list"))
    if tal_options:
        problem = f"{problem}{tal_options}"

    for key in ["answer", "answer_value", "ground_truth", "final_answer", "target", "output"]:
        if key in item and isinstance(item[key], (str, int, float)):
            gt = str(item[key]).strip()
            if gt:
                break
    else:
        gt = ""

    for key in ["id", "unique_id", "qid", "queId"]:
        if key in item and str(item[key]).strip():
            sample_id = str(item[key]).strip()
            break
    else:
        sample_id = ""

    return problem, gt, sample_id


def build_sampling_params(args):
    from vllm import SamplingParams

    temperature = args.temperature if args.do_sample else 0.0
    top_p = args.top_p if args.do_sample else 1.0

    stop_tokens = [
        "<|im_end|>",
        "<|im_start|>",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<｜User｜>",
        "<｜Assistant｜>",
        "User:",
        "Assistant:",
    ]

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=stop_tokens,
    )


def generate_completions_vllm(llm, prompts: List[str], sampling_params, batch_size: int) -> List[str]:
    outputs: List[str] = []

    if batch_size <= 0:
        batch_size = len(prompts)

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_out = llm.generate(batch_prompts, sampling_params)
        for one in batch_out:
            if one.outputs:
                outputs.append(one.outputs[0].text.strip())
            else:
                outputs.append("")

    return outputs


def evaluate_and_save(args) -> None:
    try:
        from vllm import LLM
    except Exception as e:
        raise RuntimeError(
            "Failed to import vLLM. Install it in the running python env first. "
            f"Original error: {e}"
        )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    relaxed_hit = load_relaxed_hit(args.relaxed_hit_py)
    instruction = extract_training_instruction(args.training_prompt_source)

    examples: List[Dict] = []
    for ex in iter_dataset(args.data_path):
        examples.append(ex)
        if args.max_examples is not None and len(examples) >= args.max_examples:
            break

    if not examples:
        raise RuntimeError(f"No examples found in {args.data_path}")

    prompts: List[str] = []
    problems: List[str] = []
    gts: List[str] = []
    ids: List[str] = []

    for ex in examples:
        problem, gt, sample_id = get_problem_and_gt(ex)
        prompt = format_prefill_prompt(
            instruction=instruction,
            problem_text=problem,
            prefill_prefix=args.prefill_prefix,
            append_im_end=args.append_im_end,
        )
        prompts.append(prompt)
        problems.append(problem)
        gts.append(gt)
        ids.append(sample_id)

    sampling_params = build_sampling_params(args)

    print("[vLLM] Initializing merged model...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )

    completions = generate_completions_vllm(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
    )

    correct = 0
    total = 0
    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, (sample_id, problem, gt, out) in enumerate(zip(ids, problems, gts, completions), start=1):
            ok, reason = relaxed_hit({"ground_truth": gt, "problem": problem, "model_output": out})
            total += 1
            if ok:
                correct += 1
            rec = {
                "id": sample_id or i,
                "ground_truth": gt,
                "problem": problem,
                "model_output": out,
                "correct": bool(ok),
                "reason": reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    acc = correct / total if total else 0.0
    print(f"ACC: {correct}/{total} ({acc:.2%})")
    print(f"Saved: {args.output_path}")


def build_parser(default_data_path: str, default_output_path: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate merged Qwen model on one dataset with vLLM.")
    p.add_argument("--model_path", default="/root/autodl-tmp/Qwen3-8B-prune_4000")
    p.add_argument("--data_path", default=default_data_path)
    p.add_argument("--output_path", default=default_output_path)
    p.add_argument("--relaxed_hit_py", default="/root/autodl-tmp/all_results/evaluate_results.py")

    p.add_argument(
        "--training_prompt_source",
        default="/root/autodl-tmp/PruneRL/datasets/thinkprune_700_vanguard_relaxed.json",
        help="Used to extract the exact training instruction prefix (text before 'User:').",
    )
    p.add_argument("--prefill_prefix", default="**Theorems**：")
    p.add_argument("--append_im_end", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--max_new_tokens", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)

    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_examples", type=int, default=None)

    return p


def main(default_data_path: str, default_output_path: str) -> None:
    parser = build_parser(default_data_path=default_data_path, default_output_path=default_output_path)
    args = parser.parse_args()
    evaluate_and_save(args)


if __name__ == "__main__":
    main(
        default_data_path="/root/autodl-tmp/math-500.jsonl",
        default_output_path="/root/autodl-tmp/PruneRL/qwen3/math500_results_4000L_merged.jsonl",
    )
