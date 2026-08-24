'''
nohup bash -c '
{ 
  tail --pid=11194 -f /dev/null && \
  python3 -u grpo_2.py --use_prefill --model_name /root/autodl-tmp/Qwen3-8B-L5800_innovation --data_path /root/autodl-tmp/PruneRL/datasets/thinkprune_700_vanguard_relaxed_2.json --current_limit 5800 --gen_max_new_tokens 6000 --num_generations 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --temperature 0.6 --top_p 0.9 --repetition_penalty 1.1 --num_iterations 4 --max_steps 96 --logging_steps 1 --save_steps 4 --early_stop --output_dir ./grpo_2_qwen3_L5800_stage2 && \
  VLLM_USE_V1=0 bash /root/autodl-tmp/PruneRL/code/sweep_math500_qwen3_grpo_2_checkpoints.sh
} > run_group2.log 2>&1
' &
Qwen3
python3 -u grpo_3_stage2.py \
  --use_prefill \
  --model_name /root/autodl-tmp/Qwen3-8B-L6100-innovation_stage2 \
  --data_path /root/autodl-tmp/PruneRL/datasets/thinkprune_mini_20.json \
  --current_limit 5000 \
  --gen_max_new_tokens 5500 \
  --num_generations 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --temperature 0.6 \
  --top_p 0.9 \
  --repetition_penalty 1.2 \
  --num_iterations 1 \
  --max_steps 40 \
  --logging_steps 1 \
  --save_steps 4 \
  --debug_group_rewards_path ./grpo_3_qwen_rewards_stage3_innovation.jsonl \
  --debug_group_rewards_max_calls 0 \
  --output_dir ./grpo_3_qwen3_L5000_stage3_innovation


GLM-Z1
python3 -u grpo_3_stage2.py \
  --use_prefill \
  --prompt_style glmz1 \
  --model_name /root/autodl-tmp/GLM-Z1-9B-L5400-innovation_stage2 \
  --data_path /root/autodl-tmp/PruneRL/datasets/thinkprune_mini_20.json \
  --current_limit 4400 --gen_max_new_tokens 4900 \
  --num_generations 4 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --temperature 0.6 --top_p 0.9 --repetition_penalty 1.2 \
  --num_iterations 1 \
  --max_steps 40 --logging_steps 1 --save_steps 4 \
  --early_stop \
  --output_dir /root/autodl-tmp/PruneRL/code/grpo_3_glmz1_L4400_stage3_innovation \
  --debug_group_rewards_path /root/autodl-tmp/PruneRL/code/grpo_3_glmz1_rewards_stage3_innovation.jsonl

R1-Distill-Qwen
python3 -u grpo_3_stage2.py \
  --use_prefill \
  --prompt_style r1distill_qwen \
  --append_r1_eos \
  --model_name /root/autodl-tmp/R1-Distill-Qwen-7B-L5100-innovation_stage2 \
  --data_path /root/autodl-tmp/PruneRL/datasets/thinkprune_mini_20.json \
  --current_limit 4100 --gen_max_new_tokens 4600 \
  --num_generations 4 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --temperature 0.6 --top_p 0.9 --repetition_penalty 1.2 \
  --num_iterations 1 \
  --max_steps 40 --logging_steps 1 --save_steps 4 \
  --output_dir /root/autodl-tmp/PruneRL/code/grpo_3_r1distill_L4100_stage3_innovation \
  --debug_group_rewards_path /root/autodl-tmp/PruneRL/code/grpo_3_r1distill_rewards_stage3_innovation.jsonl
'''

import argparse
import hashlib
import importlib.util
import json
import os
import random

from collections import deque
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import math
import tiktoken


def _load_relaxed_hit(relaxed_hit_py: str):
    spec = importlib.util.spec_from_file_location("evaluate_results", relaxed_hit_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {relaxed_hit_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "relaxed_hit"):
        raise RuntimeError(f"`relaxed_hit` not found in: {relaxed_hit_py}")
    return module.relaxed_hit


class ThinkPruneLogCallback(TrainerCallback):
    def __init__(self, current_limit: int):
        self.current_limit = int(current_limit)
        self.reward_window = deque(maxlen=50)
        self.acc_window = deque(maxlen=50)
        self.len_window = deque(maxlen=50)
        self.kl_window = deque(maxlen=50)

    def safe(self, x, default=0.0):
        return default if x is None else x

    def _extract_reward_stats(self, logs):
        reward = logs.get("reward")
        reward_std = logs.get("reward_std")

        if reward is None:
            for k, v in logs.items():
                if k.startswith("rewards/") and k.endswith("/mean"):
                    reward = v
                    break
        if reward_std is None:
            for k, v in logs.items():
                if k.startswith("rewards/") and k.endswith("/std"):
                    reward_std = v
                    break
        return reward, reward_std

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        if "completions/mean_length" not in logs:
            return control

        reward_raw, reward_std_raw = self._extract_reward_stats(logs)
        if reward_raw is None or reward_std_raw is None:
            return control
        reward = self.safe(reward_raw)
        reward_std = self.safe(reward_std_raw)

        length = self.safe(logs.get("completions/mean_length"))
        clip_ratio = self.safe(logs.get("completions/clipped_ratio"))
        kl = self.safe(logs.get("kl"))
        entropy = self.safe(logs.get("entropy"))
        acc = self.safe(logs.get("accuracy"))

        self.reward_window.append(reward)
        self.len_window.append(length)
        self.kl_window.append(kl)
        self.acc_window.append(acc)

        def mean(x):
            return sum(x) / len(x) if len(x) else 0.0

        def std(x):
            if len(x) < 2:
                return 0.0
            m = mean(x)
            return math.sqrt(sum((i - m) ** 2 for i in x) / len(x))

        reward_trend = mean(self.reward_window)
        len_trend = mean(self.len_window)
        kl_trend = mean(self.kl_window)
        acc_trend = mean([x for x in self.acc_window if x != 0.0])

        reward_collapse = reward_std < 0.05
        learning_active = kl > 1e-4
        length_pressure = length / self.current_limit if self.current_limit > 0 else 0.0
        reward_efficiency = reward / (length / 1000 + 1e-6)

        print("\n" + "=" * 60)
        print(f"[Step {state.global_step}] ThinkPrune Diagnostics")
        print("-" * 60)
        print(f"🎯 Reward: {reward:.4f} | std={reward_std:.4f} | trend={reward_trend:.4f}")
        print(f"📏 Length: {length:.1f} ({length_pressure*100:.1f}% of limit) | trend={len_trend:.1f}")
        print(f"⚖️ KL: {kl:.6f} | trend={kl_trend:.6f} | active={learning_active}")
        print(f"🧠 Entropy: {entropy:.4f}")
        if acc != 0.0:
            print(f"🎯 Accuracy: {acc:.4f} | trend={acc_trend:.4f}")
        print(f"⚡ Reward/Length: {reward_efficiency:.4f}")
        print(f"⚠️ Clip ratio: {clip_ratio*100:.1f}%")

        if reward_collapse:
            print("🚨 WARNING: reward collapsed (std < 0.05)")
        if clip_ratio > 0.8:
            print("🚨 WARNING: severe truncation pressure")
        if kl < 1e-5:
            print("🚨 WARNING: model nearly not updating (KL ~ 0)")
        if 0 < reward_std < 0.1:
            print("⚠️ reward signal too weak / low variance")
        print("=" * 60 + "\n")
        return control


class ThinkPruneEarlyStopCallback(TrainerCallback):
    """Early stop when the current length limit is saturated.

    Stops training once reward is consistently high, clipped_ratio is low,
    and mean completion length is far below the current limit.
    """

    def __init__(
        self,
        *,
        current_limit: int,
        enabled: bool,
        window: int,
        min_steps: int,
        reward_mean_threshold: float,
        reward_std_threshold: float,
        clipped_ratio_threshold: float,
        mean_len_ratio_threshold: float,
    ):
        self.current_limit = int(current_limit)
        self.enabled = bool(enabled)
        self.window = int(window)
        self.min_steps = int(min_steps)
        self.reward_mean_threshold = float(reward_mean_threshold)
        self.reward_std_threshold = float(reward_std_threshold)
        self.clipped_ratio_threshold = float(clipped_ratio_threshold)
        self.mean_len_ratio_threshold = float(mean_len_ratio_threshold)
        self._recent: List[Dict[str, float]] = []

    def _push(self, item: Dict[str, float]) -> None:
        self._recent.append(item)
        if len(self._recent) > self.window:
            self._recent = self._recent[-self.window :]

    def _should_stop(self) -> bool:
        if self.window <= 0 or len(self._recent) < self.window:
            return False
        for r in self._recent:
            if r["reward_mean"] < self.reward_mean_threshold:
                return False
            if r["reward_std"] > self.reward_std_threshold:
                return False
            if r["clipped_ratio"] > self.clipped_ratio_threshold:
                return False
            if r["mean_length"] > self.mean_len_ratio_threshold * float(self.current_limit):
                return False
        return True

    def _extract_reward_stats(self, logs):
        reward = logs.get("reward")
        reward_std = logs.get("reward_std")

        if reward is None:
            for k, v in logs.items():
                if k.startswith("rewards/") and k.endswith("/mean"):
                    reward = v
                    break
        if reward_std is None:
            for k, v in logs.items():
                if k.startswith("rewards/") and k.endswith("/std"):
                    reward_std = v
                    break
        return reward, reward_std

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled:
            return control
        if state.global_step < self.min_steps:
            return control
        if not logs:
            return control

        reward_mean_raw, reward_std_raw = self._extract_reward_stats(logs)
        if reward_mean_raw is None or reward_std_raw is None:
            return control

        reward_mean = float(reward_mean_raw)
        reward_std = float(reward_std_raw)
        mean_length = float(logs.get("completions/mean_length", 0.0))
        clipped_ratio = float(logs.get("completions/clipped_ratio", 0.0))

        self._push(
            {
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "mean_length": mean_length,
                "clipped_ratio": clipped_ratio,
            }
        )

        if self._should_stop():
            msg = (
                "\n"
                + "!" * 50
                + "\n"
                + f"🛑 EarlyStop triggered at step={state.global_step} | "
                + f"reward_mean>={self.reward_mean_threshold}, reward_std<={self.reward_std_threshold}, "
                + f"clipped_ratio<={self.clipped_ratio_threshold}, "
                + f"mean_len<={self.mean_len_ratio_threshold}*L (L={self.current_limit}).\n"
                + f"Recent window={self.window}: {self._recent}\n"
                + "!" * 50
                + "\n"
            )
            print(msg)
            control.should_training_stop = True

        return control


def _clip_to_n_tokens(text: str, tokenizer, n: int) -> str:
    if not text:
        return ""
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= n:
        return text
    return tokenizer.decode(token_ids[:n], skip_special_tokens=True)


# def think_prune_reward_fn_builder(tokenizer, relaxed_hit, current_limit: int):
#     current_limit = int(current_limit)

#     def think_prune_reward_fn(completions, ground_truth, problem, **kwargs):
#         """ThinkPrune reward: 1 iff correct answer is found within L tokens; else 0.

#         Correctness is evaluated by `evaluate_results.py:relaxed_hit` on the clipped (<=L) completion.
#         """

#         rewards = []
#         for completion, gt, prob in zip(completions, ground_truth, problem):
#             clipped_text = _clip_to_n_tokens(completion, tokenizer, current_limit)
#             ok, _reason = relaxed_hit(
#                 {
#                     "ground_truth": gt,
#                     "problem": prob,
#                     "model_output": clipped_text,
#                 }
#             )
#             rewards.append(1.0 if ok else 0.0)
#         return rewards

#     return think_prune_reward_fn

def think_prune_reward_fn_builder(
    tokenizer,
    relaxed_hit,
    current_limit: int,
    *,
    debug_group_rewards_path: str = "",
    debug_group_rewards_max_calls: int = 0,
):
    current_limit = int(current_limit)
    debug_group_rewards_path = (debug_group_rewards_path or "").strip()
    debug_group_rewards_max_calls = int(debug_group_rewards_max_calls)
    call_index = 0
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # 【原逻辑：一字未改】提取结构分 (满分 1.0)
    def get_structure_feature(text: str) -> float:
        t = text.lower()
        score = 0.0
        if "**key points**" in t: score += 0.15
        if "**solution**" in t: score += 0.15
        if "</think>" in t: score += 0.15
        if "**final answer**" in t or "\\boxed" in t: score += 0.55
        return score

    def think_prune_reward_fn(completions, ground_truth, problem, **kwargs):
        nonlocal call_index
        call_index += 1
        rewards = []
        debug_rows = []

        ids = kwargs.get("id")
        if not isinstance(ids, list):
            ids = [None] * len(completions)

        for idx, (completion, gt, prob) in enumerate(zip(completions, ground_truth, problem)):
            # -------- 步骤 0: 极速物理截断 --------
            token_ids = enc.encode(completion)
            token_count = len(token_ids)
            if token_count > current_limit:
                clipped = enc.decode(token_ids[:current_limit])
                token_count = current_limit
            else:
                clipped = completion

            # -------- 步骤 1: 判题与特征提取 --------
            ok, _ = relaxed_hit({
                "ground_truth": gt,
                "problem": prob,
                "model_output": clipped,
            })
            
            x3 = get_structure_feature(clipped)
            len_ratio = token_count / float(current_limit)

            # --- 【状态提取：防作弊与外迁】 ---
            has_think_close = "</think>" in clipped
            is_migrated_outside = False
            
            if has_think_close:
                after_think = clipped.split("</think>")[-1].lower()
                if "\\boxed" in after_think or "**final answer**" in after_think:
                    is_migrated_outside = True
            
            # 🚨 新增：精准抓捕早退作弊（没关门，但输出了最终答案标志）
            is_early_exit_cheat = (not has_think_close) and ("\\boxed" in clipped.lower() or "**final answer**" in clipped.lower())

            # -------- 步骤 2: 混合双轨制 Reward 核心门控 --------
            if ok:
                # 【修改点 1：长度加成与闭合强绑定，且拉高系数至 1.5】
                # 只有写了 </think> 才能享受压缩奖励，没关门的一律没有长度分！
                if has_think_close:
                    length_bonus = 1.5 * (1.0 - len_ratio)
                else:
                    length_bonus = 0.0

                # 基础分 1.0 + 原结构分 + 强效压缩分
                reward = 1.0 + (0.5 * x3) + length_bonus
                
                # --- 追加逻辑 ---
                if is_migrated_outside:
                    reward += 0.5  # 完美迁出的顶薪
                
                if not has_think_close:
                    # 区分：是撞墙了，还是早退作弊了？
                    if is_early_exit_cheat:
                        reward -= 0.8  # 早退作弊，重罚 (最低仍保底 0.475)
                    else:
                        reward -= 0.5  # 撞墙没关门，普通罚 (最低仍保底 0.5)

            else:
                # 【错题逻辑：彻底压制】
                if len_ratio >= 1.0:
                    base_fail_score = -0.5
                else:
                    base_fail_score = 0.05 if (x3 >= 0.55) else -0.1
                
                length_penalty = 1.5 * len_ratio
                reward = base_fail_score - length_penalty
                
                if is_migrated_outside:
                    reward += 0.1  
                
                if not has_think_close:
                    if is_early_exit_cheat:
                        reward -= 1.0  # 错题还敢早退，罪加一等
                    else:
                        reward -= 0.5  
                
                # 🚨 【核心红线隔离：强制降维打击】
                reward -= 0.5  
                if reward > -0.5:
                    reward = -0.5  # 斩断一切错题得高分的幻想

            reward = float(reward)
            rewards.append(reward)

            # -------- 步骤 3: 保持原有的 Debug 写入逻辑 --------
            if debug_group_rewards_path and (
                debug_group_rewards_max_calls <= 0 or call_index <= debug_group_rewards_max_calls
            ):
                has_kp = "**key points**" in clipped.lower()
                has_sol = "**solution**" in clipped.lower()
                has_final = ("**final answer**" in clipped.lower()) or ("\\boxed" in clipped.lower()) or ("答案" in clipped)
                prob_str = "" if prob is None else str(prob)
                sid = ids[idx] if idx < len(ids) else None
                
                import hashlib
                import os
                import json
                
                debug_rows.append(
                    {
                        "pid": os.getpid(),
                        "batch_call": call_index,
                        "gen_idx": idx,
                        "sample_id": sid,
                        "problem": prob_str,
                        "problem_md5": hashlib.md5(prob_str.encode("utf-8", errors="ignore")).hexdigest(),
                        "token_len": token_count,
                        "len_ratio": round(len_ratio, 6),
                        "ref_len": float(current_limit),
                        "correct": int(bool(ok)),
                        "has_kp": int(has_kp),
                        "has_sol": int(has_sol),
                        "has_final": int(has_final),
                        "has_think_close": int(has_think_close),
                        "reward": reward,
                        "model_output": clipped,
                    }
                )

        if debug_rows:
            import os
            import json
            debug_dir = os.path.dirname(debug_group_rewards_path)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
            with open(debug_group_rewards_path, "a", encoding="utf-8") as f:
                for row in debug_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return rewards

    return think_prune_reward_fn

def _extract_instruction_from_raw_prompt(raw_prompt: str) -> str:
    if not raw_prompt:
        return "Solve the problem."
    if "User:" in raw_prompt:
        return raw_prompt.split("User:", 1)[0].strip()
    return raw_prompt.strip()


def _format_prefill_prompt_qwen3(instruction: str, problem_text: str, prefill_prefix: str) -> str:
    actual_prefill = f"<think>\n{prefill_prefix}"
    return (
        f"<|im_start|>user\n{instruction}\n{problem_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{actual_prefill}"
    )


def _format_prefill_prompt_glmz1(
    instruction: str,
    problem_text: str,
    prefill_prefix: str,
    # system_prompt: str,
) -> str:
    actual_prefill = f"<think>\n{prefill_prefix}"
    user_content = f"{instruction}\n{problem_text}".strip() if instruction else problem_text.strip()
    return (
        "[gMASK]<sop>"
        # f"<|system|>\n{system_prompt.strip()}\n"
        f"<|user|>\n{user_content}\n"
        f"<|assistant|>\n{actual_prefill}"
    )


def _format_prefill_prompt_r1distill_qwen(
    instruction: str,
    problem_text: str,
    prefill_prefix: str,
    system_prompt: str,
) -> str:
    actual_prefill = f"<think>\n{prefill_prefix}"
    user_content = f"{instruction}\n{problem_text}".strip() if instruction else problem_text.strip()
    system_prefix = system_prompt.strip() if system_prompt else ""
    return (
        f"<｜begin▁of▁sentence｜>{system_prefix}"
        f"<｜User｜>{user_content}\n"
        f"<｜Assistant｜>{actual_prefill}"
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_name",
        default="/root/autodl-tmp/model/wangsijin/Qwen3-8B-7cat",
        help="HF model path",
    )
    p.add_argument(
        "--data_path",
        default="/root/autodl-tmp/PruneRL/datasets/thinkprune_700_vanguard_relaxed.json",
        help="JSON list dataset file (must include ground_truth & problem)",
    )
    p.add_argument(
        "--relaxed_hit_py",
        default="/root/autodl-tmp/all_results/evaluate_results.py",
        help="Path to evaluate_results.py (provides relaxed_hit)",
    )

    # prefill prompt format (model-dependent template)
    p.add_argument("--use_prefill", action="store_true", help="Enable chat prefill prompt formatting.")
    p.add_argument(
        "--prompt_style",
        type=str,
        choices=["qwen3", "glmz1", "r1distill_qwen"],
        default="qwen3",
        help="Prompt template style when --use_prefill is enabled. Default is qwen3.",
    )
    p.add_argument(
        "--prefill_prefix",
        default="**Theorems**：",
        help="Prefill prefix after ('\\n, e.g. **Theorems**：",
    )
    p.add_argument(
        "--instruction",
        default="",
        help="If set, overrides instruction; otherwise extracted from the raw dataset prompt prefix.",
    )
    p.add_argument(
        "--glm_system_prompt",
        default="",
        help="System prompt used only for --prompt_style glmz1.",
    )
    p.add_argument(
        "--r1_system_prompt",
        default="",
        help="Optional system prompt text prepended before <｜User｜> for --prompt_style r1distill_qwen.",
    )

    # curriculum / length
    p.add_argument("--current_limit", type=int, default=4096, help="L: max completion tokens for reward")
    p.add_argument("--num_generations", type=int, default=4)

    # generation cost controls
    p.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=0,
        help="If >0, cap generation length (<= current_limit recommended for speed)",
    )

    # speed / stability knobs
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Reuse the same rollouts for multiple optimizer steps (reduces regen frequency).",
    )
    p.add_argument("--learning_rate", type=float, default=1e-5)

    # sampling
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.05)

    # early stop (encourage EOS token in chat format)
    p.add_argument(
        "--append_im_end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --prompt_style qwen3, append an instruction to end with <|im_end|> to reduce runaway generations.",
    )
    p.add_argument(
        "--append_glm_eop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --prompt_style glmz1, append an instruction to end with <eop>.",
    )
    p.add_argument(
        "--append_r1_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --prompt_style r1distill_qwen, append an instruction to end with <｜end▁of▁sentence｜>.",
    )
    p.add_argument(
        "--glm_role_stop_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --prompt_style glmz1, add GLM role tokens to eos_token_id to avoid max-length truncation.",
    )
    p.add_argument(
        "--r1_role_stop_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --prompt_style r1distill_qwen, add R1 role tokens to eos_token_id to avoid max-length truncation.",
    )

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)

    # Early stopping (stop when current L is saturated)
    p.add_argument(
        "--early_stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable early stop when reward saturates and length is far below L.",
    )
    p.add_argument("--early_stop_window", type=int, default=3, help="Consecutive heavy-step window size.")
    p.add_argument("--early_stop_min_steps", type=int, default=8, help="Do not early-stop before this step.")
    p.add_argument("--early_stop_reward_mean", type=float, default=0.95)
    p.add_argument("--early_stop_reward_std", type=float, default=0.0)
    p.add_argument("--early_stop_clipped_ratio", type=float, default=0.02)
    p.add_argument(
        "--early_stop_mean_len_ratio",
        type=float,
        default=0.4,
        help="Stop if mean_length <= ratio * current_limit for all window steps.",
    )
    p.add_argument(
        "--debug_group_rewards_path",
        type=str,
        default="",
        help="If set, append per-generation reward details to this jsonl file.",
    )
    p.add_argument(
        "--debug_group_rewards_max_calls",
        type=int,
        default=0,
        help="If >0, only write first N reward_fn calls to debug jsonl.",
    )


    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"DATA_PATH not found: {args.data_path}")
    if not os.path.exists(args.relaxed_hit_py):
        raise FileNotFoundError(f"relaxed_hit_py not found: {args.relaxed_hit_py}")

    if args.num_generations < 2:
        raise ValueError("GRPO requires num_generations >= 2")

    # Important constraint from TRL: generation_batch_size must be divisible by num_generations.
    # By default generation_batch_size = per_device_train_batch_size * gradient_accumulation_steps (world_size=1).
    default_generation_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if default_generation_batch_size % args.num_generations != 0:
        raise ValueError(
            "Invalid config: per_device_train_batch_size * gradient_accumulation_steps must be divisible by "
            f"num_generations. Got {args.per_device_train_batch_size}*{args.gradient_accumulation_steps}="
            f"{default_generation_batch_size}, num_generations={args.num_generations}. "
            "Try gradient_accumulation_steps=4/8/12/16 when num_generations=4."
        )

    relaxed_hit = _load_relaxed_hit(args.relaxed_hit_py)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )


    current_limit = int(args.current_limit)
    gen_max_new_tokens = (
        int(args.gen_max_new_tokens)
        if args.gen_max_new_tokens and args.gen_max_new_tokens > 0
        else current_limit
    )

    output_dir = args.output_dir or (
        f"./grpo_thinkprune_L{current_limit}_gen{gen_max_new_tokens}_n{args.num_generations}_it{args.num_iterations}"
    )

    eos_token_id_for_generation = tokenizer.eos_token_id
    if args.use_prefill and args.prompt_style == "glmz1" and args.glm_role_stop_tokens:
        eos_ids = []

        if isinstance(tokenizer.eos_token_id, int):
            eos_ids.append(tokenizer.eos_token_id)
        elif isinstance(tokenizer.eos_token_id, (list, tuple)):
            eos_ids.extend([int(x) for x in tokenizer.eos_token_id])

        for tok in ["<eop>", "<|user|>", "<|observation|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)

        eos_ids = sorted(set(eos_ids))
        if eos_ids:
            eos_token_id_for_generation = eos_ids
            print(f"[GLM] eos_token_id for generation: {eos_token_id_for_generation}")

    if args.use_prefill and args.prompt_style == "r1distill_qwen" and args.r1_role_stop_tokens:
        eos_ids = []

        if isinstance(tokenizer.eos_token_id, int):
            eos_ids.append(tokenizer.eos_token_id)
        elif isinstance(tokenizer.eos_token_id, (list, tuple)):
            eos_ids.extend([int(x) for x in tokenizer.eos_token_id])

        for tok in ["<｜end▁of▁sentence｜>", "<｜User｜>", "<|EOT|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.append(tid)

        eos_ids = sorted(set(eos_ids))
        if eos_ids:
            eos_token_id_for_generation = eos_ids
            print(f"[R1] eos_token_id for generation: {eos_token_id_for_generation}")

    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        sync_ref_model=False,
        beta=0.05,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        max_completion_length=current_limit,
        generation_kwargs={
            "max_new_tokens": gen_max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "eos_token_id": eos_token_id_for_generation,
            "pad_token_id": tokenizer.pad_token_id,
        },
        remove_unused_columns=False,
    )

    train_dataset = load_dataset("json", data_files=args.data_path)["train"]

    # IMPORTANT: Different models can require different chat templates.
    if args.use_prefill:

        def _map_prompt(ex: dict):
            raw_prompt = ex.get("prompt", "")
            instruction = args.instruction.strip() or _extract_instruction_from_raw_prompt(raw_prompt)
            problem_text = ex.get("problem", "") or raw_prompt

            if args.prompt_style == "qwen3":
                if args.append_im_end and "<|im_end|>" not in instruction:
                    instruction = instruction.rstrip() + "\n\nAfter you provide the final answer, end the assistant message with <|im_end|>."
                ex["prompt"] = _format_prefill_prompt_qwen3(instruction, problem_text, args.prefill_prefix)
            elif args.prompt_style == "glmz1":
                if args.append_glm_eop and "<eop>" not in instruction:
                    instruction = instruction.rstrip() + "\n\nAfter you provide the final answer, end the assistant message with <eop>."
                ex["prompt"] = _format_prefill_prompt_glmz1(
                    instruction,
                    problem_text,
                    args.prefill_prefix,
                    # args.glm_system_prompt,
                )
            elif args.prompt_style == "r1distill_qwen":
                if args.append_r1_eos and "<｜end▁of▁sentence｜>" not in instruction:
                    instruction = (
                        instruction.rstrip()
                        + "\n\nAfter you provide the final answer, end the assistant message with <｜end▁of▁sentence｜>."
                    )
                ex["prompt"] = _format_prefill_prompt_r1distill_qwen(
                    instruction,
                    problem_text,
                    args.prefill_prefix,
                    args.r1_system_prompt,
                )
            else:
                raise ValueError(f"Unsupported prompt_style: {args.prompt_style}")
            return ex

        train_dataset = train_dataset.map(_map_prompt)

    reward_fn = think_prune_reward_fn_builder(
        tokenizer,
        relaxed_hit,
        current_limit,
        debug_group_rewards_path=args.debug_group_rewards_path,
        debug_group_rewards_max_calls=args.debug_group_rewards_max_calls,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            ThinkPruneLogCallback(current_limit),
            ThinkPruneEarlyStopCallback(
                current_limit=current_limit,
                enabled=args.early_stop,
                window=args.early_stop_window,
                min_steps=args.early_stop_min_steps,
                reward_mean_threshold=args.early_stop_reward_mean,
                reward_std_threshold=args.early_stop_reward_std,
                clipped_ratio_threshold=args.early_stop_clipped_ratio,
                mean_len_ratio_threshold=args.early_stop_mean_len_ratio,
            ),
        ],
    )

    trainer.train()


if __name__ == "__main__":
    main()
