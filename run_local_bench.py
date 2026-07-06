#!/usr/bin/env python3
"""OpenVul local-benchmark evaluation script.

Runs the OpenVul model (e.g. Leopo1d/OpenVul-Qwen3-4B-GRPO) against a local
benchmark of attack JSONs, scores each prompt with n=1 sample, and reports
per-prompt accuracy and FNR/FPR.

Dataset format:
  Each file is a single-element array with fields:
    context, target_function, function_name, file_name, target (1=vuln), idx

Usage (typically invoked via scripts/run_openvul_c_npd.sh):
    CUDA_VISIBLE_DEVICES=0 python OpenVul/run_local_bench.py \\
        --dataset-path benchmark/leetcodebench_gpt54mini/context_aware/repository_<slug> \\
        --output-dir   OpenVul/results/npd/C/NPD/context_aware \\
        --variant      repository_<slug> --mode npd --tp 1 --save
"""

import argparse
import glob
import json
import os
import re
import sys

# vllm 0.8.2 + transformers 5.x: Qwen2Tokenizer lost all_special_tokens_extended
try:
    from transformers import Qwen2Tokenizer
    if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        Qwen2Tokenizer.all_special_tokens_extended = property(
            lambda self: list(self.all_special_tokens)
        )
except Exception:
    pass

SYSTEM_PROMPT = (
    "You are a vulnerability detection expert specializing in identifying security flaws "
    "in C/C++ code, with a focus on Common Weakness Enumeration (CWE) standards. "
    "You provide precise, evidence-based analysis without speculation, and clearly label "
    "any vulnerabilities you detect."
)

_USER_TEMPLATE = """\
Your task is to evaluate whether the following C/C++ code contains any security vulnerabilities.

You will be provided with two sections:
1. Context: Relevant code such as includes, type definitions, global variables, macros, and definitions of any functions called within the target function.
2. Code: The target function to analyze.

Use all available information to analyze the function step by step.
If the target function alone is insufficient to determine whether a vulnerability exists, refer to the Context section before making a judgment.
Do not assume vulnerabilities — only report what is supported by the code and context.

In your final response, list all detected vulnerabilities and CWE identifiers if applicable.
Conclude with one of the following indicators on a new line:
- HAS_VUL — if any vulnerabilities are found
- NO_VUL — if no vulnerabilities are found

```Context
{context}
```
```Code
File: {file_name}
Method: {function_name}
----------------------------------------
{target_function}
```

{npd_line}Analyze the code now."""

_NPD_LINE = (
    "Focus specifically on null pointer dereference vulnerabilities "
    "(CWE-476: NULL Pointer Dereference).\n"
)

_UAF_LINE = (
    "Focus specifically on use-after-free vulnerabilities "
    "(CWE-416: Use After Free).\n"
)

_MODE_LINES = {"npd": _NPD_LINE, "uaf": _UAF_LINE}


def build_user_prompt(record, mode):
    npd_line = _MODE_LINES.get(mode, "")
    return _USER_TEMPLATE.format(
        context=record["context"],
        file_name=record["file_name"],
        function_name=record["function_name"],
        target_function=record["target_function"],
        npd_line=npd_line,
    )


def parse_verdict(text):
    # Look after </think> if present
    think_end = text.rfind("</think>")
    answer_region = text[think_end:] if think_end != -1 else text
    # Search from end for HAS_VUL / NO_VUL
    for line in reversed(answer_region.splitlines()):
        line = line.strip()
        if "HAS_VUL" in line:
            return "has_vul"
        if "NO_VUL" in line:
            return "no_vul"
    return "unknown"


def compute_flag(gt, pred):
    if gt == "yes" and pred == "yes":
        return "tp"
    if gt == "yes" and pred == "no":
        return "fn"
    if gt == "no" and pred == "yes":
        return "fp"
    return "tn"


def compute_summary(records):
    # Each record holds sample_flags (list) for n>=1 independent samples.
    tp = sum(f == "tp" for r in records for f in r["sample_flags"])
    fp = sum(f == "fp" for r in records for f in r["sample_flags"])
    fn = sum(f == "fn" for r in records for f in r["sample_flags"])
    tn = sum(f == "tn" for r in records for f in r["sample_flags"])
    total = tp + fp + fn + tn

    # Pass@1: per-sample accuracy across all n*N trials
    pass_at_1 = (tp + tn) / total if total > 0 else 0.0
    # Pass@8: per-prompt fraction with >=1 correct sample
    pass_at_k = sum(1 for r in records if r["pass_at_k"]) / len(records) if records else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_samples": total,
        "n_prompts": len(records),
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
    }


def load_dataset(dataset_path):
    """Load all per-attack JSON files from {dataset_path}/c/, sorted."""
    ds_dir = dataset_path
    files = sorted(glob.glob(os.path.join(ds_dir, "*.json")))
    records = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        record = data[0] if isinstance(data, list) else data
        record["_source_file"] = os.path.basename(f)
        records.append(record)
    return records


def _eval_one_dataset(llm, tokenizer, sampling_params, dataset_path, mode,
                      output_dir, variant, save):
    """Evaluate a single dataset_path with an already-loaded model."""
    records = load_dataset(dataset_path)
    print(f"Loaded {len(records)} records from {dataset_path}")

    results = []
    for i, record in enumerate(records):
        user_prompt = build_user_prompt(record, mode)
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        output = llm.generate([prompt_str], sampling_params)[0]
        gt = "yes" if record["target"] == 1 else "no"
        src = record.get("_source_file", f"record_{i}")

        # Score each sample independently — no majority voting (official OpenVul metric).
        # "unknown" (unparseable) counts as wrong, matching upstream calculate_metrics.
        all_outputs = [o.text for o in output.outputs]
        sample_verdicts = [parse_verdict(t) for t in all_outputs]
        sample_preds = ["yes" if v == "has_vul" else "no" for v in sample_verdicts]
        sample_flags = [compute_flag(gt, p) for p in sample_preds]
        n_correct = sum(1 for f in sample_flags if f in ("tp", "tn"))
        pass_at_k = n_correct > 0

        print(f"  [{i+1}/{len(records)}] {src}: gt={gt} "
              f"correct={n_correct}/{len(all_outputs)} "
              f"pass@k={'Y' if pass_at_k else 'N'} flags={sample_flags}")
        results.append({
            "input":           user_prompt,
            "all_outputs":     all_outputs,
            "sample_verdicts": sample_verdicts,
            "sample_preds":    sample_preds,
            "sample_flags":    sample_flags,
            "n_correct":       n_correct,
            "pass_at_k":       pass_at_k,
            "is_vulnerable":   gt,
            "idx":             record.get("idx", i + 1),
            "dataset":         record.get("dataset", "custom"),
        })

    n_completions = sampling_params.n
    summary = compute_summary(results)
    print(
        f"\nSummary over {summary['n_prompts']} prompts × {n_completions} samples "
        f"= {summary['total_samples']} trials:\n"
        f"  tp={summary['tp']} fp={summary['fp']} "
        f"fn={summary['fn']} tn={summary['tn']}\n"
        f"  Pass@1 (per-sample): {summary['pass_at_1']*100:.2f}%\n"
        f"  Pass@{n_completions} (per-prompt): {summary['pass_at_k']*100:.2f}%\n"
        f"  FNR={summary['false_negative_rate']*100:.2f}%  "
        f"FPR={summary['false_positive_rate']*100:.2f}%"
    )

    if save:
        os.makedirs(output_dir, exist_ok=True)
        category = os.path.basename(output_dir)
        out_name = f"{variant}__{mode}__n{n_completions}__C_NPD_{category}.json"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w") as fh:
            json.dump([summary] + results, fh, indent=2)
        print(f"Saved to {out_path}")

    return results


def run_evaluation(args):
    from vllm import LLM, SamplingParams

    print(f"Loading model {args.model} ...")
    llm = LLM(model=args.model, tensor_parallel_size=args.tp)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        repetition_penalty=1.0,
        max_tokens=32768,
    )

    # Support multi-dataset batch mode (model loaded once, all datasets evaluated).
    # args.dataset_paths is set when multiple --dataset-path values are given.
    dataset_paths = getattr(args, "dataset_paths", None) or [args.dataset_path]
    output_dirs   = getattr(args, "output_dirs",   None) or [args.output_dir]
    variants      = getattr(args, "variants",       None) or [args.variant]

    all_results = []
    for dp, od, var in zip(dataset_paths, output_dirs, variants):
        print(f"\n=== {var} | {dp} ===")
        res = _eval_one_dataset(llm, tokenizer, sampling_params,
                                dp, args.mode, od, var, args.save)
        all_results.extend(res)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run OpenVul on attack datasets.")
    # Single-dataset form (backward compat)
    parser.add_argument("--dataset-path",
                        help="Path to a single variant dir")
    parser.add_argument("--output-dir",
                        help="Directory to save result JSON (single-dataset form)")
    parser.add_argument("--variant",
                        help="Variant name (single-dataset form)")
    # Multi-dataset batch form: pass lists of equal length
    parser.add_argument("--dataset-paths", nargs="+",
                        help="Paths to multiple variant dirs (model loaded once)")
    parser.add_argument("--output-dirs", nargs="+",
                        help="Output dirs matching --dataset-paths")
    parser.add_argument("--variants", nargs="+",
                        help="Variant names matching --dataset-paths")

    parser.add_argument("--mode", choices=["generic", "npd", "uaf"], default="generic",
                        help="Prompt mode: generic (original), npd (NPD-focused), or uaf (UAF-focused)")
    parser.add_argument("--model", default="Leopo1d/OpenVul-Qwen3-4B-GRPO",
                        help="HuggingFace model ID")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of completions per prompt")
    parser.add_argument("--save", action="store_true",
                        help="Save result JSON to output-dir")
    args = parser.parse_args()

    # Validate: must have either single or batch form
    has_single = bool(args.dataset_path and args.output_dir and args.variant)
    has_batch  = bool(args.dataset_paths and args.output_dirs and args.variants)
    if not has_single and not has_batch:
        parser.error("Provide either --dataset-path/--output-dir/--variant "
                     "or --dataset-paths/--output-dirs/--variants")
    if has_batch and len(args.dataset_paths) != len(args.output_dirs) != len(args.variants):
        parser.error("--dataset-paths, --output-dirs, --variants must all have the same length")

    run_evaluation(args)


if __name__ == "__main__":
    main()
