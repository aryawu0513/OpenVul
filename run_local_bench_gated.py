#!/usr/bin/env python3
"""
run_local_bench_gated.py — OpenVul run with baseline gating.

For each slug:
  1) Run baseline (buggy) sample.
  2) If baseline is correctly flagged (tp), run context_aware variants.
  3) Otherwise, skip context_aware for that slug (broken for this auditor).
"""
import argparse
from pathlib import Path

from OpenVul.run_local_bench import _eval_one_dataset  # noqa: E402


def _load_flag(item: dict) -> str | None:
    flags = item.get("sample_flags")
    if flags:
        return flags[0]
    return item.get("flag")


def baseline_ok(results: list[dict]) -> bool:
    for item in results:
        flag = _load_flag(item)
        if flag is None:
            continue
        return flag == "tp"
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Root containing baseline/ and context_aware/ dirs.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root to write results under {mode}/C/NPD/{category}/.")
    parser.add_argument("--slugs", nargs="+", required=True,
                        help="Slug IDs (e.g. 069A7F404506).")
    parser.add_argument("--mode", default="npd", choices=["npd", "generic"])
    parser.add_argument("--model", default="Leopo1d/OpenVul-Qwen3-4B-GRPO")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

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

    base_out = args.output_root / args.mode / "C" / "NPD" / "baseline"
    ctx_out = args.output_root / args.mode / "C" / "NPD" / "context_aware"

    for slug in args.slugs:
        variant = f"repository_{slug}"
        base_path = args.dataset_root / "baseline" / variant
        ctx_path = args.dataset_root / "context_aware" / variant

        print(f"\n=== {variant} | baseline ===")
        if not base_path.exists():
            print(f"  SKIP (missing): {base_path}")
            continue
        base_res = _eval_one_dataset(
            llm, tokenizer, sampling_params,
            str(base_path), args.mode, str(base_out), variant, args.save
        )
        if not baseline_ok(base_res):
            print("  SKIP context_aware (baseline not tp)")
            continue

        print(f"=== {variant} | context_aware ===")
        if not ctx_path.exists():
            print(f"  SKIP (missing): {ctx_path}")
            continue
        _eval_one_dataset(
            llm, tokenizer, sampling_params,
            str(ctx_path), args.mode, str(ctx_out), variant, args.save
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
