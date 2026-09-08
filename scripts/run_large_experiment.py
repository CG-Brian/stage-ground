"""Thin CLI for a large (e.g. 1,000-case) ablation run with resumable,
concurrent extraction (`stageground.extraction.checkpoint`), then the SAME
tested post-processing pipeline as the pilot (`stageground.evaluate.run_evaluation`).

    uv run python scripts/run_large_experiment.py \
        --arms A_zero_shot C_constrained C_plus_unknown D_grounded \
        --sample-size 1000 --seed 42 --model gpt-4o \
        --max-workers 10 --n-boot 5000 \
        --experiment-id 20260908T000000_gpt-4o_seed42_n1000

Add --dry-run to print the preflight summary (case/arm/request counts,
requested-vs-effective model config) and exit without any API calls.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stageground.config import ExperimentArm, ExperimentConfig, ModelConfig
from stageground.evaluate import run_evaluation
from stageground.extraction.checkpoint import (
    CheckpointedRunner,
    build_text_keyed_cache,
    run_extraction_with_checkpoint,
)
from stageground.extraction.llm_client import resolve_effective_model_config


def _sanitize_experiment_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", raw)


def _default_experiment_id(model: str, seed: int, sample_size: int, timestamp: str) -> str:
    stamp = timestamp.split(".")[0].replace(":", "").replace("-", "")
    return _sanitize_experiment_id(f"{stamp}_{model}_seed{seed}_n{sample_size}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True, choices=[a.value for a in ExperimentArm])
    ap.add_argument("--sample-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--response-format", default="json_object")
    ap.add_argument("--dataset", default="data/processed/dataset.parquet")
    ap.add_argument("--pool", choices=["all-three", "any"], default="all-three")
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--output-root", default="results")
    ap.add_argument("--checkpoint", default=None, help="path to the resumable extraction checkpoint jsonl")
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap replicates for paired arm comparisons")
    ap.add_argument("--dry-run", action="store_true", help="print the preflight summary and exit, no API calls")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    gold_cols = ["gold_T", "gold_N", "gold_M"]
    pool_df = df[
        df[gold_cols].notna().all(axis=1) if args.pool == "all-three" else df[gold_cols].notna().any(axis=1)
    ].copy()

    print("=== PREFLIGHT ===")
    print(f"eligible pool ('{args.pool}'): {len(pool_df)} cases")
    if len(pool_df) < args.sample_size:
        print(
            f"ERROR: requested --sample-size {args.sample_size} but only {len(pool_df)} eligible "
            f"cases exist in the '{args.pool}' pool. Stopping rather than silently reducing the "
            "sample size or eligibility criteria -- rerun with --sample-size <= "
            f"{len(pool_df)}, or --pool any if that's an acceptable relaxation."
        )
        return

    n_cases = args.sample_size
    n_arms = len(args.arms)
    n_requests = n_cases * n_arms
    print(f"cases requested: {n_cases}")
    print(f"arms ({n_arms}): {args.arms}")
    print(f"expected API requests (1 per case-arm; each call returns T+N+M together): {n_requests}")
    print(f"expected PredictionRecords (requests x 3 targets): {n_requests * 3}")
    print(f"model={args.model} temperature={args.temperature} max_tokens={args.max_tokens} retries={args.retries}")
    print(f"worst-case HTTP attempts if every request needed all retries: {n_requests * args.retries}")

    model_config = ModelConfig(
        model=args.model, provider=args.provider, temperature=args.temperature,
        max_tokens=args.max_tokens, retries=args.retries, response_format=args.response_format,
    )
    effective_model = resolve_effective_model_config(model_config)
    if effective_model == model_config:
        print("Confirmed: requested model config == effective model config.")
    else:
        print(f"NOTE: effective model config differs from requested.\n  requested: {model_config}\n  effective: {effective_model}")

    if args.dry_run:
        print("\n--dry-run: exiting before any API calls or checkpoint I/O.")
        return

    from stageground.evaluation.sampling import stratified_sample

    sample_df, _ = stratified_sample(pool_df, n=args.sample_size, seed=args.seed, gold_cols=tuple(gold_cols))

    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_id = _sanitize_experiment_id(
        args.experiment_id or _default_experiment_id(args.model, args.seed, args.sample_size, timestamp)
    )

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else Path(args.output_root) / "_checkpoints" / f"{experiment_id}.jsonl"
    )
    print(f"\nextraction checkpoint: {checkpoint_path}")

    cache_by_case = run_extraction_with_checkpoint(
        sample_df, args.arms, model_config=model_config,
        checkpoint_path=checkpoint_path, max_workers=args.max_workers,
    )
    n_invalid = sum(1 for r in cache_by_case.values() if r.invalid)
    print(f"\n=== EXTRACTION COMPLETE === total={len(cache_by_case)} invalid={n_invalid}")
    by_arm_invalid: dict[str, int] = {}
    for (_, arm), result in cache_by_case.items():
        if result.invalid:
            by_arm_invalid[arm] = by_arm_invalid.get(arm, 0) + 1
    if by_arm_invalid:
        print(f"invalid results by arm: {by_arm_invalid}")

    runner = CheckpointedRunner(build_text_keyed_cache(sample_df, cache_by_case))

    config = ExperimentConfig(
        experiment_id=experiment_id,
        arms=[ExperimentArm(a) for a in args.arms],
        sample_size=args.sample_size,
        seed=args.seed,
        model=model_config,
        prompt_version="v1",
        schema_version="v1",
        timestamp=timestamp,
    )

    outdir = run_evaluation(config, pool_df, output_root=args.output_root, runner=runner, n_boot=args.n_boot)
    print(f"\nwrote experiment results -> {outdir}")


if __name__ == "__main__":
    main()
