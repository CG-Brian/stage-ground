"""Evaluation orchestrator CLI (spec §13):

    uv run python -m stageground.evaluate \
        --arms A_zero_shot C_constrained C_plus_unknown D_grounded \
        --sample-size 200 --seed 42 --model gpt-4o

Wires sampling -> extraction -> standardized records -> metrics -> tables ->
figures -> bootstrap comparisons -> results/<experiment_id>/. The one place
that touches the LLM is the `runner` parameter of `run_evaluation` (defaults
to `stageground.extraction.extractors.run_one`); tests substitute a
deterministic fake so the whole pipeline is exercised with no API calls.

`config.model` (an `ExperimentConfig`'s `ModelConfig`) is what actually
reaches every extraction call -- see `resolve_effective_model_config` below
and `stageground.extraction.llm_client`/`extractors.run_one`. Both the
as-requested and as-resolved ("effective") model configuration are recorded
in `config.json`, so a run's config file can never silently diverge from
what was actually sent to the provider.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Callable

import pandas as pd

from stageground.config import ExperimentArm, ExperimentConfig, ModelConfig
from stageground.evaluation.bootstrap import paired_bootstrap_compare
from stageground.evaluation.metrics import compute_all_metrics
from stageground.evaluation.plots import (
    plot_accuracy_vs_semantic_unsupported,
    plot_accuracy_vs_span_unsupported,
    plot_coverage_vs_semantic_supported_accuracy,
    plot_error_breakdown,
)
from stageground.evaluation.records import PredictionRecord, build_record, to_jsonl
from stageground.evaluation.sampling import stratified_sample
from stageground.evaluation.tables import build_comparison_table, write_tables
from stageground.extraction.extractors import Result, run_one
from stageground.extraction.llm_client import resolve_effective_model_config

TARGETS = ("T", "N", "M")

Runner = Callable[..., Result]  # (arm: str, report_text: str, *, model_config: ModelConfig | None) -> Result

# Fixed comparisons spec §7 asks for; skipped (not crashed on) if either arm
# wasn't included in the run.
BOOTSTRAP_COMPARISONS = [
    (ExperimentArm.GROUNDED, ExperimentArm.CONSTRAINED),
    (ExperimentArm.CONSTRAINED_UNKNOWN, ExperimentArm.CONSTRAINED),
    (ExperimentArm.GROUNDED, ExperimentArm.CONSTRAINED_UNKNOWN),
]

BOOTSTRAP_METRICS = [
    "accuracy",
    "semantic_supported_accuracy_over_evaluable",
    "span_unsupported_rate_over_evaluable",
    "semantic_unsupported_rate_over_evaluable",
    "abstention_rate",
    "coverage",
]

MIN_M_FOR_DEDICATED_FIGURE = 10


def _per_case_value(rec: PredictionRecord, metric_name: str) -> float | None:
    """Per-case 0/1 indicator underlying an aggregate metric, or None if this
    record isn't evaluable (no ground truth) -- used only for bootstrap
    resampling, never for the point-estimate metrics themselves (see metrics.py)."""
    if rec.ground_truth is None:
        return None
    if metric_name == "accuracy":
        return 1.0 if rec.correct else 0.0
    if metric_name == "abstention_rate":
        return 1.0 if rec.abstained else 0.0
    if metric_name == "coverage":
        return 0.0 if rec.abstained else 1.0
    if metric_name == "span_unsupported_rate_over_evaluable":
        return 1.0 if rec.evidence_span_found is False else 0.0
    if metric_name == "semantic_unsupported_rate_over_evaluable":
        return 1.0 if (
            rec.evidence_span_found is False or rec.evidence_semantically_supports_prediction is False
        ) else 0.0
    if metric_name == "semantic_supported_accuracy_over_evaluable":
        return 1.0 if (
            rec.correct and rec.evidence_span_found and rec.evidence_semantically_supports_prediction
        ) else 0.0
    raise ValueError(f"unknown bootstrap metric: {metric_name}")


def _per_case_values(
    records: list[PredictionRecord], arm: str, target: str | None, metric_name: str
) -> dict[str, float]:
    subset = [r for r in records if r.arm == arm and (target is None or r.target == target)]
    out = {}
    for r in subset:
        v = _per_case_value(r, metric_name)
        if v is None:
            continue
        key = r.case_id if target is not None else f"{r.case_id}|{r.target}"
        out[key] = v
    return out


def _build_bootstrap_json(
    records: list[PredictionRecord], arms_present: set[ExperimentArm], seed: int
) -> dict:
    result: dict = {}
    for scope in (None, "T", "N", "M"):
        scope_key = "overall" if scope is None else scope
        result[scope_key] = {}
        for metric_name in BOOTSTRAP_METRICS:
            comparisons = []
            for arm_a, arm_b in BOOTSTRAP_COMPARISONS:
                if arm_a not in arms_present or arm_b not in arms_present:
                    continue
                values_a = _per_case_values(records, arm_a.value, scope, metric_name)
                values_b = _per_case_values(records, arm_b.value, scope, metric_name)
                cmp = paired_bootstrap_compare(
                    values_a, values_b, metric_name=metric_name,
                    arm_a=arm_a.value, arm_b=arm_b.value, seed=seed,
                )
                comparisons.append(cmp.to_dict())
            result[scope_key][metric_name] = comparisons
    return result


def run_evaluation(
    config: ExperimentConfig,
    dataset_df: pd.DataFrame,
    *,
    output_root: str | Path = "results",
    runner: Runner = run_one,
    gold_cols: tuple[str, str, str] = ("gold_T", "gold_N", "gold_M"),
) -> Path:
    """Run one experiment: sample, extract (via `runner`), score, and write
    the standardized results/<experiment_id>/ layout (spec §8). Returns the
    output directory.

    `config.model` is resolved to an "effective" configuration up front
    (`resolve_effective_model_config`) -- e.g. normalizing an explicit
    temperature that a reasoning model would reject -- and that resolved
    config, not `config.model` directly, is what's passed to every `runner`
    call. Both are recorded in config.json (`model` = as requested,
    `effective_model` = as actually used) so nothing is silently substituted.
    """
    effective_model = resolve_effective_model_config(config.model)
    if effective_model != config.model:
        print(
            f"NOTE: normalized model config for '{config.model.model}' "
            f"(requested temperature={config.model.temperature!r} -> "
            f"effective temperature={effective_model.temperature!r}); "
            "both are recorded in config.json."
        )

    sample_df, sampling_summary = stratified_sample(
        dataset_df, n=config.sample_size, seed=config.seed, gold_cols=gold_cols
    )

    all_records: list[PredictionRecord] = []
    for row in sample_df.itertuples():
        case_id = row.patient_filename
        gold = {}
        for i, target in enumerate(TARGETS):
            v = getattr(row, gold_cols[i])
            gold[target] = None if pd.isna(v) else v

        for arm in config.arms:
            result = runner(arm.value, row.text, model_config=effective_model)
            for target in TARGETS:
                if result.invalid or result.extraction is None:
                    rec = build_record(
                        case_id=case_id, arm=arm.value, target=target,
                        ground_truth=gold[target], raw_output=None,
                        report_text=row.text, schema_valid=False,
                    )
                else:
                    field = getattr(result.extraction, f"{target}_stage")
                    raw_output = {
                        "value": field.value, "evidence": field.evidence,
                        "confidence": field.confidence, "reason": field.reason,
                    }
                    rec = build_record(
                        case_id=case_id, arm=arm.value, target=target,
                        ground_truth=gold[target], raw_output=raw_output,
                        report_text=row.text, schema_valid=True,
                    )
                all_records.append(rec)

    outdir = Path(output_root) / config.experiment_id
    outdir.mkdir(parents=True, exist_ok=True)

    config_payload = config.to_dict()
    config_payload["effective_model"] = asdict(effective_model)
    config_payload["sampling"] = sampling_summary.to_dict()
    (outdir / "config.json").write_text(json.dumps(config_payload, indent=2))

    to_jsonl(all_records, outdir / "predictions.jsonl")

    metrics_json = {
        arm.value: compute_all_metrics([r for r in all_records if r.arm == arm.value], target=None)
        for arm in config.arms
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    metrics_by_target_json = {
        arm.value: {
            target: compute_all_metrics([r for r in all_records if r.arm == arm.value], target=target)
            for target in TARGETS
        }
        for arm in config.arms
    }
    (outdir / "metrics_by_target.json").write_text(json.dumps(metrics_by_target_json, indent=2))

    error_breakdown: dict = {}
    for arm in config.arms:
        error_breakdown[arm.value] = {}
        for target in TARGETS:
            counts: Counter = Counter()
            for r in all_records:
                if r.arm == arm.value and r.target == target:
                    counts.update(r.errors)
            error_breakdown[arm.value][target] = dict(counts)
    (outdir / "error_breakdown.json").write_text(json.dumps(error_breakdown, indent=2))

    bootstrap_json = _build_bootstrap_json(all_records, set(config.arms), config.seed)
    (outdir / "bootstrap.json").write_text(json.dumps(bootstrap_json, indent=2))

    write_tables(all_records, outdir)

    overall_table = build_comparison_table(all_records, target=None)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_vs_span_unsupported(overall_table, figures_dir / "fig1a_accuracy_vs_span_unsupported.png")
    plot_accuracy_vs_semantic_unsupported(overall_table, figures_dir / "fig1b_accuracy_vs_semantic_unsupported.png")
    plot_coverage_vs_semantic_supported_accuracy(
        overall_table, figures_dir / "fig2_coverage_vs_semantic_supported_accuracy.png"
    )
    plot_error_breakdown(all_records, figures_dir / "fig3_error_breakdown_overall.png")

    m_evaluable = sum(1 for r in all_records if r.target == "M" and r.ground_truth is not None)
    if m_evaluable >= MIN_M_FOR_DEDICATED_FIGURE:
        plot_error_breakdown(all_records, figures_dir / "fig3_error_breakdown_M.png", target="M")

    return outdir


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
    ap.add_argument("--pool", choices=["all-three", "any"], default="all-three",
                     help="'all-three' = rows with T,N,M gold all present; 'any' = at least one")
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--output-root", default="results")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    gold_cols = ["gold_T", "gold_N", "gold_M"]
    df = df[df[gold_cols].notna().all(axis=1) if args.pool == "all-three" else df[gold_cols].notna().any(axis=1)].copy()

    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_id = args.experiment_id or _default_experiment_id(
        args.model, args.seed, args.sample_size, timestamp
    )

    config = ExperimentConfig(
        experiment_id=_sanitize_experiment_id(experiment_id),
        arms=[ExperimentArm(a) for a in args.arms],
        sample_size=args.sample_size,
        seed=args.seed,
        model=ModelConfig(
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            response_format=args.response_format,
        ),
        prompt_version="v1",
        schema_version="v1",
        timestamp=timestamp,
    )

    outdir = run_evaluation(config, df, output_root=args.output_root)
    print(f"wrote experiment results -> {outdir}")


if __name__ == "__main__":
    main()
