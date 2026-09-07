import json
import re

import pandas as pd

from stageground.config import ExperimentArm, ExperimentConfig, ModelConfig
from stageground.evaluate import run_evaluation
from stageground.evaluation.records import from_jsonl
from stageground.extraction.extractors import Result
from stageground.extraction.schemas import RawExtraction, RawField

_TOKEN = {
    "T": re.compile(r"\bT[0-4X]\b"),
    "N": re.compile(r"\bN[0-3X]\b"),
    "M": re.compile(r"\bM[01X]\b"),
}


def fake_runner(arm: str, text: str, *, model_config=None) -> Result:
    """Deterministic, offline stand-in for extractors.run_one: no LLM call.
    Extracts a stage token from the text via regex if present, else abstains.
    Accepts (and ignores) model_config for interface compatibility with the
    real run_one, which run_evaluation always calls with model_config set."""
    fields = {}
    for target, pattern in _TOKEN.items():
        match = pattern.search(text)
        value = match.group(0) if match else None
        fields[f"{target}_stage"] = RawField(
            value=value,
            evidence=value,
            confidence="high" if value else "low",
            reason=None if value else "no basis in report",
        )
    return Result(extraction=RawExtraction(**fields), raw="{}", invalid=False)


def _synthetic_dataset(n=20):
    rows = []
    for i in range(n):
        has_t = i % 5 != 0
        has_n = i % 3 != 0
        has_m = i % 4 == 0  # M is rarer, like the real corpus
        t_val = f"T{(i % 4) + 1}" if has_t else None
        n_val = f"N{i % 3}" if has_n else None
        m_val = "M0" if has_m else None
        text_parts = []
        if t_val:
            text_parts.append(t_val)
        if n_val:
            text_parts.append(n_val)
        if m_val:
            text_parts.append(m_val)
        text = f"Report {i}: " + " ".join(text_parts) + " findings noted."
        rows.append({
            "patient_filename": f"TCGA-TEST-{i:03d}",
            "text": text,
            "gold_T": t_val,
            "gold_N": n_val,
            "gold_M": m_val,
        })
    return pd.DataFrame(rows)


def _config(tmp_path, arms, sample_size=8, seed=42):
    return ExperimentConfig(
        experiment_id="test_exp",
        arms=arms,
        sample_size=sample_size,
        seed=seed,
        model=ModelConfig(model="fake-model", temperature=0.0),
        prompt_version="v1",
        schema_version="v1",
        timestamp="2026-09-07T00:00:00+00:00",
    )


def test_run_evaluation_produces_expected_files(tmp_path):
    df = _synthetic_dataset(20)
    arms = [ExperimentArm.CONSTRAINED, ExperimentArm.CONSTRAINED_UNKNOWN, ExperimentArm.GROUNDED]
    config = _config(tmp_path, arms, sample_size=8)

    outdir = run_evaluation(config, df, output_root=tmp_path / "results", runner=fake_runner)

    assert outdir == tmp_path / "results" / "test_exp"
    for fname in (
        "config.json", "predictions.jsonl", "metrics.json",
        "metrics_by_target.json", "error_breakdown.json", "bootstrap.json",
    ):
        assert (outdir / fname).exists(), fname

    assert (outdir / "tables" / "overall.csv").exists()
    assert (outdir / "tables" / "T.csv").exists()
    assert (outdir / "figures" / "fig1a_accuracy_vs_span_unsupported.png").exists()
    assert (outdir / "figures" / "fig1b_accuracy_vs_semantic_unsupported.png").exists()
    assert (outdir / "figures" / "fig2_coverage_vs_semantic_supported_accuracy.png").exists()
    assert (outdir / "figures" / "fig3_error_breakdown_overall.png").exists()


def test_predictions_jsonl_has_expected_row_count(tmp_path):
    df = _synthetic_dataset(20)
    arms = [ExperimentArm.CONSTRAINED, ExperimentArm.GROUNDED]
    sample_size = 6
    config = _config(tmp_path, arms, sample_size=sample_size)

    outdir = run_evaluation(config, df, output_root=tmp_path / "results", runner=fake_runner)

    records = from_jsonl(outdir / "predictions.jsonl")
    assert len(records) == sample_size * len(arms) * 3  # 3 targets per case


def test_config_json_roundtrips_through_experiment_config(tmp_path):
    df = _synthetic_dataset(20)
    arms = [ExperimentArm.CONSTRAINED]
    config = _config(tmp_path, arms, sample_size=5)

    outdir = run_evaluation(config, df, output_root=tmp_path / "results", runner=fake_runner)

    payload = json.loads((outdir / "config.json").read_text())
    restored = ExperimentConfig.from_dict(payload)
    assert restored == config
    assert "sampling" in payload
    assert payload["sampling"]["n_sampled"] == 5
    assert "effective_model" in payload
    # "fake-model" isn't an OpenAI reasoning model -> no normalization needed
    assert payload["effective_model"] == payload["model"]


def test_metrics_json_has_one_entry_per_arm(tmp_path):
    df = _synthetic_dataset(20)
    arms = [ExperimentArm.CONSTRAINED, ExperimentArm.GROUNDED]
    config = _config(tmp_path, arms, sample_size=8)

    outdir = run_evaluation(config, df, output_root=tmp_path / "results", runner=fake_runner)

    metrics = json.loads((outdir / "metrics.json").read_text())
    assert set(metrics.keys()) == {"C_constrained", "D_grounded"}
    assert "accuracy" in metrics["C_constrained"]


def test_bootstrap_json_skips_missing_arms_without_crashing(tmp_path):
    df = _synthetic_dataset(20)
    # Only one of the three fixed comparison arms present -> every comparison
    # involving it should be skipped, not crash.
    arms = [ExperimentArm.ZERO_SHOT, ExperimentArm.FEW_SHOT]
    config = _config(tmp_path, arms, sample_size=6)

    outdir = run_evaluation(config, df, output_root=tmp_path / "results", runner=fake_runner)

    bootstrap = json.loads((outdir / "bootstrap.json").read_text())
    assert bootstrap["overall"]["accuracy"] == []  # no fixed-comparison arms present


def test_effective_model_config_is_passed_to_runner(tmp_path):
    from stageground.extraction.llm_client import resolve_effective_model_config

    received = []

    def spy_runner(arm, text, *, model_config=None):
        received.append(model_config)
        return fake_runner(arm, text, model_config=model_config)

    df = _synthetic_dataset(10)
    arms = [ExperimentArm.CONSTRAINED]
    config = _config(tmp_path, arms, sample_size=3)

    run_evaluation(config, df, output_root=tmp_path / "results", runner=spy_runner)

    expected = resolve_effective_model_config(config.model)
    assert len(received) == 3  # one call per sampled case (one arm)
    assert all(mc == expected for mc in received)


def test_config_json_records_both_requested_and_effective_model(tmp_path):
    df = _synthetic_dataset(10)
    config = ExperimentConfig(
        experiment_id="reasoning_test",
        arms=[ExperimentArm.CONSTRAINED],
        sample_size=3,
        seed=1,
        model=ModelConfig(model="gpt-5-mini", temperature=0.7),
        prompt_version="v1",
        schema_version="v1",
        timestamp="2026-09-07T00:00:00+00:00",
    )

    outdir = run_evaluation(config, df, output_root=tmp_path / "results", runner=fake_runner)

    payload = json.loads((outdir / "config.json").read_text())
    assert payload["model"]["temperature"] == 0.7  # requested, preserved verbatim
    assert payload["effective_model"]["temperature"] is None  # normalized: gpt-5 rejects non-default
    assert payload["effective_model"]["model"] == "gpt-5-mini"


def test_does_not_write_into_real_results_directory(tmp_path):
    df = _synthetic_dataset(10)
    arms = [ExperimentArm.CONSTRAINED]
    config = _config(tmp_path, arms, sample_size=4)
    run_evaluation(config, df, output_root=tmp_path / "isolated_results", runner=fake_runner)
    # nothing under the real repo results/ directory bears this experiment_id
    from pathlib import Path
    assert not (Path("results") / "test_exp").exists()
