# StageGround Ablation + Rigorous Evaluation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** This plan is being executed **inline, in the same session that wrote it**, by an engineer (Claude) who already has full repo context from direct inspection. Per-task code is pinned at the interface level (exact signatures, exact file paths, exact JSON/field names) so nothing drifts, but boilerplate test scaffolding is not repeated three times over — tests are written first per superpowers:test-driven-development, immediately before the implementation they check.

**Goal:** Turn StageGround from a pilot (n=200, 4 conflated arms, ad-hoc metrics) into a reproducible research pipeline with a clean 5-arm ablation (A/B/C/C+/D), a centralized metrics/error-taxonomy layer, deterministic scalable sampling, bootstrap CIs, audit tooling, and comparison tables/figures — without breaking the existing pilot scripts or overwriting existing results.

**Architecture:** Keep `scripts/01-06` and `results/{cases,metrics,audit}` from the v0 pilot completely untouched (they stay reproducible on their own). Add a new, parallel layer: `src/stageground/config.py` (arm registry) is the single source of truth for prompts/schema requirements per arm; `src/stageground/evaluation/*` gains small composable modules (metrics, error_taxonomy, sampling, bootstrap, mstage, audit, tables, plots) that operate on plain dicts/DataFrames so they need no LLM/API access and are fully unit-testable; a new `src/stageground/evaluate.py` CLI (`python -m stageground.evaluate`) wires sampling → extraction (reusing existing `extractors.run_one`) → metrics → standardized `results/<experiment_id>/` output. New arm keys (`A_zero_shot`, `B_few_shot`, `C_constrained`, `C_plus_unknown`, `D_grounded`) are added to the existing `ARMS` dict in `extractors.py` alongside the old letter keys (`A`/`B`/`C`/`D`), so old scripts keep working unmodified.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, pytest, matplotlib (new dep, evaluation-only), stdlib `random`/`statistics` for bootstrap (no scipy dependency added).

---

## Decisions pinned up front (so later tasks don't drift)

1. **Arm naming.** New `ExperimentArm(str, Enum)` values are exactly: `A_zero_shot`, `B_few_shot`, `C_constrained`, `C_plus_unknown`, `D_grounded`. These are added as **additional keys** in `stageground.extraction.extractors.ARMS` (which already maps `"A"→build_zero_shot_prompt` etc.) — old keys stay so `scripts/02_run_extraction.py --arm A` keeps working.
2. **`C_constrained` = old arm C** (`build_allowed_values_prompt`, unchanged). **`D_grounded` = old arm D** (`build_schema_guided_prompt`, unchanged). Only `C_plus_unknown` is genuinely new prompt text.
3. **"Evaluable case"** for a target (T/N/M) = ground truth for that target is not null. This is the denominator for `accuracy`, `abstention_rate`, `unsupported_rate` (variant 1), `supported_accuracy` (variant 1), `coverage`.
4. **`abstained`** = normalized prediction (`stageground.evaluation.normalize.norm_pred`) == `"unknown"`.
5. **`supported`** (boolean, used for Supported Accuracy) = `not abstained AND evidence is not None AND grounded(evidence, report_text)`. This is intentionally identical to the existing pilot's `unsupported_assertion_rate` logic (`asserted & ~is_grounded` in `scripts/03_score.py`) — the scientific definition already validated by the 92%-agreement audit is not changed. Arms with no evidence requirement (`C_constrained`) will show `supported=False` for essentially all assertions whenever the model omits evidence, which is the expected, intended signal, not a bug. `supported` is `None` (not `False`) when `abstained` is `True` — abstentions are neither supported nor unsupported, they're out of that population. This mirrors the existing audit/pilot semantics and keeps `unsupported_rate` well-defined.
6. **`evidence_span_found`** vs **`evidence_semantically_supports_prediction`** are reported separately (§3 of spec) and are *not* combined into `supported` — `supported` stays defined purely by `evidence_span_found` (item 5) to avoid silently changing a metric the pilot's audit already validated. `evidence_semantically_supports_prediction` is a second, independent, weaker heuristic (regex stage-token match inside the evidence string) documented as a proxy, not full NLU — consistent with "do not over-engineer a classifier."
7. **Error taxonomy is multi-label** (`list[str]`), computed by `classify_errors(...)` in `error_taxonomy.py`, one call per prediction record.
8. **Bootstrap:** stdlib-only (`random.Random(seed)`), resampling case indices with replacement, `n_boot=2000` default, percentile CI. Paired comparisons resample the *same* indices for both arms (same reports evaluated across arms — see spec §7).
9. **Results layout:** `results/<experiment_id>/{config.json, predictions.jsonl, metrics.json, metrics_by_target.json, error_breakdown.json, bootstrap.json, tables/*.csv, tables/*.md, figures/*.png}`. `experiment_id` = `f"{timestamp}_{model}_{seed}_n{sample_size}"` sanitized, or user-supplied `--experiment-id`.
10. **No new result overwrites old ones.** `results/cases/`, `results/metrics/summary.parquet`, `results/metrics/track_c.parquet`, `results/audit/*` (the v0 pilot outputs) are never written to by any new code.

---

## File Structure

New files:
- `src/stageground/config.py` — `ExperimentArm` enum, `ArmConfig` dataclass, `ARM_CONFIGS` registry, `ModelConfig` dataclass, `ExperimentConfig` dataclass (serializable).
- `src/stageground/evaluation/metrics.py` — per-target + aggregate metric functions, all operating on a `list[PredictionRecord]`.
- `src/stageground/evaluation/records.py` — `PredictionRecord` dataclass (the standardized per-prediction JSON schema from spec §8) + `build_record(...)` constructor + `to_jsonl` / `from_jsonl` helpers.
- `src/stageground/evaluation/error_taxonomy.py` — `classify_errors(record, text) -> list[str]`.
- `src/stageground/evaluation/sampling.py` — `stratified_sample(df, n, seed) -> (sample_df, sampling_summary_dict)`.
- `src/stageground/evaluation/bootstrap.py` — `bootstrap_ci(values, statistic, seed, n_boot) -> BootstrapResult`; `paired_bootstrap_compare(records_a, records_b, metric_fn, seed, n_boot) -> PairedBootstrapResult`.
- `src/stageground/evaluation/mstage_analysis.py` — `classify_m_evidence(text) -> str` (one of 4 categories) + `build_mstage_annotation_sheet(records, texts) -> list[dict]`.
- `src/stageground/evaluation/audit.py` — `build_audit_sheet(records, texts, n, seed) -> list[dict]` (JSONL rows with blank human fields) + `score_audit(audit_jsonl_path) -> dict` (percent agreement + Cohen's kappa).
- `src/stageground/evaluation/tables.py` — `build_comparison_table(records, target=None) -> pd.DataFrame` + `write_tables(records, outdir)`.
- `src/stageground/evaluation/plots.py` — `plot_accuracy_vs_unsupported`, `plot_coverage_vs_supported_accuracy`, `plot_error_breakdown`.
- `src/stageground/evaluate.py` — CLI entry point (`python -m stageground.evaluate`), the orchestrator.
- `tests/test_metrics.py`, `tests/test_error_taxonomy.py`, `tests/test_sampling.py`, `tests/test_bootstrap.py`, `tests/test_config.py`, `tests/test_records.py`, `tests/test_evidence_matching_edge_cases.py`, `tests/test_mstage_analysis.py`, `tests/test_audit.py`.

Modified files:
- `src/stageground/extraction/prompts.py` — add `build_constrained_unknown_prompt` (C_plus_unknown).
- `src/stageground/extraction/extractors.py` — extend `ARMS` dict with the 5 new enum-string keys (aliases), no behavior change to existing keys.
- `pyproject.toml` — add `matplotlib>=3.8` under a new `[project.optional-dependencies] analysis` extra (keeps core deps light; `dev` extra also pulls it in for tests that import plots).
- `README.md` — new sections per spec §13.

Untouched (explicitly out of scope): `scripts/01-06`, `src/stageground/api/*`, `src/stageground/demo/*`, `web/`, existing `results/`.

---

## Task 1: Centralize arm configuration (`config.py`)

**Files:**
- Create: `src/stageground/config.py`
- Modify: `src/stageground/extraction/prompts.py`
- Modify: `src/stageground/extraction/extractors.py`
- Test: `tests/test_config.py`

**Design:**

```python
# src/stageground/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable


class ExperimentArm(str, Enum):
    ZERO_SHOT = "A_zero_shot"
    FEW_SHOT = "B_few_shot"
    CONSTRAINED = "C_constrained"
    CONSTRAINED_UNKNOWN = "C_plus_unknown"
    GROUNDED = "D_grounded"


@dataclass(frozen=True)
class ArmConfig:
    arm: ExperimentArm
    legacy_key: str          # the old letter key in extractors.ARMS ("A".."D"); C_plus_unknown has no legacy key
    allow_unknown: bool      # is "unknown" a sanctioned output for this arm?
    requires_evidence: bool  # must non-unknown predictions carry a verbatim evidence span?
    encourages_unknown: bool # does the prompt explicitly tell the model to prefer unknown over guessing?
    prompt_version: str
    schema_version: str = "v1"
    description: str = ""


ARM_CONFIGS: dict[ExperimentArm, ArmConfig] = {
    ExperimentArm.ZERO_SHOT: ArmConfig(
        arm=ExperimentArm.ZERO_SHOT, legacy_key="A", allow_unknown=True,
        requires_evidence=False, encourages_unknown=False, prompt_version="v1",
        description="Free-form zero-shot, JSON shape only, no allowed-value list, no rules.",
    ),
    ExperimentArm.FEW_SHOT: ArmConfig(
        arm=ExperimentArm.FEW_SHOT, legacy_key="B", allow_unknown=True,
        requires_evidence=False, encourages_unknown=False, prompt_version="v1",
        description="Zero-shot + worked examples (one abstention example), no explicit rules.",
    ),
    ExperimentArm.CONSTRAINED: ArmConfig(
        arm=ExperimentArm.CONSTRAINED, legacy_key="C", allow_unknown=True,
        requires_evidence=False, encourages_unknown=False, prompt_version="v1",
        description="Allowed-value list enforced; no evidence rule; no abstention encouragement beyond the value list including 'unknown'.",
    ),
    ExperimentArm.CONSTRAINED_UNKNOWN: ArmConfig(
        arm=ExperimentArm.CONSTRAINED_UNKNOWN, legacy_key=None, allow_unknown=True,
        requires_evidence=False, encourages_unknown=True, prompt_version="v1",
        description="Same as C_constrained, plus an explicit instruction that 'unknown' is preferred over guessing when the report does not support a stage. No mandatory evidence span.",
    ),
    ExperimentArm.GROUNDED: ArmConfig(
        arm=ExperimentArm.GROUNDED, legacy_key="D", allow_unknown=True,
        requires_evidence=True, encourages_unknown=True, prompt_version="v1",
        description="Allowed values + explicit unknown rule + mandatory verbatim evidence span for every non-unknown prediction.",
    ),
}


@dataclass(frozen=True)
class ModelConfig:
    model: str
    temperature: float | None = None   # None = provider default (see llm_client.py note on gpt-5/o-series)
    max_tokens: int | None = None
    retries: int = 3


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    arms: list[ExperimentArm]
    sample_size: int
    seed: int
    model: ModelConfig
    prompt_version: str
    schema_version: str
    timestamp: str  # ISO 8601, set at construction

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arms"] = [a.value for a in self.arms]
        d["model"] = asdict(self.model)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            experiment_id=d["experiment_id"],
            arms=[ExperimentArm(a) for a in d["arms"]],
            sample_size=d["sample_size"],
            seed=d["seed"],
            model=ModelConfig(**d["model"]),
            prompt_version=d["prompt_version"],
            schema_version=d["schema_version"],
            timestamp=d["timestamp"],
        )
```

`prompts.py` addition:

```python
def build_constrained_unknown_prompt(report_text: str) -> tuple[str, str]:  # C_plus_unknown
    system = (
        "You extract TNM pathologic staging from pathology reports.\n"
        "Allowed values (use EXACTLY one per field):\n"
        "  T: T0 T1 T2 T3 T4 TX unknown\n"
        "  N: N0 N1 N2 N3 NX unknown\n"
        "  M: M0 M1 MX unknown\n"
        "If the report does not clearly support a specific stage value, you should "
        "output \"unknown\" rather than guess. 'unknown' is a normal, encouraged answer "
        "when the evidence is not there — it is not a failure.\n"
        + _SHAPE
    )
    return system, f"Pathology report:\n\n{report_text}"
```

`extractors.py` addition (append to the existing `ARMS` dict, do not remove old keys):

```python
from stageground.config import ARM_CONFIGS, ExperimentArm
from stageground.extraction.prompts import build_constrained_unknown_prompt

ARMS.update({
    ExperimentArm.ZERO_SHOT.value: build_zero_shot_prompt,
    ExperimentArm.FEW_SHOT.value: build_few_shot_prompt,
    ExperimentArm.CONSTRAINED.value: build_allowed_values_prompt,
    ExperimentArm.CONSTRAINED_UNKNOWN.value: build_constrained_unknown_prompt,
    ExperimentArm.GROUNDED.value: build_schema_guided_prompt,
})
```

- [ ] Write `tests/test_config.py`: `ExperimentArm` has exactly 5 members with the spec's string values; every `ExperimentArm` member has an `ArmConfig` in `ARM_CONFIGS`; `ArmConfig.requires_evidence` is `True` only for `GROUNDED`; `ExperimentConfig.to_dict()`/`from_dict()` round-trips (construct one, serialize, deserialize, assert equality field-by-field).
- [ ] Implement `config.py`, edit `prompts.py`, edit `extractors.py` as above.
- [ ] Run `uv run pytest tests/test_config.py -v` — expect PASS. Run full `uv run pytest -q` — expect all old + new tests green (confirms `extractors.ARMS` extension didn't break `scripts/02_run_extraction.py`'s `--arm` choices).
- [ ] Commit: `git add src/stageground/config.py src/stageground/extraction/prompts.py src/stageground/extraction/extractors.py tests/test_config.py && git commit -m "Add centralized ExperimentArm config and C_plus_unknown arm"`

## Task 2: Standardized prediction record (`records.py`)

**Files:**
- Create: `src/stageground/evaluation/records.py`
- Test: `tests/test_records.py`

**Design:** exact JSON shape from spec §8:

```python
@dataclass
class PredictionRecord:
    case_id: str
    arm: str                 # ExperimentArm.value
    target: str               # "T" | "N" | "M"
    ground_truth: str | None  # canonicalized gold, or None if no gold
    prediction: str            # canonicalized value or "unknown" or "INVALID"
    evidence: str | None
    correct: bool | None       # None if ground_truth is None (not evaluable)
    abstained: bool
    supported: bool | None     # None when abstained (see plan decision 5)
    errors: list[str]
    raw_model_output: dict

def build_record(*, case_id, arm, target, ground_truth, raw_value, evidence, report_text) -> PredictionRecord: ...
def to_jsonl(records: list[PredictionRecord], path) -> None: ...
def from_jsonl(path) -> list[PredictionRecord]: ...
```

`build_record` normalizes `raw_value` via `stageground.evaluation.normalize.norm_pred`, computes `abstained`, `correct` (only if `ground_truth is not None`), `supported` per decision 5 using `stageground.evaluation.normalize.grounded`, and calls `error_taxonomy.classify_errors` (Task 4) to fill `errors`. `raw_model_output` stores the full un-normalized dict (`{"value":..., "evidence":..., "confidence":..., "reason":...}`) — never discarded (spec §8).

- [ ] Write `tests/test_records.py` covering: ground truth present + correct + supported; ground truth present + incorrect; no ground truth (`correct is None`); abstained (`supported is None`, `correct` computed against `"unknown"` only if gold is also unknown — normally `False`); malformed/`None` raw_value defaults to abstained; JSONL round-trip preserves all fields including nested `raw_model_output`.
- [ ] Implement `records.py` (depends on `error_taxonomy.py` — implement a minimal `classify_errors` stub first if sequencing requires it, then flesh out fully in Task 4; simplest is to do Task 4 before finishing this task's implementation step, see reordering note below).
- [ ] Run `uv run pytest tests/test_records.py -v` — PASS.
- [ ] Commit.

*(Reordering note: implement Task 4's `error_taxonomy.py` first, since `records.py` calls it. Tasks are listed in spec-requested order; execute 4 before 2's implementation step, tests can still be written in this order.)*

## Task 3: Metrics module (`metrics.py`)

**Files:**
- Create: `src/stageground/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Design:** every function takes `records: list[PredictionRecord]` (already filtered to one arm/target by the caller, or an explicit `target` filter param) and returns a `dict` or `float`. Names match spec exactly:

```python
def accuracy(records: list[PredictionRecord]) -> float:
    """correct predictions / evaluable ground-truth cases"""

def abstention_rate(records: list[PredictionRecord]) -> float:
    """unknown predictions / evaluable cases"""

def coverage(records: list[PredictionRecord]) -> float:
    """1 - abstention_rate"""

def unsupported_rate_over_evaluable(records) -> float:
    """unsupported predictions / all evaluable cases"""

def unsupported_rate_over_asserted(records) -> float:
    """unsupported predictions / non-abstained predictions"""

def supported_accuracy_over_evaluable(records) -> float:
    """supported_correct / all evaluable cases"""

def supported_accuracy_over_asserted(records) -> float:
    """supported_correct / non-abstained cases"""

def allowed_value_compliance(records, allowed_values: set[str]) -> float:
    """fraction of raw (pre-canonicalization) values in the allowed domain (existing pilot semantics, ported)"""

def evidence_span_found_rate(records) -> float:
    """fraction of non-abstained predictions whose evidence string is a verbatim (OCR-noise-tolerant) substring of the report"""

def evidence_semantic_support_rate(records, texts: dict[str, str]) -> float:
    """fraction of non-abstained predictions whose evidence contains a stage token canonicalizing to the prediction (heuristic proxy, see module docstring)"""

def compute_all_metrics(records, *, target: str | None = None) -> dict:
    """Bundles every metric above + n_evaluable, n_asserted, n_total for one target
    (or aggregated across all three if target is None). This is what evaluate.py
    calls to build metrics.json / metrics_by_target.json."""
```

All denominators of 0 return `float("nan")`, never raise or divide-by-zero crash — required by spec §12 edge cases ("all predictions unknown", "no valid ground truth").

`evidence_semantic_support_rate` needs report text per case_id (records don't carry the report text, only evidence/prediction), hence the `texts: dict[str, str]` param — reuse the same regex-token approach as `source_boundary.text_supports_gold`, applied to the `evidence` substring rather than the full report, checking whether any token found *within the evidence string* canonicalizes to `record.prediction`.

- [ ] Write `tests/test_metrics.py` with the required edge cases from spec §12: all-unknown (accuracy/supported_accuracy over evaluable = 0, abstention_rate = 1, coverage = 0, `unsupported_rate_over_asserted` = nan since denominator 0), no-abstentions (coverage = 1), no-valid-ground-truth (accuracy = nan, `n_evaluable` = 0), prediction-correct-but-unsupported (accuracy counts it, supported_accuracy does not), prediction-incorrect-but-evidence-found (evidence_span_found_rate counts it, accuracy does not), mixed realistic set with hand-computed expected values for every metric.
- [ ] Implement `metrics.py`.
- [ ] Run `uv run pytest tests/test_metrics.py -v` — PASS.
- [ ] Commit.

## Task 4: Error taxonomy (`error_taxonomy.py`)

**Files:**
- Create: `src/stageground/evaluation/error_taxonomy.py`
- Test: `tests/test_error_taxonomy.py`

**Design:**

```python
def classify_errors(
    *, ground_truth: str | None, prediction: str, evidence: str | None,
    report_text: str, raw_value: object, schema_valid: bool,
) -> list[str]:
    """Returns zero or more of:
    hallucinated_stage, wrong_stage_with_supporting_evidence, evidence_span_not_found,
    evidence_does_not_support_prediction, missed_explicit_stage, over_abstention,
    invalid_normalization, invalid_schema_output, other
    Multiple flags may coexist (spec §4)."""
```

Rules (each independently checked, order doesn't matter since it's a set):
- `invalid_schema_output`: `not schema_valid` (the raw model JSON failed Pydantic shape validation upstream).
- `invalid_normalization`: `prediction == "INVALID"` (raw value present but `canonicalize()` raised — a real, non-null value that isn't a recognizable stage token).
- abstention-only checks (only when `prediction == "unknown"`):
  - `over_abstention`: `ground_truth is not None and text_supports_gold-style check says report_text contains an explicit token canonicalizing to ground_truth` (reuse `stageground.evaluation.source_boundary.text_supports_gold`).
  - `missed_explicit_stage`: same condition as `over_abstention` but framed as "the report had an explicit, findable stage token and the model still abstained" — **this is deliberately the same underlying check as `over_abstention`**; both flags are set together when it fires (distinguishes "clinically named" over_abstention from a more specific "explicit token literally present" case for M-stage analysis in Task 8). Document this overlap explicitly in the docstring so it isn't mistaken for a bug.
- assertion-only checks (only when `prediction not in ("unknown", "INVALID")`):
  - `evidence_span_not_found`: `evidence is None or not grounded(evidence, report_text)`.
  - `evidence_does_not_support_prediction`: evidence *is* grounded (span found) but the semantic heuristic (Task 3's token check) says the evidence text doesn't canonicalize to `prediction`.
  - `hallucinated_stage`: `ground_truth is not None and prediction != ground_truth and "evidence_span_not_found" in flags` — asserted a value with zero textual basis and it happens to be wrong vs. gold.
  - `wrong_stage_with_supporting_evidence`: `ground_truth is not None and prediction != ground_truth and "evidence_span_not_found" not in flags` — asserted a *grounded* value that still disagrees with gold (candidate registry-discordance case, same spirit as existing `report_gold_discordance` Track-C tag).
- `other`: fallback when `prediction not in ("unknown","INVALID")` and none of the above fired but `ground_truth is not None and prediction != ground_truth` — should be rare; keep it so the taxonomy always has a bucket per spec's "other".

- [ ] Write `tests/test_error_taxonomy.py`: one test per flag firing in isolation (minimal fixture), one test asserting multiple flags coexist for a single record (hallucinated M1 with no evidence, wrong vs gold → both `hallucinated_stage` and `evidence_span_not_found`), one test for schema-invalid input, one test for a fully correct+supported record → `errors == []`.
- [ ] Implement `error_taxonomy.py`.
- [ ] Run `uv run pytest tests/test_error_taxonomy.py -v` — PASS.
- [ ] Now finish Task 2 (`records.py` implementation calling this), run `uv run pytest tests/test_records.py -v` — PASS.
- [ ] Commit both.

## Task 5: Deterministic + stratified sampling (`sampling.py`)

**Files:**
- Create: `src/stageground/evaluation/sampling.py`
- Test: `tests/test_sampling.py`

**Design:**

```python
@dataclass(frozen=True)
class SamplingSummary:
    total_reports: int
    n_sampled: int
    n_with_T: int
    n_with_N: int
    n_with_M: int
    n_with_all_TNM: int
    seed: int

    def to_dict(self) -> dict: ...


def stratified_sample(
    df: pd.DataFrame, n: int, seed: int, *,
    gold_cols: tuple[str, str, str] = ("gold_T", "gold_N", "gold_M"),
) -> tuple[pd.DataFrame, SamplingSummary]:
    """Deterministic (seeded) sample of size min(n, len(df)).
    Stratifies on the label-availability pattern (which subset of T/N/M gold is
    present for a row -> 8 strata incl. 'none'), proportionally allocating n
    across the strata that are actually present, using a numpy Generator(seed)
    for both stratum allocation and within-stratum sampling so it's exactly
    reproducible. Falls back to plain random sampling if n >= len(df) (returns
    the whole df, still summarized)."""
```

`n` is always a caller-supplied parameter — never hardcoded (spec §2). The summary is computed over the **input `df`** (i.e., "total number of reports" = the pool sampling drew from, not the raw 9,523-report corpus, since evaluate.py may already be working from a pre-filtered pool) — `evaluate.py` documents which pool it passes in.

- [ ] Write `tests/test_sampling.py`: same `(df, n, seed)` called twice yields identical sample (row order + content) — determinism; different seeds yield different samples (for a large-enough df); `n >= len(df)` returns everything without error; a df with only some rows having M gold still returns *some* M-labeled rows if any exist (stratification isn't accidentally dropping the rare stratum) unless `n` is too small to allocate one row to every stratum, in which case verify no crash; `SamplingSummary` counts match manual `notna().sum()` on the returned sample.
- [ ] Implement `sampling.py`.
- [ ] Run `uv run pytest tests/test_sampling.py -v` — PASS.
- [ ] Commit.

## Task 6: Bootstrap CIs (`bootstrap.py`)

**Files:**
- Create: `src/stageground/evaluation/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Design:**

```python
@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_boot: int
    seed: int
    def to_dict(self) -> dict: ...

@dataclass(frozen=True)
class PairedBootstrapResult:
    metric_name: str
    arm_a: str
    arm_b: str
    diff: float          # metric(arm_a) - metric(arm_b)
    ci_low: float
    ci_high: float
    p_value: float | None  # two-sided empirical p from the bootstrap diff distribution
    n_boot: int
    seed: int
    n_paired_cases: int
    def to_dict(self) -> dict: ...


def bootstrap_ci(values: list[float], *, seed: int, n_boot: int = 2000, ci: float = 0.95) -> BootstrapResult:
    """values is a list of 0/1 (or float) per-case indicators for one metric
    (e.g. per-case 'correct'). Resamples indices with replacement n_boot times,
    recomputes the mean, returns percentile CI. NaNs in `values` are dropped
    before resampling (documented); if fewer than 2 usable values, returns a
    BootstrapResult with ci_low=ci_high=point_estimate (degenerate, not a crash)."""


def paired_bootstrap_compare(
    values_a: dict[str, float], values_b: dict[str, float], *,
    metric_name: str, arm_a: str, arm_b: str, seed: int, n_boot: int = 2000,
) -> PairedBootstrapResult:
    """values_a/values_b are {case_id: per-case metric value} for the SAME metric
    on the SAME reports under two arms. Only case_ids present in both dicts are
    used (paired). Resamples the shared case_id list with replacement (same
    resampled indices applied to both arms per bootstrap iteration -- this is
    what makes it 'paired' per spec §7), computes diff of means each iteration,
    returns percentile CI on the diff + empirical two-sided p-value (fraction of
    bootstrap diffs on the opposite side of 0 from the observed diff, doubled,
    capped at 1.0)."""
```

- [ ] Write `tests/test_bootstrap.py`: `bootstrap_ci` on constant values (all 1.0) returns point=1.0, ci=(1.0,1.0); on `[0,1]*50` returns point≈0.5 with a sane CI width; degenerate single-value input doesn't crash; `paired_bootstrap_compare` on identical `values_a == values_b` returns `diff≈0` and CI straddling 0; on `values_a` all better than `values_b` returns a positive diff with CI not straddling 0 and small p-value; mismatched case_id sets are intersected correctly (`n_paired_cases` reflects the intersection); output shape test — every field of `BootstrapResult`/`PairedBootstrapResult` is JSON-serializable via `to_dict()`.
- [ ] Implement `bootstrap.py`.
- [ ] Run `uv run pytest tests/test_bootstrap.py -v` — PASS.
- [ ] Commit.

## Task 7: M-stage analysis (`mstage_analysis.py`)

**Files:**
- Create: `src/stageground/evaluation/mstage_analysis.py`
- Test: `tests/test_mstage_analysis.py`

**Design:** reproducible, regex-based, explicitly a heuristic first pass (spec §5: "do not over-engineer a classifier initially").

```python
M_CATEGORY_EXPLICIT_TOKEN = "explicit_m_token"          # e.g. 'pM1', 'M0' literally present
M_CATEGORY_METASTATIC_DESCRIBED = "metastatic_described_no_token"  # words like 'metastatic', 'metastasis' present, no explicit M-token
M_CATEGORY_NO_EVIDENCE = "no_m_evidence"                 # neither
M_CATEGORY_AMBIGUOUS = "ambiguous_insufficient"          # both signals present but conflicting, or token present but not clearly M (reserved for manual override)

def classify_m_evidence(report_text: str) -> str:
    """Regex-only classification into one of the four M_CATEGORY_* constants."""

def build_mstage_annotation_sheet(
    records: list[PredictionRecord], texts: dict[str, str], *, target: str = "M",
) -> list[dict]:
    """One row per M-target record: case_id, arm, report_excerpt (text truncated
    around the first M-relevant match or first 1000 chars), automated category
    (classify_m_evidence), prediction, ground_truth, supported, blank
    'human_category' and 'notes' fields for manual override -- reproducible
    annotation format per spec §5, not a trained classifier."""
```

- [ ] Write `tests/test_mstage_analysis.py`: text with `"pM1"` → `explicit_m_token`; text with `"metastatic disease noted"` and no token → `metastatic_described_no_token`; plain text with neither → `no_m_evidence`; text with both an M-token AND separately worded metastatic language that could read either way → `ambiguous_insufficient` (pick one concrete conflicting example and pin the expected category); `build_mstage_annotation_sheet` output has all required keys and `human_category`/`notes` are empty strings.
- [ ] Implement `mstage_analysis.py`.
- [ ] Run tests — PASS.
- [ ] Commit.

## Task 8: Audit infrastructure (`audit.py`)

**Files:**
- Create: `src/stageground/evaluation/audit.py`
- Test: `tests/test_audit.py`

**Design:** JSONL (fits the existing project's JSON-heavy `results/cases/*.json` convention better than CSV for nested/optional fields).

```python
def build_audit_sheet(
    records: list[PredictionRecord], texts: dict[str, str], *, n: int, seed: int,
) -> list[dict]:
    """Deterministic (seeded) sample of n records (or all, if fewer). Each row:
    case_id, arm, target, report_excerpt (full text if <=4000 chars else
    truncated + '[...truncated for audit]'), ground_truth, prediction, evidence,
    automated_supported (records[i].supported), automated_errors (records[i].errors),
    human_supported: None, human_evidence_correct: None, human_prediction_correct: None,
    human_error_type: None, reviewer_notes: \"\".
    Blank human_* fields use JSON null / empty string, ready for a reviewer to
    fill in and re-save as JSONL."""

def write_audit_jsonl(rows: list[dict], path) -> None: ...
def read_audit_jsonl(path) -> list[dict]: ...

def score_audit(rows: list[dict]) -> dict:
    """Compares automated_supported (bool) vs human_supported (bool, after
    filling), and automated 'correct' proxy vs human_prediction_correct, on
    rows where the human field is filled (not None). Returns:
    {'n_scored': int, 'percent_agreement_supported': float,
     'cohens_kappa_supported': float | None, ...same two keys for
     'prediction_correct' if that human field is populated}.
    Uses sklearn.metrics.cohen_kappa_score (already a project dependency).
    Returns None for kappa if fewer than 2 rows or a category has zero variance
    (kappa undefined) rather than raising."""
```

- [ ] Write `tests/test_audit.py`: `build_audit_sheet` respects `n` and is seed-deterministic (two calls, same seed, identical case_id order); all rows have the 5 blank human_* keys; `score_audit` on rows with `human_supported` unfilled (`None`) for all → `n_scored == 0`, no crash; on rows with perfect agreement → 100% / kappa==1.0 (or `None` if human labels are constant, tested as an explicit degenerate case); on rows with one disagreement out of N → percent agreement matches hand count.
- [ ] Implement `audit.py`.
- [ ] Run tests — PASS.
- [ ] Commit.

## Task 9: Comparison tables (`tables.py`)

**Files:**
- Create: `src/stageground/evaluation/tables.py`
- Test: covered by `tests/test_metrics.py` extension (`test_tables.py` if it grows large — start inline, split out only if the file exceeds ~150 lines)

**Design:**

```python
def build_comparison_table(records: list[PredictionRecord], *, target: str | None = None) -> pd.DataFrame:
    """One row per arm. Columns: arm, n_evaluable, accuracy, supported_accuracy_over_evaluable,
    unsupported_rate_over_evaluable, abstention_rate, coverage. target=None aggregates
    across T/N/M (pools records); otherwise filters to records where record.target == target."""

def write_tables(records: list[PredictionRecord], outdir: Path) -> None:
    """Writes overall.csv/.md, T.csv/.md, N.csv/.md, M.csv/.md into outdir/tables/,
    using DataFrame.to_csv and DataFrame.to_markdown (needs 'tabulate' -- check
    if already available; if not, hand-roll a minimal markdown table writer to
    avoid adding a new dependency for this alone)."""
```

- [ ] Check whether `tabulate` is importable in the current env (`uv run python -c "import tabulate"`); if not, implement a small `_to_markdown(df) -> str` helper in `tables.py` instead of depending on `DataFrame.to_markdown`, to honor "avoid unnecessary frameworks" (spec §14).
- [ ] Write test: build a small synthetic `records` list spanning 2 arms x 3 targets with known correctness/support, call `build_comparison_table(records, target="T")`, assert exact expected numbers per arm; call with `target=None` and assert row count == number of arms and values equal the pooled computation.
- [ ] Implement `tables.py`.
- [ ] Run tests — PASS.
- [ ] Commit.

## Task 10: Figures (`plots.py`)

**Files:**
- Create: `src/stageground/evaluation/plots.py`
- Modify: `pyproject.toml` (add `matplotlib` to `dev`/new `analysis` extra)
- Test: `tests/test_plots.py` (smoke test only — assert the function runs and produces a non-empty PNG file; do not assert on pixel content)

**Design:**

```python
def plot_accuracy_vs_unsupported(table_by_arm: pd.DataFrame, outpath: Path) -> None:
    """Scatter, one point per arm (arm label annotated next to point).
    x = unsupported_rate_over_evaluable, y = accuracy."""

def plot_coverage_vs_supported_accuracy(table_by_arm: pd.DataFrame, outpath: Path) -> None:
    """x = coverage, y = supported_accuracy_over_evaluable."""

def plot_error_breakdown(records: list[PredictionRecord], outpath: Path, *, target: str | None = None) -> None:
    """Grouped bar chart: x = error category (from error_taxonomy flags, counted
    per arm), grouped/colored by arm. target=None -> overall; target='M' -> M-only
    subset, only called by evaluate.py when M sample size is large enough
    (>=10 evaluable M records) to be worth a dedicated figure, per spec §10."""
```

- [ ] Add `matplotlib>=3.8` to `pyproject.toml` under `[project.optional-dependencies] analysis = ["matplotlib>=3.8"]`, and add `"stageground[analysis]"`-equivalent by listing `matplotlib` in `dev` too (`dev = ["pytest>=8.0", "ruff>=0.5", "matplotlib>=3.8"]`) so `uv sync --extra dev` covers tests that import `plots.py`.
- [ ] Run `uv sync --extra dev` to pull in matplotlib.
- [ ] Write `tests/test_plots.py`: synthetic 2-arm table/records, call each plot function writing to a tmp_path, assert file exists and size > 0. Use `matplotlib.use("Agg")` at module import time in `plots.py` to avoid needing a display backend in CI.
- [ ] Implement `plots.py`.
- [ ] Run tests — PASS.
- [ ] Commit.

## Task 11: Evaluation orchestrator (`evaluate.py`)

**Files:**
- Create: `src/stageground/evaluate.py`
- Test: `tests/test_evaluate_cli.py` (uses a fake/injected extraction function — **no live API calls**, per spec §12)

**Design:** CLI shape from spec §13 example, adapted to the repo's actual `data/processed/dataset.parquet` input:

```bash
uv run python -m stageground.evaluate \
  --arms A_zero_shot C_constrained C_plus_unknown D_grounded \
  --sample-size 200 \
  --seed 42 \
  --model gpt-4o \
  --dataset data/processed/dataset.parquet \
  --pool all-three   # or 'any' -- which rows are eligible before stratified sampling
```

Internals:
1. Load `--dataset` parquet (same shape as `scripts/01_build_dataset.py` output: `patient_filename, text, gold_T, gold_N, gold_M, easy_T, easy_N, easy_M, cancer_type`).
2. Filter to the eligibility pool (`all-three` = has all of T/N/M gold, matching the existing `01b_sample_eval.py` behavior; `any` = has at least one).
3. `sampling.stratified_sample(pool_df, n=args.sample_size, seed=args.seed)` → sample df + `SamplingSummary`.
4. For each arm in `args.arms`, for each row in the sample: call `stageground.extraction.extractors.run_one(arm, row.text)` (this is the one place that touches the LLM — injectable via a `runner: Callable[[str, str], Result] = run_one` parameter on the orchestrator function so tests substitute a deterministic fake).
5. Build a `PredictionRecord` per (row, arm, target in T/N/M) via `records.build_record`.
6. Write `results/<experiment_id>/predictions.jsonl` (via `records.to_jsonl`).
7. Compute `metrics.compute_all_metrics` overall and per-target → `metrics.json`, `metrics_by_target.json`.
8. Compute error breakdown (count of each taxonomy flag per arm per target) → `error_breakdown.json`.
9. Compute `bootstrap.paired_bootstrap_compare` for the fixed set of comparisons spec §7 asks for (`D_grounded vs C_constrained`, `C_plus_unknown vs C_constrained`, `D_grounded vs C_plus_unknown`) on `accuracy`, `supported_accuracy_over_evaluable`, `unsupported_rate_over_evaluable`, `abstention_rate`, `coverage` — skip any comparison where one of the two arms wasn't in `args.arms` (log a note, don't crash) → `bootstrap.json`.
10. `tables.write_tables(...)` → `results/<experiment_id>/tables/`.
11. `plots.plot_*(...)` → `results/<experiment_id>/figures/` (M-specific error plot only if `n_evaluable_M >= 10`).
12. Write `config.json` from an `ExperimentConfig` built from the CLI args + `datetime.now(UTC).isoformat()`.

Refactor `run_arm`-style extraction into a small internal function `_extract_for_sample(sample_df, arm, runner) -> list[dict]` so the orchestrator function itself (`run_evaluation(config: ExperimentConfig, dataset_df, *, runner=run_one) -> Path`) is the unit tested entry point, and `main()` (argparse) is a thin wrapper — this is what makes the CLI testable without hitting the API.

- [ ] Write `tests/test_evaluate_cli.py`: build a small synthetic `dataset_df` (10-20 rows with T/N/M gold in various availability patterns + report text containing/missing explicit tokens), a fake `runner(arm, text) -> Result` returning canned `RawExtraction`-like results (reuse `stageground.extraction.extractors.Result`), call `run_evaluation(...)` with `sample_size` smaller than the pool, assert: output directory created under a tmp `results/` path (parametrize the output root so the test doesn't write into the real `results/`), all 6 expected files exist (`config.json`, `predictions.jsonl`, `metrics.json`, `metrics_by_target.json`, `error_breakdown.json`, `bootstrap.json`) plus `tables/overall.csv` and at least one figure PNG, `config.json` round-trips through `ExperimentConfig.from_dict`, `predictions.jsonl` line count == `sample_size * len(arms) * 3` (3 targets).
- [ ] Implement `evaluate.py`.
- [ ] Run `uv run pytest tests/test_evaluate_cli.py -v` — PASS.
- [ ] Run full suite `uv run pytest -q` — confirm all green, no regressions.
- [ ] Commit.

## Task 12: README updates

**Files:**
- Modify: `README.md`

- [ ] Add a "Research question" section restating the source-grounding-vs-apparent-accuracy question (spec §13), reusing language already in the repo's DESIGN.md/CONTEXT.md rather than inventing new claims.
- [ ] Add an "Experimental arms" section documenting A/B/C/C+/D with the exact distinctions from Task 1's `ArmConfig.description` values (single source of truth — quote them, don't restate differently).
- [ ] Add a "Metrics" section defining every metric's exact denominator, copied from the docstrings in `metrics.py` (again: one definition, not two).
- [ ] Add "Running an experiment" with the real CLI from Task 11 (200-case pilot, 1000-case, 2000+, full dataset — just `--sample-size` changes; document that `--sample-size` larger than the eligible pool clamps to the pool size, per `sampling.stratified_sample`'s documented behavior).
- [ ] Add "Reproducing analysis" showing `tables.write_tables` / `plots.plot_*` are already invoked by `evaluate.py`, and how to regenerate them standalone from an existing `predictions.jsonl` (small snippet: `records.from_jsonl(...)` → `tables.write_tables(...)`).
- [ ] Note explicitly that `scripts/01-06` and the v0 pilot results are unchanged and still the source of the README's existing A/B/C/D results table — the new A_zero_shot/B_few_shot/C_constrained/C_plus_unknown/D_grounded pipeline is a separate, additive experiment track for the ablation study.
- [ ] Commit.

---

## Self-review checklist (performed after drafting, before execution)

- Spec coverage: §1 arm config → Task 1; §2 scalable sampling → Task 5 + Task 11 step 3; §3 metrics → Task 3; §4 taxonomy → Task 4; §5 M-stage → Task 7; §6 audit → Task 8; §7 bootstrap → Task 6 + Task 11 step 9; §8 standardized outputs → Task 2 + Task 11; §9 tables → Task 9; §10 figures → Task 10; §11 experimental control → Task 1 (`ModelConfig`) + Task 11 (`config.json` records everything); §12 tests → one test file per task above, all API-free; §13 docs → Task 12; §14 philosophy → decisions section + "untouched" list; §15 order → tasks follow the same order (with the 4↔2 reorder noted explicitly for a real dependency, not a whim).
- No placeholders: every task has concrete file paths, exact function signatures, and either full code (Task 1) or a fully specified interface + rules precise enough to implement directly (Tasks 2-11) — chosen over restating full code twice (plan + implementation) given inline same-session execution.
- Type consistency: `PredictionRecord` fields (Task 2) are the single vocabulary used verbatim by `metrics.py` (Task 3), `error_taxonomy.py` (Task 4), `tables.py` (Task 9), `plots.py` (Task 10), and `evaluate.py` (Task 11) — no renamed duplicates introduced.
