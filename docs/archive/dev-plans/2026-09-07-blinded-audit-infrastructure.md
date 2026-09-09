# Blinded Stratified Human Audit Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** Inline, same-session execution, same engineer as the prior two passes (ablation pipeline + correctness pass). This upgrades `stageground.evaluation.audit` from a single-file flat sheet into a blinded reviewer/key bundle with richer human fields, M-stage-aware sampling, and proper scoring — it does not touch `results/audit/` (the completely separate v0 pilot Milestone-1a audit, its own schema, its own scripts `05_audit_sample.py`/`06_audit_score.py`) or the 200-case pilot results already committed.

**Goal:** Let a human reviewer blindly, reproducibly audit a stratified (by target, with M-stage oversampling of the most diagnostic cases) sample of predictions, judging span-grounding, semantic support, and source sufficiency separately from correctness — with scoring that reports agreement, confusion matrices, and source-sufficiency/M0-prior analyses, all without ever requiring the reviewer to see the experimental arm or the automated judgment.

**Architecture:** New `stageground/evaluation/audit_sampling.py` owns arm-pattern classification and the stratified-with-M-oversampling sampler (pure functions over `list[PredictionRecord]`, no I/O). `stageground/evaluation/audit.py` keeps its existing (legacy, single-file) `build_audit_sheet`/`write_audit_jsonl`/`read_audit_jsonl`/`score_audit` completely unchanged for backward compatibility, and gains the new bundle-building functions (`build_reviewer_and_key`, `write_audit_bundle`) plus a `build`/`score` argparse CLI. New `stageground/evaluation/audit_scoring.py` owns the new scoring: per-field agreement + confusion matrix + source-sufficiency + M0-prior analyses, joining `reviewer.jsonl` and `key.jsonl` on `audit_id`.

**Tech Stack:** No new dependencies (stdlib `random`/`argparse`/`json`, existing `sklearn.metrics.cohen_kappa_score`).

---

## Decisions pinned up front

1. **Directory layout** (per spec): `<output>/{reviewer.jsonl, key.jsonl, config.json, INSTRUCTIONS.md}`, e.g. `audit/audit_001/`. `--output` is a plain CLI path argument — no hardcoded default location beyond whatever the caller passes (spec item 6: "do not hardcode... provide CLI/config support").
2. **Reviewer schema** (`reviewer.jsonl`, one row per audited `PredictionRecord`): `audit_id, target, report_excerpt, report_truncated, ground_truth (omitted in grounding-blind mode), prediction, evidence, human_evidence_span_found, human_semantic_support, human_prediction_correct, human_source_has_explicit_stage, human_source_has_inferential_evidence, human_source_sufficient_for_stage, human_confidence, human_error_type, reviewer_notes` — all `human_*` fields start `null`/`""` exactly like the spec's example. NO `arm`, NO `automated_*` fields, NO `case_id` (case_id is itself potentially identifying/searchable; only `audit_id` is reviewer-facing).
3. **Key schema** (`key.jsonl`, one row per audit_id, machine-only): `audit_id, case_id, arm, target, ground_truth, prediction, automated_evidence_span_found, automated_semantic_support, automated_errors, arm_pattern` (`arm_pattern` is the M-stage priority/comparison-arm classification, computed at build time so scoring never needs to re-load the original predictions.jsonl; `None` for T/N or when either priority/comparison arm is absent from the source predictions file).
4. **Two review modes** (spec item 10): `--mode grounding-blind` (default) omits `ground_truth` from `reviewer.jsonl` entirely (not just `null` — the key is absent) so a reviewer judging span/semantic/source-sufficiency isn't anchored by gold; `--mode with-gold` includes it, for a pass focused on `human_prediction_correct`. `key.jsonl` always has `ground_truth` regardless of mode (it's machine-only).
5. **`audit_id` format**: `f"A{i:04d}"` (`A0001`, `A0002`, ...), assigned in the FINAL shuffled row order (not sampling-pool order) so the sequence itself doesn't leak sampling-tier information.
6. **M-stage oversampling** (spec item 7): `audit_sampling.classify_arm_pattern` computes, per `case_id`, one of `both_predict`, `priority_predicts_comparison_abstains`, `priority_abstains_comparison_predicts`, `both_abstain`, `pattern_unknown` (either arm missing from the input records) using two configurable arm names (default `priority_arm="C_constrained"`, `comparison_arm="D_grounded"`). For the M quota: up to `m_priority_fraction` (default `0.5`) of the M quota is drawn from `priority_arm` records where pattern is `priority_predicts_comparison_abstains` (spec's single most informative category); of the remainder, up to `m_semantic_balance_fraction` (default `0.3`) each is drawn from the asserted-M pool split by `automated_semantic_support in (True, False)` (so both heuristic classes are represented, spec item 7's "avoid...only machine-flagged failures"); whatever's left is filled by uniform random draw from the remaining M pool. All draws are seeded and deterministic; under-filled tiers (small pools) simply yield fewer rows, never a crash, and the actual achieved composition is recorded in `config.json` for transparency.
7. **T/N sampling**: plain deterministic stratified random sample of `target_counts[target]` records from all `target`-matching records (any arm) — no special-case logic, since spec items 6-7 only call for oversampling *within M*.
8. **Report excerpt** (spec item 9): if `len(text) <= max_chars` (default `4000`) show the full report (`report_truncated=False`). Else, if `evidence` is present and locatable (case-insensitive substring search — this is for *display windowing*, not scoring, so it doesn't need `grounded()`'s OCR-noise tolerance) show a window of `evidence` plus `window_chars` (default `1500`) of surrounding context on each side, clipped to the text and prefixed/suffixed with `...` when clipped (`report_truncated=True`). Else (abstained / no evidence / evidence not locatable) fall back to the first `max_chars` characters (`report_truncated=True`) — a deterministic, documented strategy, not silently dropping potentially-relevant tail content without a signal.
9. **Legacy audit.py functions are untouched** — `build_audit_sheet`/`write_audit_jsonl`/`read_audit_jsonl`/`score_audit` keep their exact current signatures and behavior (already tested in `tests/test_audit.py`, which is not modified by this plan). `write_audit_jsonl`/`read_audit_jsonl` are also reused as-is for the new `reviewer.jsonl`/`key.jsonl` (they're already generic `list[dict] <-> jsonl`).
10. **Scoring join**: `score_reviewed_audit(reviewer_rows, key_rows)` joins on `audit_id` (a dict built from `key_rows`), then compares `human_evidence_span_found` vs `key.automated_evidence_span_found` and `human_semantic_support` vs `key.automated_semantic_support`, each: overall + per-target (T/N/M), reporting `n`, `percent_agreement`, `cohens_kappa`, and a confusion matrix (`tp/fp/tn/fn` with **automated as the predicted class, human as the reference/actual class** — spec item 13's framing, "is the regex heuristic under-detecting") plus derived `precision/recall/specificity`. Rows where either side is `None` are excluded from that specific comparison's `n`, exactly like the legacy scorer's null-skipping convention.
11. **Source-sufficiency + M0-prior analyses** are separate functions (`analyze_source_sufficiency`, `analyze_m0_prior_prediction`) in `audit_scoring.py`, both operating on the same joined reviewer+key rows — kept separate from `score_reviewed_audit` since they answer a different question (spec items 14-15) and so a caller can request only what they need.
12. **No fabricated labels, no new experiment run.** Nothing in this plan writes to any `human_*` field. No `evaluate.py` invocation.

---

## File Structure

New files:
- `src/stageground/evaluation/audit_sampling.py`
- `src/stageground/evaluation/audit_scoring.py`
- `tests/test_audit_sampling.py`
- `tests/test_audit_bundle.py` (new-schema build/blinding/reproducibility tests)
- `tests/test_audit_scoring.py`

Modified files:
- `src/stageground/evaluation/audit.py` — add `build_reviewer_and_key`, `write_audit_bundle`, `_report_excerpt`, `generate_audit_id`, CLI (`build`/`score` subcommands + `main()`). Legacy functions untouched (verified by not touching `tests/test_audit.py`, which must still pass unmodified).
- `README.md` — new "Blinded human audit" subsection with exact commands.

Untouched: `results/audit/*`, `scripts/05_audit_sample.py`, `scripts/06_audit_score.py`, `tests/test_audit.py`, the 200-case pilot results already committed.

---

## Task 1: Arm-pattern classification + stratified/M-oversampled audit sampler

**Files:** Create `src/stageground/evaluation/audit_sampling.py`; Test `tests/test_audit_sampling.py`

```python
# Pattern constants
BOTH_PREDICT = "both_predict"
PRIORITY_PREDICTS_COMPARISON_ABSTAINS = "priority_predicts_comparison_abstains"
PRIORITY_ABSTAINS_COMPARISON_PREDICTS = "priority_abstains_comparison_predicts"
BOTH_ABSTAIN = "both_abstain"
PATTERN_UNKNOWN = "pattern_unknown"

def classify_arm_pattern(
    case_records: dict[str, PredictionRecord], *, priority_arm: str, comparison_arm: str,
) -> str: ...

def stratified_audit_sample(
    records: list[PredictionRecord], *, target_counts: dict[str, int], seed: int,
    m_priority_arm: str = "C_constrained", m_comparison_arm: str = "D_grounded",
    m_priority_fraction: float = 0.5, m_semantic_balance_fraction: float = 0.3,
) -> tuple[list[PredictionRecord], dict]:
    """Returns (selected_records, sampling_report). sampling_report documents
    the ACHIEVED composition per target (and, for M, per tier) so config.json
    can record exactly what was drawn, not just what was requested."""
```

- [ ] Write `tests/test_audit_sampling.py`: `classify_arm_pattern` returns each of the 5 constants for constructed fixtures (both predict, priority predicts, comparison predicts, both abstain, one arm missing from the dict). `stratified_audit_sample` on synthetic T/N/M records: (a) same `(records, target_counts, seed)` twice → identical selected `(case_id, arm, target)` triples in the same order (determinism); (b) `target_counts={"T": 3, "N": 0, "M": 5}` returns exactly 3 T-records and 0 N-records; (c) construct an M pool where 4 cases are `priority_predicts_comparison_abstains`, request `target_counts={"M": 4}` with `m_priority_fraction=0.5` → at least 2 of the selected M records come from that tier (assert via `arm==m_priority_arm` and pattern check, recomputing pattern in the test); (d) construct an M pool with only `automated_semantic_support=True` records available in the general tier plus a couple `False` ones, request enough M quota to exhaust the priority tier, and assert BOTH `True` and `False` semantic-support records appear in the result if both exist in the pool; (e) requesting more than available in a tier doesn't crash, just under-fills (assert `len(selected) <= sum(target_counts.values())` and `sampling_report` reflects the shortfall).
- [ ] Implement `audit_sampling.py`.
- [ ] Run `uv run pytest tests/test_audit_sampling.py -v` — PASS.
- [ ] Commit.

## Task 2: New reviewer/key bundle builder + report-excerpt windowing

**Files:** Modify `src/stageground/evaluation/audit.py`; Test `tests/test_audit_bundle.py`

```python
AUDIT_MODES = ("grounding-blind", "with-gold")

def generate_audit_id(i: int) -> str: return f"A{i:04d}"

def _report_excerpt(text: str, evidence: str | None, *, max_chars=4000, window=1500) -> tuple[str, bool]: ...

def build_reviewer_and_key(
    records: list[PredictionRecord], texts: dict[str, str], *,
    target_counts: dict[str, int], seed: int, mode: str = "grounding-blind",
    m_priority_arm: str = "C_constrained", m_comparison_arm: str = "D_grounded",
) -> tuple[list[dict], list[dict], dict]:
    """Returns (reviewer_rows, key_rows, sampling_report). Uses
    audit_sampling.stratified_audit_sample internally, then shuffles the
    combined result with random.Random(seed) before assigning audit_ids
    (decision 5) and building the two row schemas (decisions 2-4)."""

def write_audit_bundle(
    reviewer_rows: list[dict], key_rows: list[dict], config: dict, outdir: str | Path,
) -> None:
    """Writes reviewer.jsonl, key.jsonl, config.json (indent=2), and a static
    INSTRUCTIONS.md (Task 5) into outdir, creating it if needed."""
```

- [ ] Write `tests/test_audit_bundle.py`:
  - **Blinding — no arm**: every reviewer row lacks an `"arm"` key.
  - **Blinding — no automated judgments**: every reviewer row lacks `"automated_evidence_span_found"`, `"automated_semantic_support"`, `"automated_errors"` keys.
  - **grounding-blind hides gold**: with `mode="grounding-blind"`, no reviewer row has a `"ground_truth"` key at all (not even `null`).
  - **with-gold shows it**: with `mode="with-gold"`, every reviewer row has `"ground_truth"` matching the corresponding key row's value.
  - **key preserves mapping**: every `key_row["audit_id"]` maps to a real `(case_id, arm, target)` that exists in the original `records` list, and `len(key_rows) == len(reviewer_rows)`, and audit_ids match 1:1 between the two files.
  - **reviewer schema fields present**: every reviewer row has exactly the field set from decision 2 (for the active mode).
  - **reproducibility**: same `(records, texts, target_counts, seed, mode)` twice → byte-identical `reviewer_rows`/`key_rows` (including audit_id assignment).
  - **long-report evidence-window extraction**: construct a >8000-char synthetic report with a unique evidence phrase near the middle; call `_report_excerpt` directly; assert the evidence phrase IS in the returned excerpt, `report_truncated is True`, and the excerpt is meaningfully shorter than the full text.
  - **short report not truncated**: `_report_excerpt("short text", None)` → `(text, False)`.
  - **abstention handling in build_reviewer_and_key**: an abstained M record (`prediction == "unknown"`) still gets a reviewer row with `human_source_has_explicit_stage`/`human_source_has_inferential_evidence`/`human_source_sufficient_for_stage` present as `null` (fillable) and `human_evidence_span_found`/`human_semantic_support` present as `null` too (evidence is `None` for abstentions per `records.py`, so the excerpt-windowing falls back to the full-report/first-N-chars strategy, not an evidence-window).
  - **legacy loading untouched**: import `build_audit_sheet`/`score_audit` from `audit.py` in this same test file and confirm they still work on a tiny fixture exactly as before (a smoke test proving the legacy path wasn't broken by the new additions) — full behavioral coverage stays in the untouched `tests/test_audit.py`.
- [ ] Implement `_report_excerpt`, `generate_audit_id`, `build_reviewer_and_key`, `write_audit_bundle` in `audit.py`, importing `stratified_audit_sample` from `audit_sampling.py`. Do not modify any existing function in the file.
- [ ] Run `uv run pytest tests/test_audit_bundle.py tests/test_audit.py -v` — PASS (both new and legacy).
- [ ] Commit.

## Task 3: INSTRUCTIONS.md content

**Files:** Modify `src/stageground/evaluation/audit.py` (add `_INSTRUCTIONS_TEMPLATE` + include in `write_audit_bundle`)

- [ ] Add a module-level `_INSTRUCTIONS_TEMPLATE` string (spec item 11): defines every reviewer field in plain language, and includes the exact worked examples from the spec (semantic support: prediction M1 / evidence "No distant metastasis identified" → `human_evidence_span_found=true, human_semantic_support=false`; evidence absent: prediction N1 / evidence string not in report → both `false`; abstention: prediction `unknown` → evidence judgments left `null`, source-sufficiency judgments still completed). Note which review `mode` the bundle was generated in (interpolated from `config["mode"]` at write time) so the instructions match what the reviewer actually sees.
- [ ] `write_audit_bundle` writes `outdir/INSTRUCTIONS.md` with the template, substituting the mode.
- [ ] Extend `tests/test_audit_bundle.py` with one assertion: `INSTRUCTIONS.md` exists after `write_audit_bundle` and contains the strings `"human_semantic_support"` and the configured mode name.
- [ ] Run tests — PASS. Commit (can combine with Task 2's commit if done together; otherwise a small standalone commit).

## Task 4: New scoring — agreement, confusion matrices, source-sufficiency, M0-prior analyses

**Files:** Create `src/stageground/evaluation/audit_scoring.py`; Test `tests/test_audit_scoring.py`

```python
def _confusion_and_agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs = [(automated, human), ...]. Returns n, percent_agreement,
    cohens_kappa, tp, fp, tn, fn, precision, recall, specificity. automated
    is the 'predicted' class, human is the 'actual' class (spec item 13).
    precision/recall/specificity are None when their denominator is 0."""

def score_reviewed_audit(reviewer_rows: list[dict], key_rows: list[dict]) -> dict:
    """Joins on audit_id. Returns {'overall': {...}, 'T': {...}, 'N': {...}, 'M': {...}},
    each with 'evidence_span_found': _confusion_and_agreement(...) and
    'semantic_support': _confusion_and_agreement(...), built from
    (key.automated_*, reviewer.human_*) pairs where BOTH sides are non-null."""

def analyze_source_sufficiency(reviewer_rows: list[dict], key_rows: list[dict]) -> dict:
    """Per target (T/N/M): n_reviewed, pct_source_sufficient, pct_explicit_stage,
    pct_inferential_evidence (each computed over rows where the corresponding
    human_* field is filled). PLUS, for M specifically, a breakdown by
    arm_pattern (from key_rows) with the same three percentages within each
    pattern group -- spec item 14's "C predicts/D abstains" vs "C predicts/D predicts"."""

def analyze_m0_prior_prediction(reviewer_rows: list[dict], key_rows: list[dict], *, priority_arm: str = "C_constrained") -> dict:
    """Among key_rows where arm == priority_arm, target == 'M', prediction == 'M0':
    accuracy (vs key.ground_truth), pct_source_sufficient, pct_semantic_support,
    pct_explicit_stage -- all among the subset with a filled human field, each
    with its own n. Language in the returned dict's own field NAMES stays
    neutral (e.g. 'pct_explicit_m_token_present'), not 'prior_guessing_rate' --
    spec item 15's "do not call it prior guessing automatically"."""
```

- [ ] Write `tests/test_audit_scoring.py`:
  - `_confusion_and_agreement`: hand-built pairs list covering all 4 confusion cells, assert exact tp/fp/tn/fn and derived precision/recall/specificity; all-same-class pairs → kappa `None` (zero variance) but percent_agreement still computed; empty pairs list → `n=0`, all rates `None`/NaN, no crash.
  - `score_reviewed_audit`: synthetic reviewer+key rows spanning T/N/M with known agreement/disagreement counts on both `evidence_span_found` and `semantic_support`; assert `overall` aggregates correctly and each target's sub-dict matches hand-computed values; rows with `human_evidence_span_found=None` (unreviewed) are excluded from that field's `n` but don't prevent the OTHER field from being scored on the same row.
  - `analyze_source_sufficiency`: synthetic rows with known `human_source_sufficient_for_stage`/etc. values; assert per-target percentages; construct M rows with two different `arm_pattern` values in `key_rows` and assert the M breakdown separates them correctly.
  - `analyze_m0_prior_prediction`: synthetic key/reviewer rows where some `C_constrained`/M/`M0` predictions are gold-correct and some aren't, with varying `human_source_sufficient_for_stage`; assert accuracy and the sufficiency percentage match hand computation; assert non-M0, non-`C_constrained`, or non-M rows are excluded.
  - **abstention handling**: a joined row with `human_evidence_span_found=None`/`human_semantic_support=None` (abstained case) doesn't crash any of the four functions above and is correctly excluded from grounding-agreement stats while still counting toward source-sufficiency stats if those human fields ARE filled.
- [ ] Implement `audit_scoring.py`.
- [ ] Run `uv run pytest tests/test_audit_scoring.py -v` — PASS.
- [ ] Commit.

## Task 5: CLI (`build` / `score` subcommands)

**Files:** Modify `src/stageground/evaluation/audit.py` (add `main()` + argparse); Test: extend `tests/test_audit_bundle.py` with one CLI-level smoke test

```bash
python -m stageground.evaluation.audit build \
  --predictions results/<experiment_id>/predictions.jsonl \
  --dataset data/processed/dataset.parquet \
  --target-count T=20 N=20 M=40 \
  --seed 42 \
  --mode grounding-blind \
  --output audit/audit_001

python -m stageground.evaluation.audit score \
  --audit-dir audit/audit_001
```

- `build`: loads `records.from_jsonl(--predictions)`, loads `pd.read_parquet(--dataset)` for `{patient_filename: text}`, parses `--target-count` (`nargs="+"`, each `KEY=VALUE`) into a dict, calls `build_reviewer_and_key(...)`, writes the bundle via `write_audit_bundle`, and additionally writes `config.json` with `seed, target_counts, m_priority_arm, m_comparison_arm, m_priority_fraction, m_semantic_balance_fraction, mode, source_experiment_id (parsed from the predictions path's parent dir name), source_prediction_file (the --predictions path), timestamp, achieved_composition (from the sampler's sampling_report)` — spec item 8.
- `score`: loads `reviewer.jsonl`/`key.jsonl` from `--audit-dir`, runs `score_reviewed_audit` + `analyze_source_sufficiency` + `analyze_m0_prior_prediction`, prints a human-readable summary and writes `<audit-dir>/scored.json` with all three results.

- [ ] Write one smoke test: build a tiny synthetic `predictions.jsonl` + `dataset.parquet` in `tmp_path`, invoke `main(["build", "--predictions", ..., "--dataset", ..., "--target-count", "T=2", "M=2", "--seed", "1", "--output", str(tmp_path / "audit_out")])`, assert `reviewer.jsonl`/`key.jsonl`/`config.json`/`INSTRUCTIONS.md` all exist and `config.json` round-trips as JSON with the expected keys from spec item 8. Then invoke `main(["score", "--audit-dir", str(tmp_path / "audit_out")])` on the (unfilled, all-null) reviewer file and assert it runs without crashing and writes `scored.json` (all `n_scored`-type fields will be 0/NaN since nothing is filled in — this is expected and exactly why the functions must be NaN/None-safe, not a test bug).
- [ ] Implement `main()` + argparse in `audit.py`.
- [ ] Run the new test + full suite `uv run pytest -q` — PASS.
- [ ] Commit.

## Task 6: README — document the blinded audit workflow

**Files:** Modify `README.md`

- [ ] Add a "Blinded human audit" subsection under the existing "Manual audit" section (or replacing/supplementing it — the existing one documents the single-file legacy `audit.py` functions used as a Python snippet, not a CLI; keep that as "quick single-file audit" and add this as the recommended path): the exact `build`/`score` commands from Task 5 (including the concrete 80-case recommendation `T=20 N=20 M=40`), a short explanation of blinding (reviewer never sees `arm` or automated judgments; gold is hidden by default), and a one-paragraph explanation of the M-stage oversampling rationale (tests whether `C_constrained`'s higher M accuracy reflects genuine textual evidence or dataset priors, by concentrating review on cases where `C_constrained` asserts a value and `D_grounded` abstains on the same report).
- [ ] Commit.

## Task 7: Final verification

- [ ] `uv run pytest -q` — all green (old + new).
- [ ] `uv run ruff check src/ tests/` — clean.
- [ ] Confirm `git status` shows no changes under `results/` or `audit/` (this plan creates no real audit run — only code + tests).
- [ ] Confirm no `OPENAI_API_KEY`/network access was needed anywhere in the test run.

---

## Self-review checklist

- Spec coverage: item 1→Task 2 (schema), item 2→Task 2, item 3→Task 2 (fields) + Task 4 (scoring must not crash on `human_confidence`, it's just data), item 4→Task 2 (blinding), item 5→Task 2 (blinding, no automated_* in reviewer), item 6→Task 1+5 (target-count CLI), item 7→Task 1 (M oversampling), item 8→Task 5 (config.json fields), item 9→Task 2 (`_report_excerpt`), item 10→Task 2 (modes), item 11→Task 3, item 12→Task 4, item 13→Task 4 (confusion matrix), item 14→Task 4 (`analyze_source_sufficiency`), item 15→Task 4 (`analyze_m0_prior_prediction`, conservative naming), item 16→Task 5 (`score` subcommand does the join, reviewer never touches key.jsonl), item 17→Task 5, item 18→ distributed across Tasks 1/2/4/5 test files (every category from the spec's list maps to a specific test named above), item 19→ no test or code fabricates a human label anywhere, item 20→ no `evaluate.py` invocation anywhere in this plan.
- No placeholders: every task has exact file paths, exact function signatures/schemas, and precise rules (fractions, field lists, join keys) pinned in the decisions section.
- Type consistency: `arm_pattern` constants, reviewer/key field names, and `audit_id` format are used identically from Task 1 through Task 5.
