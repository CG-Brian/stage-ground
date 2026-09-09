# StageGround Correctness & Reproducibility Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** Inline, same-session execution by the engineer who wrote
> both this plan and the prior ablation-pipeline plan
> (`docs/superpowers/plans/2026-09-07-ablation-eval-pipeline.md`). This is a
> **fix-it pass**, not a redesign: three concrete defects (model-config not
> reaching the API call, span-grounding conflated with semantic support,
> duplicated arm→prompt dispatch), plus the renames/tests/docs that follow
> directly from fixing them. No new subsystems.

**Goal:** Make `ExperimentConfig.model` the sole authority over the actual OpenAI request, split "evidence span exists" from "evidence semantically supports the prediction" as two explicit `PredictionRecord` fields with correctly renamed metrics, and make `ARM_CONFIGS` the only place arm→prompt dispatch is defined — all backed by tests that fail if any of these regress, with zero live API calls.

**Architecture:** `config.py` gains `provider`/`response_format` on `ModelConfig`; `llm_client.py` gains a parameterized `complete_json(..., model=, temperature=, max_tokens=, retries=)` plus `resolve_effective_model_config`/`is_reasoning_model`; `extractors.py`'s `ARMS` dict is *derived* from `ARM_CONFIGS` (which now owns `prompt_builder` directly) instead of being separately populated; `records.py`'s `PredictionRecord.supported` is replaced by `evidence_span_found` + `evidence_semantically_supports_prediction`, with a `from_dict` shim for old JSONL; `metrics.py` gets precise `span_*`/`semantic_*` names with the old names kept as documented deprecated aliases; `tables.py`/`plots.py`/`evaluate.py`/`audit.py`/`mstage_analysis.py` follow the field rename.

**Tech Stack:** No new dependencies. `unittest.mock.MagicMock` + `monkeypatch` for the OpenAI-client-level tests (Test A–D), all offline.

---

## Decisions pinned up front

1. **`ModelConfig` new fields** `provider: str = "openai"`, `response_format: str = "json_object"` — added with defaults so existing `ModelConfig(...)` call sites and the `test_config.py` roundtrip test keep working unmodified.
2. **`complete_json` signature becomes** `complete_json(system, user, *, model=DEFAULT_MODEL, temperature=None, max_tokens=None, retries=3, response_format="json_object")`, where `DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")` computed once at import (unchanged legacy env-default behavior when called with no kwargs). `temperature`/`max_tokens` are only included in the actual API kwargs when not `None` (preserves existing gpt-5/o-series compatibility note).
3. **`run_one(arm, report_text, *, model_config: ModelConfig | None = None)`** — when `model_config` is given, its `model/temperature/max_tokens/retries/response_format` are passed explicitly to `complete_json`, overriding any environment default. When `None` (legacy call, e.g. from old ad-hoc scripts), falls back to `complete_json`'s own env-sourced default — this is the one sanctioned legacy path per spec item 1.
4. **Effective config resolution lives in `llm_client.py`** (`is_reasoning_model`, `resolve_effective_model_config`) since the "gpt-5/o-series reject non-default temperature" fact is OpenAI-API-specific domain knowledge that already lived as a comment in that module. `resolve_effective_model_config` never mutates its input; returns a new `ModelConfig` via `dataclasses.replace`.
5. **`run_evaluation` resolves once, up front**: `effective_model = resolve_effective_model_config(config.model)`, prints a one-line note if it differs from `config.model`, passes `effective_model` to every `runner(...)` call, and writes **both** `config["model"]` (as requested) and `config["effective_model"]` (as actually used) into `config.json` — never silently substituting one for the other.
6. **`PredictionRecord.supported` is replaced**, not kept alongside, by `evidence_span_found: bool | None` and `evidence_semantically_supports_prediction: bool | None`. Both `None` for abstentions. For non-abstained predictions: `evidence_span_found = evidence is not None and grounded(evidence, report_text)`; `evidence_semantically_supports_prediction = False` if `not evidence_span_found` (fabricated/absent evidence cannot semantically support anything — this also makes semantic support a strict subset of span-found, so `semantic_supported_accuracy ⊆ span_grounded_accuracy` by construction), else `value_supported_by_text(prediction, evidence)`. `PredictionRecord.from_dict` maps a legacy `supported` key (no `evidence_span_found` present) onto `evidence_span_found`, with `evidence_semantically_supports_prediction` defaulting to `None` (legacy records never computed it — `None` means "not available," not "False").
7. **Metrics renamed, old names kept as deprecated aliases** (all in `metrics.py`, all still computed since nothing has shipped real results under the old names yet, but the user asked to preserve them if kept — cheap to keep, documented as deprecated):
   - `span_unsupported_rate_over_evaluable` / `_over_asserted` (= old `unsupported_rate_over_evaluable`/`_over_asserted`, kept as deprecated aliases).
   - `semantic_unsupported_rate_over_evaluable` / `_over_asserted` — **new**: non-abstained prediction where `evidence_span_found is False` OR `evidence_semantically_supports_prediction is False`.
   - `span_grounded_accuracy_over_evaluable` / `_over_asserted` (= old `supported_accuracy_over_evaluable`/`_over_asserted`, kept as deprecated aliases).
   - `semantic_supported_accuracy_over_evaluable` / `_over_asserted` — **new**: `correct AND evidence_span_found AND evidence_semantically_supports_prediction`.
   - `evidence_semantic_support_rate` is simplified to average the now-precomputed `evidence_semantically_supports_prediction` field instead of recomputing the same regex check a second time.
8. **`ARM_CONFIGS` becomes the single source of prompt dispatch**: `ArmConfig` gains a `prompt_builder: Callable[[str], tuple[str, str]]` field (Option B from the spec — clean one-directional dependency, `config.py` imports `stageground.extraction.prompts`, nothing imports back). `extractors.ARMS` becomes `{arm.value: cfg.prompt_builder for arm, cfg in ARM_CONFIGS.items()}` plus legacy letter keys derived from `cfg.legacy_key`, deleting the previous manually-duplicated `ARMS.update({...})` block entirely.
9. **Downstream field-rename fallout** (audit.py, mstage_analysis.py, tables.py, plots.py, evaluate.py) is fixed to use the new field/metric names directly — these are pure functions over `PredictionRecord`/tables with no persisted-artifact backward-compat requirement (only the JSONL *record* schema itself needs the legacy shim, per spec item 7).
10. **No new experiment run.** Only unit/offline-integration tests, ruff, and a synthetic mocked end-to-end run (already exists in `test_evaluate_cli.py`, extended here) are executed.

---

## File Structure

Modified files:
- `src/stageground/config.py` — `ModelConfig` gains `provider`/`response_format`; `ArmConfig` gains `prompt_builder`; `ARM_CONFIGS` entries pass the real builder functions (imported from `prompts.py`).
- `src/stageground/extraction/llm_client.py` — parameterized `complete_json`; add `is_reasoning_model`, `resolve_effective_model_config`.
- `src/stageground/extraction/extractors.py` — `ARMS` derived from `ARM_CONFIGS`; `run_one` takes `model_config`.
- `src/stageground/evaluation/records.py` — `supported` → `evidence_span_found` + `evidence_semantically_supports_prediction`; `from_dict` legacy shim.
- `src/stageground/evaluation/metrics.py` — renamed metrics + deprecated aliases + new semantic metrics.
- `src/stageground/evaluation/audit.py` — field rename in `build_audit_sheet`/`score_audit`.
- `src/stageground/evaluation/mstage_analysis.py` — field rename in `build_mstage_annotation_sheet`.
- `src/stageground/evaluation/tables.py` — new columns (`semantic_supported_accuracy`, `span_unsupported_rate`, `semantic_unsupported_rate`), pretty Markdown labels.
- `src/stageground/evaluation/plots.py` — `plot_accuracy_vs_span_unsupported` (new name for old fn) + `plot_accuracy_vs_semantic_unsupported` (new headline fig) + `plot_coverage_vs_semantic_supported_accuracy` (renamed).
- `src/stageground/evaluate.py` — effective-config resolution + `effective_model` in `config.json`; `runner` type takes `model_config`; `BOOTSTRAP_METRICS`/`_per_case_value` updated to new metric names; CLI gains `--max-tokens`/`--retries`/`--provider`/`--response-format`; figure calls updated.
- `README.md` — scientific-language audit per item 14.
- Existing tests touched by the rename: `test_records.py`, `test_metrics.py`, `test_mstage_analysis.py`, `test_evaluate_cli.py`, `test_tables.py`, `test_plots.py`.

New files:
- `tests/test_arm_invariants.py` — item 9 (prompt/config fragment invariants per arm).
- `tests/test_model_config_wiring.py` — item 11 Tests A–D (mocked OpenAI client, no live calls).
- `tests/test_grounding_semantics.py` — item 12 Cases 1–5.

---

## Task 1: Wire `ModelConfig` through to the actual API call

**Files:** `src/stageground/config.py`, `src/stageground/extraction/llm_client.py`, `src/stageground/extraction/extractors.py`, `tests/test_model_config_wiring.py`, `tests/test_config.py` (extend)

- [ ] Add `provider: str = "openai"` and `response_format: str = "json_object"` fields to `ModelConfig` in `config.py`.
- [ ] Rewrite `llm_client.py`: rename module constant to `DEFAULT_MODEL`, parameterize `complete_json` per decision 2, add `is_reasoning_model(model: str) -> bool` (regex `^(o[1-9](-mini)?|gpt-5)`, case-insensitive) and `resolve_effective_model_config(model_config: ModelConfig) -> ModelConfig` (decision 4), importing `ModelConfig` from `stageground.config`.
- [ ] Rewrite `run_one` in `extractors.py` to accept `*, model_config: ModelConfig | None = None` and thread it into `complete_json` per decision 3.
- [ ] Write `tests/test_model_config_wiring.py` with Tests A–D from spec item 11 (mock `llm_client._client` directly via `monkeypatch.setattr`, never touch a real key): explicit model reaches the request; env `OPENAI_MODEL` does not override an explicit `model_config`; legacy call with no `model_config` uses `DEFAULT_MODEL`; `temperature`/`max_tokens` are omitted from the request kwargs when `None`; `retries` governs actual retry attempts on a transient failure.
- [ ] Extend `tests/test_config.py` with a `resolve_effective_model_config` test: a reasoning-model name (`gpt-5-mini`) with `temperature=0.7` normalizes to `temperature=None`; a non-reasoning model (`gpt-4o`) with `temperature=0.7` is unchanged; the function never mutates its input (`ModelConfig` is frozen, so this is really "returns a distinct equal-otherwise instance").
- [ ] Run `uv run pytest tests/test_model_config_wiring.py tests/test_config.py -v` — PASS.
- [ ] Commit.

## Task 2: `ARM_CONFIGS` as the sole arm→prompt registry

**Files:** `src/stageground/config.py`, `src/stageground/extraction/extractors.py`, `tests/test_config.py` (extend), `tests/test_arm_invariants.py`

- [ ] Add `prompt_builder: Callable[[str], tuple[str, str]]` field to `ArmConfig`; import the five builder functions from `stageground.extraction.prompts` into `config.py` and pass them into each `ARM_CONFIGS` entry.
- [ ] Rewrite `extractors.py`'s `ARMS` to be derived from `ARM_CONFIGS` per decision 8, deleting the manual `ARMS.update({...})` block and the now-redundant direct imports of the five builder functions (extractors.py no longer needs to import `prompts` directly at all).
- [ ] Extend `tests/test_config.py`: every `ArmConfig.prompt_builder` is callable and, called with a sample report string, returns a `(system, user)` tuple of two non-empty strings; `extractors.ARMS` contains exactly the 5 new keys + the 4 legacy letter keys (9 total) and every value is exactly the `prompt_builder` from the corresponding `ARM_CONFIGS` entry (proves single-source-of-truth: no drift possible since it's the same object).
- [ ] Write `tests/test_arm_invariants.py` (spec item 9) asserting fragment-level invariants, not full prompt strings:
  - `C_constrained`: system prompt contains `"Allowed values"`; does NOT contain the evidence-mandate fragment `"MUST be a verbatim substring"`; does NOT contain the abstention-encouragement fragment `"should output"` (i.e. no explicit unknown-preference instruction beyond listing it as a value).
  - `C_plus_unknown`: contains `"Allowed values"` AND the abstention-encouragement fragment (e.g. `"rather than guess"`); does NOT contain `"MUST be a verbatim substring"`.
  - `D_grounded`: contains `"Allowed values"`, the evidence-mandate fragment `"MUST be a verbatim substring"`, AND an abstention-encouragement fragment (e.g. `"Do NOT guess"`).
  - Cross-check against `ARM_CONFIGS` flags: `ARM_CONFIGS[arm].requires_evidence` is `True` only for `GROUNDED`; `encourages_unknown` is `True` for `CONSTRAINED_UNKNOWN` and `GROUNDED`, `False` for `CONSTRAINED` (already covered in `test_config.py` from the prior pass, referenced here for completeness, not re-tested).
- [ ] Run `uv run pytest tests/test_config.py tests/test_arm_invariants.py -v` and full `uv run pytest -q` (extractors.py changed, must not break `test_evaluate_cli.py`/old scripts' `--arm` choices) — PASS.
- [ ] Commit.

## Task 3: Split `evidence_span_found` / `evidence_semantically_supports_prediction`

**Files:** `src/stageground/evaluation/records.py`, `tests/test_records.py` (rewrite affected assertions), `tests/test_grounding_semantics.py`

- [ ] Rewrite `PredictionRecord` per decision 6: remove `supported`, add the two new fields (both typed `bool | None`).
- [ ] Rewrite `build_record`'s computation per decision 6 (semantic support forced `False`, not independently computed, whenever span isn't found).
- [ ] Rewrite `from_dict` per decision 6's legacy shim (map old `supported` → `evidence_span_found`, default `evidence_semantically_supports_prediction` to `None`; silently drop a stray `supported` key if the new keys are already present; ignore any other unknown keys for forward-compat). Never mutate the caller's dict.
- [ ] Update `tests/test_records.py`'s existing assertions (`rec.supported is True/None/False`) to the two new fields.
- [ ] Add a `from_dict` backward-compat test: construct a dict shaped like an OLD PredictionRecord JSON line (has `supported: True`, no `evidence_span_found`/`evidence_semantically_supports_prediction` keys), call `PredictionRecord.from_dict(...)`, assert `evidence_span_found is True` and `evidence_semantically_supports_prediction is None`.
- [ ] Write `tests/test_grounding_semantics.py` — spec item 12, Cases 1–5, using `build_record` directly:
  - Case 1: prediction M1, evidence "M1 metastatic disease" grounded in a report containing that phrase → `evidence_span_found=True`, `evidence_semantically_supports_prediction=True`.
  - Case 2: prediction M1, evidence "No distant metastasis identified" (grounded, but doesn't contain an M1-supporting token) → `span_found=True`, `semantic_support=False`.
  - Case 3: evidence string absent from the report entirely → `span_found=False`, `semantic_support=False` (not independently re-derived from the fabricated evidence's own content).
  - Case 4: prediction `unknown` → both fields `None`.
  - Case 5: correct prediction (matches gold), evidence grounded but irrelevant to the predicted stage (e.g. predicts T2, evidence is "Nodes 0/12") → `span_found=True`, `semantic_support=False`; assert this record does NOT count toward `semantic_supported_accuracy_over_evaluable` when run through `metrics.py` (cross-check with Task 4, written after Task 4 lands — see reordering note).
- [ ] Run `uv run pytest tests/test_records.py -v` — PASS (Case 5's metrics cross-check deferred to Task 4).
- [ ] Commit records.py + its own tests; hold `test_grounding_semantics.py`'s Case 5 assertion for Task 4's commit.

## Task 4: Rename metrics, add semantic variants, keep deprecated aliases

**Files:** `src/stageground/evaluation/metrics.py`, `tests/test_metrics.py` (extend/rename), `tests/test_grounding_semantics.py` (finish Case 5)

- [ ] Rewrite `metrics.py` per decision 7: `span_unsupported_rate_over_evaluable`/`_over_asserted`, `semantic_unsupported_rate_over_evaluable`/`_over_asserted`, `span_grounded_accuracy_over_evaluable`/`_over_asserted`, `semantic_supported_accuracy_over_evaluable`/`_over_asserted` as the precise primary names; `unsupported_rate_over_evaluable`/`_over_asserted` and `supported_accuracy_over_evaluable`/`_over_asserted` kept as thin deprecated-alias functions (docstring: "Deprecated alias for `span_*`; kept for any code written against the pre-split schema. New code should call the `span_*` name directly."); `evidence_semantic_support_rate` simplified to average `evidence_semantically_supports_prediction` over the asserted population. `compute_all_metrics` includes every one of these keys (old + new) so nothing reading the dict breaks.
- [ ] Update `tests/test_metrics.py`: rename the mixed-scenario assertions to the new primary names, add assertions that the deprecated aliases return identical values to their `span_*` equivalents, add a semantic-specific case distinguishing `span_grounded_accuracy` from `semantic_supported_accuracy` (reuse the metrics-mixed-scenario record set, adding one record where evidence is grounded but doesn't support the prediction — expect it to count toward `span_grounded_accuracy` but not `semantic_supported_accuracy`).
- [ ] Finish `tests/test_grounding_semantics.py` Case 5 by running the record through `metrics.compute_all_metrics([record], target=...)` and asserting `semantic_supported_accuracy_over_evaluable == 0.0` while `span_grounded_accuracy_over_evaluable == 1.0` (record is correct + span found).
- [ ] Run `uv run pytest tests/test_metrics.py tests/test_grounding_semantics.py -v` — PASS.
- [ ] Commit records.py + metrics.py + all grounding/metrics tests together (they're one coherent change).

## Task 5: Propagate the rename through audit / mstage / tables / plots

**Files:** `src/stageground/evaluation/audit.py`, `src/stageground/evaluation/mstage_analysis.py`, `src/stageground/evaluation/tables.py`, `src/stageground/evaluation/plots.py`, their existing tests

- [ ] `audit.py`: replace `"automated_supported": rec.supported` with `"automated_evidence_span_found": rec.evidence_span_found, "automated_semantic_support": rec.evidence_semantically_supports_prediction`; update `score_audit`'s automated-key lookup to `r.get("automated_evidence_span_found")` (paired against the existing `human_supported` field name — not renamed in this pass, flagged as a follow-up in the final report, out of this task's explicit scope). Update `tests/test_audit.py`'s row fixtures/assertions accordingly.
- [ ] `mstage_analysis.py`: replace `"supported": rec.supported` with `"evidence_span_found": rec.evidence_span_found, "evidence_semantically_supports_prediction": rec.evidence_semantically_supports_prediction` in `build_mstage_annotation_sheet`. Update `tests/test_mstage_analysis.py`'s field-presence assertions.
- [ ] `tables.py`: `build_comparison_table` columns become `arm, n_evaluable, accuracy, semantic_supported_accuracy, span_unsupported_rate, semantic_unsupported_rate, abstention_rate, coverage` (item 13); add a `PRETTY_LABELS` dict used only by `_to_markdown` (CSV keeps precise snake_case headers, Markdown gets human-readable ones, e.g. `"semantic_supported_accuracy" -> "Semantic Supported Accuracy"`). Update `tests/test_tables.py`'s expected column values.
- [ ] `plots.py`: keep `plot_accuracy_vs_unsupported` renamed to two functions — `plot_accuracy_vs_span_unsupported` and `plot_accuracy_vs_semantic_unsupported` (the latter becomes the headline "Fig 1" per item 13, titled "Accuracy vs Semantic Unsupported Rate (heuristic)"); rename `plot_coverage_vs_supported_accuracy` to `plot_coverage_vs_semantic_supported_accuracy` (x=coverage, y=semantic_supported_accuracy). Update `tests/test_plots.py`'s function names/table fixture columns.
- [ ] Run `uv run pytest tests/test_audit.py tests/test_mstage_analysis.py tests/test_tables.py tests/test_plots.py -v` — PASS.
- [ ] Commit.

## Task 6: Wire effective config + renamed metrics into `evaluate.py`

**Files:** `src/stageground/evaluate.py`, `tests/test_evaluate_cli.py` (extend)

- [ ] Resolve `effective_model = resolve_effective_model_config(config.model)` once at the top of `run_evaluation`; print a one-line note when it differs from `config.model`; pass `model_config=effective_model` to every `runner(arm.value, row.text, model_config=effective_model)` call; write both `config["model"]` (requested, from `config.to_dict()`) and `config["effective_model"]` (via `dataclasses.asdict(effective_model)`) into `config.json`, alongside the existing `sampling` key.
- [ ] Update `BOOTSTRAP_METRICS` to `["accuracy", "semantic_supported_accuracy_over_evaluable", "span_unsupported_rate_over_evaluable", "semantic_unsupported_rate_over_evaluable", "abstention_rate", "coverage"]` and `_per_case_value` to compute each from `evidence_span_found`/`evidence_semantically_supports_prediction` per decision 7's definitions.
- [ ] Update the figure-generation calls to the Task 5 function names, generating both the span and semantic accuracy-vs-unsupported figures (`fig1a_accuracy_vs_span_unsupported.png`, `fig1b_accuracy_vs_semantic_unsupported.png`) plus the renamed coverage figure (`fig2_coverage_vs_semantic_supported_accuracy.png`).
- [ ] Add `--max-tokens`, `--retries`, `--provider`, `--response-format` CLI flags to `main()`, threading them into the constructed `ModelConfig`.
- [ ] Update `tests/test_evaluate_cli.py`'s `fake_runner` to accept `*, model_config=None` (ignored by default, since the fake doesn't call a real API).
- [ ] Add `test_effective_model_config_is_passed_to_runner`: a spy runner records every `model_config` it receives; assert they all equal `resolve_effective_model_config(config.model)`.
- [ ] Add `test_config_json_records_both_requested_and_effective_model` (spec item 11 Test E): build an `ExperimentConfig` with `model=ModelConfig(model="gpt-5-mini", temperature=0.7)`, run the (fake-runner) evaluation, load `config.json`, assert `payload["model"]["temperature"] == 0.7` (untouched, as requested) and `payload["effective_model"]["temperature"] is None` (normalized, as actually used) and `payload["effective_model"]["model"] == "gpt-5-mini"`.
- [ ] Run `uv run pytest tests/test_evaluate_cli.py -v` and full `uv run pytest -q` — PASS.
- [ ] Commit.

## Task 7: README scientific-language audit

**Files:** `README.md`

- [ ] In the "Ablation study" section added in the prior pass, replace every metric name/table column that used `supported_accuracy`/`unsupported_rate` with the new `span_*`/`semantic_*` names, and add one paragraph explicitly distinguishing span grounding from semantic support (item 14's suggested framing verbatim or near-verbatim), stating semantic support is currently an automated heuristic (token-regex proxy) pending larger-scale manual audit validation via `audit.py`.
- [ ] Scan the v0-pilot section of the README (the original Track A/B/C results, arms A–D) for phrasing implying "evidence span exists = truly supported" — the pilot's own `grounded()`-based claims are historically validated and unchanged in code, so this is a *language* audit only (add a clarifying footnote if a sentence conflates the two), not a re-derivation of pilot numbers.
- [ ] Commit.

## Task 8: Final verification

- [ ] `uv run pytest -q` — all green.
- [ ] `uv run ruff check src/ tests/` — clean.
- [ ] Confirm no `results/` artifacts were created or modified (`git status` shows nothing under `results/`).
- [ ] Confirm no network/API calls were made (no `OPENAI_API_KEY` needed anywhere in the test run).

---

## Self-review checklist

- Spec coverage: item 1→Task 1, item 2→Task 1+6, item 3→Task 1 (ModelConfig fields), item 4→Task 3, item 5→Task 4, item 6→Task 4, item 7→Task 3, item 8→Task 2, item 9→Task 2, item 10→Task 6, item 11→Task 1+6, item 12→Task 3+4, item 13→Task 5, item 14→Task 7, item 15→ no experiment run anywhere in this plan, item 16→Task 8.
- No placeholders: every task names exact files, exact new field/function names, and precise semantics (span-found-implies-semantic-eligibility rule stated once in decision 6, reused everywhere).
- Type consistency: `evidence_span_found`/`evidence_semantically_supports_prediction` are the two names used verbatim from Task 3 onward through metrics, audit, mstage, evaluate — no re-spelling.
