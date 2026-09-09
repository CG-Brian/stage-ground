# Evaluation

StageGround keeps output validity, evidence-span grounding, semantic support, and abstention as separate measurements rather than folding them into one reliability score. This document defines each one, the bootstrap comparison method, the error taxonomy, and the audit process used to check the automated semantic-support heuristic against a human reviewer.

## The four axes

Every non-abstained prediction carries two distinct grounding fields — never one ambiguous flag:

- **`evidence_span_found`** — a syntactic fact. Does the model's `evidence` string appear verbatim (OCR-noise-tolerant) in the source report? This is unchanged from the pilot's validated `grounded()` check.
- **`evidence_semantically_supports_prediction`** — does that evidence actually say what was predicted? This is a conservative, rule-based heuristic ([`semantic_support.py`](../src/stageground/evaluation/semantic_support.py)), explicitly not equivalent to human judgment and not a clinical staging engine. It never independently assigns a stage from the report — it only asks whether the model's own offered evidence plausibly supports the model's own prediction. Fabricated evidence can't semantically support anything, so this field is forced `False` whenever `evidence_span_found` is `False`. Both fields are `None` for abstained predictions.

A schema can constrain the answer space. It cannot make missing evidence appear in the report.

**Recognized evidence categories** (see the module docstring for full rationale): an explicit stage token (`T2`, `pN1`, `M0` — the primary signal for every target); regional-lymph-node status (negative counts support `N0`, positive support `N1` — `N2`/`N3` still require an explicit token, since count-to-stage cutoffs are cancer-type-dependent); local-invasion language for T (validates the `T2`/`T3`/`T4` group generically, never a specific value); tumor-size mentions (recognized but never independently justify a specific T value); distant-metastasis phrasing for M. Regional-node evidence is structurally never consulted for M-target predictions, so a lymph-node finding can never be mistaken for M1 support.

## Metric definitions

Every metric ([`metrics.py`](../src/stageground/evaluation/metrics.py)) is scoped to *evaluable* cases — reports where ground truth exists for that target — so every number in a comparison table shares the same denominator population. All metrics return `NaN` rather than raising when a denominator is zero.

| Metric | Definition |
|---|---|
| `accuracy` | correct predictions / evaluable cases |
| `abstention_rate` | `unknown` predictions / evaluable cases |
| `coverage` | `1 - abstention_rate` |
| `span_unsupported_rate_over_evaluable` / `_over_asserted` | non-abstained predictions with no evidence span found / evaluable (or asserted) cases |
| `semantic_unsupported_rate_over_evaluable` / `_over_asserted` | non-abstained predictions where the span is absent or doesn't semantically support the prediction / evaluable (or asserted) cases |
| `span_grounded_accuracy_*` | correct **and** span-found — span-only, doesn't require the evidence to say what was predicted |
| `semantic_supported_accuracy_*` | correct **and** span-found **and** semantically supports — the closest automated proxy to "true grounding" this project computes, still not a substitute for manual audit |
| `allowed_value_compliance` | fraction whose raw, pre-canonicalization output is already an allowed-domain token |
| `evidence_span_found_rate` / `evidence_semantic_support_rate` | span-found / semantic-support rate among non-abstained predictions |

`unsupported_rate_*` and `supported_accuracy_*` exist as deprecated aliases of the `span_*` metrics (their original names were ambiguous about being span-only) — new code should use the `span_*` names.

Each problematic prediction is also tagged with zero or more **error taxonomy** flags — `hallucinated_stage`, `wrong_stage_with_supporting_evidence`, `evidence_span_not_found`, `evidence_does_not_support_prediction`, `missed_explicit_stage`, `over_abstention`, `invalid_normalization`, `invalid_schema_output` ([`error_taxonomy.py`](../src/stageground/evaluation/error_taxonomy.py)). Flags aren't mutually exclusive; a single prediction can carry several.

## Bootstrap comparisons

`bootstrap.json` in each experiment directory reports paired-bootstrap (same cases across arms, not independent resampling) diff + 95% CI + empirical two-sided p-value for the three adjacent arm comparisons (`D vs C+`, `C+ vs C`, `D vs C`), on accuracy, semantic-supported accuracy, span-unsupported rate, semantic-unsupported rate, abstention rate, and coverage — both pooled and per T/N/M target. A comparison is silently omitted if one of its two arms wasn't included in a given run.

Read the confidence interval before the p-value. `p < 0.0002` and `p < 1/5000` mean the same thing — zero of 5,000 bootstrap resamples crossed the null — and the CI tells you the size of the effect, not just that one exists.

## Blinded human audit

Automated grounding checks rest on heuristics. [`audit.py`](../src/stageground/evaluation/audit.py) produces a **blinded** reviewer/key bundle to check them against a human: `reviewer.jsonl` never contains `arm`, `case_id`, or any automated judgment, so a reviewer scores each row with no signal about which prompting strategy produced it or what the pipeline already concluded. In the default `grounding-blind` mode, `ground_truth` is also omitted — not shown as `null`, the key is absent — so span/semantic judgments aren't anchored by knowing the right answer.

Reviewer fields: `human_evidence_span_found`, `human_semantic_support`, `human_prediction_correct`, `human_source_has_explicit_stage`, `human_source_has_inferential_evidence`, `human_source_sufficient_for_stage`, `human_confidence`, `human_error_type`, `reviewer_notes`.

M-stage sampling within the audit isn't uniform: it preferentially selects cases where `C_constrained` asserts a value and `D_grounded` abstains on the identical report — the most diagnostic cases for testing whether `C_constrained`'s higher M accuracy reflects genuine textual evidence or the M0 base rate.

Scoring (`stageground.evaluation.audit score`) joins `reviewer.jsonl` + `key.jsonl` on a blind ID and reports:

- **agreement** — percent agreement, Cohen's kappa, and a confusion matrix (automated vs. human) for both `evidence_span_found` and `semantic_support`, overall and per target.
- **source_sufficiency** — % source-sufficient / explicit-stage / inferential-evidence per target, plus an M-only breakdown by which arm asserted vs. abstained.
- **m0_prior_prediction** — among `C_constrained`'s M0 predictions: accuracy, source-sufficiency, semantic-support rate, explicit-token rate. Field names are deliberately neutral — whether high M0 accuracy reflects extraction or prior is left for a human reading the numbers to conclude, not asserted automatically.

### Current audit status

The heuristic used in the main study was checked against an earlier blinded 80-case audit (`audit/audit_001`) run against a prior version of the heuristic: the literal-token-only predecessor had precision 1.0 but recall ~0.53 overall (N ~0.29, T ~0.56); the current heuristic raises recall to ~0.71 (N ~0.64, T ~0.69) while precision stays at 1.0 at every target. M-stage recall (~0.88) was already high and is unchanged.

**Every row in that audit run is marked `[MODEL-ASSISTED PROVISIONAL]`** in `reviewer_notes` — it was not an independent human pass in the strict sense the tooling is designed for. Read the numbers above as directional and heuristic-corroborating, not as settled ground truth. An independent human re-review of `audit/audit_001`, or a fresh audit, is the natural next step before treating the semantic-support heuristic as fully validated.
