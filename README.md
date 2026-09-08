# StageGround

**Source-Boundary-Aware TNM Extraction from Pathology Reports**

> Evaluating not only whether LLMs match registry labels, but whether their staging
> claims are actually supported by the clinical document.

A clinical-AI validation research project. Registry gold labels reflect information
from *outside* the pathology report (e.g. imaging for M stage); the model sees only
the report. The real question is not just accuracy — it is whether the model can tell
**what the source document can and cannot support**. Schema-guided extraction is the
*method*; source-boundary-aware evaluation is the *subject*.

Concretely: when an LLM extracts TNM stage from TCGA reports, does schema-guided
extraction (allowed values + forced evidence spans + explicit abstain rule) produce
extractions **better grounded in the source text** than free-form — even when raw
accuracy is similar?

See [DESIGN.md](DESIGN.md) for the full spec.

## Dual-track evaluation

- **Track A** — accuracy vs gold labels (P / R / F1, invalid-output rate).
- **Track B** — groundedness vs report text (evidence-grounding rate,
  unsupported-assertion rate, appropriate-abstention rate). This is what
  distinguishes the work from prior TNM-extraction papers.

## Setup

```bash
uv sync                      # core deps
uv sync --extra dev          # + pytest / ruff
cp .env.example .env         # add your LLM API key
```

## Pipeline (DESIGN §7)

```bash
uv run python scripts/01_build_dataset.py    # load, join, canonicalize -> data/processed/*.parquet
uv run python scripts/01b_sample_eval.py     # stratified 200-patient eval subset
uv run python scripts/02_run_extraction.py --arm A   # run an arm -> results/predictions/<arm>.parquet
uv run python scripts/02_run_extraction.py --arm D
uv run python scripts/03_score.py            # Track A + B -> results/metrics/*.parquet
uv run python scripts/04_export_cases.py     # Track C + per-case JSON -> results/cases/*.json
```

## Research landing page (web/)

`web/` hosts a static Next.js research landing page presenting StageGround's
1,000-case ablation results (source-grounded evaluation, not a TNM staging
product) for a recruiter/researcher audience. It has no backend and no live
LLM calls — every number is exported at data-prep time from the committed
result artifacts under `results/20260908T022659_gpt-4o_seed42_n1000/`.

```bash
# regenerate the frontend's data (only needed after a new experiment run)
uv run python scripts/export_frontend_data.py
uv run python scripts/export_case_examples.py

cd web
npm install
npm run dev      # http://localhost:3000
npm run build    # production build
```

Data provenance: `web/src/data/stageground.ts` documents the source result
file for every metric it exposes; nothing displayed is hand-typed.

A separate FastAPI serving layer exists at `src/stageground/api/` (reads
`results/cases` + `results/metrics`, a v0-pilot artifact layout) for an
earlier planned case-by-case review UI. It is currently unwired to any
frontend and unrelated to the landing page above.

## Layout

```
src/stageground/   library code (data, extraction, evaluation, demo)
scripts/           pipeline entry points (run order)
data/              raw/ (gitignored) + processed/ parquet
results/           predictions/ + metrics/ parquet
tests/             unit tests (canonicalize, schema, evidence_grounded)
notebooks/         exploration + error analysis
```

## Ablation study (A / B / C / C+ / D)

The pilot above (Arms A–D, n=200) established that allowed-value constraints alone
do not improve grounding. The **ablation study** isolates *which* individual
intervention is responsible: is it the allowed-value list, the permission to
abstain, or the mandatory evidence span?

**Research question.** Does structured output improve *actual source grounding*,
or does it only improve *apparent extraction accuracy*? A pilot finding is that
allowed-value-constrained output (`C_constrained`) can reach very high apparent
accuracy while still producing many unsupported predictions — and that explicit
abstention and evidence binding are what actually reduce unsupported predictions,
not the value constraint itself.

### Experimental arms

The v0 pilot's Arm D bundled "allowed values + explicit unknown + mandatory
evidence" together, making it impossible to tell which piece did the work. The
ablation splits that into five arms, defined once in
[`src/stageground/config.py`](src/stageground/config.py) (`ExperimentArm` +
`ARM_CONFIGS`) so prompt text, evidence requirements, and evaluation logic all
read from the same source instead of branching on arm-name string literals:

| Arm | Allowed values | Evidence required | `unknown` encouraged |
| --- | :-: | :-: | :-: |
| `A_zero_shot` | – | – | – |
| `B_few_shot` | – | – | – (one worked abstention example, no rule) |
| `C_constrained` | ✓ | – | – |
| `C_plus_unknown` | ✓ | – | ✓ |
| `D_grounded` | ✓ | ✓ (every non-unknown prediction) | ✓ |

`C_constrained` and `D_grounded` reuse the v0 pilot's Arm C and Arm D prompts
unchanged (`stageground.extraction.prompts.build_allowed_values_prompt` /
`build_schema_guided_prompt`); `C_plus_unknown` is new
(`build_constrained_unknown_prompt`) and isolates "permission to abstain" from
"mandatory evidence." All other model settings (model, temperature, retries,
JSON output shape) are held constant across arms and recorded in every
experiment's `config.json` (`ModelConfig` in `config.py`) — the only thing that
varies between arms is the constraint being tested.

### Metrics

**StageGround distinguishes output validity, evidence-span grounding,
semantic support, and abstention rather than treating them as equivalent
notions of reliability.** Concretely, every non-abstained prediction carries
TWO separate grounding fields on its `PredictionRecord`, never one ambiguous
flag:

- **`evidence_span_found`** — a syntactic fact: does the model's `evidence`
  string literally appear (verbatim, OCR-noise-tolerant) in the source
  report? This is exactly the v0 pilot's validated, 92%-agreement-audited
  `grounded()` check — unchanged, not redefined.
- **`evidence_semantically_supports_prediction`** — does that evidence
  actually say what was predicted? This is a **conservative, rule-based
  heuristic** ([`semantic_support.py`](src/stageground/evaluation/semantic_support.py)),
  explicitly **not** equivalent to human semantic judgment and **not** a
  clinical staging engine — it never independently assigns a TNM stage from
  the report text; it only asks whether the model's *own offered evidence*
  plausibly supports the model's *own prediction*. Fabricated evidence can't
  semantically support anything, so this field is always `False` whenever
  `evidence_span_found` is `False` — it's never computed independently. Both
  fields are `None` for abstained predictions.

  **Recognized evidence categories** (see the module docstring for full
  rationale): an explicit stage token (`T2`, `pN1`, `M0`, unchanged from the
  original heuristic and still the primary signal for every target);
  regional-lymph-node status (negative counts/phrasing support `N0`,
  positive support `N1` — `N2`/`N3` still require an explicit token, since
  node-count-to-stage cutoffs are cancer-type-dependent); local-invasion
  language for T ("invades muscularis propria", "extrathyroidal extension",
  etc. — validates the `T2`/`T3`/`T4` group generically, never a specific
  value, and never validates `T0`/`T1`); tumor-size mentions (recognized as
  a category, but never independently justify a specific T value — no
  "size > X → T2" rule, for the same cancer-type-dependence reason);
  distant-metastasis positive/negative phrasing for M. Regional-node
  evidence is structurally never consulted for M-target predictions, so
  "metastatic carcinoma in a regional lymph node" can never be mistaken for
  `M1` support.

  **Validated against the blinded 80-case audit** (`audit/audit_001`,
  `scripts/rescore_audit_semantic_support.py`): the earlier
  literal-token-only version had precision 1.0 but recall ~0.53 overall,
  driven by N-stage (~0.29) and T-stage (~0.56); the current heuristic
  raises overall recall to ~0.71 (N to ~0.64, T to ~0.69) while precision
  and specificity remain exactly 1.0 at every target — zero new false
  positives introduced. M-stage (already ~0.88 recall) is unchanged.
  **Caveat:** every row in the current `audit/audit_001` run is marked
  `[MODEL-ASSISTED PROVISIONAL]` in `reviewer_notes` — it was not an
  independent human pass in the strict sense the audit tooling is designed
  for, so these numbers are directional, not final validation. Treat
  `semantic_*` metrics below as a heuristic, provisionally-audit-corroborated
  signal, not a settled ground truth — an independent human re-review of
  `audit/audit_001` (or a fresh audit) is the natural next step before
  treating this heuristic as fully validated.

Every metric (defined once in
[`src/stageground/evaluation/metrics.py`](src/stageground/evaluation/metrics.py))
is scoped to **evaluable cases** — reports where ground truth exists for that
target (T/N/M) — so every number in a comparison table shares the same
denominator population. All metrics return `NaN` rather than raising when a
denominator is zero (e.g. an all-abstained sample), so they never crash a run.

| Metric | Definition |
| --- | --- |
| `accuracy` | correct predictions / evaluable cases |
| `abstention_rate` | `unknown` predictions / evaluable cases |
| `coverage` | `1 - abstention_rate` |
| `span_unsupported_rate_over_evaluable` / `_over_asserted` | non-abstained predictions with no evidence span found / evaluable cases (or / non-abstained predictions) |
| `semantic_unsupported_rate_over_evaluable` / `_over_asserted` | non-abstained predictions where the span is absent OR the evidence doesn't semantically support the prediction (heuristic) / evaluable (or asserted) cases |
| `span_grounded_accuracy_over_evaluable` / `_over_asserted` | correct **and** evidence-span-found / evaluable (or asserted) cases — span-only, does not require the evidence to actually say what was predicted |
| `semantic_supported_accuracy_over_evaluable` / `_over_asserted` | correct **and** span-found **and** semantically-supports (heuristic) / evaluable (or asserted) cases — the closest automated proxy to "true grounding" this project computes, still not a substitute for manual audit |
| `allowed_value_compliance` | fraction whose *raw*, pre-canonicalization value is already an exact allowed-domain token (format compliance, independent of correctness or grounding) |
| `evidence_span_found_rate` | fraction of non-abstained predictions whose evidence is a verbatim (OCR-noise-tolerant) substring of the report |
| `evidence_semantic_support_rate` | fraction of non-abstained predictions whose evidence semantically supports the prediction (heuristic proxy) |

`unsupported_rate_over_evaluable`/`_over_asserted` and
`supported_accuracy_over_evaluable`/`_over_asserted` still exist as
**deprecated aliases** of the `span_*` metrics above (their original
semantics were always span-only, despite the ambiguous name) — new code
should call the `span_*` names directly.

Each incorrect/problematic prediction is also tagged with zero or more
**error taxonomy** flags (`hallucinated_stage`,
`wrong_stage_with_supporting_evidence`, `evidence_span_not_found`,
`evidence_does_not_support_prediction`, `missed_explicit_stage`,
`over_abstention`, `invalid_normalization`, `invalid_schema_output`) by
[`error_taxonomy.py`](src/stageground/evaluation/error_taxonomy.py) — multiple
flags can and do coexist on one prediction.

### Running an experiment

```bash
uv sync --extra dev            # + matplotlib, for tables/figures
uv sync --extra serve  # not required for evaluate.py; only if also running the API

# 200-case pilot
uv run python -m stageground.evaluate \
  --arms A_zero_shot C_constrained C_plus_unknown D_grounded \
  --sample-size 200 --seed 42 --model gpt-4o

# 1,000-case experiment
uv run python -m stageground.evaluate \
  --arms C_constrained C_plus_unknown D_grounded \
  --sample-size 1000 --seed 42 --model gpt-4o

# 2,000+ / full available dataset (sample-size >= pool size returns everything)
uv run python -m stageground.evaluate \
  --arms C_constrained C_plus_unknown D_grounded \
  --sample-size 100000 --seed 42 --model gpt-4o
```

`--sample-size` is never hardcoded in the evaluation logic — it's always a
required CLI argument, plumbed straight into
`stageground.evaluation.sampling.stratified_sample`, which deterministically
(same `df`, `n`, `seed` → identical sample) and proportionally stratifies on
the T/N/M gold-*availability* pattern. `--pool all-three` (default) restricts
to reports with all of T/N/M gold, matching `scripts/01b_sample_eval.py`'s
pilot behavior; `--pool any` uses reports with at least one target's gold.

Each run writes a self-contained, timestamped
`results/<experiment_id>/` directory — it never touches or overwrites the v0
pilot's `results/cases/`, `results/metrics/`, or `results/audit/`:

```
results/<experiment_id>/
  config.json             # arms, model, temperature, seed, sample size, sampling summary, timestamp
  predictions.jsonl       # one PredictionRecord per (case, arm, target); raw model output preserved
  metrics.json            # per-arm metrics, pooled across T/N/M
  metrics_by_target.json  # per-arm metrics, split by T/N/M
  error_breakdown.json    # per-arm, per-target error-taxonomy counts
  bootstrap.json          # paired-bootstrap arm comparisons (see below)
  tables/                 # overall/T/N/M comparison tables, .csv + .md
  figures/                # fig1a (accuracy vs span-unsupported, diagnostic),
                          # fig1b (accuracy vs semantic-unsupported, headline),
                          # fig2 (coverage vs semantic-supported accuracy), fig3 (error breakdown)
```

`config.json` records both the **requested** model config (`model`, exactly
as passed via `--model`/`--temperature`/etc.) and the **effective** one
(`effective_model`, what was actually sent to the provider after
normalization — e.g. an OpenAI reasoning model like `gpt-5-mini` silently
rejects any non-default temperature, so `resolve_effective_model_config`
drops it to `null` and records that explicitly rather than sending a request
that would fail or, worse, silently diverging from what `config.json` claims).

**Bootstrap comparisons.** `bootstrap.json` reports paired-bootstrap (same
reports across arms, not independent resampling) diff + 95% CI + empirical
two-sided p-value for `D_grounded vs C_constrained`, `C_plus_unknown vs
C_constrained`, and `D_grounded vs C_plus_unknown`, on `accuracy`,
`semantic_supported_accuracy_over_evaluable`, `span_unsupported_rate_over_evaluable`,
`semantic_unsupported_rate_over_evaluable`, `abstention_rate`, and `coverage`
— both pooled ("overall") and per T/N/M target. A comparison is silently
omitted (not an error) if one of its two arms wasn't included in `--arms`.

### Reproducing analysis

Tables and figures are generated automatically by `evaluate.py`, but can also
be regenerated standalone from an existing `predictions.jsonl` (e.g. after
manually editing/re-running only part of a pipeline):

```python
from stageground.evaluation.records import from_jsonl
from stageground.evaluation.tables import write_tables
from stageground.evaluation.plots import plot_error_breakdown

records = from_jsonl("results/<experiment_id>/predictions.jsonl")
write_tables(records, "results/<experiment_id>")
plot_error_breakdown(records, "results/<experiment_id>/figures/fig3_M.png", target="M")
```

### M-stage analysis

M-stage shows the largest accuracy/grounding tradeoff (lowest coverage,
M0-dominated, often needs imaging the report doesn't contain — DESIGN §3.3).
[`mstage_analysis.py`](src/stageground/evaluation/mstage_analysis.py)
regex-classifies each M-target report into `explicit_m_token`,
`metastatic_described_no_token`, `no_m_evidence`, or `ambiguous_insufficient`,
and `build_mstage_annotation_sheet(...)` produces a reproducible sheet with
blank `human_category`/`notes` fields for manual review — this is
intentionally a heuristic-plus-audit-trail, not a trained classifier.

### Manual audit — quick single-file (legacy)

The original single-file audit tool still exists for quick, small,
non-blinded checks:

```python
from stageground.evaluation.audit import build_audit_sheet, write_audit_jsonl, score_audit, read_audit_jsonl

rows = build_audit_sheet(records, texts, n=75, seed=42)
write_audit_jsonl(rows, "results/<experiment_id>/audit_sheet.jsonl")
report = score_audit(read_audit_jsonl("results/<experiment_id>/audit_sheet.jsonl"))
```

This path shows the reviewer the arm and the automated judgment in the same
row (no blinding) and uses a single ambiguous `human_supported` field. It's
kept only for backward compatibility with any audit files already generated
this way — **use the blinded workflow below for all new audits**, especially
anything meant to validate `semantic_supported_accuracy` / M-stage findings.

### Blinded, stratified human audit (recommended)

[`audit.py`](src/stageground/evaluation/audit.py)'s `build`/`score` CLI
produces a **blinded** reviewer/key bundle instead of one flat file, with
explicit fields separating span-grounding, semantic support, correctness,
and source sufficiency — this is the mechanism that should be used to
validate (or correct) `semantic_supported_accuracy` / `semantic_unsupported_rate`
before treating them as settled results. **The automated heuristic is a
starting point for triage, not a substitute for review** — nothing in this
repo fabricates or simulates a human label; a real reviewer must fill in the
`human_*` fields.

**Generate the recommended 80-case audit** (T=20, N=20, M=40) from an
existing experiment's predictions:

```bash
uv run python -m stageground.evaluation.audit build \
  --predictions results/<experiment_id>/predictions.jsonl \
  --dataset data/processed/dataset.parquet \
  --target-count T=20 N=20 M=40 \
  --seed 42 \
  --mode grounding-blind \
  --output audit/audit_001
```

This writes:

```
audit/audit_001/
  reviewer.jsonl      # what the reviewer sees and fills in -- see below
  key.jsonl           # hidden audit_id -> case_id/arm/ground_truth/automated-judgment mapping
  config.json         # seed, target_counts, M-oversampling params, source experiment, timestamp
  INSTRUCTIONS.md      # field definitions + worked examples, generated for this bundle's mode
```

**Blinding.** `reviewer.jsonl` never contains `arm`, `case_id`, or any
`automated_*` field — the reviewer judges each row independently, with no
signal about which experimental arm produced it or what the pipeline already
concluded. In the default `grounding-blind` mode, `ground_truth` is also
omitted entirely (not shown as `null` — the key is absent), so span/semantic/
source-sufficiency judgments aren't anchored by knowing the right answer;
`--mode with-gold` includes `ground_truth` for a pass focused on
`human_prediction_correct`. Only `key.jsonl` (which the reviewer never needs
to open) can map an `audit_id` back to a real case.

**Reviewer fields** (per row — see `INSTRUCTIONS.md` for full definitions and
worked examples): `human_evidence_span_found`, `human_semantic_support`,
`human_prediction_correct`, `human_source_has_explicit_stage`,
`human_source_has_inferential_evidence`, `human_source_sufficient_for_stage`,
`human_confidence` (`high`/`medium`/`low`), `human_error_type`,
`reviewer_notes`. For abstained (`unknown`) predictions, the evidence-related
fields are left `null` (there's no evidence to judge) but the
source-sufficiency fields should still be completed — that's precisely the
question of whether abstaining was the right call.

**M-stage oversampling.** Within the M quota, sampling isn't uniform random:
it preferentially selects cases where `C_constrained` asserts a value and
`D_grounded` abstains on the *same report* (`--m-priority-arm`/
`--m-comparison-arm`, default `C_constrained`/`D_grounded`) — these are the
most diagnostic cases for testing whether `C_constrained`'s higher M accuracy
reflects genuine textual evidence or dataset priors (M0 is the dominant gold
value). The remaining quota ensures both `automated_semantic_support` classes
(`True` and `False`) are represented, then fills randomly — so the sample is
never just machine-flagged failures. The achieved tier composition is
recorded in `config.json`'s `achieved_composition`.

**Reproducibility.** The same `(predictions, dataset, target_counts, seed,
mode, M-oversampling params)` always produces the identical sample and
`audit_id` assignment — `config.json` records every one of these inputs plus
`source_experiment_id`/`source_prediction_file`/`timestamp`.

**Score a completed audit** (after a human has filled in `reviewer.jsonl`
and saved it back to the same path):

```bash
uv run python -m stageground.evaluation.audit score --audit-dir audit/audit_001
```

This joins `reviewer.jsonl` + `key.jsonl` on `audit_id` (the only unblinding
step, and it happens automatically — the reviewer never touches `key.jsonl`)
and writes `audit/audit_001/scored.json` with three sections:

- **`agreement`** — overall + per-target (T/N/M) percent agreement, Cohen's
  kappa, and a full confusion matrix (automated=predicted, human=reference)
  with precision/recall/specificity, for both `evidence_span_found` and
  `semantic_support`. This is where you find out whether the regex heuristic
  systematically under-detects semantic support, and whether M behaves worse
  than T/N.
- **`source_sufficiency`** — % source-sufficient / explicit-stage /
  inferential-evidence per target, plus an M-only breakdown by
  `arm_pattern` (e.g. `priority_predicts_comparison_abstains` vs
  `both_predict`).
- **`m0_prior_prediction`** — among `C_constrained`'s M0 predictions:
  accuracy, source-sufficiency rate, semantic-support rate, and
  explicit-M-token rate. Field names are deliberately neutral (not
  "prior_guessing_rate") — whether high M0 accuracy reflects genuine
  extraction or dataset priors is an interpretation for a human reading these
  numbers to draw, not a conclusion this repo asserts automatically.

### Scope note

`scripts/01-06` and the existing `results/cases/`, `results/metrics/`,
`results/audit/` (and the results table below) are the v0 pilot and are
unchanged by the ablation study — they remain independently reproducible. The
ablation pipeline (`stageground.evaluate`, `stageground.config`, and
`stageground.evaluation.{metrics,records,error_taxonomy,sampling,bootstrap,
mstage_analysis,audit,tables,plots}`) is a separate, additive experiment track
built for the A/B/C/C+/D comparison above.

## Results

A **constraint ladder** of four prompting arms, same model, same JSON output shape,
200 stratified fully-labeled TCGA patients. The point is not "D wins" — it is
decomposing *which* kind of constraint reduces *which* kind of failure. Numbers from
`results/metrics/` (reproduce: `scripts/03_score.py`, `scripts/04_export_cases.py`).

| Arm | Constraint |
| --- | ---------- |
| A | zero-shot free-form (none) |
| B | few-shot — examples only, incl. one abstention example, no rules |
| C | allowed-values only — value domain listed, no evidence/abstention rule |
| D | schema-guided — allowed values + verbatim-evidence + abstain-with-reason rule |

### The decomposition (M-stage — the discriminating stage)

| Metric | A | B | C | D |
| ------ | --- | --- | --- | --- |
| Format compliance (allowed-value rate) ↑ | .21 | .88 | **1.00** | **1.00** |
| Evidence-grounding rate ↑ | .69 | **1.00** | .70 | **1.00** |
| Unsupported-assertion rate ↓ | .31 | **.00** | .30 | **.00** |
| Abstention rate | .00 | .51 | .01 | .47 |
| Track C source-boundary success ↑ | .07 | **.53** | .01 | .47 |
| Unsupported errors (count) | 33 | 0 | **56** | 0 |

Three findings, each a decomposed lever:

- **C (allowed-values only) breaks the intuition that "just give it a schema" makes it
  safe.** C maximizes *format* compliance (1.00) but its groundedness is no better than
  free-form — in fact its unsupported errors *rise to 56 (vs A's 33)*. Forced to emit an
  allowed value with no permission to abstain, the model hedges with **`MX`** (all 56 are
  `MX`; 52 have no M descriptor anywhere in the report). Constraining the value set alone
  produces *confident* unsupported assertions.
- **B (few-shot) teaches a safety behavior, not just accuracy.** A single abstention
  example induced strong abstention: unsupported-assertion drops to 0 and groundedness
  reaches 1.00. Few-shot abstention *unexpectedly matched or exceeded* rule-guided
  prompting on M-stage source-boundary success (.53 vs D's .47), suggesting even one
  abstention example can strongly shape behavior. (B is less conservative than D, which
  may favor it under the success definition — stated as suggestive, not decisive.)
- **D (explicit rules) is the most conservative and safest:** zero unsupported errors,
  100% evidence-grounding, but the highest abstention — trading some gold accuracy for it.

**Key factor:** *not constraint strength, but whether the prompt gives the model permission
and guidance to abstain when the source text does not support a value.* Listing allowed
values does not do this; showing or telling it to abstain does.

### Track C — source-boundary agreement

Every gold-mismatch tagged by *why* it disagrees (M-stage, n=200):

| arm | correct | boundary abstention ✓ | over-abstention | unsupported error ✗ | discordance | **success rate** |
| --- | ------- | --------------------- | --------------- | ------------------- | ----------- | ---------------- |
| A   | 52      | 10                    | 0               | 33                  | 105         | 0.07             |
| B   | 10      | 101                   | 1               | 0                   | 88          | **0.53**         |
| C   | 33      | 2                     | 0               | 56                  | 109         | 0.01             |
| D   | 7       | 91                    | 2               | 0                   | 100         | 0.47             |

The guardrail holds: only an *abstention* against a text-unsupportable gold is a
source-boundary success; an asserted wrong value is an `unsupported_error`. The high
`discordance` counts (text supports a value the registry contradicts) flag likely
out-of-document registry labels — the most clinically interesting review candidates.

### Manual audit (Milestone 1a)

Automated Track-C tags rest on `grounded()`, a verbatim-substring proxy — worth checking
against independent judgment before trusting the numbers above. A **stratified, blinded
25-case audit** (arm and existing tag hidden; report text, gold value, predicted value, and
submitted evidence shown) was scored by an independent LLM judge (Claude — a different model
family than the GPT extraction models) against a 4-question rubric (does the report support
gold? support the prediction? does the submitted evidence support it? is an explicit M token
present?), reconstructing each case's expected tag from the answers.

- **Design:** 9 `source_boundary_abstention` / 8 `unsupported_error` / 8 `report_gold_discordance`,
  arm-balanced within availability constraints. Reproduce: `scripts/05_audit_sample.py` builds
  the blinded sheet; `scripts/06_audit_score.py` scores it against `results/audit/judgments.json`.
- **Result:** **23 / 25 (92%) agreement.** `source_boundary_abstention` and
  `report_gold_discordance` were **100%** (9/9, 8/8) — the Track-C *tagging logic* was correct
  on every sampled case. All disagreement was concentrated in `unsupported_error` (6/8, 75%).
- **Root cause, not noise.** Both disagreements traced to the *same* mechanism: TCGA reports are
  OCR'd PDFs where line-wraps insert a stray period mid-phrase (`"under. investigation"`,
  `"cannot be. assessed."`). The model's evidence was semantically and substantively correct —
  it quoted the right passage — but the strict verbatim-substring check failed on the punctuation
  noise, mislabeling a grounded prediction as unsupported. **Fixed:** `grounded()` now strips
  periods/commas before matching (`src/stageground/evaluation/normalize.py`); all metrics above
  reflect the fix (`results/audit/` has the pre-fix sampling, judgments, and scoring for
  reproducibility). A larger reviewer study (N=100, per the original design) remains future work.

### Track A — accuracy stays roughly flat, except by design on M

| Stage | A | B | C | D |
| ----- | --- | --- | --- | --- |
| T | .89 | .90 | .91 | .90 |
| N | .80 | .80 | .79 | .79 |
| M | .26 | .05 | .17 | .035 † |

† Arms that abstain (B, D) score low on M **because they refuse to guess the dominant
`M0`** when the report gives no basis — not because they are wrong. Track A alone would
reward blindly guessing `M0`; Tracks B and C are the corrective.

## Scope

- **v0 (complete)**: data pipeline, Arms A (zero-shot) vs D (schema-guided),
  Track A (accuracy) + Track B (groundedness). The research claim, done.
- **Extension**:
  - ✅ Arms B (few-shot), C (allowed-values only) — constraint ladder complete (A/B/C/D).
  - ⬜ Arm E (supervised reference — trained-model ceiling; Track A/C only).
  - **Track C — source-boundary agreement**: crosses accuracy × text-support × abstention
    to separate principled abstention from genuine error (see below).
  - **Clinical review UI** (Next.js): a reviewer tool, not a dashboard — inspect *why* each
    prediction is grounded / unsupported / a source-boundary case.
  - **Serving**: FastAPI + Docker (`POST /extract`, `GET /cases`, `GET /case/{id}`, `GET /metrics`).
  - All three layers share one **per-case JSON contract** (`results/cases/<id>.json`) so the
    UI can be built on static JSON first and swapped to the API with no rewrite.
- **v2**: external validation on non-TCGA reports; additional variables.

### Track C — source-boundary agreement

Disagreement with gold is tagged by *why* it disagrees:

| Prediction vs gold | Condition | Tag |
| --- | --- | --- |
| matches | — | `correct` |
| abstains (`unknown`) | text has no basis for gold | **`source_boundary_abstention`** ✓ |
| abstains (`unknown`) | text supports gold | `over_abstention` |
| asserts other value | no verbatim evidence | `unsupported_error` ✗ |
| asserts other value | evidence supports it | `report_gold_discordance` |

The guardrail: only an *abstention* against an unsupportable gold counts as a source-boundary
success — an asserted wrong value with no evidence is an error, not a virtue. Headline metric:
**source-boundary success rate** = principled abstentions ÷ all gold-mismatch cases.
