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

## Clinical review app (extension)

Two servers: the FastAPI serving layer and the Next.js review UI.

```bash
# terminal 1 — API (reads results/cases + results/metrics; POST /extract hits the LLM)
uv sync --extra serve
uv run uvicorn stageground.api.app:app --reload    # http://localhost:8000/docs

# terminal 2 — review UI (fetches from the API above)
cd web && npm install && npm run dev               # http://localhost:3000
```

The UI reads only the `/case/{id}` + `/cases` contract, so the API and UI stay decoupled.
Point the UI at another API with `API_URL=... npm run dev`.

## Layout

```
src/stageground/   library code (data, extraction, evaluation, demo)
scripts/           pipeline entry points (run order)
data/              raw/ (gitignored) + processed/ parquet
results/           predictions/ + metrics/ parquet
tests/             unit tests (canonicalize, schema, evidence_grounded)
notebooks/         exploration + error analysis
```

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
