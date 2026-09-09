> **Historical design document.** This is the original project plan, written
> before the ablation study existed. It describes a five-arm design
> (A/B/C/D/E), a Streamlit demo, and an "Arm E" that isn't how the project
> ended up using an external classifier (see `external_baselines/bbten/`
> instead). Kept for the record, not as current documentation — for what
> actually shipped, start at the [README](../../README.md), then
> [`docs/methods.md`](../methods.md).

# StageGround

**Source-Boundary-Aware TNM Extraction from Pathology Reports**

*Evaluating not only whether LLMs match registry labels, but whether their staging
claims are actually supported by the clinical document.*

Design document. Working repo name: `stage-ground` (changeable).

> **Positioning.** "Schema-guided extraction" is the *method*; the *research subject*
> is **source-boundary-aware clinical extraction evaluation** — distinguishing what a
> clinical document can and cannot support. Registry gold labels (`ajcc_pathologic_*`)
> reflect information from *outside* the pathology report (e.g. whole-body imaging for
> M stage); the model sees only the report. Track C measures exactly this boundary.

> **Scope split.**
> - **v0 (complete):** data pipeline, Arms A & D, Track A + Track B. The research claim.
> - **Extension (portfolio):** Arms B/C/E, Track C (source-boundary), clinical review UI,
>   FastAPI + Docker serving. Built on top of a finished v0, not because v0 is incomplete.

---

## 0. TL;DR

When an LLM extracts TNM / pathologic stage from TCGA pathology reports, does **schema-guided extraction** (allowed values + forced evidence spans + explicit abstain rule) produce extractions that are **better grounded in the source text and less prone to unsupported assertions** than **free-form** extraction — even if raw accuracy is similar?

This is a **clinical-AI validation research project**, not a product. The demo and any infrastructure exist only to display and reproduce the research result.

The headline contribution is a **dual-track evaluation**: Track A measures accuracy against gold labels; Track B measures whether each extraction is actually supported by the report text. Track B is what distinguishes this work from prior TNM-extraction papers.

---

## 1. Research question & hypotheses

**Primary question.** In no-fine-tuning LLM extraction of pathologic T/N/M stage from pathology report text, how do prompting strategies differ on (1) accuracy, (2) rate of unsupported assertions, and (3) rate of fabricated evidence?

Each method arm carries a specific hypothesis, and each hypothesis is mapped 1:1 to a metric:

| Arm                      | Hypothesis being tested                                                                             | Metric that answers it                                                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Zero-shot free-form      | How well does an unconstrained LLM do? (floor)                                                      | F1 (reference point for other arms)                                                         |
| Few-shot extraction      | Do a few examples improve format/accuracy?                                                          | ΔF1 + Δ invalid-output rate vs zero-shot                                                  |
| Schema-guided extraction | Do allowed-values + evidence-span + unknown-rule constraints reduce hallucination / invalid output? | **evidence-grounding rate ↑, unsupported-assertion rate ↓, invalid-output rate ↓** |

**Framing as a "constraint ladder."** The three arms increase constraint: none → weak nudge (examples) → explicit enforcement. The expected story is not "schema-guided wins" but **"as constraint increases, accuracy stays roughly flat while groundedness and safety improve."** This is the clinically meaningful and research-credible conclusion.

**Reference baseline (not a primary arm).** The Tatonetti-lab fine-tuned classifier (BERT/BigBird family) positions our no-fine-tuning results against a trained model. It is context, not the object of study.

---

## 2. Project identity (do not drift)

- This is a **research project**. Output ~90% research, ~10% thin demo.
- The demo is a **single-file Streamlit app** that visualizes the research claim (side-by-side extraction + evidence-span highlight). It is not a product.
- **FastAPI / Postgres / Docker are NOT part of v0.** A single `Dockerfile` for reproducibility is the only acceptable infra, and it is optional. Anything heavier is portfolio polish to be added *after* the research is done, if at all.

---

## 3. Data

All data comes from a single public source.

**Source.** `https://github.com/tatonetti-lab/tnm-stage-classifier`
**License.** MIT (free to use, modify, redistribute — safe for portfolio/GitHub).

### 3.1 Files used

| Path in repo                                    | Contents                                 |
| ----------------------------------------------- | ---------------------------------------- |
| `TCGA_Pathology_Reports/TCGA_Reports.csv.zip` | 9,523 machine-readable pathology reports |
| `TCGA_Metadata/TCGA_T14_patients.csv`         | gold`ajcc_pathologic_t`                |
| `TCGA_Metadata/TCGA_N03_patients.csv`         | gold`ajcc_pathologic_n`                |
| `TCGA_Metadata/TCGA_M01_patients.csv`         | gold`ajcc_pathologic_m`                |

`TCGA_Reports.csv` columns: `patient_filename`, `text`. (Reports are clean narrative prose, e.g. "Tumor Size: Greatest diameter is 2.4 cm. Fuhrman Nuclear Grade: Nuclear grade II/IV.")

Gold metadata columns: `ajcc_pathologic_t` / `_n` / `_m`, plus `project_id` (cancer type, e.g. `TCGA-BRCA`), `case_id`, `case_submitter_id`.

### 3.2 Join key (verified)

```python
reports['submitter_id'] = reports['patient_filename'].str.split('.').str[0]
# 'TCGA-BP-5195.25c0b433-...' -> 'TCGA-BP-5195'
# join to metadata on case_submitter_id  (clean 1:1, no fuzzy matching needed)
```

### 3.3 Label coverage (verified — this shapes the study)

|                        | patients with gold | % of 9,523 reports |
| ---------------------- | ------------------ | ------------------ |
| T                      | 6,966              | 73.1%              |
| N                      | 5,678              | 59.6%              |
| M                      | 4,608              | 48.4%              |
| **all of T,N,M** | **3,907**    | 41.0%              |
| any of T,N,M           | 6,994              | 73.4%              |

Coverage falls T > N > M. **This is not noise — it reflects that M stage often cannot be determined from a pathology report alone** (it needs whole-body imaging). M is therefore the natural stage for testing the abstention hypothesis.

### 3.4 Raw value distributions (verified)

```
T: T2(1722) T3(1378) T1(628) T3a(441) T1b(429) T1a(363) T4a(311) T3b(289)
   T2a(250) T4(228) T1c(218) T2b(197) T2c(156) T4b(147) TX(76) T1b1(71)
   T1b2(30) T2a2(11) T2a1(7) T4d(5) T3c(4) T0(3) T4c(1) T4e(1)
N: N0(3329) N1(1056) N2(434) N1a(276) N2b(127) N1b(123) N3(97) N3a(92)
   N2a(84) N2c(48) N3b(6) N1c(5) N3c(1)
M: M0(4295) M1(283) M1a(19) M1b(10) M1c(1)
```

Note `TX`(76), `T0`(3) exist in gold — these are real "evaluated but indeterminate" labels, useful for the abstention analysis. Subtypes (T3a, N2c, M1a...) are sparse; v0 collapses them (§4.1).

---

## 4. Extraction schema

Value domains are **fixed to AJCC 8th-edition pathologic staging** — not arbitrary. The gold labels are already `ajcc_pathologic_*`, so the schema is anchored to the same standard, eliminating circularity.

### 4.1 Canonicalization (applied identically to gold AND model output)

Both sides are folded to **major category** before scoring. Applying it to only one side would be unfair.

```python
def canonicalize(raw: str) -> str:
    # 'T3a'->'T3', 'T1b1'->'T1', 'N2c'->'N2', 'M1a'->'M1'
    # rule: keep leading letter + first digit, drop lowercase suffixes
    # preserve TX/NX/MX and T0/N0/M0 as-is
    ...
```

Rationale: sparse subtypes (T2a2 has 11 cases, N3c has 1) make per-subtype F1 meaningless in v0. Subtype-level extraction is **v1 future work**.

### 4.2 Value domains (after canonicalization)

```
T: {T0, T1, T2, T3, T4, TX, unknown}
N: {N0, N1, N2, N3, NX, unknown}
M: {M0, M1, MX, unknown}
```

### 4.3 Two distinct kinds of "don't know" — the spine of the study

These are **not** the same and must never be conflated:

- **`TX` / `NX` / `MX`** — a *gold value* meaning "officially assessed as indeterminate under the staging system." When gold is X, the correct prediction is X.
- **`unknown`** — a *model behavior* (abstention) meaning "no basis for a value exists in this report text." Not a gold value.
  The hypothesis ("guided abstains instead of asserting without evidence") is measured precisely at this distinction.

### 4.4 Output schema (all arms must return this JSON)

```python
# Pydantic v2
class Field(BaseModel):
    value: Literal[...]              # from the value domain for T/N/M
    evidence: str | None             # verbatim span copied from report text, or null
    confidence: Literal["high","medium","low"]
    reason: str | None = None        # required when value == "unknown"
 
class Extraction(BaseModel):
    T_stage: Field
    N_stage: Field
    M_stage: Field
    pathologic_stage: Field          # overall AJCC group stage (e.g. "Stage IIB"); optional in v0
```

**Output-format control (fairness).** *All* arms are required to emit this JSON shape. The only thing that varies across arms is **constraint strength**, not output format — otherwise "schema-guided wins because the others produced unparseable text" becomes a confound.

- Zero-shot free-form: asked for the JSON shape only.
- Few-shot: JSON shape + 3–5 worked examples.
- Schema-guided: JSON shape + examples + enforced allowed-values + evidence-required + explicit unknown rule ("if the report contains no basis for a value, output `unknown` with a reason; do not guess").

---

## 5. Methods (the four arms)

| Arm                        | Prompt                                                 | What it isolates          |
| -------------------------- | ------------------------------------------------------ | ------------------------- |
| A. Zero-shot free-form     | "Extract TNM stage as JSON"                            | LLM floor                 |
| B. Few-shot                | A + 3–5 examples                                      | value of examples         |
| C. JSON-structured         | strict JSON enforced                                   | invalid-output reduction  |
| D.**Schema-guided**  | allowed values + evidence span required + unknown rule | **core experiment** |
| E. Supervised reference | Tatonetti-style fine-tuned classifier                | **trained-model ceiling** |

Arms C and D may be merged in v0 if time is tight (start with A vs D — see §8).

**Arm E (extension).** The Tatonetti fine-tuned classifier, promoted from "reference
baseline" to a full arm: it sets the *ceiling* — how close can no-fine-tuning prompting
get to a trained model? Arm E produces no evidence span, so it is scored on **Track A
and Track C only**, never Track B.

---

## 6. Evaluation — dual track

This is the heart of the project. Two independent scoring tracks.

### 6.1 Track A — accuracy vs gold

Standard. Predicted (canonicalized) value vs canonicalized gold.

- Per-stage (T/N/M) precision, recall, F1.
- Invalid-output rate (schema violations / unparseable).
  This is what prior work measures. **On its own it is misleading**: a model that blindly outputs the majority class (e.g. M0) for reports with no M evidence looks *more* accurate while being *less* trustworthy.

### 6.2 Track B — groundedness vs text (the signature)

Every prediction is cross-classified by whether the report text actually supports it:

|                                  | text HAS basis | text has NO basis                   |
| -------------------------------- | -------------- | ----------------------------------- |
| **asserts a value**        | grounded ✓    | **unsupported assertion ✗**  |
| **abstains (`unknown`)** | over-cautious  | **appropriate abstention ✓** |

Primary Track-B metrics:

- **Evidence-grounding rate** — fraction of asserted values whose `evidence` span is found verbatim in the report text. Fabricated `evidence` (model writes `"pT2"` but that string is absent from the report) is the most damning failure and is caught here.
- **Unsupported-assertion rate** — fraction of predictions that assert a value with no textual basis. *Hypothesis: high for free-form, low for guided.*
- **Appropriate-abstention rate** — fraction of no-basis cases correctly returned as `unknown`. *Hypothesis: higher for guided.*
  Automatic grounding check (no manual labeling needed for the bulk):

```python
def evidence_grounded(field, report_text: str) -> bool:
    if field.evidence is None:
        return False                                   # no evidence offered
    return field.evidence.lower() in report_text.lower()  # verbatim presence
```

Manual spot-check ~100 cases to validate the automatic check; report agreement.

### 6.2b Track C — source-boundary agreement (the headline of the extension)

Track A (vs gold) and Track B (vs text) are **crossed**, not read separately. The
insight lives in the disagreement cases: when a prediction differs from gold, *why*?
Each prediction with gold present is tagged into exactly one type:

| Prediction relative to gold | Evidence / text condition | Track-C tag |
| --------------------------- | ------------------------- | ----------- |
| `pred == gold`              | —                         | `correct` |
| `pred != gold`, `pred == unknown` | text has **no** basis for gold | **`source_boundary_abstention`** ✓ |
| `pred != gold`, `pred == unknown` | text **does** support gold | `over_abstention` (a real miss) |
| `pred != gold`, asserted value | prediction has **no** verbatim evidence | `unsupported_error` ✗ |
| `pred != gold`, asserted value | evidence **supports** the prediction | `report_gold_discordance` (candidate) |

**Why the split matters (guardrail).** Labeling every "gold mismatch + no evidence" as
a source-boundary success would launder genuine errors into virtue. The tag must depend
on **prediction type**: only an *abstention* (`unknown`) against a gold value the text
cannot support is a `source_boundary_abstention`. An *asserted wrong value* with no
evidence is an `unsupported_error`, full stop. And an asserted value the text *does*
support while gold disagrees is a `report_gold_discordance` candidate — possibly the
registry encoding out-of-document information, worth manual review.

The headline metric is the **source-boundary success rate**: fraction of gold-mismatch
cases that are principled abstentions rather than errors. Expectation: high for D, ~0 for
A (A never abstains). This is the case that is *wrong by accuracy but right by clinical
reasoning* — e.g. gold `M0`, prediction `unknown`, no M descriptor in the report.

*Text support for gold* is approximated by whether the gold token (e.g. `p?M0`) appears
verbatim in the report; documented as a proxy, validated by the same spot-check as Track B.

### 6.3 The expected trade-off (this is the result)

Schema-guided may **lose a little Track-A recall** (it refuses to guess M0 when the text is silent) but should **dominate Track B**. Conclusion to aim for: *"guided trades a small amount of accuracy for substantially better groundedness and safe abstention"* — which is the clinically correct trade, since in medicine an unsupported assertion is more dangerous than an honest "unknown."

### 6.4 Analysis splits

- **easy vs hard** (see §6.5): expect arms to converge on easy, diverge on hard.
- **per stage** T/N/M: expect M to be the headline (low coverage, M0-dominated, little textual basis).
- **per cancer type** (`project_id`).

### 6.5 easy / hard tagging

```python
# easy: explicit stage token present in text
easy = bool(re.search(r'\bp?T[0-4X]', text))   # analogous for pN, pM
# hard: no explicit token -> requires inference from narrative
```

Report all metrics split by easy/hard. M will skew hard by construction.

---

## 7. Pipeline

```
1. Load        unzip TCGA_Reports.csv; load 3 metadata CSVs
2. Join        submitter_id = patient_filename.split('.')[0]  <-> case_submitter_id
3. Clean       canonicalize() gold T/N/M to major category
4. Tag         easy/hard per stage; attach cancer type
5. Sample      stratified eval subset (100-300 in v0) from the 3,907 fully-labeled patients
6. Extract     run arms A..D through the LLM -> Extraction JSON
7. Validate    Pydantic parse; allowed-values check; evidence-in-text check; unsupported-assertion flag
8. Score       Track A (vs gold) + Track B (vs text)
9. Analyze     splits (easy/hard, T/N/M, cancer type); method disagreement; M-hallucination case study
10. Demo       Streamlit: paste report -> A vs D side-by-side + evidence highlight + gold compare
```

---

## 8. Scope layers

**v0 (sufficient on its own)**

- Data load/join/clean/EDA on coverage.
- Eval subset of 100–300 fully-labeled patients.
- **Arms A (zero-shot free-form) vs D (schema-guided) only.**
- Report, as first-class results: F1 **and** evidence-grounding rate **and** unsupported-assertion rate.
  **v1**
- Add arms B (few-shot) and C (JSON-structured).
- Strengthen abstention analysis; subtype-level extraction.
- Streamlit demo with evidence-span highlight.
  **v2 (state as future work)**
- External validation on non-TCGA reports (generalization).
- Additional cancer types / additional variables (histology, grade, ECOG).

---

## 9. Repo structure

```
stageground/
  data/
    raw/                      # TCGA_Reports.csv, metadata CSVs (gitignored)
    processed/                # joined+canonicalized parquet, eval subset
  src/
    data/
      build_dataset.py        # load, join, canonicalize, easy/hard tag
      clean_labels.py         # canonicalize()
    extraction/
      schemas.py              # Pydantic Field / Extraction
      prompts.py              # one prompt builder per arm
      llm_client.py           # provider wrapper, retries, JSON parse
      extractors.py           # run an arm over a batch
    evaluation/
      track_a.py              # accuracy / P / R / F1 / invalid-rate
      track_b.py              # grounding / unsupported-assertion / abstention
      error_analysis.py       # splits, disagreement, M case study
    demo/
      app.py                  # single-file Streamlit
  tests/
    test_clean_labels.py      # canonicalize correctness
    test_schema_validation.py # Pydantic + allowed-values
    test_track_b.py           # evidence_grounded() edge cases
  notebooks/
    01_data_exploration.ipynb
    02_error_analysis.ipynb
  README.md
  pyproject.toml              # uv
  Dockerfile                  # optional, reproducibility only
```

---

## 10. Tech stack

**Core (the only thing that matters for v0):** Python + pandas; Pydantic v2 (schema enforcement / invalid detection); an LLM API (OpenAI/Azure); scikit-learn + pytest (metrics, tests); Streamlit (thin demo); SQLite or parquet (store runs/predictions/results); uv (env).

**Future / portfolio polish (optional, not v0):** FastAPI, Postgres, Docker beyond a single reproducibility Dockerfile.

---

## 11. Risks & guardrails

- **Circularity** — avoided by (a) anchoring the schema to AJCC (same standard as gold) and (b) Track B scoring against *text*, independent of the gold labels.
- **Format confound** — avoided by requiring the same JSON output shape from all arms; only constraint strength varies.
- **Majority-class illusion** — Track A alone rewards blindly guessing M0; Track B is the corrective.
- **Sparse subtypes** — collapsed to major category in v0; subtype work deferred.
- **Fabricated evidence** — explicitly detected (`evidence` span must appear verbatim in text).
- **Scope creep into product** — infra is footnoted, not built; research is ~90% of effort.

---

## 12. Narrative (for the application, not the code)

Industrial experience maps directly onto this clinical work:

- **Messy-data LLM normalization (C&E matrix)** → structured extraction from messy pathology text.
- **Explainable / auditable guided troubleshooting system** → schema-guided extraction with enforced evidence spans (the evidence-span highlight in the demo is the clinical port of "why was this flagged").
  **DHI fit.** Turning unstructured clinical text into auditable, structured, research-ready data assets; rigorous validation; groundedness and auditability — aligned with DHI's emphasis on data-into-action, validation, and deployment.

---

## Appendix A — verified facts to rely on

- 9,523 reports; columns `patient_filename`, `text`.
- Gold columns `ajcc_pathologic_{t,n,m}`; cancer type in `project_id`.
- Join: `patient_filename.split('.')[0]` == `case_submitter_id` (clean 1:1).
- Coverage: T 6,966 / N 5,678 / M 4,608 / all-three 3,907.
- License: MIT.
- `TX`(76), `T0`(3) present in gold; subtypes sparse.
- M is M0-dominated (4,295 / 4,608) and lowest-coverage → headline stage for abstention.

