"""Manual audit infrastructure (spec §6).

Generates a JSONL sheet (fits this project's JSON-heavy `results/cases/`
convention better than CSV, since fields like `automated_errors` are lists)
for a larger blind/manual audit than the v0 pilot's 25-case Track-C audit,
plus a scorer comparing automated judgments to filled-in human labels.

Two audit builders coexist:

  - `build_audit_sheet` (LEGACY, unchanged): one flat file, `human_supported`
    is ambiguous about span-vs-semantic, arm is visible to the reviewer. Kept
    only so previously generated audit files remain loadable/scorable; do
    not use for new audits.
  - `build_reviewer_and_key` (CURRENT): a blinded reviewer/key bundle with
    explicit `human_evidence_span_found` / `human_semantic_support` /
    `human_prediction_correct` / source-sufficiency fields, target-stratified
    (with M-stage oversampling) sampling, and no `arm` or `automated_*`
    fields visible to the reviewer. Use this for all new audits -- see the
    `build`/`score` CLI at the bottom of this module.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from stageground.evaluation.audit_sampling import classify_arm_pattern, group_by_case, stratified_audit_sample
from stageground.evaluation.records import PredictionRecord

MAX_EXCERPT_CHARS = 4000


def build_audit_sheet(
    records: list[PredictionRecord], texts: dict[str, str], *, n: int, seed: int
) -> list[dict]:
    """Deterministic (seeded) sample of `min(n, len(records))` rows. Each row
    carries the automated judgments (`automated_evidence_span_found`,
    `automated_semantic_support`, `automated_errors`) plus blank human_*
    fields for a reviewer to fill in and re-save."""
    k = min(n, len(records))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), k))

    rows = []
    for i in indices:
        rec = records[i]
        text = texts.get(rec.case_id, "")
        excerpt = (
            text if len(text) <= MAX_EXCERPT_CHARS
            else text[:MAX_EXCERPT_CHARS] + "\n\n[... truncated for audit sheet length ...]"
        )
        rows.append({
            "case_id": rec.case_id,
            "arm": rec.arm,
            "target": rec.target,
            "report_excerpt": excerpt,
            "ground_truth": rec.ground_truth,
            "prediction": rec.prediction,
            "evidence": rec.evidence,
            "automated_evidence_span_found": rec.evidence_span_found,
            "automated_semantic_support": rec.evidence_semantically_supports_prediction,
            "automated_errors": list(rec.errors),
            "human_supported": None,
            "human_evidence_correct": None,
            "human_prediction_correct": None,
            "human_error_type": "",
            "reviewer_notes": "",
        })
    return rows


def write_audit_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def read_audit_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _paired_bool_labels(rows: list[dict], *, automated_key, human_key: str) -> tuple[list[bool], list[bool]]:
    autos, humans = [], []
    for row in rows:
        human_val = row.get(human_key)
        if human_val is None:
            continue
        auto_val = automated_key(row)
        if auto_val is None:
            continue
        autos.append(bool(auto_val))
        humans.append(bool(human_val))
    return autos, humans


def _score_pair(autos: list[bool], humans: list[bool], *, label: str) -> dict:
    n_scored = len(autos)
    if n_scored == 0:
        return {
            f"n_scored_{label}": 0,
            f"percent_agreement_{label}": float("nan"),
            f"cohens_kappa_{label}": None,
        }

    agree = sum(1 for a, h in zip(autos, humans) if a == h)
    percent_agreement = agree / n_scored

    kappa = None
    if n_scored >= 2 and len(set(autos) | set(humans)) >= 2:
        from sklearn.metrics import cohen_kappa_score

        value = float(cohen_kappa_score(autos, humans))
        kappa = None if value != value else value  # NaN (zero-variance) -> None

    return {
        f"n_scored_{label}": n_scored,
        f"percent_agreement_{label}": percent_agreement,
        f"cohens_kappa_{label}": kappa,
    }


def score_audit(rows: list[dict]) -> dict:
    """Compares automated judgments to filled-in human labels wherever a
    human field is populated (not None), reporting percent agreement + Cohen's
    kappa for two judgment types: 'supported' and 'prediction_correct'
    (derived automatically as `prediction == ground_truth`). Rows with an
    unfilled human field for a given comparison are simply excluded from that
    comparison's `n_scored`, never causing a crash.

    NOTE: the 'supported' comparison pairs `human_supported` against
    `automated_evidence_span_found` (span-only), not the semantic-support
    field -- `human_supported`'s own name is ambiguous about which notion of
    grounding the reviewer judged and is a candidate for a future rename to
    e.g. `human_evidence_span_found` / `human_semantic_support`, out of
    scope for this pass."""
    supported_autos, supported_humans = _paired_bool_labels(
        rows, automated_key=lambda r: r.get("automated_evidence_span_found"), human_key="human_supported"
    )
    correct_autos, correct_humans = _paired_bool_labels(
        rows,
        automated_key=lambda r: (
            r.get("prediction") == r.get("ground_truth") if r.get("ground_truth") is not None else None
        ),
        human_key="human_prediction_correct",
    )

    result = {}
    result.update(_score_pair(supported_autos, supported_humans, label="supported"))
    result.update(_score_pair(correct_autos, correct_humans, label="prediction_correct"))
    return result


# ============================================================================
# Blinded reviewer/key bundle (CURRENT audit workflow, spec items 1-11)
# ============================================================================

AUDIT_MODES = ("grounding-blind", "with-gold")
DEFAULT_EXCERPT_WINDOW = 1500


def generate_audit_id(i: int) -> str:
    return f"A{i:04d}"


def _find_case_insensitive(text: str, needle: str) -> int | None:
    idx = text.lower().find(needle.lower())
    return idx if idx >= 0 else None


def _report_excerpt(
    text: str, evidence: str | None, *, max_chars: int = MAX_EXCERPT_CHARS, window: int = DEFAULT_EXCERPT_WINDOW
) -> tuple[str, bool]:
    """Spec item 9: don't blindly take the first N characters if that risks
    excluding relevant staging information.

    - Report fits within `max_chars`: show it in full (`truncated=False`).
    - Report is long AND `evidence` is locatable in it: show a window of
      `window` characters of context on each side of the evidence occurrence
      (`truncated=True`) -- a case-insensitive substring search, since this
      is display windowing, not the OCR-noise-tolerant grounding check.
    - Otherwise (abstained / no evidence / evidence not locatable): fall back
      to the first `max_chars` characters (`truncated=True`), a deterministic,
      documented strategy rather than silently guessing.
    """
    if len(text) <= max_chars:
        return text, False

    if evidence:
        idx = _find_case_insensitive(text, evidence)
        if idx is not None:
            start = max(0, idx - window)
            end = min(len(text), idx + len(evidence) + window)
            prefix = "...\n" if start > 0 else ""
            suffix = "\n..." if end < len(text) else ""
            return prefix + text[start:end] + suffix, True

    return text[:max_chars] + "\n\n[... truncated for audit sheet length ...]", True


def build_reviewer_and_key(
    records: list[PredictionRecord],
    texts: dict[str, str],
    *,
    target_counts: dict[str, int],
    seed: int,
    mode: str = "grounding-blind",
    m_priority_arm: str = "C_constrained",
    m_comparison_arm: str = "D_grounded",
    m_priority_fraction: float = 0.5,
    m_semantic_balance_fraction: float = 0.3,
) -> tuple[list[dict], list[dict], dict]:
    """Build the blinded reviewer-facing rows and the machine-only key rows.

    Blinding (spec items 4-5): reviewer rows carry NO `arm`, NO `case_id`,
    and NO `automated_*` judgments. In `mode="grounding-blind"` (recommended
    default) reviewer rows also omit `ground_truth` entirely, so a reviewer
    judging span/semantic/source-sufficiency isn't anchored by gold;
    `mode="with-gold"` includes it, for a pass focused on
    `human_prediction_correct`. `key.jsonl` always has everything
    (`case_id`, `arm`, `ground_truth`, automated judgments, and the M-stage
    `arm_pattern` -- see `stageground.evaluation.audit_sampling`), and is the
    only place `audit_id` is mapped back to a real case/arm.

    Returns `(reviewer_rows, key_rows, sampling_report)`.
    """
    if mode not in AUDIT_MODES:
        raise ValueError(f"unknown audit mode: {mode!r}, expected one of {AUDIT_MODES}")

    selected, sampling_report = stratified_audit_sample(
        records, target_counts=target_counts, seed=seed,
        m_priority_arm=m_priority_arm, m_comparison_arm=m_comparison_arm,
        m_priority_fraction=m_priority_fraction, m_semantic_balance_fraction=m_semantic_balance_fraction,
    )

    # Deterministic shuffle of final row order so the audit_id sequence
    # itself doesn't leak which sampling tier a case came from.
    rng = random.Random(seed)
    shuffled = list(selected)
    rng.shuffle(shuffled)

    # Arm-pattern is computed against the FULL pool (not just what got
    # sampled), since the comparison arm's record for a given case may not
    # itself have been selected for audit.
    by_case = group_by_case(records)

    reviewer_rows: list[dict] = []
    key_rows: list[dict] = []
    for i, rec in enumerate(shuffled, start=1):
        audit_id = generate_audit_id(i)
        text = texts.get(rec.case_id, "")
        excerpt, truncated = _report_excerpt(text, rec.evidence)

        reviewer_row = {
            "audit_id": audit_id,
            "target": rec.target,
            "report_excerpt": excerpt,
            "report_truncated": truncated,
            "prediction": rec.prediction,
            "evidence": rec.evidence,
            "human_evidence_span_found": None,
            "human_semantic_support": None,
            "human_prediction_correct": None,
            "human_source_has_explicit_stage": None,
            "human_source_has_inferential_evidence": None,
            "human_source_sufficient_for_stage": None,
            "human_confidence": "",
            "human_error_type": "",
            "reviewer_notes": "",
        }
        if mode == "with-gold":
            reviewer_row["ground_truth"] = rec.ground_truth
        reviewer_rows.append(reviewer_row)

        arm_pattern = (
            classify_arm_pattern(by_case.get(rec.case_id, {}), priority_arm=m_priority_arm, comparison_arm=m_comparison_arm)
            if rec.target == "M" else None
        )
        key_rows.append({
            "audit_id": audit_id,
            "case_id": rec.case_id,
            "arm": rec.arm,
            "target": rec.target,
            "ground_truth": rec.ground_truth,
            "prediction": rec.prediction,
            "automated_evidence_span_found": rec.evidence_span_found,
            "automated_semantic_support": rec.evidence_semantically_supports_prediction,
            "automated_errors": list(rec.errors),
            "arm_pattern": arm_pattern,
        })

    return reviewer_rows, key_rows, sampling_report


_INSTRUCTIONS_TEMPLATE = """\
# Audit Instructions

Review mode: **{mode}**
{mode_note}

Fill in every `human_*` field in `reviewer.jsonl` for each row and re-save
the file (one JSON object per line, same order/count). Leave a field `null`
if you are genuinely unsure or it doesn't apply (see "Abstention" below) --
do not force a confident guess.

## Fields

- **human_evidence_span_found** (true/false/null): Does the `evidence` text
  the model gave actually appear (verbatim or near-verbatim) somewhere in
  `report_excerpt`?
- **human_semantic_support** (true/false/null): ASSUMING the evidence span is
  real, does that evidence actually support the model's `prediction`? A
  quoted sentence can be real text from the report and still say the
  opposite of, or something unrelated to, the predicted stage.
- **human_prediction_correct** (true/false/null): Does `prediction` match the
  gold label? (Only answerable if `ground_truth` is shown to you -- see mode
  above; leave `null` if it isn't.)
- **human_source_has_explicit_stage** (true/false/null): Does the report
  contain an explicit stage token equivalent to the target value (e.g. "pT2",
  "N1", "M0" or a directly equivalent explicit statement)?
- **human_source_has_inferential_evidence** (true/false/null): Does the
  report contain clinical/pathologic information that COULD support a stage
  determination even without a literal token -- e.g. "tumor invades
  muscularis propria," "regional lymph node metastasis identified,"
  "distant metastasis identified"? This is a judgment call, not a rule
  lookup -- use your clinical/pathology reasoning.
- **human_source_sufficient_for_stage** (true/false/null): Stepping back --
  could a qualified reviewer reasonably determine the target TNM stage from
  this report text ALONE (no imaging, no other records)? This is especially
  important for M-stage, which often requires imaging the pathology report
  doesn't contain.
- **human_confidence** ("high" / "medium" / "low"): How confident are you in
  your judgments on this row overall?
- **human_error_type** (free text): Optional short tag for what went wrong,
  if anything (e.g. "fabricated_evidence", "wrong_stage", "ambiguous_report").
- **reviewer_notes** (free text): Anything else worth recording.

## Worked examples

**Semantic support (span found, but doesn't support the prediction):**
prediction = `M1`, evidence = `"No distant metastasis identified."`
-> `human_evidence_span_found = true`, `human_semantic_support = false`
(the quoted text is real, but it says the OPPOSITE of M1).

**Evidence absent:**
prediction = `N1`, evidence string does not appear anywhere in the report.
-> `human_evidence_span_found = false`, `human_semantic_support = false`.

**Abstention:**
prediction = `unknown`.
-> Leave `human_evidence_span_found` and `human_semantic_support` as `null`
(there is no evidence to judge). STILL complete the source-sufficiency
fields (`human_source_has_explicit_stage`,
`human_source_has_inferential_evidence`, `human_source_sufficient_for_stage`)
-- these describe the REPORT, not the model's behavior, and matter most for
exactly this case (was abstaining the right call?).

## Blinding

You are not shown which experimental arm produced each prediction, nor any
automated judgment about it. Please judge each row independently, based only
on the report excerpt and the stated prediction/evidence.
"""

_MODE_NOTES = {
    "grounding-blind": (
        "Gold labels are HIDDEN in this file. Judge span-grounding, semantic "
        "support, and source sufficiency without knowing the correct answer -- "
        "this avoids anchoring your grounding judgment on whether the "
        "prediction happens to be right. Leave `human_prediction_correct` "
        "`null` unless you already know the gold label independently."
    ),
    "with-gold": (
        "Gold labels ARE shown (`ground_truth` field) in this file. Use this "
        "mode for a pass focused on `human_prediction_correct`; try not to let "
        "the gold label bias your span/semantic-support judgments."
    ),
}


def _render_instructions(mode: str) -> str:
    return _INSTRUCTIONS_TEMPLATE.format(mode=mode, mode_note=_MODE_NOTES.get(mode, ""))


def write_audit_bundle(reviewer_rows: list[dict], key_rows: list[dict], config: dict, outdir: str | Path) -> None:
    """Writes `reviewer.jsonl`, `key.jsonl`, `config.json`, and `INSTRUCTIONS.md`
    into `outdir` (created if needed)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_audit_jsonl(reviewer_rows, outdir / "reviewer.jsonl")
    write_audit_jsonl(key_rows, outdir / "key.jsonl")
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    (outdir / "INSTRUCTIONS.md").write_text(_render_instructions(config.get("mode", "grounding-blind")))
