# Audit Instructions

Review mode: **grounding-blind**
Gold labels are HIDDEN in this file. Judge span-grounding, semantic support, and source sufficiency without knowing the correct answer -- this avoids anchoring your grounding judgment on whether the prediction happens to be right. Leave `human_prediction_correct` `null` unless you already know the gold label independently.

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
