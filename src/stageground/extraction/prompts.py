"""Per-arm prompt builders. v0: Arm A (zero-shot free-form) and D (schema-guided)."""

from __future__ import annotations

# Same required JSON SHAPE for every arm (DESIGN §4.4 fairness).
_SHAPE = """Return ONLY a JSON object with this shape:
{
  "T_stage": {"value": ..., "evidence": ..., "confidence": "high|medium|low", "reason": ...},
  "N_stage": {...},
  "M_stage": {...}
}
"evidence" is a verbatim span copied from the report, or null.
"reason" may be null."""


def build_zero_shot_prompt(report_text: str) -> tuple[str, str]: # Arm A
    system = "You extract TNM pathologic staging from pathology reports. " + _SHAPE
    return system, f"Pathology report:\n\n{report_text}"


# Few-shot examples for Arm B. Deliberately include ONE abstention case (M unknown)
# so we can test whether examples ALONE induce abstention, with no explicit rule.
_FEWSHOT = """Examples:

Report: "Primary tumor pT2 invades muscularis. Nodes: 0/9 negative. No distant metastasis identified (pM0)."
{"T_stage":{"value":"T2","evidence":"Primary tumor pT2 invades muscularis","confidence":"high","reason":null},
 "N_stage":{"value":"N0","evidence":"Nodes: 0/9 negative","confidence":"high","reason":null},
 "M_stage":{"value":"M0","evidence":"No distant metastasis identified (pM0)","confidence":"high","reason":null}}

Report: "Invasive carcinoma, pT3. 2 of 14 regional lymph nodes involved."
{"T_stage":{"value":"T3","evidence":"Invasive carcinoma, pT3","confidence":"high","reason":null},
 "N_stage":{"value":"N1","evidence":"2 of 14 regional lymph nodes involved","confidence":"high","reason":null},
 "M_stage":{"value":"unknown","evidence":null,"confidence":"low","reason":"report does not mention distant metastasis"}}
"""


def build_few_shot_prompt(report_text: str) -> tuple[str, str]: # Arm B
    # A's instruction + worked examples; no explicit allowed-values list, no rules.
    system = (
        "You extract TNM pathologic staging from pathology reports. "
        + _SHAPE
        + "\n\n"
        + _FEWSHOT
    )
    return system, f"Pathology report:\n\n{report_text}"


def build_allowed_values_prompt(report_text: str) -> tuple[str, str]: # Arm C
    # Value domain constrained, but NO evidence-verbatim rule and NO explicit
    # abstention/reason rule. Isolates the effect of value-constraint alone.
    system = (
        "You extract TNM pathologic staging from pathology reports.\n"
        "Allowed values (use EXACTLY one per field):\n"
        "  T: T0 T1 T2 T3 T4 TX unknown\n"
        "  N: N0 N1 N2 N3 NX unknown\n"
        "  M: M0 M1 MX unknown\n"
        + _SHAPE
    )
    return system, f"Pathology report:\n\n{report_text}"


def build_schema_guided_prompt(report_text: str) -> tuple[str, str]: # Arm D
    system = (
        "You extract TNM pathologic staging from pathology reports.\n"
        "Allowed values (use EXACTLY one per field):\n"
        "  T: T0 T1 T2 T3 T4 TX unknown\n"
        "  N: N0 N1 N2 N3 NX unknown\n"
        "  M: M0 M1 MX unknown\n"
        "Rules:\n"
        "- 'evidence' MUST be a verbatim substring copied from the report. "
        "Do not paraphrase or invent it.\n"
        "- If the report contains NO basis for a value, output \"unknown\" with a "
        "non-null 'reason'. Do NOT guess.\n"
        "- TX/NX/MX mean the report explicitly says the stage was assessed but "
        "indeterminate. This is different from 'unknown'.\n"
        + _SHAPE
    )
    return system, f"Pathology report:\n\n{report_text}"