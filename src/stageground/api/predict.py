"""Live extraction for POST /extract: run an arm on arbitrary report text.

No gold label here (ad-hoc text), so it returns value + evidence + grounding,
but no Track-A/Track-C tag. Same per-stage shape as the stored case contract.
"""

from __future__ import annotations

from stageground.evaluation.normalize import grounded, norm_pred
from stageground.extraction.extractors import run_one

STAGES = ["T", "N", "M"]


def extract(arm: str, text: str) -> dict:
    r = run_one(arm, text)
    preds = {}
    for s in STAGES:
        f = getattr(r.extraction, f"{s}_stage") if r.extraction else None
        raw = f.value if f else None
        evidence = f.evidence if f else None
        preds[s] = {
            "value": raw,
            "value_norm": norm_pred(raw),
            "evidence": evidence,
            "confidence": f.confidence if f else None,
            "reason": f.reason if f else None,
            "grounded": bool(grounded(evidence, text)),
        }
    return {"arm": arm, "invalid": r.invalid, "predictions": preds}
