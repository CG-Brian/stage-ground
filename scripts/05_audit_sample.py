"""05: build a stratified, BLINDED sample for manual Track-C audit (Milestone 1a).

Produces two files that must be kept separate during judging:
  results/audit/blinded_sheet.json — report text, gold_M, pred_M value + evidence.
                                      NO arm name, NO pipeline tag, NO regex result.
  results/audit/answer_key.json    — same case_ids -> arm, patient_id, pipeline tag.
                                      Not to be opened until judging is complete.

Quotas (M-stage), scaled to a 25-case pilot and constrained by population size
(B unsupported_error=1, D unsupported_error=0, C source_boundary_abstention=2):
  source_boundary_abstention (9): A=2 B=3 C=2 D=2
  unsupported_error          (8): A=3 B=1 C=4 D=0
  report_gold_discordance    (8): A=2 B=2 C=2 D=2
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from stageground.evaluation.normalize import grounded, norm_pred
from stageground.evaluation.source_boundary import tag_case

PRED_DIR = Path("results/predictions")
DATASET = Path("data/processed/dataset.parquet")
OUTDIR = Path("results/audit")
ARMS = ["A", "B", "C", "D"]

QUOTAS = {
    "source_boundary_abstention": {"A": 2, "B": 3, "C": 2, "D": 2},
    "unsupported_error": {"A": 3, "B": 1, "C": 4, "D": 0},
    "report_gold_discordance": {"A": 2, "B": 2, "C": 2, "D": 2},
}

MAX_CHARS = 6000  # truncate pathological outliers; note if truncated


def _clean(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def build_pool() -> pd.DataFrame:
    data = pd.read_parquet(DATASET)[["patient_filename", "text", "gold_M"]]
    rows = []
    for arm in ARMS:
        pred = pd.read_parquet(PRED_DIR / f"{arm}.parquet")
        df = pred.merge(data, on="patient_filename", how="left")
        for r in df.itertuples():
            g = _clean(r.gold_M)
            if g is None:
                continue
            raw = _clean(r.pred_M_value)
            ev = _clean(r.pred_M_evidence)
            norm = norm_pred(raw)
            gr = grounded(ev, r.text)
            tag = tag_case(g, norm, gr, r.text)
            rows.append({
                "arm": arm, "patient_id": r.patient_filename, "text": r.text,
                "gold_M": g, "pred_M_raw": raw, "pred_M_norm": norm,
                "pred_M_evidence": ev, "pipeline_tag": tag,
            })
    return pd.DataFrame(rows)


def sample(pool: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    picked = []
    for tag, arm_quota in QUOTAS.items():
        for arm, n in arm_quota.items():
            if n == 0:
                continue
            cell = pool[(pool["pipeline_tag"] == tag) & (pool["arm"] == arm)]
            n = min(n, len(cell))
            picked.append(cell.sample(n, random_state=rng.randint(0, 10_000)))
    return pd.concat(picked, ignore_index=True)


def main() -> None:
    pool = build_pool()
    s = sample(pool)
    s = s.sample(frac=1, random_state=11).reset_index(drop=True)  # shuffle order

    OUTDIR.mkdir(parents=True, exist_ok=True)
    blinded, answer_key = [], []
    for i, r in enumerate(s.itertuples(), 1):
        case_id = f"case_{i:02d}"
        text = r.text
        truncated = len(text) > MAX_CHARS
        if truncated:
            text = text[:MAX_CHARS] + "\n\n[... truncated for audit sheet length ...]"
        blinded.append({
            "case_id": case_id,
            "report_text": text,
            "truncated": truncated,
            "gold_M": r.gold_M,
            "predicted_M_value": r.pred_M_norm,
            "predicted_M_raw": r.pred_M_raw,
            "submitted_evidence": r.pred_M_evidence,
        })
        answer_key.append({
            "case_id": case_id, "arm": r.arm, "patient_id": r.patient_id,
            "pipeline_tag": r.pipeline_tag,
        })

    (OUTDIR / "blinded_sheet.json").write_text(json.dumps(blinded, indent=2, ensure_ascii=False))
    (OUTDIR / "answer_key.json").write_text(json.dumps(answer_key, indent=2, ensure_ascii=False))

    print(f"wrote {len(blinded)} blinded cases -> {OUTDIR}/blinded_sheet.json")
    print(f"answer key (DO NOT open until judging is done) -> {OUTDIR}/answer_key.json")
    print("\ntag distribution in sample:")
    print(s["pipeline_tag"].value_counts().to_string())
    print("\narm distribution in sample:")
    print(s["arm"].value_counts().to_string())


if __name__ == "__main__":
    main()
