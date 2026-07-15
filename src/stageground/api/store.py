"""Read-side store: serves the same artifacts the review UI reads from disk.

`GET /cases`, `GET /case/{id}`, `GET /metrics` are backed by this. The per-case
JSON is the shared contract (identical shape to results/cases/<id>.json).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repo root
CASES_DIR = ROOT / "results" / "cases"
METRICS_DIR = ROOT / "results" / "metrics"


def list_case_ids() -> list[str]:
    return sorted(p.stem for p in CASES_DIR.glob("*.json"))


def get_case(case_id: str) -> dict | None:
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


ARMS = ["A", "B", "C", "D"]


@lru_cache(maxsize=1)
def case_summaries() -> list[dict]:
    """Lightweight list for the triage view (M-stage focus, all arms)."""
    out = []
    for cid in list_case_ids():
        c = get_case(cid)
        if c is None:
            continue
        row = {"id": cid, "gold_M": c["gold"]["M"]}
        for arm in ARMS:
            m = c["predictions"].get(arm, {}).get("M")
            row[f"{arm}_M"] = m["value_norm"] if m else None
            row[f"{arm}_tag"] = m["boundary_tag"] if m else None
        out.append(row)
    return out


def _read_parquet(name: str) -> list[dict]:
    path = METRICS_DIR / name
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return json.loads(df.to_json(orient="records"))  # NaN -> null


def metrics() -> dict:
    return {
        "track_ab": _read_parquet("summary.parquet"),
        "track_c": _read_parquet("track_c.parquet"),
    }
