"""06: score the blind-judge audit against the pipeline's Track C tags (Milestone 1a).

Compares judge answers (results/audit/judgments.json) to the pipeline tag each
case actually received (results/audit/answer_key.json), and writes a report.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

AUDIT_DIR = Path("results/audit")


def load(name: str) -> list[dict]:
    return json.loads((AUDIT_DIR / name).read_text())


def main() -> None:
    judgments = {j["case_id"]: j for j in load("judgments.json")}
    answers = {a["case_id"]: a for a in load("answer_key.json")}

    rows = []
    for cid, j in judgments.items():
        a = answers[cid]
        agree = j["reconstructed_tag"] == a["pipeline_tag"]
        rows.append({
            "case_id": cid, "arm": a["arm"], "pipeline_tag": a["pipeline_tag"],
            "judge_tag": j["reconstructed_tag"], "agree": agree,
            "caveat": j.get("caveat"),
        })

    n = len(rows)
    n_agree = sum(r["agree"] for r in rows)
    print(f"Overall agreement: {n_agree}/{n} = {n_agree/n:.0%}\n")

    print("Per-tag agreement (pipeline tag as reference):")
    by_tag = Counter(r["pipeline_tag"] for r in rows)
    for tag in by_tag:
        cell = [r for r in rows if r["pipeline_tag"] == tag]
        a = sum(r["agree"] for r in cell)
        print(f"  {tag:30s} {a}/{len(cell)} = {a/len(cell):.0%}")

    print("\nConfusion (pipeline_tag -> judge_tag) for disagreements:")
    for r in rows:
        if not r["agree"]:
            print(f"  {r['case_id']} (arm {r['arm']}): pipeline={r['pipeline_tag']} judge={r['judge_tag']}")

    caveats = [r for r in rows if r["caveat"]]
    if caveats:
        print(f"\n{len(caveats)} case(s) flagged with caveats (agree despite limitation):")
        for r in caveats:
            print(f"  {r['case_id']}: {r['caveat']}")

    (AUDIT_DIR / "audit_scored.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {AUDIT_DIR}/audit_scored.json")


if __name__ == "__main__":
    main()
