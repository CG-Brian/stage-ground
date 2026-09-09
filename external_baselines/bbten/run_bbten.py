"""Run the published BB-TEN classifiers (Tatonetti lab, jkefeli/CancerStage_*)
on StageGround's cases that overlap BB-TEN's own official TCGA held-out test
split -- see ../README.md for how that overlap was derived and verified.

This is intentionally a standalone script outside `src/stageground`: BB-TEN
is an external supervised baseline, not part of the StageGround ablation, and
its dependencies (torch, transformers) are deliberately NOT added to the main
project's environment. Install external_baselines/bbten/requirements.txt in
a separate virtualenv to run this.

Reads:
  - external_baselines/bbten/data/overlap_{T,N,M}.csv (case_id, target, gold)
  - data/processed/dataset.parquet (report text only -- StageGround's own
    dataset, not re-fetched from BB-TEN's repo, so the exact same text
    StageGround's GPT arms saw is what BB-TEN sees here too)

Writes:
  - external_baselines/bbten/bbten_predictions.csv

No StageGround experiment file is read for anything other than report text
lookup, and none is written to.

    python run_bbten.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, BigBirdForSequenceClassification

ROOT = Path(__file__).resolve().parent
SG_ROOT = ROOT.parent.parent

TOKENIZER_ID = "yikuan8/Clinical-BigBird"
MODEL_IDS = {
    "T": "jkefeli/CancerStage_Classifier_T",
    "N": "jkefeli/CancerStage_Classifier_N",
    "M": "jkefeli/CancerStage_Classifier_M",
}
# Confirmed from the official repo's T14_example.sh / N03_example.sh /
# M01_example.sh invocations -- NOT a uniform 2048 across all three targets.
MAX_TOKENS = {"T": 2048, "N": 2048, "M": 1024}
NUM_LABELS = {"T": 4, "N": 4, "M": 2}
LABEL_MAPS = {
    "T": {0: "T1", 1: "T2", 2: "T3", 3: "T4"},
    "N": {0: "N0", 1: "N1", 2: "N2", 3: "N3"},
    "M": {0: "M0", 1: "M1"},
}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    dataset = pd.read_parquet(SG_ROOT / "data" / "processed" / "dataset.parquet").set_index("patient_filename")

    all_results = []
    timings = {}

    for target in ["T", "N", "M"]:
        overlap = pd.read_csv(ROOT / "data" / f"overlap_{target}.csv")
        print(f"\n=== {target}: {len(overlap)} cases, loading {MODEL_IDS[target]} ===")

        t0 = time.time()
        model = BigBirdForSequenceClassification.from_pretrained(MODEL_IDS[target], num_labels=NUM_LABELS[target])
        model.eval()
        load_time = time.time() - t0
        print(f"model load time: {load_time:.1f}s")

        infer_times = []
        for _, row in overlap.iterrows():
            case_id = row["case_id"]
            gold = row["gold"]
            text = dataset.loc[case_id, "text"]

            # Token count WITHOUT truncation, to know whether this case was
            # actually truncated by the official max_length below.
            full_len = len(tokenizer.encode(text, truncation=False))
            truncated = full_len > MAX_TOKENS[target]

            t0 = time.time()
            inputs = tokenizer(text, truncation=True, max_length=MAX_TOKENS[target], padding=True, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            infer_time = time.time() - t0
            infer_times.append(infer_time)

            pred_label = LABEL_MAPS[target][pred_idx]
            all_results.append({
                "case_id": case_id,
                "target": target,
                "gold": gold,
                "bbten_prediction": pred_label,
                "bbten_confidence": round(float(probs[pred_idx]), 4),
                "bbten_correct": pred_label == gold,
                "input_token_count": full_len,
                "was_truncated": truncated,
            })

        del model
        timings[target] = {
            "load_seconds": round(load_time, 1),
            "mean_infer_seconds": round(sum(infer_times) / len(infer_times), 3),
            "total_seconds": round(load_time + sum(infer_times), 1),
            "n": len(infer_times),
        }
        print(f"{target} done: {timings[target]}")

    out_df = pd.DataFrame(all_results)
    out_path = ROOT / "bbten_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {len(out_df)} predictions -> {out_path}")

    for target, t in timings.items():
        print(f"  {target}: n={t['n']} load={t['load_seconds']}s mean_infer={t['mean_infer_seconds']}s total={t['total_seconds']}s")
    total_wall = sum(t["total_seconds"] for t in timings.values())
    print(f"  TOTAL wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)")


if __name__ == "__main__":
    main()
