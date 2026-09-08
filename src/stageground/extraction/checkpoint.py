"""Resumable, concurrent extraction for large experiments (e.g. 1,000+ cases).

`stageground.evaluate.run_evaluation` calls its `runner` once per (case, arm)
sequentially and holds everything in memory until the very end -- fine for a
200-case pilot, risky for a 1,000-case run where a single interruption partway
through would otherwise lose all completed work and require redoing every
API call. This module adds a checkpointed pre-pass: extract every (case_id,
arm) pair (with modest thread-pool concurrency, since the OpenAI call is
I/O-bound), appending each result to an on-disk JSONL checkpoint as it
completes, then hand the fully-populated cache to `run_evaluation` via a
`CheckpointedRunner` so every downstream step (records, metrics, tables,
figures, bootstrap, config.json) is the SAME already-tested code path used
for the 200-case pilot -- nothing about post-processing is reimplemented here.

Re-running `run_extraction_with_checkpoint` against the same checkpoint file
after an interruption skips every (case_id, arm) pair already on disk and
only calls the runner for what's left -- no wasted API calls, no lost work.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
from openai import RateLimitError

from stageground.config import ModelConfig
from stageground.extraction.extractors import Result, run_one
from stageground.extraction.llm_client import QuotaExhaustedError
from stageground.extraction.schemas import RawExtraction

Runner = Callable[..., Result]

_RATE_LIMITED = object()  # sentinel: retry-later, never checkpointed as a final result


def serialize_result(case_id: str, arm: str, result: Result) -> dict:
    return {
        "case_id": case_id,
        "arm": arm,
        "invalid": result.invalid,
        "raw": result.raw,
        "extraction": result.extraction.model_dump() if result.extraction is not None else None,
    }


def deserialize_result(row: dict) -> Result:
    extraction = RawExtraction.model_validate(row["extraction"]) if row["extraction"] is not None else None
    return Result(extraction=extraction, raw=row["raw"], invalid=row["invalid"])


def load_checkpoint(path: Path) -> dict[tuple[str, str], Result]:
    """Returns `{(case_id, arm): Result}` for every row already on disk, or
    an empty dict if the checkpoint doesn't exist yet."""
    if not path.exists():
        return {}
    cache: dict[tuple[str, str], Result] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[(row["case_id"], row["arm"])] = deserialize_result(row)
    return cache


def run_extraction_with_checkpoint(
    sample_df: pd.DataFrame,
    arms: list[str],
    *,
    model_config: ModelConfig,
    checkpoint_path: Path,
    max_workers: int = 10,
    runner: Runner = run_one,
    progress_every: int = 50,
    max_rounds: int = 20,
    round_sleep_seconds: float = 20.0,
) -> dict[tuple[str, str], Result]:
    """Extracts `{(case_id, arm): Result}` for every case in `sample_df` x
    `arms`. Resumes from `checkpoint_path` if it already has rows; appends
    each freshly-computed row as soon as it completes (thread-safe), so an
    interruption at any point leaves a valid, resumable checkpoint.

    Runs in bounded ROUNDS: a request that fails with a rate-limit error
    (HTTP 429) even after the runner's own internal retries is NOT
    checkpointed as a final result -- it's skipped and retried in the next
    round, after `round_sleep_seconds`. This matters because OpenAI's
    tokens-per-minute limit is an infrastructure throttle, not a property of
    that particular case; permanently recording it as `invalid` would
    silently corrupt the experiment's real invalid/error rate. A genuine,
    non-rate-limit failure (e.g. persistent malformed output) IS recorded as
    invalid immediately, since retrying won't fix it. Rounds stop early once
    nothing is rate-limited; `max_rounds` bounds the total wait if the
    account's rate limit never clears.
    """
    cache = load_checkpoint(checkpoint_path)
    all_pairs = [
        (row.patient_filename, arm, row.text)
        for row in sample_df.itertuples()
        for arm in arms
    ]
    print(f"[extraction] {len(cache)}/{len(all_pairs)} already checkpointed")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    def _work(case_id: str, arm: str, text: str):
        try:
            result = runner(arm, text, model_config=model_config)
        except QuotaExhaustedError:
            raise  # permanent, account-level -- must abort the whole run, not just this case
        except RateLimitError:
            return case_id, arm, _RATE_LIMITED
        except Exception as exc:  # persistent, non-rate-limit failure
            return case_id, arm, Result(extraction=None, raw=f"ERROR: {exc!r}", invalid=True)
        return case_id, arm, result

    for round_num in range(1, max_rounds + 1):
        todo = [(cid, arm, text) for cid, arm, text in all_pairs if (cid, arm) not in cache]
        if not todo:
            break

        print(f"[extraction] round {round_num}: {len(todo)} remaining (max_workers={max_workers})")
        n_invalid = 0
        n_rate_limited = 0
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_work, case_id, arm, text) for case_id, arm, text in todo]
            with checkpoint_path.open("a") as f:
                for future in as_completed(futures):
                    case_id, arm, result = future.result()
                    if result is _RATE_LIMITED:
                        n_rate_limited += 1
                        continue
                    cache[(case_id, arm)] = result
                    if result.invalid:
                        n_invalid += 1
                    with write_lock:
                        f.write(json.dumps(serialize_result(case_id, arm, result), ensure_ascii=False) + "\n")
                        f.flush()
                    done += 1
                    if progress_every and (done % progress_every == 0 or done == len(todo)):
                        print(f"[extraction] {done}/{len(todo)} done this round ({n_invalid} invalid, {n_rate_limited} rate-limited so far)")

        if n_rate_limited == 0:
            break
        print(f"[extraction] {n_rate_limited} rate-limited this round; sleeping {round_sleep_seconds}s before retrying")
        time.sleep(round_sleep_seconds)

    return cache


def build_text_keyed_cache(
    sample_df: pd.DataFrame, cache_by_case: dict[tuple[str, str], Result]
) -> dict[tuple[str, str], Result]:
    """Remaps `{(case_id, arm): Result}` to `{(arm, report_text): Result}` --
    the shape `CheckpointedRunner` needs, since `run_evaluation` only passes
    `(arm, report_text)` to its runner, never `case_id`."""
    text_by_case = dict(zip(sample_df["patient_filename"], sample_df["text"]))
    return {
        (arm, text_by_case[case_id]): result
        for (case_id, arm), result in cache_by_case.items()
        if case_id in text_by_case
    }


class CheckpointedRunner:
    """Injectable `runner` for `stageground.evaluate.run_evaluation` that
    looks up a pre-computed `Result` instead of calling the API. Pass an
    instance as `run_evaluation(..., runner=CheckpointedRunner(cache))`."""

    def __init__(self, cache_by_text: dict[tuple[str, str], Result]):
        self._cache = cache_by_text

    def __call__(self, arm: str, report_text: str, *, model_config: ModelConfig | None = None) -> Result:
        key = (arm, report_text)
        try:
            return self._cache[key]
        except KeyError as exc:
            raise RuntimeError(
                f"No checkpointed result for arm={arm!r}: this report text was not part of the "
                "extraction pre-pass. run_evaluation's internal sample must exactly match the "
                "sample the checkpoint was built from (same dataset_df, sample_size, seed)."
            ) from exc
