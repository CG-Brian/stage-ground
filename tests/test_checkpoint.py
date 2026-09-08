import json

import httpx
import pandas as pd
import pytest
from openai import RateLimitError

from stageground.config import ModelConfig
from stageground.extraction import checkpoint as checkpoint_module
from stageground.extraction.checkpoint import (
    CheckpointedRunner,
    build_text_keyed_cache,
    deserialize_result,
    load_checkpoint,
    run_extraction_with_checkpoint,
    serialize_result,
)
from stageground.extraction.extractors import Result
from stageground.extraction.schemas import RawExtraction, RawField


def _fake_rate_limit_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _sample_df(n=5):
    return pd.DataFrame([
        {"patient_filename": f"case{i}", "text": f"Report text for case {i}, T2 N0 M0."}
        for i in range(n)
    ])


def _fake_extraction(value="T2"):
    return RawExtraction(
        T_stage=RawField(value=value, evidence=value, confidence="high", reason=None),
        N_stage=RawField(value="N0", evidence="N0", confidence="high", reason=None),
        M_stage=RawField(value="M0", evidence="M0", confidence="high", reason=None),
    )


def _counting_fake_runner(call_log: list):
    def runner(arm, text, *, model_config=None):
        call_log.append((arm, text))
        return Result(extraction=_fake_extraction(), raw="{}", invalid=False)
    return runner


MODEL_CONFIG = ModelConfig(model="fake-model")


def test_serialize_deserialize_roundtrip():
    result = Result(extraction=_fake_extraction(), raw='{"T_stage": {}}', invalid=False)
    row = serialize_result("case0", "C_constrained", result)
    restored = deserialize_result(row)
    assert restored.invalid is False
    assert restored.raw == result.raw
    assert restored.extraction.T_stage.value == "T2"


def test_serialize_deserialize_invalid_result():
    result = Result(extraction=None, raw="not json", invalid=True)
    row = serialize_result("case0", "A_zero_shot", result)
    restored = deserialize_result(row)
    assert restored.invalid is True
    assert restored.extraction is None


def test_load_checkpoint_missing_file_returns_empty(tmp_path):
    assert load_checkpoint(tmp_path / "nonexistent.jsonl") == {}


def test_fresh_run_computes_everything_and_writes_checkpoint(tmp_path):
    df = _sample_df(3)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    call_log = []
    runner = _counting_fake_runner(call_log)

    cache = run_extraction_with_checkpoint(
        df, ["A_zero_shot", "C_constrained"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=2, runner=runner,
    )

    assert len(cache) == 3 * 2
    assert len(call_log) == 3 * 2
    assert checkpoint_path.exists()
    lines = checkpoint_path.read_text().strip().splitlines()
    assert len(lines) == 6


def test_resume_skips_already_checkpointed_pairs(tmp_path):
    df = _sample_df(3)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    call_log_1 = []
    run_extraction_with_checkpoint(
        df, ["A_zero_shot", "C_constrained"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=2, runner=_counting_fake_runner(call_log_1),
    )
    assert len(call_log_1) == 6

    # "resume": call again with the same checkpoint -- must call the runner ZERO times
    call_log_2 = []
    cache = run_extraction_with_checkpoint(
        df, ["A_zero_shot", "C_constrained"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=2, runner=_counting_fake_runner(call_log_2),
    )
    assert len(call_log_2) == 0
    assert len(cache) == 6


def test_resume_only_computes_the_missing_arm(tmp_path):
    df = _sample_df(2)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    # first pass: only one arm
    run_extraction_with_checkpoint(
        df, ["A_zero_shot"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=2, runner=_counting_fake_runner([]),
    )
    # second pass: two arms -- only the NEW arm's 2 cases should be computed
    call_log = []
    cache = run_extraction_with_checkpoint(
        df, ["A_zero_shot", "D_grounded"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=2, runner=_counting_fake_runner(call_log),
    )
    assert len(call_log) == 2
    assert all(arm == "D_grounded" for arm, _ in call_log)
    assert len(cache) == 4  # 2 cases x 2 arms total


def test_persistent_runner_failure_recorded_as_invalid_not_crashed(tmp_path):
    df = _sample_df(2)
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    def flaky_runner(arm, text, *, model_config=None):
        if "case 0" in text:
            raise RuntimeError("simulated persistent API failure")
        return Result(extraction=_fake_extraction(), raw="{}", invalid=False)

    cache = run_extraction_with_checkpoint(
        df, ["A_zero_shot"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=2, runner=flaky_runner,
    )
    assert len(cache) == 2
    assert cache[("case0", "A_zero_shot")].invalid is True
    assert "simulated persistent API failure" in cache[("case0", "A_zero_shot")].raw
    assert cache[("case1", "A_zero_shot")].invalid is False


def test_build_text_keyed_cache_remaps_correctly():
    df = _sample_df(2)
    cache_by_case = {
        ("case0", "A_zero_shot"): Result(extraction=_fake_extraction(), raw="{}", invalid=False),
        ("case1", "A_zero_shot"): Result(extraction=None, raw="bad", invalid=True),
    }
    text_cache = build_text_keyed_cache(df, cache_by_case)
    assert text_cache[("A_zero_shot", "Report text for case 0, T2 N0 M0.")].invalid is False
    assert text_cache[("A_zero_shot", "Report text for case 1, T2 N0 M0.")].invalid is True


def test_checkpointed_runner_returns_cached_result():
    result = Result(extraction=_fake_extraction(), raw="{}", invalid=False)
    runner = CheckpointedRunner({("A_zero_shot", "some report text"): result})
    assert runner("A_zero_shot", "some report text") is result


def test_checkpointed_runner_raises_clear_error_on_missing_key():
    runner = CheckpointedRunner({})
    with pytest.raises(RuntimeError, match="No checkpointed result"):
        runner("A_zero_shot", "unseen report text")


def test_checkpoint_file_is_valid_jsonl_after_run(tmp_path):
    df = _sample_df(2)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    run_extraction_with_checkpoint(
        df, ["A_zero_shot"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=2, runner=_counting_fake_runner([]),
    )
    for line in checkpoint_path.read_text().strip().splitlines():
        row = json.loads(line)
        assert set(row.keys()) == {"case_id", "arm", "invalid", "raw", "extraction"}


# --- rate-limit round-retry behavior ---

def test_rate_limited_case_is_retried_in_a_later_round_not_marked_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda s: None)
    df = _sample_df(1)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    attempts = {"n": 0}

    def flaky_then_success(arm, text, *, model_config=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _fake_rate_limit_error()
        return Result(extraction=_fake_extraction(), raw="{}", invalid=False)

    cache = run_extraction_with_checkpoint(
        df, ["A_zero_shot"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=1, runner=flaky_then_success,
    )
    assert attempts["n"] == 2  # first attempt rate-limited, second succeeded
    result = cache[("case0", "A_zero_shot")]
    assert result.invalid is False  # NOT permanently marked invalid by the rate limit


def test_rate_limited_case_is_never_written_to_checkpoint_file(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda s: None)
    df = _sample_df(1)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    attempts = {"n": 0}

    def flaky_then_success(arm, text, *, model_config=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _fake_rate_limit_error()
        return Result(extraction=_fake_extraction(), raw="{}", invalid=False)

    run_extraction_with_checkpoint(
        df, ["A_zero_shot"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=1, runner=flaky_then_success,
    )
    lines = checkpoint_path.read_text().strip().splitlines()
    assert len(lines) == 1  # only the eventual success was ever written, not the 429


def test_permanent_rate_limiting_stops_after_max_rounds_without_hanging(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda s: None)
    df = _sample_df(1)
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    def always_rate_limited(arm, text, *, model_config=None):
        raise _fake_rate_limit_error()

    cache = run_extraction_with_checkpoint(
        df, ["A_zero_shot"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=1, runner=always_rate_limited,
        max_rounds=3,
    )
    assert cache == {}  # never succeeded, nothing cached, but returned instead of hanging
    assert checkpoint_path.read_text().strip() == ""  # nothing ever written


def test_genuine_non_rate_limit_failure_still_recorded_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda s: None)
    df = _sample_df(1)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    attempts = {"n": 0}

    def always_generic_error(arm, text, *, model_config=None):
        attempts["n"] += 1
        raise RuntimeError("persistent non-rate-limit failure")

    cache = run_extraction_with_checkpoint(
        df, ["A_zero_shot"], model_config=MODEL_CONFIG,
        checkpoint_path=checkpoint_path, max_workers=1, runner=always_generic_error,
        max_rounds=5,
    )
    assert attempts["n"] == 1  # recorded immediately, NOT retried across rounds like a rate limit
    assert cache[("case0", "A_zero_shot")].invalid is True
