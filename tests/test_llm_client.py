"""Tests for the permanent-quota-exhaustion vs transient-rate-limit distinction
in `complete_json` (see the comment above `_PERMANENT_QUOTA_ERROR_CODES` in
`stageground.extraction.llm_client`): both conditions surface as the same
`openai.RateLimitError` (HTTP 429), but only one of them is worth retrying."""

import httpx
import pytest
from openai import RateLimitError

from stageground.extraction import llm_client
from stageground.extraction.llm_client import QuotaExhaustedError, _is_permanent_quota_error


def _rate_limit_error(code: str | None = None):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    body = {"message": "rate limited", "type": "rate_limit_exceeded", "code": code}
    return RateLimitError("rate limited", response=response, body=body)


def _install_fake_client(monkeypatch, *, side_effect):
    from unittest.mock import MagicMock

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = side_effect
    monkeypatch.setattr(llm_client, "_client", fake_client)
    return fake_client


def test_is_permanent_quota_error_true_for_insufficient_quota():
    assert _is_permanent_quota_error(_rate_limit_error("insufficient_quota")) is True


def test_is_permanent_quota_error_true_for_credit_balance_exhausted():
    assert _is_permanent_quota_error(_rate_limit_error("credit_balance_exhausted")) is True


def test_is_permanent_quota_error_false_for_generic_rate_limit():
    assert _is_permanent_quota_error(_rate_limit_error("rate_limit_exceeded")) is False


def test_is_permanent_quota_error_false_for_no_code():
    assert _is_permanent_quota_error(_rate_limit_error(None)) is False


def test_insufficient_quota_raises_quota_exhausted_without_retrying(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_quota_exhausted(**kwargs):
        calls["n"] += 1
        raise _rate_limit_error("insufficient_quota")

    _install_fake_client(monkeypatch, side_effect=always_quota_exhausted)

    with pytest.raises(QuotaExhaustedError, match="no usable quota/credit left"):
        llm_client.complete_json("sys", "user", model="m", retries=3)

    assert calls["n"] == 1  # must NOT retry a permanent condition


def test_credit_balance_exhausted_raises_quota_exhausted_without_retrying(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_quota_exhausted(**kwargs):
        calls["n"] += 1
        raise _rate_limit_error("credit_balance_exhausted")

    _install_fake_client(monkeypatch, side_effect=always_quota_exhausted)

    with pytest.raises(QuotaExhaustedError):
        llm_client.complete_json("sys", "user", model="m", retries=3)

    assert calls["n"] == 1


def test_transient_rate_limit_without_quota_code_still_retries_and_succeeds(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky_then_success(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _rate_limit_error("rate_limit_exceeded")
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="{}"))]
        return resp

    _install_fake_client(monkeypatch, side_effect=flaky_then_success)

    result = llm_client.complete_json("sys", "user", model="m", retries=3)

    assert result == "{}"
    assert calls["n"] == 2  # retried past the transient rate limit, no QuotaExhaustedError
