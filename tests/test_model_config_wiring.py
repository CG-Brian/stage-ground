"""Spec item 11, Tests A-D: prove ExperimentConfig.model reaches the actual
OpenAI request layer, with no live API calls (the OpenAI client is mocked
directly at the module level `llm_client._client`)."""

from unittest.mock import MagicMock

from stageground.config import ModelConfig
from stageground.extraction import llm_client
from stageground.extraction.extractors import run_one


def _fake_response(content="{}"):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


def _install_fake_client(monkeypatch, *, side_effect=None):
    captured = {}
    fake_client = MagicMock()

    def default_create(**kwargs):
        captured.update(kwargs)
        return _fake_response()

    fake_client.chat.completions.create.side_effect = side_effect or default_create
    monkeypatch.setattr(llm_client, "_client", fake_client)
    return captured


# --- Test A: explicit model reaches the request ---

def test_explicit_model_config_reaches_request(monkeypatch):
    captured = _install_fake_client(monkeypatch)

    run_one(
        "C_constrained", "Report text",
        model_config=ModelConfig(model="gpt-9-experiment", temperature=0.3, max_tokens=50, retries=1),
    )

    assert captured["model"] == "gpt-9-experiment"
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 50


# --- Test B: OPENAI_MODEL env var does not override an explicit model_config ---

def test_env_var_does_not_override_explicit_model(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "should-not-be-used")
    captured = _install_fake_client(monkeypatch)

    run_one("C_constrained", "Report text", model_config=ModelConfig(model="explicit-model", retries=1))

    assert captured["model"] == "explicit-model"


# --- Test C: legacy call with no model_config still works via the default ---

def test_legacy_call_without_model_config_uses_default(monkeypatch):
    captured = _install_fake_client(monkeypatch)

    run_one("C_constrained", "Report text")  # no model_config -> legacy path

    assert captured["model"] == llm_client.DEFAULT_MODEL


# --- Test D: temperature/max_tokens/retries handled consistently ---

def test_none_temperature_and_max_tokens_are_omitted_from_request(monkeypatch):
    captured = _install_fake_client(monkeypatch)

    run_one(
        "C_constrained", "Report text",
        model_config=ModelConfig(model="m", temperature=None, max_tokens=None, retries=1),
    )

    assert "temperature" not in captured
    assert "max_tokens" not in captured


def test_retries_are_respected_on_transient_failure(monkeypatch):
    attempts = {"n": 0}

    def flaky_create(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient")
        return _fake_response()

    _install_fake_client(monkeypatch, side_effect=flaky_create)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)  # skip real backoff

    result = llm_client.complete_json("sys", "user", model="m", retries=3)

    assert attempts["n"] == 2
    assert result == "{}"


def test_retries_exhausted_raises(monkeypatch):
    def always_fails(**kwargs):
        raise RuntimeError("persistent failure")

    _install_fake_client(monkeypatch, side_effect=always_fails)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)

    import pytest
    with pytest.raises(RuntimeError, match="persistent failure"):
        llm_client.complete_json("sys", "user", model="m", retries=2)
