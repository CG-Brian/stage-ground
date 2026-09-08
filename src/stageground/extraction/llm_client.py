"""OpenAI wrapper: call + retry, returns RAW json string (parsing done in extractors).

`complete_json`'s `model`/`temperature`/`max_tokens`/`retries`/`response_format`
parameters are what an `ExperimentConfig.model` (see `stageground.config`)
ultimately controls -- an experiment's recorded `ModelConfig` must be the
thing that actually reaches this call, not a module-level environment
default, or `config.json` could describe a run that never happened
(spec: "config.json says model A but the actual API call uses model B" must
not be possible). `DEFAULT_MODEL`/keyword defaults below exist ONLY for
legacy/standalone callers that don't have an `ExperimentConfig` at all.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import replace

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from stageground.config import ModelConfig

load_dotenv()

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

_client: OpenAI | None = None

# OpenAI's o-series/gpt-5 "reasoning" models reject any explicit temperature
# other than the provider default (1) -- a request with temperature=0, for
# example, is rejected outright rather than silently clamped.
_REASONING_MODEL_RE = re.compile(r"^(o[1-9](-mini)?|gpt-5)", re.IGNORECASE)


def is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_RE.match(model))


def resolve_effective_model_config(model_config: ModelConfig) -> ModelConfig:
    """Resolve exactly what will be sent to the provider, normalizing known
    incompatible combinations up front rather than sending a request the API
    would reject. Never mutates `model_config`; returns a new `ModelConfig`.
    Callers (e.g. `stageground.evaluate.run_evaluation`) must record BOTH the
    originally-requested config and this resolved one, so nothing is silently
    substituted after the fact."""
    temperature = model_config.temperature
    if is_reasoning_model(model_config.model) and temperature not in (None, 1.0):
        temperature = None  # fall back to the provider default
    return replace(model_config, temperature=temperature)


# Without an explicit timeout, a request whose underlying TCP connection
# dies silently (e.g. the machine sleeps mid-request during a long unattended
# run) hangs forever rather than raising -- which, under a ThreadPoolExecutor
# with a small worker count, can permanently occupy every worker and stall
# the entire extraction with no error, no retry, and no progress. Discovered
# during the 1000-case run: the process survived a laptop sleep/wake cycle
# but every in-flight request never returned, silently halting all progress
# for hours while still showing as "running." 120s is generous for a normal
# chat completion but still bounded.
REQUEST_TIMEOUT_SECONDS = 120.0


def _get_client() -> OpenAI:
    """Lazily construct the client on first real use, not at import time --
    modules that only need e.g. `run_one`'s type/dict shape (evaluation code,
    tests injecting a fake runner) can import this package without an
    OPENAI_API_KEY set."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=REQUEST_TIMEOUT_SECONDS)
    return _client


def complete_json(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retries: int = 3,
    response_format: str = "json_object",
) -> str:
    """Return the model's raw response text (expected to be JSON). Retries on
    transient errors. `model`/`temperature`/`max_tokens`/`retries` default to
    legacy environment-derived behavior only when the caller passes none of
    them explicitly -- an explicit `ModelConfig` (via `run_one`) always wins.

    Rate-limit errors (HTTP 429) get a much longer, dedicated backoff than
    other transient errors: OpenAI's tokens-per-minute limits reset on a
    ~60s window, so the previous flat `2**attempt` backoff (max ~4s) could
    exhaust `retries` entirely without ever waiting long enough for the
    window to clear -- which would silently mislabel a rate-limit throttle
    as a genuine model/parse failure to every caller of this function.
    """
    for attempt in range(retries):
        try:
            kwargs: dict = {
                "model": model,
                "response_format": {"type": response_format},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = _get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except RateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 15 * (attempt + 1)))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")
