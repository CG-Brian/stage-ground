"""OpenAI wrapper: call + retry, returns RAW json string (parsing done in extractors)."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazily construct the client on first real use, not at import time --
    modules that only need e.g. `run_one`'s type/dict shape (evaluation code,
    tests injecting a fake runner) can import this package without an
    OPENAI_API_KEY set."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def complete_json(system:str, user:str, *, retries: int = 3) -> str:
    """Return the model's raw response text (expected to be JSON). Retries on transient errors."""
    for attempt in range(retries):
        try:
            resp = _get_client().chat.completions.create(
                model=MODEL,
                # NOTE: gpt-5/o-series models only allow the default temperature (1);
                # they reject temperature=0. Left at default for cross-model compatibility.
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")