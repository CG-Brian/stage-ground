"""Run a single arm over a batch of reports -> list[Extraction] (DESIGN §7 step 6)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from stageground.extraction.llm_client import complete_json
from stageground.extraction.prompts import (
    build_allowed_values_prompt,
    build_few_shot_prompt,
    build_schema_guided_prompt,
    build_zero_shot_prompt,
)
from stageground.extraction.schemas import RawExtraction


ARMS = {
    "A": build_zero_shot_prompt,      # zero-shot free-form
    "B": build_few_shot_prompt,       # few-shot (examples, no rules)
    "C": build_allowed_values_prompt, # allowed-values only
    "D": build_schema_guided_prompt,  # schema-guided (values + evidence + abstain)
}


@dataclass
class Result:
    extraction: RawExtraction | None
    raw: str
    invalid: bool


def run_one(arm: str, report_text: str) -> Result:
    system, user = ARMS[arm](report_text)
    raw = complete_json(system, user)
    try:
        ext = RawExtraction.model_validate_json(raw)  # shape only
        return Result(ext, raw, invalid=False)
    except (ValidationError, json.JSONDecodeError):
        return Result(None, raw, invalid=True)