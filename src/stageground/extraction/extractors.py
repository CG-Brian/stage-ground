"""Run a single arm over a batch of reports -> list[Extraction] (DESIGN §7 step 6)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from stageground.config import ARM_CONFIGS, ModelConfig
from stageground.extraction.llm_client import complete_json
from stageground.extraction.schemas import RawExtraction

# ARMS is DERIVED from ARM_CONFIGS -- not a second, separately-maintained
# mapping. Adding or changing an arm's prompt only requires editing
# `ARM_CONFIGS` in config.py; nothing here needs to change in step.
# Includes both the new ablation arm keys (ExperimentArm.value strings) and
# each arm's legacy letter key where one exists, so
# `scripts/02_run_extraction.py --arm A` keeps working unmodified.
ARMS = {arm.value: cfg.prompt_builder for arm, cfg in ARM_CONFIGS.items()}
ARMS.update({
    cfg.legacy_key: cfg.prompt_builder
    for cfg in ARM_CONFIGS.values()
    if cfg.legacy_key is not None
})


@dataclass
class Result:
    extraction: RawExtraction | None
    raw: str
    invalid: bool


def run_one(arm: str, report_text: str, *, model_config: ModelConfig | None = None) -> Result:
    """Run one arm's prompt against the LLM. When `model_config` is given
    (the normal path from `stageground.evaluate.run_evaluation`), its
    model/temperature/max_tokens/retries/response_format are passed through
    explicitly to `complete_json`, which is what actually reaches the API
    request -- an experiment's recorded config, not a module-level
    environment variable, decides what gets sent. `model_config=None` is the
    legacy path for standalone/ad-hoc callers with no `ExperimentConfig`."""
    system, user = ARMS[arm](report_text)
    if model_config is None:
        raw = complete_json(system, user)
    else:
        raw = complete_json(
            system, user,
            model=model_config.model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            retries=model_config.retries,
            response_format=model_config.response_format,
        )
    try:
        ext = RawExtraction.model_validate_json(raw)  # shape only
        return Result(ext, raw, invalid=False)
    except (ValidationError, json.JSONDecodeError):
        return Result(None, raw, invalid=True)