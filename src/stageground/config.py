"""Centralized experiment-arm configuration.

Single source of truth for which prompt/schema requirements define each
experimental arm, so prompts, schema requirements, and evaluation settings
are not scattered across the codebase. Every new arm-conditional behavior
(prompting, evidence requirements, metrics) should read from `ARM_CONFIGS`
rather than branching on arm name string literals.

`ExperimentArm` values are the *new* ablation arm names. The v0 pilot's
letter-coded arms (A/B/C/D in `stageground.extraction.extractors.ARMS`) are
preserved unmodified for backward compatibility; `ArmConfig.legacy_key` links
each new arm to its old key where one exists (`C_plus_unknown` is genuinely
new and has no legacy counterpart).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable

from stageground.extraction.prompts import (
    build_allowed_values_prompt,
    build_constrained_unknown_prompt,
    build_few_shot_prompt,
    build_schema_guided_prompt,
    build_zero_shot_prompt,
)

PromptBuilder = Callable[[str], tuple[str, str]]


class ExperimentArm(str, Enum):
    ZERO_SHOT = "A_zero_shot"
    FEW_SHOT = "B_few_shot"
    CONSTRAINED = "C_constrained"
    CONSTRAINED_UNKNOWN = "C_plus_unknown"
    GROUNDED = "D_grounded"


@dataclass(frozen=True)
class ArmConfig:
    arm: ExperimentArm
    legacy_key: str | None  # old letter key in extractors.ARMS, or None if new
    prompt_builder: PromptBuilder  # the ONE place arm -> prompt dispatch is defined
    allow_unknown: bool  # is "unknown" a sanctioned output for this arm?
    requires_evidence: bool  # must non-unknown predictions carry an evidence span?
    encourages_unknown: bool  # does the prompt tell the model to prefer unknown over guessing?
    prompt_version: str
    schema_version: str = "v1"
    description: str = ""


ARM_CONFIGS: dict[ExperimentArm, ArmConfig] = {
    ExperimentArm.ZERO_SHOT: ArmConfig(
        arm=ExperimentArm.ZERO_SHOT,
        legacy_key="A",
        prompt_builder=build_zero_shot_prompt,
        allow_unknown=True,
        requires_evidence=False,
        encourages_unknown=False,
        prompt_version="v1",
        description="Free-form zero-shot, JSON shape only, no allowed-value list, no rules.",
    ),
    ExperimentArm.FEW_SHOT: ArmConfig(
        arm=ExperimentArm.FEW_SHOT,
        legacy_key="B",
        prompt_builder=build_few_shot_prompt,
        allow_unknown=True,
        requires_evidence=False,
        encourages_unknown=False,
        prompt_version="v1",
        description="Zero-shot + worked examples (one abstention example), no explicit rules.",
    ),
    ExperimentArm.CONSTRAINED: ArmConfig(
        arm=ExperimentArm.CONSTRAINED,
        legacy_key="C",
        prompt_builder=build_allowed_values_prompt,
        allow_unknown=True,
        requires_evidence=False,
        encourages_unknown=False,
        prompt_version="v1",
        description=(
            "Allowed-value list enforced; no evidence rule; no abstention "
            "encouragement beyond 'unknown' being one of the listed values."
        ),
    ),
    ExperimentArm.CONSTRAINED_UNKNOWN: ArmConfig(
        arm=ExperimentArm.CONSTRAINED_UNKNOWN,
        legacy_key=None,
        prompt_builder=build_constrained_unknown_prompt,
        allow_unknown=True,
        requires_evidence=False,
        encourages_unknown=True,
        prompt_version="v1",
        description=(
            "Same as C_constrained, plus an explicit instruction that 'unknown' "
            "is preferred over guessing when the report does not support a "
            "stage. No mandatory evidence span."
        ),
    ),
    ExperimentArm.GROUNDED: ArmConfig(
        arm=ExperimentArm.GROUNDED,
        legacy_key="D",
        prompt_builder=build_schema_guided_prompt,
        allow_unknown=True,
        requires_evidence=True,
        encourages_unknown=True,
        prompt_version="v1",
        description=(
            "Allowed values + explicit unknown rule + mandatory verbatim "
            "evidence span for every non-unknown prediction."
        ),
    ),
}


@dataclass(frozen=True)
class ModelConfig:
    model: str
    provider: str = "openai"  # provenance only; llm_client.py is OpenAI-specific
    temperature: float | None = None  # None = provider default; see llm_client.py
    max_tokens: int | None = None
    retries: int = 3
    response_format: str = "json_object"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    arms: list[ExperimentArm]
    sample_size: int
    seed: int
    model: ModelConfig
    prompt_version: str
    schema_version: str
    timestamp: str  # ISO 8601

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arms"] = [a.value for a in self.arms]
        d["model"] = asdict(self.model)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            experiment_id=d["experiment_id"],
            arms=[ExperimentArm(a) for a in d["arms"]],
            sample_size=d["sample_size"],
            seed=d["seed"],
            model=ModelConfig(**d["model"]),
            prompt_version=d["prompt_version"],
            schema_version=d["schema_version"],
            timestamp=d["timestamp"],
        )
