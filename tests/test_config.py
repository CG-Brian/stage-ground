from stageground.config import ARM_CONFIGS, ExperimentArm, ExperimentConfig, ModelConfig
from stageground.extraction.extractors import ARMS
from stageground.extraction.llm_client import resolve_effective_model_config


def test_arm_enum_has_five_members_with_spec_values():
    values = {a.value for a in ExperimentArm}
    assert values == {
        "A_zero_shot",
        "B_few_shot",
        "C_constrained",
        "C_plus_unknown",
        "D_grounded",
    }


def test_every_arm_has_a_config():
    for arm in ExperimentArm:
        assert arm in ARM_CONFIGS
        assert ARM_CONFIGS[arm].arm == arm


def test_only_grounded_requires_evidence():
    for arm, cfg in ARM_CONFIGS.items():
        expected = arm == ExperimentArm.GROUNDED
        assert cfg.requires_evidence is expected, arm


def test_constrained_unknown_encourages_unknown_but_no_evidence():
    cfg = ARM_CONFIGS[ExperimentArm.CONSTRAINED_UNKNOWN]
    assert cfg.encourages_unknown is True
    assert cfg.requires_evidence is False


def test_constrained_does_not_encourage_unknown():
    cfg = ARM_CONFIGS[ExperimentArm.CONSTRAINED]
    assert cfg.encourages_unknown is False


def test_legacy_keys_map_to_old_letter_arms():
    assert ARM_CONFIGS[ExperimentArm.ZERO_SHOT].legacy_key == "A"
    assert ARM_CONFIGS[ExperimentArm.FEW_SHOT].legacy_key == "B"
    assert ARM_CONFIGS[ExperimentArm.CONSTRAINED].legacy_key == "C"
    assert ARM_CONFIGS[ExperimentArm.GROUNDED].legacy_key == "D"
    assert ARM_CONFIGS[ExperimentArm.CONSTRAINED_UNKNOWN].legacy_key is None


def test_experiment_config_roundtrip():
    cfg = ExperimentConfig(
        experiment_id="exp1",
        arms=[ExperimentArm.CONSTRAINED, ExperimentArm.GROUNDED],
        sample_size=200,
        seed=42,
        model=ModelConfig(model="gpt-4o", temperature=0.0, max_tokens=None, retries=3),
        prompt_version="v1",
        schema_version="v1",
        timestamp="2026-09-07T00:00:00+00:00",
    )
    d = cfg.to_dict()
    assert d["arms"] == ["C_constrained", "D_grounded"]
    assert d["model"]["model"] == "gpt-4o"

    restored = ExperimentConfig.from_dict(d)
    assert restored == cfg


def test_model_config_defaults_include_provider_and_response_format():
    cfg = ModelConfig(model="gpt-4o")
    assert cfg.provider == "openai"
    assert cfg.response_format == "json_object"


# --- ARM_CONFIGS as the single source of prompt dispatch ---

def test_every_arm_config_has_a_callable_prompt_builder():
    for arm, cfg in ARM_CONFIGS.items():
        system, user = cfg.prompt_builder("Sample pathology report text.")
        assert isinstance(system, str) and system
        assert isinstance(user, str) and user


def test_extractors_arms_derived_from_arm_configs_exactly():
    # 5 new keys + 4 legacy letter keys (C_plus_unknown has no legacy key) = 9
    assert len(ARMS) == 9
    for arm, cfg in ARM_CONFIGS.items():
        assert ARMS[arm.value] is cfg.prompt_builder  # same object -> no drift possible
        if cfg.legacy_key is not None:
            assert ARMS[cfg.legacy_key] is cfg.prompt_builder


# --- resolve_effective_model_config ---

def test_reasoning_model_temperature_is_normalized():
    requested = ModelConfig(model="gpt-5-mini", temperature=0.7)
    effective = resolve_effective_model_config(requested)
    assert effective.temperature is None
    assert effective.model == "gpt-5-mini"
    assert requested.temperature == 0.7  # input untouched


def test_non_reasoning_model_temperature_is_unchanged():
    requested = ModelConfig(model="gpt-4o", temperature=0.7)
    effective = resolve_effective_model_config(requested)
    assert effective.temperature == 0.7


def test_reasoning_model_default_temperature_is_left_alone():
    requested = ModelConfig(model="o1-mini", temperature=None)
    effective = resolve_effective_model_config(requested)
    assert effective.temperature is None
