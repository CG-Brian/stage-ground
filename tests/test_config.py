from stageground.config import ARM_CONFIGS, ExperimentArm, ExperimentConfig, ModelConfig


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
