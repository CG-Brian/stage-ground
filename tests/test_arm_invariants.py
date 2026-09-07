"""Spec item 9: fail if prompt/config behavior drifts between arms.

Deliberately checks required fragments, not full exact prompt strings, so
these tests stay meaningful (and don't turn into change-detector noise) as
prompt wording is tuned.
"""

from stageground.config import ARM_CONFIGS, ExperimentArm

SAMPLE_REPORT = "Primary tumor pT2 invades muscularis. Nodes: 0/9 negative."

EVIDENCE_MANDATE_FRAGMENT = "MUST be a verbatim substring"
ABSTENTION_ENCOURAGEMENT_FRAGMENTS = ("rather than guess", "do not guess", "do not guess.")


def _system_prompt(arm: ExperimentArm) -> str:
    system, _ = ARM_CONFIGS[arm].prompt_builder(SAMPLE_REPORT)
    return system


def test_constrained_has_allowed_values_but_no_evidence_mandate_or_encouragement():
    system = _system_prompt(ExperimentArm.CONSTRAINED)
    assert "Allowed values" in system
    assert EVIDENCE_MANDATE_FRAGMENT not in system
    assert "rather than guess" not in system.lower()
    assert ARM_CONFIGS[ExperimentArm.CONSTRAINED].requires_evidence is False
    assert ARM_CONFIGS[ExperimentArm.CONSTRAINED].encourages_unknown is False


def test_constrained_unknown_encourages_abstention_but_no_evidence_mandate():
    system = _system_prompt(ExperimentArm.CONSTRAINED_UNKNOWN)
    assert "Allowed values" in system
    assert "rather than guess" in system.lower()
    assert EVIDENCE_MANDATE_FRAGMENT not in system
    assert ARM_CONFIGS[ExperimentArm.CONSTRAINED_UNKNOWN].requires_evidence is False
    assert ARM_CONFIGS[ExperimentArm.CONSTRAINED_UNKNOWN].encourages_unknown is True


def test_grounded_has_both_evidence_mandate_and_abstention_encouragement():
    system = _system_prompt(ExperimentArm.GROUNDED)
    assert "Allowed values" in system
    assert EVIDENCE_MANDATE_FRAGMENT in system
    assert "do not guess" in system.lower()
    assert ARM_CONFIGS[ExperimentArm.GROUNDED].requires_evidence is True
    assert ARM_CONFIGS[ExperimentArm.GROUNDED].encourages_unknown is True


def test_zero_shot_and_few_shot_have_no_allowed_value_list():
    for arm in (ExperimentArm.ZERO_SHOT, ExperimentArm.FEW_SHOT):
        system = _system_prompt(arm)
        assert "Allowed values" not in system
        assert ARM_CONFIGS[arm].requires_evidence is False
        assert ARM_CONFIGS[arm].encourages_unknown is False
