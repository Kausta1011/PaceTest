"""Tests for the LoopConfig container."""
from dataclasses import FrozenInstanceError

from pacetest.config import LoopConfig


def test_defaults_match_week4_baseline():
    """Defaults must reproduce Week 4 baseline behaviour.

    If this test fails, any run that omits a config argument silently
    changes behaviour and breaks comparison with earlier logs.

    self_judgement_weight defaults to 0.5 (balanced), which is the
    closest continuous value to the Week 4 rewriter's neutral behaviour
    (its meta-prompt referenced neither the oracle nor the agent's
    self-critique explicitly).
    """
    c = LoopConfig()
    assert c.feedback_strength == 1.0
    assert c.self_judgement_weight == 0.5
    assert c.update_asymmetry == 0.5


def test_valid_custom_values_accepted():
    """Any values in [0.0, 1.0] should be accepted without error."""
    c = LoopConfig(
        feedback_strength=0.25,
        self_judgement_weight=0.75,
        update_asymmetry=1.0,
    )
    assert c.feedback_strength == 0.25
    assert c.self_judgement_weight == 0.75
    assert c.update_asymmetry == 1.0


def test_too_high_values_rejected():
    """Values above 1.0 must raise ValueError."""
    for field, value in [
        ("feedback_strength", 1.5),
        ("self_judgement_weight", 2.0),
        ("update_asymmetry", 100.0),
    ]:
        try:
            LoopConfig(**{field: value})
            assert False, f"Should have rejected {field}={value}"
        except ValueError:
            pass


def test_negative_values_rejected():
    """Values below 0.0 must raise ValueError."""
    for field, value in [
        ("feedback_strength", -0.01),
        ("self_judgement_weight", -1.0),
        ("update_asymmetry", -0.5),
    ]:
        try:
            LoopConfig(**{field: value})
            assert False, f"Should have rejected {field}={value}"
        except ValueError:
            pass


def test_non_numeric_values_rejected():
    """Passing a string or None for a knob must raise ValueError."""
    for bad in ["medium", None, [0.5]]:
        try:
            LoopConfig(feedback_strength=bad)
            assert False, f"Should have rejected feedback_strength={bad!r}"
        except ValueError:
            pass


def test_config_is_frozen():
    """Mutating a LoopConfig after creation must raise FrozenInstanceError."""
    c = LoopConfig()
    try:
        c.feedback_strength = 0.5
        assert False, "Should have raised on mutation."
    except FrozenInstanceError:
        pass


def test_asdict_returns_all_fields():
    """asdict() must return a plain dict with the three knob values."""
    c = LoopConfig(
        feedback_strength=0.5,
        self_judgement_weight=0.5,
        update_asymmetry=0.5,
    )
    d = c.asdict()
    assert d == {
        "feedback_strength": 0.5,
        "self_judgement_weight": 0.5,
        "update_asymmetry": 0.5,
    }


if __name__ == "__main__":
    test_defaults_match_week4_baseline()
    print("test_defaults_match_week4_baseline passed")
    test_valid_custom_values_accepted()
    print("test_valid_custom_values_accepted passed")
    test_too_high_values_rejected()
    print("test_too_high_values_rejected passed")
    test_negative_values_rejected()
    print("test_negative_values_rejected passed")
    test_non_numeric_values_rejected()
    print("test_non_numeric_values_rejected passed")
    test_config_is_frozen()
    print("test_config_is_frozen passed")
    test_asdict_returns_all_fields()
    print("test_asdict_returns_all_fields passed")
