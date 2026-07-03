"""Tests for the knob-application helpers."""
from pacetest.config import LoopConfig
from pacetest.knobs import (
    should_include_oracle,
    should_update_agent,
    should_update_tool,
    judgement_emphasis,
)


# ---- Determinism ----

def test_should_include_oracle_is_deterministic():
    """Same round + same config must give the same answer every call."""
    cfg = LoopConfig(feedback_strength=0.5)
    for r in range(20):
        first = should_include_oracle(r, cfg)
        second = should_include_oracle(r, cfg)
        assert first == second, f"round {r} not deterministic."


def test_update_predicates_are_deterministic():
    """Update predicates must also be deterministic per round."""
    cfg = LoopConfig(update_asymmetry=0.3)
    for r in range(20):
        a1, a2 = should_update_agent(r, cfg), should_update_agent(r, cfg)
        t1, t2 = should_update_tool(r, cfg), should_update_tool(r, cfg)
        assert a1 == a2, f"agent round {r} not deterministic."
        assert t1 == t2, f"tool round {r} not deterministic."


# ---- Extreme values ----

def test_feedback_strength_1_always_shows_oracle():
    cfg = LoopConfig(feedback_strength=1.0)
    for r in range(50):
        assert should_include_oracle(r, cfg) is True


def test_feedback_strength_0_never_shows_oracle():
    cfg = LoopConfig(feedback_strength=0.0)
    for r in range(50):
        assert should_include_oracle(r, cfg) is False


def test_update_asymmetry_0_tool_only():
    """At update_asymmetry=0.0, tool always updated, agent never."""
    cfg = LoopConfig(update_asymmetry=0.0)
    for r in range(50):
        assert should_update_agent(r, cfg) is False, f"agent updated at round {r}"
        assert should_update_tool(r, cfg) is True, f"tool skipped at round {r}"


def test_update_asymmetry_1_agent_only():
    """At update_asymmetry=1.0, agent always updated, tool never."""
    cfg = LoopConfig(update_asymmetry=1.0)
    for r in range(50):
        assert should_update_agent(r, cfg) is True, f"agent skipped at round {r}"
        assert should_update_tool(r, cfg) is False, f"tool updated at round {r}"


def test_update_asymmetry_half_updates_both():
    """Week 4 baseline: both artefacts rewritten every round."""
    cfg = LoopConfig(update_asymmetry=0.5)
    for r in range(50):
        assert should_update_agent(r, cfg) is True
        assert should_update_tool(r, cfg) is True


# ---- Fractional + emphasis ----

def test_feedback_strength_half_is_roughly_half():
    """fs=0.5 should give roughly 50% True over 200 rounds.

    Loose bounds (30-70%) so a hash-based pseudo-random sample does not
    flake. If this ever fails, the hash distribution is skewed and we
    have a real problem.
    """
    cfg = LoopConfig(feedback_strength=0.5)
    n_true = sum(should_include_oracle(r, cfg) for r in range(200))
    assert 60 <= n_true <= 140, f"got {n_true}/200 (expected ~100)"


def test_judgement_emphasis_branches():
    """Each emphasis branch returns a text mentioning the right side."""
    low = judgement_emphasis(LoopConfig(self_judgement_weight=0.0))
    mid = judgement_emphasis(LoopConfig(self_judgement_weight=0.5))
    high = judgement_emphasis(LoopConfig(self_judgement_weight=1.0))
    assert "oracle" in low.lower() and "secondary" in low.lower()
    assert "equal emphasis" in mid.lower() or "equally" in mid.lower()
    assert "agent" in high.lower() and "secondary" in high.lower()
    assert low != mid != high != low


if __name__ == "__main__":
    for name, fn in [
        ("test_should_include_oracle_is_deterministic", test_should_include_oracle_is_deterministic),
        ("test_update_predicates_are_deterministic", test_update_predicates_are_deterministic),
        ("test_feedback_strength_1_always_shows_oracle", test_feedback_strength_1_always_shows_oracle),
        ("test_feedback_strength_0_never_shows_oracle", test_feedback_strength_0_never_shows_oracle),
        ("test_update_asymmetry_0_tool_only", test_update_asymmetry_0_tool_only),
        ("test_update_asymmetry_1_agent_only", test_update_asymmetry_1_agent_only),
        ("test_update_asymmetry_half_updates_both", test_update_asymmetry_half_updates_both),
        ("test_feedback_strength_half_is_roughly_half", test_feedback_strength_half_is_roughly_half),
        ("test_judgement_emphasis_branches", test_judgement_emphasis_branches),
    ]:
        fn()
        print(f"{name} passed")
