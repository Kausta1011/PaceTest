"""Tests for the embedding-based cycling metrics.

These tests will trigger the one-time sentence-transformers model load
on first run (about 2-5 seconds). Subsequent runs use the cached model.
"""
from pacetest.embedding_metrics import (
    semantic_distance_agent,
    semantic_distance_tool,
    joint_trajectory_variance,
)


def _r(agent_prompt: str, tool_doc: str = "d"):
    return {"agent_prompt": agent_prompt, "tool_doc": tool_doc}


def test_identical_agent_prompts_zero_distance():
    """Same agent_prompt every round -> semantic_distance_agent ~ 0."""
    rs = [_r("You are an arithmetic agent.") for _ in range(4)]
    d = semantic_distance_agent(rs)
    assert d < 1e-5, f"expected ~0, got {d}"


def test_different_agent_prompts_positive_distance():
    """Different agent_prompts each round -> distance > 0."""
    rs = [
        _r("You are an arithmetic agent."),
        _r("Always use the calculator tool."),
        _r("Divide by zero should be avoided."),
    ]
    d = semantic_distance_agent(rs)
    assert d > 0.0, f"expected positive, got {d}"


def test_semantic_distance_deterministic():
    """Same input produces the same output across calls."""
    rs = [_r("hello"), _r("world"), _r("foobar")]
    a = semantic_distance_agent(rs)
    b = semantic_distance_agent(rs)
    assert abs(a - b) < 1e-6


def test_semantic_distance_single_round_is_zero():
    """Fewer than 2 rounds means no transitions -> 0.0."""
    assert semantic_distance_agent([_r("only one")]) == 0.0
    assert semantic_distance_agent([]) == 0.0


def test_semantic_distance_tool_symmetric_shape():
    """Same properties hold for the tool version."""
    rs = [_r("p", "same doc") for _ in range(3)]
    assert semantic_distance_tool(rs) < 1e-5
    rs2 = [_r("p", "doc one"), _r("p", "totally different second doc")]
    assert semantic_distance_tool(rs2) > 0.0


def test_joint_variance_zero_for_static_trajectory():
    """Identical (prompt, tool_doc) every round -> variance ~ 0."""
    rs = [_r("static prompt", "static doc") for _ in range(4)]
    v = joint_trajectory_variance(rs)
    assert v < 1e-5, f"expected ~0, got {v}"


def test_joint_variance_positive_for_drifting_trajectory():
    """Different (prompt, tool_doc) each round -> variance > 0."""
    rs = [
        _r("first prompt", "first doc"),
        _r("second entirely different prompt", "second entirely different doc"),
        _r("third prompt about oceans", "third doc about mountains"),
    ]
    v = joint_trajectory_variance(rs)
    assert v > 0.0, f"expected positive, got {v}"


if __name__ == "__main__":
    for name, fn in [
        ("test_identical_agent_prompts_zero_distance", test_identical_agent_prompts_zero_distance),
        ("test_different_agent_prompts_positive_distance", test_different_agent_prompts_positive_distance),
        ("test_semantic_distance_deterministic", test_semantic_distance_deterministic),
        ("test_semantic_distance_single_round_is_zero", test_semantic_distance_single_round_is_zero),
        ("test_semantic_distance_tool_symmetric_shape", test_semantic_distance_tool_symmetric_shape),
        ("test_joint_variance_zero_for_static_trajectory", test_joint_variance_zero_for_static_trajectory),
        ("test_joint_variance_positive_for_drifting_trajectory", test_joint_variance_positive_for_drifting_trajectory),
    ]:
        fn()
        print(f"{name} passed")
