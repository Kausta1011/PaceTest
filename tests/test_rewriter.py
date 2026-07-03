"""Tests for the rewriter meta-prompt builder.

These tests exercise the string-shaping helpers directly and never call
the LLM, so they run in milliseconds. Behaviour of the full rewriter
(with LLM call + sanity fallback) is covered by the loop smoke test.
"""
from pacetest.config import LoopConfig
from pacetest.rewriter import (
    _build_agent_meta_prompt,
    _build_tool_meta_prompt,
)


def _sample_result_with_oracle():
    return {
        "task": "What is 5 + 3?",
        "agent_response": "TOOL_CALL: calculator(\"5 + 3\")\nANSWER: 8",
        "tool_call_expr": "5 + 3",
        "tool_output": 8.0,
        "tool_error": None,
        "agent_answer": 8.0,
        "success": True,
        "correct": True,
    }


def _sample_result_without_oracle():
    r = _sample_result_with_oracle()
    del r["correct"]
    return r


def test_agent_meta_prompt_includes_current_prompt():
    """The current agent prompt must appear verbatim in the meta-prompt."""
    text = _build_agent_meta_prompt(
        "You are an arithmetic agent...",
        _sample_result_with_oracle(),
        LoopConfig(),
    )
    assert "You are an arithmetic agent..." in text


def test_agent_meta_prompt_shows_oracle_when_present():
    """`correct` must appear as True/False when present in the result."""
    text = _build_agent_meta_prompt(
        "prompt",
        _sample_result_with_oracle(),
        LoopConfig(),
    )
    assert "Oracle correctness: True" in text
    assert "not available" not in text


def test_agent_meta_prompt_marks_oracle_missing():
    """When the loop stripped `correct`, the meta-prompt must say so."""
    text = _build_agent_meta_prompt(
        "prompt",
        _sample_result_without_oracle(),
        LoopConfig(),
    )
    assert "(not available this round)" in text


def test_agent_meta_prompt_includes_emphasis_directive():
    """The judgement_emphasis text for the config must appear."""
    text_low = _build_agent_meta_prompt(
        "prompt",
        _sample_result_with_oracle(),
        LoopConfig(self_judgement_weight=0.0),
    )
    text_high = _build_agent_meta_prompt(
        "prompt",
        _sample_result_with_oracle(),
        LoopConfig(self_judgement_weight=1.0),
    )
    assert "REWRITE GUIDANCE:" in text_low
    assert "REWRITE GUIDANCE:" in text_high
    # The two emphasis regimes must produce different text.
    assert text_low != text_high


def test_agent_meta_prompt_preserves_format_markers():
    """The TOOL_CALL and ANSWER format markers must survive every rewrite.

    Without them the sanity fallback rejects everything the LLM ever
    produces, and the loop never rewrites at all.
    """
    text = _build_agent_meta_prompt(
        "prompt",
        _sample_result_with_oracle(),
        LoopConfig(),
    )
    assert "TOOL_CALL: calculator" in text
    assert "ANSWER:" in text


def test_tool_meta_prompt_includes_current_doc_and_emphasis():
    """The tool doc meta-prompt must include the current doc and the emphasis."""
    text = _build_tool_meta_prompt(
        "calculator(expr: str) -> float",
        _sample_result_with_oracle(),
        LoopConfig(),
    )
    assert "calculator(expr: str) -> float" in text
    assert "REWRITE GUIDANCE:" in text


def test_tool_meta_prompt_shows_oracle_state():
    """Tool doc meta-prompt must also reflect oracle presence/absence."""
    with_oracle = _build_tool_meta_prompt(
        "doc", _sample_result_with_oracle(), LoopConfig()
    )
    without_oracle = _build_tool_meta_prompt(
        "doc", _sample_result_without_oracle(), LoopConfig()
    )
    assert "Oracle correctness: True" in with_oracle
    assert "(not available this round)" in without_oracle


if __name__ == "__main__":
    for name, fn in [
        ("test_agent_meta_prompt_includes_current_prompt", test_agent_meta_prompt_includes_current_prompt),
        ("test_agent_meta_prompt_shows_oracle_when_present", test_agent_meta_prompt_shows_oracle_when_present),
        ("test_agent_meta_prompt_marks_oracle_missing", test_agent_meta_prompt_marks_oracle_missing),
        ("test_agent_meta_prompt_includes_emphasis_directive", test_agent_meta_prompt_includes_emphasis_directive),
        ("test_agent_meta_prompt_preserves_format_markers", test_agent_meta_prompt_preserves_format_markers),
        ("test_tool_meta_prompt_includes_current_doc_and_emphasis", test_tool_meta_prompt_includes_current_doc_and_emphasis),
        ("test_tool_meta_prompt_shows_oracle_state", test_tool_meta_prompt_shows_oracle_state),
    ]:
        fn()
        print(f"{name} passed")
