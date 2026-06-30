"""Tests for the correctness oracle."""
from pacetest.oracle import is_correct, score_round
from pacetest.tasks import Task


def test_is_correct_matches_exact():
    """Equal numbers are scored correct."""
    assert is_correct(8.0, 8.0) is True
    assert is_correct(0.0, 0.0) is True
    assert is_correct(-5.0, -5.0) is True


def test_is_correct_rejects_mismatch():
    """Different numbers are scored incorrect."""
    assert is_correct(8.0, 9.0) is False
    assert is_correct(100.0, -100.0) is False


def test_is_correct_absorbs_float_error():
    """The tolerance must absorb IEEE 754 representation error.

    0.1 + 0.2 == 0.3 is False in raw floats; the oracle must say True.
    """
    assert is_correct(0.1 + 0.2, 0.3) is True


def test_is_correct_handles_none():
    """None on either side counts as not correct, never raises."""
    assert is_correct(None, 8.0) is False
    assert is_correct(8.0, None) is False
    assert is_correct(None, None) is False


def test_score_round_adds_fields():
    """score_round augments a result dict with task_id, gold_answer, correct."""
    task = Task(task_id="toy_0001", question="What is 5 + 3?", gold_answer=8.0)
    result = {"agent_answer": 8.0, "agent_response": "TOOL_CALL: ...\nANSWER: 8"}
    scored = score_round(result, task)
    assert scored["task_id"] == "toy_0001"
    assert scored["gold_answer"] == 8.0
    assert scored["correct"] is True
    # original fields are preserved
    assert scored["agent_answer"] == 8.0
    assert "agent_response" in scored


def test_score_round_does_not_mutate_input():
    """The oracle must not silently change the caller's result dict."""
    task = Task(task_id="toy_0002", question="What is 4 * 1?", gold_answer=4.0)
    result = {"agent_answer": 4.0}
    original_keys = set(result.keys())
    _ = score_round(result, task)
    assert set(result.keys()) == original_keys, "Input dict was mutated."


if __name__ == "__main__":
    test_is_correct_matches_exact()
    print("test_is_correct_matches_exact passed")
    test_is_correct_rejects_mismatch()
    print("test_is_correct_rejects_mismatch passed")
    test_is_correct_absorbs_float_error()
    print("test_is_correct_absorbs_float_error passed")
    test_is_correct_handles_none()
    print("test_is_correct_handles_none passed")
    test_score_round_adds_fields()
    print("test_score_round_adds_fields passed")
    test_score_round_does_not_mutate_input()
    print("test_score_round_does_not_mutate_input passed")
