"""Tests for the forward pass."""
from pacetest.forward_pass import parse_tool_call, parse_answer, run_one_task


def test_parse_tool_call():
    assert parse_tool_call('TOOL_CALL: calculator("5 + 3")') == "5 + 3"
    assert parse_tool_call('TOOL_CALL: calculator("(10 * 2)")') == "(10 * 2)"
    assert parse_tool_call("no tool call here") is None


def test_parse_answer():
    assert parse_answer("ANSWER: 8") == 8.0
    assert parse_answer("ANSWER: 3.14") == 3.14
    assert parse_answer("ANSWER: -5") == -5.0
    assert parse_answer("no answer here") is None


def test_run_one_task_smoke():
    """End-to-end smoke test. Calls the LLM; not strict on output."""
    result = run_one_task("What is 5 + 3?")
    assert isinstance(result, dict)
    assert "agent_response" in result
    print(f"  task: {result['task']}")
    print(f"  tool_call_expr: {result['tool_call_expr']!r}")
    print(f"  tool_output: {result['tool_output']}")
    print(f"  agent_answer: {result['agent_answer']}")
    print(f"  success: {result['success']}")


if __name__ == "__main__":
    test_parse_tool_call()
    print("test_parse_tool_call passed")
    test_parse_answer()
    print("test_parse_answer passed")
    test_run_one_task_smoke()
    print("test_run_one_task_smoke passed")