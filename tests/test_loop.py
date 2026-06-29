"""Smoke test for the closed rewriting loop."""
from pacetest.loop import run_loop


def test_loop_runs_3_rounds():
    """End-to-end: 3 rounds, 3 simple tasks. Not strict on agent behavior."""
    tasks = ["What is 1 + 1?", "What is 2 + 2?", "What is 3 + 3?"]
    out = run_loop(tasks, num_rounds=3, run_name="test_loop_smoke")
    assert "final_agent_prompt" in out
    assert "final_tool_doc" in out
    assert "log_path" in out
    print(f"  Final prompt length: {len(out['final_agent_prompt'])} chars")
    print(f"  Final tool doc length: {len(out['final_tool_doc'])} chars")
    print(f"  Log saved to: {out['log_path']}")


if __name__ == "__main__":
    test_loop_runs_3_rounds()
    print("test_loop_runs_3_rounds passed")