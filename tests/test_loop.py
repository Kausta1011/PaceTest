"""Smoke test for the closed rewriting loop."""
from pacetest.loop import run_loop
from pacetest.tasks import generate_tasks


def test_loop_runs_3_rounds():
    """End-to-end: 3 rounds with seeded Task objects.

    Not strict on agent behavior. The point is: the integrated pipeline
    (forward pass + oracle + rewriter + logger) does not crash and returns
    the expected fields.
    """
    tasks = generate_tasks(seed=42, n=3)
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
