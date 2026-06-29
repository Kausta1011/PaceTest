"""The main closed rewriting loop for PaceTest."""
from pacetest.forward_pass import run_one_task
from pacetest.rewriter import rewrite_agent_prompt, rewrite_tool_doc
from pacetest.prompts import AGENT_PROMPT, TOOL_DOC
from pacetest.logger import init_log, log_round


def run_loop(tasks, num_rounds: int = 20,
             agent_prompt: str = None, tool_doc: str = None,
             run_name: str = None) -> dict:
    """Run K rounds of the closed agent-tool rewriting loop.

    Args:
        tasks: A list of task strings to cycle through.
        num_rounds: Number of rounds K. Default 20.
        agent_prompt: Starting prompt. Defaults to AGENT_PROMPT.
        tool_doc: Starting docstring. Defaults to TOOL_DOC.
        run_name: Optional log file identifier.

    Returns:
        Dict with final_agent_prompt, final_tool_doc, log_path.
    """
    if agent_prompt is None:
        agent_prompt = AGENT_PROMPT
    if tool_doc is None:
        tool_doc = TOOL_DOC

    log_path = init_log(run_name=run_name)
    current_prompt = agent_prompt
    current_doc = tool_doc

    for round_num in range(num_rounds):
        task = tasks[round_num % len(tasks)]
        print(f"[Round {round_num + 1}/{num_rounds}] Task: {task[:60]}...")

        result = run_one_task(task, current_prompt, current_doc)
        log_round(log_path, round_num, result)
        print(f"  success={result['success']}, answer={result['agent_answer']}")

        # No rewrite after the last round (no next round to use it)
        if round_num < num_rounds - 1:
            current_prompt = rewrite_agent_prompt(current_prompt, result)
            current_doc = rewrite_tool_doc(current_doc, result)

    return {
        "final_agent_prompt": current_prompt,
        "final_tool_doc": current_doc,
        "log_path": str(log_path),
    }


if __name__ == "__main__":
    tasks = [
        "What is 5 + 3?",
        "What is (10 + 5) * 2?",
        "What is 100 / 4?",
    ]
    out = run_loop(tasks, num_rounds=3, run_name="loop_smoke")
    print()
    print(f"Final agent prompt (first 200 chars): {out['final_agent_prompt'][:200]}")
    print(f"Final tool doc (first 200 chars): {out['final_tool_doc'][:200]}")
    print(f"Log saved to: {out['log_path']}")