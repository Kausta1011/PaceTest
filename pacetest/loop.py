"""The main closed rewriting loop for PaceTest.

The loop now consumes a list of Task objects (from pacetest.tasks) and
augments each round's result with oracle correctness info (from pacetest.oracle)
before logging. The log file therefore carries both protocol compliance
(`success`) and actual correctness (`correct`) per round, which is what
the sycophancy decoupling metric (Section 3.3, forthcoming) consumes.
"""
from pacetest.forward_pass import run_one_task
from pacetest.rewriter import rewrite_agent_prompt, rewrite_tool_doc
from pacetest.prompts import AGENT_PROMPT, TOOL_DOC
from pacetest.logger import init_log, log_round
from pacetest.tasks import Task, generate_tasks
from pacetest.oracle import score_round


def run_loop(tasks: list[Task], num_rounds: int = 20,
             agent_prompt: str = None, tool_doc: str = None,
             run_name: str = None, task_seed: int = None) -> dict:
    """Run K rounds of the closed agent-tool rewriting loop.

    Args:
        tasks: A list of Task objects to cycle through. Each Task carries
            its question string and its known gold_answer for the oracle.
        num_rounds: Number of rounds K. Default 20.
        agent_prompt: Starting prompt. Defaults to AGENT_PROMPT.
        tool_doc: Starting docstring. Defaults to TOOL_DOC.
        run_name: Optional log file identifier.
        task_seed: Optional task seed for this run. Recorded in the log
            header as `task_seed` so the run is fully reproducible from
            the log file alone.

    Returns:
        Dict with final_agent_prompt, final_tool_doc, log_path.
    """
    if agent_prompt is None:
        agent_prompt = AGENT_PROMPT
    if tool_doc is None:
        tool_doc = TOOL_DOC

    extra_metadata = {}
    if task_seed is not None:
        extra_metadata["task_seed"] = task_seed
    extra_metadata["num_tasks"] = len(tasks)
    extra_metadata["num_rounds"] = num_rounds

    log_path = init_log(run_name=run_name, extra_metadata=extra_metadata)
    current_prompt = agent_prompt
    current_doc = tool_doc

    for round_num in range(num_rounds):
        task = tasks[round_num % len(tasks)]
        print(
            f"[Round {round_num + 1}/{num_rounds}] "
            f"{task.task_id}: {task.question[:60]}..."
        )

        result = run_one_task(task.question, current_prompt, current_doc)
        scored = score_round(result, task)
        log_round(log_path, round_num, scored)
        print(
            f"  success={scored['success']}, "
            f"correct={scored['correct']}, "
            f"answer={scored['agent_answer']} (gold={scored['gold_answer']})"
        )

        # No rewrite after the last round (no next round to use it)
        if round_num < num_rounds - 1:
            current_prompt = rewrite_agent_prompt(current_prompt, scored)
            current_doc = rewrite_tool_doc(current_doc, scored)

    return {
        "final_agent_prompt": current_prompt,
        "final_tool_doc": current_doc,
        "log_path": str(log_path),
    }


if __name__ == "__main__":
    tasks = generate_tasks(seed=42, n=3)
    out = run_loop(tasks, num_rounds=3, run_name="loop_smoke")
    print()
    print(f"Final agent prompt (first 200 chars): {out['final_agent_prompt'][:200]}")
    print(f"Final tool doc (first 200 chars): {out['final_tool_doc'][:200]}")
    print(f"Log saved to: {out['log_path']}")
