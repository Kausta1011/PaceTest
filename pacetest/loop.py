"""The main closed rewriting loop for PaceTest.

From Week 5 Day 4 onward the loop consumes a `LoopConfig` and applies
per-round knob decisions:

- `should_include_oracle(round_num, config)` decides whether the
  rewriter sees the oracle's `correct` field for this round. The logged
  round always carries `correct`; the rewriter's view of the round may
  or may not.

- `should_update_agent(round_num, config)` decides whether the agent
  prompt is rewritten at the end of this round.

- `should_update_tool(round_num, config)` decides whether the tool doc
  is rewritten at the end of this round.

The config values are recorded in the log header alongside the model,
task seed, and environment metadata, so the log file is fully
reproducible from its own header.
"""
from pacetest.forward_pass import run_one_task
from pacetest.rewriter import rewrite_agent_prompt, rewrite_tool_doc
from pacetest.prompts import AGENT_PROMPT, TOOL_DOC
from pacetest.logger import init_log, log_round
from pacetest.tasks import Task, generate_tasks
from pacetest.oracle import score_round
from pacetest.config import LoopConfig
from pacetest.knobs import (
    should_include_oracle,
    should_update_agent,
    should_update_tool,
)


def run_loop(tasks: list[Task], num_rounds: int = 20,
             agent_prompt: str = None, tool_doc: str = None,
             run_name: str = None, task_seed: int = None,
             config: LoopConfig = None) -> dict:
    """Run K rounds of the closed agent-tool rewriting loop.

    Args:
        tasks: A list of Task objects to cycle through.
        num_rounds: Number of rounds K. Default 20.
        agent_prompt: Starting prompt. Defaults to AGENT_PROMPT.
        tool_doc: Starting docstring. Defaults to TOOL_DOC.
        run_name: Optional log file identifier.
        task_seed: Optional task seed. Recorded in the log header.
        config: Optional LoopConfig with the three knob values. Defaults
            to a Week-4-like baseline (fs=1.0, sjw=0.5, ua=0.5).

    Returns:
        Dict with final_agent_prompt, final_tool_doc, log_path.
    """
    if agent_prompt is None:
        agent_prompt = AGENT_PROMPT
    if tool_doc is None:
        tool_doc = TOOL_DOC
    if config is None:
        config = LoopConfig()

    extra_metadata = {}
    if task_seed is not None:
        extra_metadata["task_seed"] = task_seed
    extra_metadata["num_tasks"] = len(tasks)
    extra_metadata["num_rounds"] = num_rounds
    # Difficulty is a property of the tasks; read it off the first one.
    if tasks:
        extra_metadata["difficulty"] = tasks[0].difficulty
    # Knob values flow into the header so the run is self-documenting.
    extra_metadata.update(config.asdict())

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
        # The full scored dict (with `correct`) is always logged, so
        # metrics can be computed post-hoc regardless of feedback_strength.
        log_round(log_path, round_num, scored)
        print(
            f"  success={scored['success']}, "
            f"correct={scored['correct']}, "
            f"answer={scored['agent_answer']} (gold={scored['gold_answer']})"
        )

        # No rewrite after the last round (no next round to use it).
        if round_num < num_rounds - 1:
            # Feedback strength: possibly hide `correct` from the rewriter.
            if should_include_oracle(round_num, config):
                rewriter_input = scored
            else:
                rewriter_input = {k: v for k, v in scored.items() if k != "correct"}

            # Update asymmetry: possibly rewrite each artefact independently.
            if should_update_agent(round_num, config):
                current_prompt = rewrite_agent_prompt(
                    current_prompt, rewriter_input, config,
                )
            if should_update_tool(round_num, config):
                current_doc = rewrite_tool_doc(
                    current_doc, rewriter_input, config,
                )

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
