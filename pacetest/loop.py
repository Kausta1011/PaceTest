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

From Week 7 Day 4 onward the loop also consults the active pacemaker
(if any) after the rewriter has produced a candidate agent prompt. The
pacemaker is chosen by `config.pacemaker` (`None`, `"freeze"`,
`"diversity"`, or `"gating"`). Its verdict is applied to the candidate
and the resolved prompt is used for the next round. The verdict is
recorded on the following round's log entry via the
`starting_pacemaker_verdict` field.

Only the agent prompt is pacemakered. Tool doc rewrites always pass
through unchanged.

The config values are recorded in the log header alongside the model,
task seed, and environment metadata, so the log file is fully
reproducible from its own header.
"""
from pacetest.forward_pass import run_one_task
from pacetest.rewriter import rewrite_agent_prompt, rewrite_tool_doc
from pacetest.prompts import AGENT_PROMPT, TOOL_DOC, DIVERSITY_SEEDS
from pacetest.logger import init_log, log_round
from pacetest.tasks import Task, generate_tasks
from pacetest.oracle import score_round, is_correct
from pacetest.config import LoopConfig
from pacetest.knobs import (
    should_include_oracle,
    should_update_agent,
    should_update_tool,
)
from pacetest.pacemakers import (
    Decision,
    PacemakerVerdict,
    TrajectoryHistory,
    variance_triggered_freeze,
    diversity_injection,
    oracle_anchored_gating,
)


def _compute_semantic_distance(text_a: str, text_b: str) -> float:
    """Cosine distance between two texts using the sentence-transformer.

    Lazily imported so runs with `config.pacemaker=None` do not load
    the sentence-transformers model. Returns a non-negative float in
    [0.0, 2.0]; clipped at 0.0 to absorb floating-point noise when the
    two texts are identical.
    """
    import numpy as np
    from pacetest.embedding_metrics import _embed
    embs = _embed([text_a, text_b])
    return max(0.0, 1.0 - float(np.dot(embs[0], embs[1])))


def _apply_pacemaker(
    config: LoopConfig,
    current_prompt: str,
    candidate_prompt: str,
    current_doc: str,
    round_num: int,
    history: TrajectoryHistory,
    held_out_tasks: list,
    gating_score_cache: dict = None,
) -> tuple[str, PacemakerVerdict]:
    """Consult the active pacemaker and return the resolved next-round prompt.

    Args:
        config: LoopConfig; the `pacemaker` field selects the pacemaker.
        current_prompt: agent prompt in force before the rewriter fired.
        candidate_prompt: agent prompt produced by the rewriter.
        current_doc: current tool documentation (needed by gating's
            evaluator; ignored by freeze and diversity).
        round_num: current round index (needed by diversity for rotation).
        history: TrajectoryHistory of previous per-round agent-prompt
            semantic distances (updated in place by freeze and diversity).
        held_out_tasks: task list used by gating (ignored by the others).

    Returns:
        (next_prompt, verdict). If `config.pacemaker` is None, next_prompt
        is candidate_prompt and verdict is PacemakerVerdict(Decision.ACCEPT).
    """
    if config.pacemaker is None:
        return candidate_prompt, PacemakerVerdict(Decision.ACCEPT)

    if config.pacemaker in ("freeze", "diversity"):
        dist = _compute_semantic_distance(current_prompt, candidate_prompt)
        if config.pacemaker == "freeze":
            verdict = variance_triggered_freeze(dist, history)
        else:  # diversity
            verdict = diversity_injection(
                dist, history, round_num=round_num, seeds=DIVERSITY_SEEDS,
            )
        history.record(dist)
    elif config.pacemaker == "gating":
        def evaluator(agent_prompt_, tool_doc_, task_):
            r = run_one_task(task_.question, agent_prompt_, tool_doc_)
            return is_correct(r.get("agent_answer"), task_.gold_answer)
        verdict = oracle_anchored_gating(
            candidate_agent_prompt=candidate_prompt,
            current_agent_prompt=current_prompt,
            tool_doc=current_doc,
            held_out_tasks=held_out_tasks or [],
            evaluator=evaluator,
            score_cache=gating_score_cache,
        )
    else:
        # Config validation prevents this; raised as a defensive guard.
        raise ValueError(f"Unknown pacemaker: {config.pacemaker!r}")

    if verdict.decision == Decision.ACCEPT:
        return candidate_prompt, verdict
    if verdict.decision == Decision.FREEZE:
        return current_prompt, verdict
    if verdict.decision == Decision.REPLACE:
        return verdict.replacement, verdict
    raise ValueError(f"Unknown Decision: {verdict.decision}")


def run_loop(tasks: list[Task], num_rounds: int = 20,
             agent_prompt: str = None, tool_doc: str = None,
             run_name: str = None, task_seed: int = None,
             config: LoopConfig = None,
             held_out_tasks: list[Task] = None) -> dict:
    """Run K rounds of the closed agent-tool rewriting loop.

    Args:
        tasks: A list of Task objects to cycle through.
        num_rounds: Number of rounds K. Default 20.
        agent_prompt: Starting prompt. Defaults to AGENT_PROMPT.
        tool_doc: Starting docstring. Defaults to TOOL_DOC.
        run_name: Optional log file identifier.
        task_seed: Optional task seed. Recorded in the log header.
        config: Optional LoopConfig with knob values and pacemaker choice.
        held_out_tasks: Optional held-out task list for oracle-anchored
            gating. Required when `config.pacemaker == "gating"`; ignored
            for other pacemaker values.

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
    if tasks:
        extra_metadata["difficulty"] = tasks[0].difficulty
    extra_metadata.update(config.asdict())

    log_path = init_log(run_name=run_name, extra_metadata=extra_metadata)
    current_prompt = agent_prompt
    current_doc = tool_doc
    history = TrajectoryHistory()
    # Cache of held-out correctness counts keyed by (agent_prompt, tool_doc).
    # Populated only when gating is the active pacemaker; None otherwise.
    # Halves gating's LLM cost on rounds where the current prompt is
    # unchanged from the previous rewrite (Section 3.6.3 addendum).
    gating_score_cache = {} if config.pacemaker == "gating" else None
    # The pacemaker verdict that produced the CURRENT prompt. Attached to
    # the next round's log entry so log readers can see which decision
    # brought each round's prompt into existence.
    pending_verdict = None

    for round_num in range(num_rounds):
        task = tasks[round_num % len(tasks)]
        print(
            f"[Round {round_num + 1}/{num_rounds}] "
            f"{task.task_id}: {task.question[:60]}..."
        )

        result = run_one_task(task.question, current_prompt, current_doc)
        scored = score_round(result, task)
        # Record the verdict that produced this round's prompt.
        scored["starting_pacemaker_verdict"] = pending_verdict
        log_round(log_path, round_num, scored)
        print(
            f"  success={scored['success']}, "
            f"correct={scored['correct']}, "
            f"answer={scored['agent_answer']} (gold={scored['gold_answer']})"
        )

        if round_num < num_rounds - 1:
            if should_include_oracle(round_num, config):
                rewriter_input = scored
            else:
                rewriter_input = {k: v for k, v in scored.items() if k != "correct"}

            if should_update_agent(round_num, config):
                candidate_prompt = rewrite_agent_prompt(
                    current_prompt, rewriter_input, config,
                )
                current_prompt, verdict = _apply_pacemaker(
                    config=config,
                    current_prompt=current_prompt,
                    candidate_prompt=candidate_prompt,
                    current_doc=current_doc,
                    round_num=round_num,
                    history=history,
                    held_out_tasks=held_out_tasks or [],
                    gating_score_cache=gating_score_cache,
                )
                # Log the verdict only when a pacemaker was actually consulted.
                # When config.pacemaker is None, distinguish that from an
                # active-pacemaker ACCEPT by recording None.
                pending_verdict = (
                    verdict.decision.value if config.pacemaker is not None else None
                )
            else:
                pending_verdict = "skipped_no_rewrite"

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
