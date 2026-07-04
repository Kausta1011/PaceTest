"""Pacemaker controllers for the closed rewriting loop.

Three pacemakers are planned in the proposal (Section 2.3, 5.2). Each is
a decision function invoked by the loop between the rewriter and the
next round, receiving trajectory history and a candidate rewrite, and
returning a decision to accept, reject (freeze), or replace the
candidate.

Week 7 Day 1 implements the first pacemaker, `variance_triggered_freeze`.
The other two (`diversity_injection`, `oracle_anchored_gating`) will be
added later this week.

The decisions are represented as a `Decision` enum with three values so
subsequent pacemakers can extend the return type without breaking the
loop's API.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Optional


class Decision(Enum):
    """Pacemaker decision for a candidate rewrite."""
    ACCEPT = "accept"
    FREEZE = "freeze"  # reject candidate; keep the current prompt
    REPLACE = "replace"  # use the replacement text provided by the pacemaker


class PacemakerVerdict(NamedTuple):
    """A pacemaker's decision plus (for REPLACE) the replacement text.

    Fields:
        decision: which action the loop should take.
        replacement: text to use in place of the candidate rewrite. Set
            only when decision is REPLACE; None for ACCEPT and FREEZE.
    """
    decision: Decision
    replacement: Optional[str] = None


@dataclass
class TrajectoryHistory:
    """State a pacemaker maintains across rounds.

    Fields:
        semantic_distances: list of floats, one per round transition.
            index i is the semantic distance between the prompt used in
            round i and the prompt used in round i+1.
    """
    semantic_distances: list[float] = field(default_factory=list)

    def record(self, distance: float) -> None:
        """Append a new semantic distance measurement."""
        self.semantic_distances.append(float(distance))


# Threshold constants for variance_triggered_freeze.
# ABS_THRESHOLD sits between the maximum semantic distance observed on
# stable Section 4.8 cells (~0.18) and the collapse-cell values (0.27, 0.14).
# See Section 3.6 of the thesis draft.
FREEZE_ABS_THRESHOLD = 0.20
# Relative threshold: freeze if the latest distance is more than
# FREEZE_REL_FACTOR times the trailing mean of previous distances. Kicks
# in even when the absolute threshold has not been crossed, if the
# rewriter suddenly produces a much larger drift than it has been.
FREEZE_REL_FACTOR = 2.0
# Minimum history length before the relative rule can fire (so a single
# large first distance does not trip the freeze).
FREEZE_REL_MIN_HISTORY = 2


def variance_triggered_freeze(
    candidate_distance: float,
    history: TrajectoryHistory,
    abs_threshold: float = FREEZE_ABS_THRESHOLD,
    rel_factor: float = FREEZE_REL_FACTOR,
    rel_min_history: int = FREEZE_REL_MIN_HISTORY,
) -> PacemakerVerdict:
    """Decide whether to freeze the artefact based on the candidate's drift.

    Returns a PacemakerVerdict with:
      * decision=Decision.FREEZE if either:
          candidate_distance >= abs_threshold, OR
          len(history) >= rel_min_history AND candidate_distance is more
          than rel_factor times the mean of the historical distances.
      * decision=Decision.ACCEPT otherwise.
    In both cases the replacement field is None (freeze never supplies text).

    Args:
        candidate_distance: cosine distance between the current prompt and
            the candidate replacement prompt (0.0 = identical, up to ~1.0
            for orthogonal).
        history: TrajectoryHistory of previous per-round distances.
        abs_threshold: absolute distance above which freeze fires.
        rel_factor: relative-spike multiplier.
        rel_min_history: minimum history length for the relative rule.
    """
    if candidate_distance >= abs_threshold:
        return PacemakerVerdict(Decision.FREEZE)
    hist = history.semantic_distances
    if len(hist) >= rel_min_history:
        mean_prev = sum(hist) / len(hist)
        if mean_prev > 0.0 and candidate_distance > rel_factor * mean_prev:
            return PacemakerVerdict(Decision.FREEZE)
    return PacemakerVerdict(Decision.ACCEPT)


# ---- Diversity injection (Week 7 Day 2) ----

# Below this per-round semantic distance a round is treated as "stuck":
# the rewriter's candidate is essentially the current prompt. If the loop
# stays stuck for a run of consecutive rounds, the loop is in a narrow
# attractor and diversity_injection fires to break it out.
DIVERSITY_STUCK_THRESHOLD = 0.05
# Number of consecutive stuck rounds required before injection fires.
# Three in a row is unlikely by accident; the loop is genuinely converged.
DIVERSITY_STUCK_STREAK = 3


def diversity_injection(
    candidate_distance: float,
    history: TrajectoryHistory,
    round_num: int,
    seeds: list[str],
    stuck_threshold: float = DIVERSITY_STUCK_THRESHOLD,
    stuck_streak: int = DIVERSITY_STUCK_STREAK,
) -> PacemakerVerdict:
    """Break the loop out of a narrow attractor by rotating to a seed prompt.

    Fires (returns Decision.REPLACE with a seed prompt as replacement)
    when the last `stuck_streak` recorded semantic distances are all below
    `stuck_threshold` AND the current candidate distance is also below
    `stuck_threshold`. In that regime the loop has been effectively
    stationary; diversity injection replaces the candidate rewrite with
    the seed prompt at index `round_num % len(seeds)` from the caller's
    seed list. Otherwise returns Decision.ACCEPT.

    Rotation is deterministic on round_num, so runs with the same seed
    and same round number always inject the same seed. This preserves
    reproducibility.

    Args:
        candidate_distance: cosine distance between the current prompt
            and the candidate replacement prompt.
        history: TrajectoryHistory of previous per-round distances.
        round_num: current round index; used for deterministic rotation.
        seeds: list of alternative seed prompts to rotate through; must
            be non-empty when this pacemaker is active.
        stuck_threshold: per-round distance below which a round counts
            as stuck.
        stuck_streak: consecutive stuck rounds required before injection
            fires.

    Returns:
        PacemakerVerdict(REPLACE, seeds[round_num % len(seeds)]) when the
        loop is judged stuck; PacemakerVerdict(ACCEPT) otherwise.

    Raises:
        ValueError if seeds is empty and injection is triggered.
    """
    if candidate_distance >= stuck_threshold:
        return PacemakerVerdict(Decision.ACCEPT)
    hist = history.semantic_distances
    if len(hist) < stuck_streak:
        return PacemakerVerdict(Decision.ACCEPT)
    recent = hist[-stuck_streak:]
    if any(d >= stuck_threshold for d in recent):
        return PacemakerVerdict(Decision.ACCEPT)
    # All recent + current are below threshold: inject.
    if not seeds:
        raise ValueError("diversity_injection triggered but seeds list is empty")
    seed_index = round_num % len(seeds)
    return PacemakerVerdict(Decision.REPLACE, seeds[seed_index])


# ---- Oracle-anchored gating (Week 7 Day 3) ----

# The evaluator callback returns True iff the given agent prompt and tool
# doc solve the given task correctly. The pacemaker itself contains no LLM
# call; the caller (the loop) provides the evaluator, which internally runs
# `run_one_task` and `score_round`. This split keeps the pacemaker unit-
# testable without an LLM setup and keeps the pacemaker deterministic in
# a given evaluator's behaviour.


def oracle_anchored_gating(
    candidate_agent_prompt: str,
    current_agent_prompt: str,
    tool_doc: str,
    held_out_tasks: list,
    evaluator,
) -> PacemakerVerdict:
    """Reject the candidate rewrite if it reduces accuracy on a held-out set.

    Evaluates the candidate prompt on each held-out task, then evaluates
    the current prompt on the same tasks. Compares the two correctness
    counts. If the candidate is strictly worse, returns FREEZE; otherwise
    ACCEPT (ties go to the candidate).

    Args:
        candidate_agent_prompt: the prompt the rewriter has just produced.
        current_agent_prompt: the prompt in force at the end of the last round.
        tool_doc: current tool documentation; not itself gated, but passed
            through to the evaluator so agent-side evaluation has the tool
            context it needs.
        held_out_tasks: list of Task instances used only for gating. Must
            be disjoint from the training pool consumed by the loop.
        evaluator: callable `(agent_prompt, tool_doc, task) -> bool` that
            returns True iff the task is solved correctly under the given
            prompts. Injected so the pacemaker is testable without an LLM.

    Returns:
        PacemakerVerdict(Decision.FREEZE) if candidate correctness < current
        correctness on the held-out set; PacemakerVerdict(Decision.ACCEPT)
        otherwise (including the empty-held-out edge case and ties).
    """
    if not held_out_tasks:
        # Fail safe: with nothing to evaluate on, do not intervene.
        return PacemakerVerdict(Decision.ACCEPT)
    candidate_correct = sum(
        1 for t in held_out_tasks
        if evaluator(candidate_agent_prompt, tool_doc, t)
    )
    current_correct = sum(
        1 for t in held_out_tasks
        if evaluator(current_agent_prompt, tool_doc, t)
    )
    if candidate_correct < current_correct:
        return PacemakerVerdict(Decision.FREEZE)
    return PacemakerVerdict(Decision.ACCEPT)
