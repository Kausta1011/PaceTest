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


class Decision(Enum):
    """Pacemaker decision for a candidate rewrite."""
    ACCEPT = "accept"
    FREEZE = "freeze"  # reject candidate; keep the current prompt
    REPLACE = "replace"  # (reserved for diversity injection)


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
) -> Decision:
    """Decide whether to freeze the artefact based on the candidate's drift.

    Freezes (returns Decision.FREEZE) if either:
      * candidate_distance >= abs_threshold, OR
      * len(history) >= rel_min_history AND candidate_distance is more than
        rel_factor times the mean of the historical distances.

    Otherwise returns Decision.ACCEPT.

    Args:
        candidate_distance: cosine distance between the current prompt and
            the candidate replacement prompt (0.0 = identical, up to ~1.0
            for orthogonal).
        history: TrajectoryHistory of previous per-round distances.
        abs_threshold: absolute distance above which freeze fires.
        rel_factor: relative-spike multiplier.
        rel_min_history: minimum history length for the relative rule.

    Returns:
        Decision.ACCEPT if the candidate is safe; Decision.FREEZE if the
        candidate should be rejected and the current prompt preserved.
    """
    if candidate_distance >= abs_threshold:
        return Decision.FREEZE
    hist = history.semantic_distances
    if len(hist) >= rel_min_history:
        mean_prev = sum(hist) / len(hist)
        if mean_prev > 0.0 and candidate_distance > rel_factor * mean_prev:
            return Decision.FREEZE
    return Decision.ACCEPT
