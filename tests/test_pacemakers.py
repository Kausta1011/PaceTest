"""Tests for the Week 7 pacemakers: variance-triggered freeze + diversity injection."""
from pacetest.pacemakers import (
    Decision,
    PacemakerVerdict,
    TrajectoryHistory,
    variance_triggered_freeze,
    diversity_injection,
    FREEZE_ABS_THRESHOLD,
    DIVERSITY_STUCK_THRESHOLD,
    DIVERSITY_STUCK_STREAK,
)


# ---- variance_triggered_freeze ----

def test_freeze_empty_history_accepts_moderate_distance():
    """First round: no prior history, distance below abs threshold -> accept."""
    v = variance_triggered_freeze(0.10, TrajectoryHistory())
    assert v.decision is Decision.ACCEPT
    assert v.replacement is None


def test_freeze_absolute_threshold_triggers():
    """A candidate distance above the abs threshold always freezes."""
    v = variance_triggered_freeze(0.30, TrajectoryHistory())
    assert v.decision is Decision.FREEZE
    assert v.replacement is None
    v2 = variance_triggered_freeze(FREEZE_ABS_THRESHOLD, TrajectoryHistory())
    assert v2.decision is Decision.FREEZE


def test_freeze_relative_spike_triggers():
    """A distance more than 2x the trailing mean triggers the relative rule."""
    h = TrajectoryHistory()
    h.record(0.02)
    h.record(0.02)
    v = variance_triggered_freeze(0.10, h)
    assert v.decision is Decision.FREEZE


def test_freeze_no_spike_when_history_too_short():
    """With only 1 prior distance, the relative rule must not fire."""
    h = TrajectoryHistory()
    h.record(0.01)
    v = variance_triggered_freeze(0.05, h)
    assert v.decision is Decision.ACCEPT


def test_freeze_modest_growth_within_history_accepts():
    """Growth within 2x of the mean must NOT trigger freeze."""
    h = TrajectoryHistory()
    h.record(0.05)
    h.record(0.06)
    v = variance_triggered_freeze(0.10, h)
    assert v.decision is Decision.ACCEPT


def test_freeze_zero_previous_mean_does_not_crash():
    """All-zero history -> relative rule skipped, absolute rule applies."""
    h = TrajectoryHistory()
    h.record(0.0)
    h.record(0.0)
    v = variance_triggered_freeze(0.05, h)
    assert v.decision is Decision.ACCEPT
    v2 = variance_triggered_freeze(0.30, h)
    assert v2.decision is Decision.FREEZE


def test_trajectory_history_records_and_appends():
    """TrajectoryHistory.record() appends a distance in order."""
    h = TrajectoryHistory()
    h.record(0.1)
    h.record(0.2)
    h.record(0.3)
    assert h.semantic_distances == [0.1, 0.2, 0.3]


# ---- diversity_injection ----

_DUMMY_SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3"]


def test_diversity_no_history_accepts():
    """No history recorded yet -> no way to be stuck -> accept."""
    v = diversity_injection(0.01, TrajectoryHistory(), round_num=0, seeds=_DUMMY_SEEDS)
    assert v.decision is Decision.ACCEPT
    assert v.replacement is None


def test_diversity_history_but_candidate_moving_accepts():
    """Even if history is stuck, a large candidate distance -> accept."""
    h = TrajectoryHistory()
    for _ in range(5):
        h.record(0.01)  # very small distances, "stuck" history
    v = diversity_injection(0.30, h, round_num=5, seeds=_DUMMY_SEEDS)
    assert v.decision is Decision.ACCEPT


def test_diversity_stuck_history_and_candidate_replaces():
    """Three consecutive stuck rounds + a stuck candidate -> REPLACE with seed."""
    h = TrajectoryHistory()
    h.record(0.01)
    h.record(0.02)
    h.record(0.01)
    v = diversity_injection(0.01, h, round_num=3, seeds=_DUMMY_SEEDS)
    assert v.decision is Decision.REPLACE
    assert v.replacement == _DUMMY_SEEDS[3 % len(_DUMMY_SEEDS)]


def test_diversity_rotation_is_deterministic_on_round_num():
    """Same round_num always injects the same seed."""
    h = TrajectoryHistory()
    h.record(0.01)
    h.record(0.01)
    h.record(0.01)
    v_a = diversity_injection(0.01, h, round_num=7, seeds=_DUMMY_SEEDS)
    v_b = diversity_injection(0.01, h, round_num=7, seeds=_DUMMY_SEEDS)
    assert v_a.replacement == v_b.replacement
    # And round_num=7 rotates to seed index 3
    assert v_a.replacement == _DUMMY_SEEDS[3]


def test_diversity_one_recent_nonstuck_prevents_injection():
    """If any of the last streak-length distances exceeds the threshold, ACCEPT."""
    h = TrajectoryHistory()
    h.record(0.01)
    h.record(0.10)  # this one is above the stuck threshold
    h.record(0.01)
    v = diversity_injection(0.01, h, round_num=3, seeds=_DUMMY_SEEDS)
    assert v.decision is Decision.ACCEPT


def test_diversity_empty_seeds_raises_when_triggered():
    """Empty seeds list must raise ValueError when injection would fire."""
    h = TrajectoryHistory()
    for _ in range(3):
        h.record(0.01)
    try:
        diversity_injection(0.01, h, round_num=0, seeds=[])
        assert False, "Should have raised ValueError on empty seeds during injection."
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in [
        ("test_freeze_empty_history_accepts_moderate_distance", test_freeze_empty_history_accepts_moderate_distance),
        ("test_freeze_absolute_threshold_triggers", test_freeze_absolute_threshold_triggers),
        ("test_freeze_relative_spike_triggers", test_freeze_relative_spike_triggers),
        ("test_freeze_no_spike_when_history_too_short", test_freeze_no_spike_when_history_too_short),
        ("test_freeze_modest_growth_within_history_accepts", test_freeze_modest_growth_within_history_accepts),
        ("test_freeze_zero_previous_mean_does_not_crash", test_freeze_zero_previous_mean_does_not_crash),
        ("test_trajectory_history_records_and_appends", test_trajectory_history_records_and_appends),
        ("test_diversity_no_history_accepts", test_diversity_no_history_accepts),
        ("test_diversity_history_but_candidate_moving_accepts", test_diversity_history_but_candidate_moving_accepts),
        ("test_diversity_stuck_history_and_candidate_replaces", test_diversity_stuck_history_and_candidate_replaces),
        ("test_diversity_rotation_is_deterministic_on_round_num", test_diversity_rotation_is_deterministic_on_round_num),
        ("test_diversity_one_recent_nonstuck_prevents_injection", test_diversity_one_recent_nonstuck_prevents_injection),
        ("test_diversity_empty_seeds_raises_when_triggered", test_diversity_empty_seeds_raises_when_triggered),
    ]:
        fn()
        print(f"{name} passed")
