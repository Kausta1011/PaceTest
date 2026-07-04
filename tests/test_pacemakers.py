"""Tests for the Week 7 Day 1 pacemaker: variance-triggered freeze."""
from pacetest.pacemakers import (
    Decision,
    TrajectoryHistory,
    variance_triggered_freeze,
    FREEZE_ABS_THRESHOLD,
)


def test_empty_history_accepts_moderate_distance():
    """First round: no prior history, distance below abs threshold -> accept."""
    d = variance_triggered_freeze(0.10, TrajectoryHistory())
    assert d is Decision.ACCEPT


def test_absolute_threshold_freezes():
    """A candidate distance above the abs threshold always freezes.

    The threshold sits between the Section 4.8 stable-cell maximum (~0.18)
    and the collapse-cell distances (0.27, 0.14). We test with 0.30 to
    stay comfortably above the threshold.
    """
    d = variance_triggered_freeze(0.30, TrajectoryHistory())
    assert d is Decision.FREEZE
    # Exactly at the threshold should also freeze (>= not >).
    d2 = variance_triggered_freeze(FREEZE_ABS_THRESHOLD, TrajectoryHistory())
    assert d2 is Decision.FREEZE


def test_relative_spike_freezes():
    """A distance more than 2x the trailing mean triggers the relative rule.

    History of small distances (0.02, 0.02) has mean 0.02. Candidate at
    0.10 is 5x the mean; should freeze.
    """
    h = TrajectoryHistory()
    h.record(0.02)
    h.record(0.02)
    d = variance_triggered_freeze(0.10, h)
    assert d is Decision.FREEZE


def test_no_spike_when_history_too_short():
    """With only 1 prior distance, relative rule must not fire yet.

    A single first distance of 0.01 and a candidate at 0.05 would trigger
    the relative rule if we allowed it, but rel_min_history=2 prevents it.
    """
    h = TrajectoryHistory()
    h.record(0.01)
    d = variance_triggered_freeze(0.05, h)
    assert d is Decision.ACCEPT


def test_modest_growth_within_history_accepts():
    """Growth within a factor of 2 of the mean must NOT trigger freeze."""
    h = TrajectoryHistory()
    h.record(0.05)
    h.record(0.06)
    # Mean is 0.055; 0.10 is less than 2x = 0.11.
    d = variance_triggered_freeze(0.10, h)
    assert d is Decision.ACCEPT


def test_zero_previous_mean_does_not_crash():
    """If all previous distances were exactly 0, relative rule is skipped.

    A history of literal zeros means the artefact never changed; the
    relative multiplier is meaningless. The absolute rule alone decides.
    """
    h = TrajectoryHistory()
    h.record(0.0)
    h.record(0.0)
    d = variance_triggered_freeze(0.05, h)
    assert d is Decision.ACCEPT
    d2 = variance_triggered_freeze(0.30, h)
    assert d2 is Decision.FREEZE


def test_trajectory_history_records_and_appends():
    """TrajectoryHistory.record() appends a distance in order."""
    h = TrajectoryHistory()
    h.record(0.1)
    h.record(0.2)
    h.record(0.3)
    assert h.semantic_distances == [0.1, 0.2, 0.3]


if __name__ == "__main__":
    for name, fn in [
        ("test_empty_history_accepts_moderate_distance", test_empty_history_accepts_moderate_distance),
        ("test_absolute_threshold_freezes", test_absolute_threshold_freezes),
        ("test_relative_spike_freezes", test_relative_spike_freezes),
        ("test_no_spike_when_history_too_short", test_no_spike_when_history_too_short),
        ("test_modest_growth_within_history_accepts", test_modest_growth_within_history_accepts),
        ("test_zero_previous_mean_does_not_crash", test_zero_previous_mean_does_not_crash),
        ("test_trajectory_history_records_and_appends", test_trajectory_history_records_and_appends),
    ]:
        fn()
        print(f"{name} passed")
