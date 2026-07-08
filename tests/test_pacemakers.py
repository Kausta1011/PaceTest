"""Tests for the Week 7 pacemakers: variance-triggered freeze + diversity injection."""
from pacetest.pacemakers import (
    Decision,
    PacemakerVerdict,
    TrajectoryHistory,
    variance_triggered_freeze,
    diversity_injection,
    oracle_anchored_gating,
    FREEZE_ABS_THRESHOLD,
    DIVERSITY_STUCK_THRESHOLD,
    DIVERSITY_STUCK_STREAK,
)
from pacetest.prompts import (
    AGENT_PROMPT,
    GSM8K_AGENT_PROMPT,
    DIVERSITY_SEEDS,
    GSM8K_DIVERSITY_SEEDS,
    diversity_seeds_for,
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


# ---- oracle_anchored_gating ----
# Mock evaluator: returns True iff the agent prompt contains the token
# `_ok_` and the task's id contains `pass`. Lets us construct scenarios
# where candidate and current have different correctness on the same tasks.

class _MockTask:
    """Minimal task stub for gating tests. Not the real Task dataclass."""
    def __init__(self, task_id: str):
        self.task_id = task_id


def _mock_eval(agent_prompt: str, tool_doc: str, task) -> bool:
    """True iff both '_ok_' is in the prompt and 'pass' is in the task_id."""
    return "_ok_" in agent_prompt and "pass" in task.task_id


def test_gating_empty_held_out_accepts():
    """No held-out tasks -> no evidence -> accept (fail safe)."""
    v = oracle_anchored_gating(
        candidate_agent_prompt="bad prompt",
        current_agent_prompt="_ok_ prompt",
        tool_doc="doc",
        held_out_tasks=[],
        evaluator=_mock_eval,
    )
    assert v.decision is Decision.ACCEPT


def test_gating_candidate_matches_current_accepts():
    """Candidate solves the same number of tasks as current -> ACCEPT (ties go to candidate)."""
    tasks = [_MockTask("t_pass_1"), _MockTask("t_pass_2"), _MockTask("t_fail")]
    v = oracle_anchored_gating(
        candidate_agent_prompt="_ok_ candidate",
        current_agent_prompt="_ok_ current",
        tool_doc="doc",
        held_out_tasks=tasks,
        evaluator=_mock_eval,
    )
    # Both prompts solve the two `pass` tasks; both fail on the `fail` task.
    assert v.decision is Decision.ACCEPT


def test_gating_candidate_worse_freezes():
    """Candidate solves fewer tasks than current -> FREEZE."""
    tasks = [_MockTask("t_pass_1"), _MockTask("t_pass_2")]
    v = oracle_anchored_gating(
        candidate_agent_prompt="bad candidate",       # no _ok_ -> solves 0
        current_agent_prompt="_ok_ current",           # solves 2
        tool_doc="doc",
        held_out_tasks=tasks,
        evaluator=_mock_eval,
    )
    assert v.decision is Decision.FREEZE


def test_gating_candidate_better_accepts():
    """Candidate solves more tasks than current -> ACCEPT."""
    tasks = [_MockTask("t_pass_1"), _MockTask("t_pass_2")]
    v = oracle_anchored_gating(
        candidate_agent_prompt="_ok_ candidate",       # solves 2
        current_agent_prompt="bad current",            # solves 0
        tool_doc="doc",
        held_out_tasks=tasks,
        evaluator=_mock_eval,
    )
    assert v.decision is Decision.ACCEPT


def test_gating_evaluator_called_n_times_per_prompt():
    """The evaluator must be called len(held_out_tasks) times for each prompt.

    Counts total evaluator invocations; with N held-out tasks and 2 prompts
    to evaluate, expect exactly 2N calls.
    """
    tasks = [_MockTask("t_pass_1"), _MockTask("t_pass_2"), _MockTask("t_pass_3")]
    call_count = 0
    def counting_eval(ap, td, t):
        nonlocal call_count
        call_count += 1
        return _mock_eval(ap, td, t)
    oracle_anchored_gating(
        candidate_agent_prompt="_ok_ candidate",
        current_agent_prompt="_ok_ current",
        tool_doc="doc",
        held_out_tasks=tasks,
        evaluator=counting_eval,
    )
    assert call_count == 6, f"Expected 6 evaluator calls (2 prompts x 3 tasks), got {call_count}"


# ---- Cache tests (Week 8 Day 2) ----

def test_gating_cache_reuses_score_on_repeated_input():
    """Same (prompt, tool_doc) across successive calls: evaluator called once."""
    tasks = [_MockTask("t_pass_1"), _MockTask("t_pass_2"), _MockTask("t_pass_3")]
    call_count = 0
    def counting_eval(ap, td, t):
        nonlocal call_count
        call_count += 1
        return _mock_eval(ap, td, t)
    cache = {}
    # First call: 2N = 6 evaluator invocations, both prompts get cached.
    oracle_anchored_gating(
        candidate_agent_prompt="_ok_ candidate",
        current_agent_prompt="_ok_ current",
        tool_doc="doc",
        held_out_tasks=tasks,
        evaluator=counting_eval,
        score_cache=cache,
    )
    first_count = call_count
    # Second call with the same prompts + same tool_doc: 0 additional
    # invocations (both hits in cache).
    oracle_anchored_gating(
        candidate_agent_prompt="_ok_ candidate",
        current_agent_prompt="_ok_ current",
        tool_doc="doc",
        held_out_tasks=tasks,
        evaluator=counting_eval,
        score_cache=cache,
    )
    assert first_count == 6
    assert call_count == 6, f"Expected 6 total calls after two identical invocations, got {call_count}"


def test_gating_cache_misses_on_different_tool_doc():
    """Same agent_prompt but different tool_doc: cache miss, re-evaluate."""
    tasks = [_MockTask("t_pass_1")]
    call_count = 0
    def counting_eval(ap, td, t):
        nonlocal call_count
        call_count += 1
        return _mock_eval(ap, td, t)
    cache = {}
    oracle_anchored_gating(
        candidate_agent_prompt="_ok_ cand",
        current_agent_prompt="_ok_ curr",
        tool_doc="doc_v1",
        held_out_tasks=tasks,
        evaluator=counting_eval,
        score_cache=cache,
    )
    first_count = call_count  # expected 2
    oracle_anchored_gating(
        candidate_agent_prompt="_ok_ cand",
        current_agent_prompt="_ok_ curr",
        tool_doc="doc_v2",  # different tool doc
        held_out_tasks=tasks,
        evaluator=counting_eval,
        score_cache=cache,
    )
    assert first_count == 2
    # Cache misses because tool_doc changed; expect 2 more calls.
    assert call_count == 4, f"Expected 4 total calls after tool_doc change, got {call_count}"


def test_gating_no_cache_matches_week7_behavior():
    """score_cache=None (default): exactly 2N evaluator calls per invocation.

    Ensures backward-compat with the Week 7 pacemaker interface. All prior
    tests using this pacemaker without the cache argument must still pass.
    """
    tasks = [_MockTask("t_pass_1"), _MockTask("t_pass_2")]
    call_count = 0
    def counting_eval(ap, td, t):
        nonlocal call_count
        call_count += 1
        return _mock_eval(ap, td, t)
    # Call twice with identical inputs, no cache. Expect 4N = 8 total calls.
    for _ in range(2):
        oracle_anchored_gating(
            candidate_agent_prompt="_ok_ cand",
            current_agent_prompt="_ok_ curr",
            tool_doc="doc",
            held_out_tasks=tasks,
            evaluator=counting_eval,
            # score_cache omitted (defaults to None)
        )
    assert call_count == 8, f"Expected 8 total calls without cache, got {call_count}"


# ---- diversity seed pools (Week 8 Day 3 fix) ----

def test_gsm8k_seeds_first_is_initial_prompt():
    """Section 3.6.2 contract: the rotation contains the run's initial prompt."""
    assert GSM8K_DIVERSITY_SEEDS[0] == GSM8K_AGENT_PROMPT
    assert DIVERSITY_SEEDS[0] == AGENT_PROMPT


def test_gsm8k_seeds_preserve_format_markers():
    """All GSM8K seeds keep the three-line response contract verbatim."""
    for seed in GSM8K_DIVERSITY_SEEDS:
        assert "REASONING:" in seed
        assert 'TOOL_CALL: calculator("' in seed
        assert "ANSWER: <number>" in seed


def test_seed_pool_selection_by_initial_prompt():
    """GSM8K initial prompt gets GSM8K seeds; anything else gets toy seeds."""
    assert diversity_seeds_for(GSM8K_AGENT_PROMPT) is GSM8K_DIVERSITY_SEEDS
    assert diversity_seeds_for(AGENT_PROMPT) is DIVERSITY_SEEDS
    assert diversity_seeds_for("some drifted prompt") is DIVERSITY_SEEDS


def test_gsm8k_seed_pool_size_matches_toy_pool():
    """Both pools have four seeds so rotation arithmetic is identical."""
    assert len(GSM8K_DIVERSITY_SEEDS) == len(DIVERSITY_SEEDS) == 4


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
        ("test_gating_empty_held_out_accepts", test_gating_empty_held_out_accepts),
        ("test_gating_candidate_matches_current_accepts", test_gating_candidate_matches_current_accepts),
        ("test_gating_candidate_worse_freezes", test_gating_candidate_worse_freezes),
        ("test_gating_candidate_better_accepts", test_gating_candidate_better_accepts),
        ("test_gating_evaluator_called_n_times_per_prompt", test_gating_evaluator_called_n_times_per_prompt),
        ("test_gating_cache_reuses_score_on_repeated_input", test_gating_cache_reuses_score_on_repeated_input),
        ("test_gating_cache_misses_on_different_tool_doc", test_gating_cache_misses_on_different_tool_doc),
        ("test_gating_no_cache_matches_week7_behavior", test_gating_no_cache_matches_week7_behavior),
        ("test_gsm8k_seeds_first_is_initial_prompt", test_gsm8k_seeds_first_is_initial_prompt),
        ("test_gsm8k_seeds_preserve_format_markers", test_gsm8k_seeds_preserve_format_markers),
        ("test_seed_pool_selection_by_initial_prompt", test_seed_pool_selection_by_initial_prompt),
        ("test_gsm8k_seed_pool_size_matches_toy_pool", test_gsm8k_seed_pool_size_matches_toy_pool),
    ]:
        fn()
        print(f"{name} passed")
