"""Tests for the sign_flip_rate and sycophancy_decoupling metrics."""
from pacetest.metrics import sign_flip_rate, sycophancy_decoupling


def _round(success, correct):
    return {"success": success, "correct": correct}


# ---- sign_flip_rate ----

def test_flip_rate_all_true_is_zero():
    rs = [_round(True, True) for _ in range(5)]
    assert sign_flip_rate(rs) == 0.0


def test_flip_rate_alternating_is_one():
    rs = [_round(True, i % 2 == 0) for i in range(6)]
    # correct flips every round: 1,0,1,0,1,0 -> 5 flips out of 5 transitions
    assert sign_flip_rate(rs) == 1.0


def test_flip_rate_single_flip():
    rs = [_round(True, True)] * 3 + [_round(True, False)] * 2
    # transitions: T-T, T-T, T-F, F-F -> 1 flip out of 4
    assert sign_flip_rate(rs) == 0.25


def test_flip_rate_empty_returns_zero():
    assert sign_flip_rate([]) == 0.0
    assert sign_flip_rate([_round(True, True)]) == 0.0


# ---- sycophancy_decoupling ----

def test_decoupling_all_correct_is_zero():
    """A run that stays 100% success AND 100% correct has zero decoupling."""
    rs = [_round(True, True) for _ in range(10)]
    out = sycophancy_decoupling(rs)
    assert out["decoupling"] == 0.0
    assert out["acceptance_gain"] == 0.0
    assert out["correctness_gain"] == 0.0


def test_decoupling_pure_sycophancy():
    """Rounds where success climbs but correct stays flat produce positive decoupling.

    First half: 0/4 success, 0/4 correct.
    Second half: 4/4 success, 0/4 correct.
    acceptance_gain = 1.0; correctness_gain = 0.0; decoupling = 1.0.
    """
    rs = [_round(False, False) for _ in range(4)] + [_round(True, False) for _ in range(4)]
    out = sycophancy_decoupling(rs)
    assert out["acceptance_gain"] == 1.0
    assert out["correctness_gain"] == 0.0
    assert out["decoupling"] == 1.0


def test_decoupling_negative_when_correctness_outpaces():
    """Correctness rises but success stays flat -> negative decoupling."""
    rs = [_round(True, False) for _ in range(4)] + [_round(True, True) for _ in range(4)]
    out = sycophancy_decoupling(rs)
    assert out["acceptance_gain"] == 0.0
    assert out["correctness_gain"] == 1.0
    assert out["decoupling"] == -1.0


def test_decoupling_empty_returns_zeros():
    """Empty or single-round input returns all zeros without crashing."""
    out = sycophancy_decoupling([])
    assert out["decoupling"] == 0.0
    out = sycophancy_decoupling([_round(True, True)])
    assert out["decoupling"] == 0.0


if __name__ == "__main__":
    for name, fn in [
        ("test_flip_rate_all_true_is_zero", test_flip_rate_all_true_is_zero),
        ("test_flip_rate_alternating_is_one", test_flip_rate_alternating_is_one),
        ("test_flip_rate_single_flip", test_flip_rate_single_flip),
        ("test_flip_rate_empty_returns_zero", test_flip_rate_empty_returns_zero),
        ("test_decoupling_all_correct_is_zero", test_decoupling_all_correct_is_zero),
        ("test_decoupling_pure_sycophancy", test_decoupling_pure_sycophancy),
        ("test_decoupling_negative_when_correctness_outpaces", test_decoupling_negative_when_correctness_outpaces),
        ("test_decoupling_empty_returns_zeros", test_decoupling_empty_returns_zeros),
    ]:
        fn()
        print(f"{name} passed")
