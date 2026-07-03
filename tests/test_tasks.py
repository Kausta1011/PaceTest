"""Tests for the toy task generator."""
import re

from pacetest.tasks import Task, generate_tasks


def test_generator_returns_n_tasks():
    """generate_tasks(n=10) returns exactly 10 Task objects."""
    tasks = generate_tasks(seed=42, n=10)
    assert len(tasks) == 10
    for t in tasks:
        assert isinstance(t, Task)


def test_generator_is_reproducible():
    """Same seed must produce identical tasks across calls.

    Works because @dataclass auto-generates an equality method for Task.
    """
    a = generate_tasks(seed=42, n=5)
    b = generate_tasks(seed=42, n=5)
    assert a == b, "Same seed produced different tasks."


def test_different_seeds_differ():
    """Different seeds should produce different task sets."""
    a = generate_tasks(seed=42, n=10)
    b = generate_tasks(seed=99, n=10)
    assert a != b, "Different seeds produced identical task sets."


def test_gold_answers_match_questions():
    """Every gold_answer must equal the arithmetic in its question.

    Parses 'What is X op Y?' out of each question, recomputes the answer
    independently, and compares. Catches generator bugs that would silently
    poison the oracle downstream.
    """
    tasks = generate_tasks(seed=42, n=20)
    for t in tasks:
        match = re.search(r"What is (.+)\?", t.question)
        assert match, f"Could not parse question: {t.question!r}"
        expr = match.group(1)
        expected = float(eval(expr, {"__builtins__": {}}, {}))
        assert t.gold_answer == expected, (
            f"{t.task_id}: gold {t.gold_answer} != computed {expected} "
            f"for expression {expr!r}"
        )


def test_task_ids_unique_and_ordered():
    """task_ids must be unique and in generation order."""
    tasks = generate_tasks(seed=42, n=10)
    ids = [t.task_id for t in tasks]
    assert ids == sorted(ids), "Task ids are not in order."
    assert len(set(ids)) == len(ids), "Task ids are not unique."


# ---- Hard-tier tests (Week 5 Day 6) ----

def test_default_difficulty_is_easy():
    """Backward compat: unspecified difficulty must still be 'easy'."""
    tasks = generate_tasks(seed=42, n=5)
    for t in tasks:
        assert t.difficulty == "easy"


def test_hard_returns_n_tasks():
    """generate_tasks(n=10, difficulty='hard') returns 10 hard Task objects."""
    tasks = generate_tasks(seed=42, n=10, difficulty="hard")
    assert len(tasks) == 10
    for t in tasks:
        assert isinstance(t, Task)
        assert t.difficulty == "hard"


def test_hard_reproducible():
    """Same seed at hard difficulty must produce identical tasks."""
    a = generate_tasks(seed=42, n=5, difficulty="hard")
    b = generate_tasks(seed=42, n=5, difficulty="hard")
    assert a == b


def test_hard_gold_answers_match_questions():
    """Hard-tier gold answers must equal the arithmetic in the question.

    Uses the same 'parse question, re-evaluate, compare' approach as the
    easy-tier test. Catches generator bugs that would poison the oracle.
    """
    tasks = generate_tasks(seed=42, n=20, difficulty="hard")
    for t in tasks:
        match = re.search(r"What is (.+)\?", t.question)
        assert match, f"Could not parse question: {t.question!r}"
        expr = match.group(1)
        expected = float(eval(expr, {"__builtins__": {}}, {}))
        assert t.gold_answer == expected, (
            f"{t.task_id}: gold {t.gold_answer} != computed {expected} "
            f"for {expr!r}"
        )


def test_hard_answers_are_integers():
    """Hard-tier gold_answer must always be a clean integer.

    Otherwise the oracle's float tolerance would be doing meaningful work
    on hard runs, which is a separate concern from the ones this tier
    exists to study.
    """
    tasks = generate_tasks(seed=42, n=20, difficulty="hard")
    for t in tasks:
        assert t.gold_answer == int(t.gold_answer), (
            f"{t.task_id}: non-integer gold {t.gold_answer} for {t.question!r}"
        )


def test_hard_questions_are_three_operand_parenthesised():
    """Every hard question must have exactly one pair of parentheses."""
    tasks = generate_tasks(seed=42, n=20, difficulty="hard")
    for t in tasks:
        assert t.question.count("(") == 1 and t.question.count(")") == 1, (
            f"{t.task_id}: question does not have exactly one paren pair: {t.question!r}"
        )


def test_invalid_difficulty_rejected():
    """Unknown difficulty strings must raise ValueError."""
    for bad in ["medium", "EASY", "", None]:
        try:
            generate_tasks(seed=42, n=5, difficulty=bad)
            assert False, f"Should have rejected difficulty={bad!r}"
        except (ValueError, TypeError):
            pass


if __name__ == "__main__":
    for name, fn in [
        ("test_generator_returns_n_tasks", test_generator_returns_n_tasks),
        ("test_generator_is_reproducible", test_generator_is_reproducible),
        ("test_different_seeds_differ", test_different_seeds_differ),
        ("test_gold_answers_match_questions", test_gold_answers_match_questions),
        ("test_task_ids_unique_and_ordered", test_task_ids_unique_and_ordered),
        ("test_default_difficulty_is_easy", test_default_difficulty_is_easy),
        ("test_hard_returns_n_tasks", test_hard_returns_n_tasks),
        ("test_hard_reproducible", test_hard_reproducible),
        ("test_hard_gold_answers_match_questions", test_hard_gold_answers_match_questions),
        ("test_hard_answers_are_integers", test_hard_answers_are_integers),
        ("test_hard_questions_are_three_operand_parenthesised", test_hard_questions_are_three_operand_parenthesised),
        ("test_invalid_difficulty_rejected", test_invalid_difficulty_rejected),
    ]:
        fn()
        print(f"{name} passed")
