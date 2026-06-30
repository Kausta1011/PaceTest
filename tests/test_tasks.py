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


if __name__ == "__main__":
    test_generator_returns_n_tasks()
    print("test_generator_returns_n_tasks passed")
    test_generator_is_reproducible()
    print("test_generator_is_reproducible passed")
    test_different_seeds_differ()
    print("test_different_seeds_differ passed")
    test_gold_answers_match_questions()
    print("test_gold_answers_match_questions passed")
    test_task_ids_unique_and_ordered()
    print("test_task_ids_unique_and_ordered passed")
