"""Toy task generator for PaceTest.

Produces a reproducible list of arithmetic problems, each carrying its own
known correct answer ('gold answer'). Same seed always produces the same
tasks; different seeds produce different problem families, which is what
multi-seed experiments vary.

Operators included in this initial version: +, -, *. Division is skipped
on purpose so every gold answer is a clean integer; a later iteration can
add division and parentheses once the oracle is in place.
"""
import random
from dataclasses import dataclass


@dataclass
class Task:
    """One toy arithmetic task with its known correct answer.

    Fields:
        task_id: short string identifier, useful when reading logs by hand.
        question: the natural-language string the agent will be asked.
        gold_answer: the correct numeric answer, computed at generation time.
    """
    task_id: str
    question: str
    gold_answer: float


def _generate_one_task(rng: random.Random, task_index: int) -> Task:
    """Make one random arithmetic task and compute its gold answer."""
    operators = ["+", "-", "*"]
    a = rng.randint(1, 20)
    b = rng.randint(1, 20)
    op = rng.choice(operators)
    expr = f"{a} {op} {b}"
    # Safe eval: only digits, spaces, and the three operators are present.
    gold = float(eval(expr, {"__builtins__": {}}, {}))
    return Task(
        task_id=f"toy_{task_index:04d}",
        question=f"What is {expr}?",
        gold_answer=gold,
    )


def generate_tasks(seed: int = 42, n: int = 20) -> list[Task]:
    """Generate n seeded arithmetic tasks.

    Args:
        seed: random seed. Same seed gives the same tasks every run.
        n: how many tasks to produce.

    Returns:
        A list of n Task objects.
    """
    rng = random.Random(seed)
    return [_generate_one_task(rng, i) for i in range(n)]


if __name__ == "__main__":
    tasks = generate_tasks(seed=42, n=5)
    for t in tasks:
        print(f"{t.task_id}: {t.question}  (gold={t.gold_answer})")
