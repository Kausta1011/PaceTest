"""Toy task generator for PaceTest.

Produces a reproducible list of arithmetic problems, each carrying its own
known correct answer ('gold answer'). Same seed always produces the same
tasks; different seeds produce different problem families.

Two difficulty tiers are supported:

- `difficulty='easy'` (default, Week 4 baseline): two operands drawn from
  [1, 20], one operator drawn from {+, -, *}. Every gold answer is a
  clean integer.

- `difficulty='hard'` (Week 5 Day 6 onward): three operands, two
  operators drawn from {+, -, *, /}, and random parenthesization. Division
  is constrained so the intermediate result is a clean integer, and the
  final answer is also a clean integer; the generator retries a template
  if either produces a non-integer.
"""
import random
from dataclasses import dataclass


EASY_OPS = ["+", "-", "*"]
HARD_OPS = ["+", "-", "*", "/"]
_MAX_RETRIES = 32


@dataclass
class Task:
    """One toy arithmetic task with its known correct answer.

    Fields:
        task_id: short string identifier, useful when reading logs by hand.
        question: the natural-language string the agent will be asked.
        gold_answer: the correct numeric answer, computed at generation time.
        difficulty: 'easy' or 'hard'. Recorded so the sweep header can
            document which family a run used.
    """
    task_id: str
    question: str
    gold_answer: float
    difficulty: str = "easy"


def _generate_easy_task(rng: random.Random, task_index: int) -> Task:
    """Single-op arithmetic on operands in [1, 20]. Integer answers only."""
    a = rng.randint(1, 20)
    b = rng.randint(1, 20)
    op = rng.choice(EASY_OPS)
    expr = f"{a} {op} {b}"
    gold = float(eval(expr, {"__builtins__": {}}, {}))
    return Task(
        task_id=f"toy_{task_index:04d}",
        question=f"What is {expr}?",
        gold_answer=gold,
        difficulty="easy",
    )


def _generate_hard_task(rng: random.Random, task_index: int) -> Task:
    """Three-operand parenthesized arithmetic. Integer answers guaranteed.

    Rejects any attempt whose intermediate or final result is non-integer
    (avoids messy decimals from division), retrying up to _MAX_RETRIES times.
    If retries are exhausted, falls back to a safe fixed template.
    """
    for _ in range(_MAX_RETRIES):
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        c = rng.randint(1, 20)
        op1 = rng.choice(HARD_OPS)
        op2 = rng.choice(HARD_OPS)
        # Randomly choose parenthesization: (a op1 b) op2 c  OR  a op1 (b op2 c).
        left_grouped = rng.random() < 0.5
        if left_grouped:
            expr = f"({a} {op1} {b}) {op2} {c}"
            # Check intermediate: a op1 b must be clean integer.
            try:
                inner = eval(f"{a} {op1} {b}", {"__builtins__": {}}, {})
            except ZeroDivisionError:
                continue
        else:
            expr = f"{a} {op1} ({b} {op2} {c})"
            try:
                inner = eval(f"{b} {op2} {c}", {"__builtins__": {}}, {})
            except ZeroDivisionError:
                continue
        if not float(inner).is_integer():
            continue
        try:
            gold = eval(expr, {"__builtins__": {}}, {})
        except ZeroDivisionError:
            continue
        if not float(gold).is_integer():
            continue
        return Task(
            task_id=f"toy_{task_index:04d}",
            question=f"What is {expr}?",
            gold_answer=float(gold),
            difficulty="hard",
        )
    # Fallback in the astronomically unlikely case we exhausted retries.
    return Task(
        task_id=f"toy_{task_index:04d}",
        question="What is (10 + 5) * 2?",
        gold_answer=30.0,
        difficulty="hard",
    )


def generate_tasks(
    seed: int = 42, n: int = 20, difficulty: str = "easy"
) -> list[Task]:
    """Generate n seeded arithmetic tasks at the requested difficulty.

    Args:
        seed: random seed. Same seed always gives the same tasks.
        n: how many tasks to produce.
        difficulty: 'easy' (default, single-op +/-/*) or 'hard'
            (three-operand parenthesised with +/-/*/, integer answers).

    Returns:
        A list of n Task objects.
    """
    if difficulty not in ("easy", "hard"):
        raise ValueError(
            f"difficulty must be 'easy' or 'hard', got {difficulty!r}"
        )
    rng = random.Random(seed)
    if difficulty == "easy":
        return [_generate_easy_task(rng, i) for i in range(n)]
    return [_generate_hard_task(rng, i) for i in range(n)]


if __name__ == "__main__":
    print("=== easy (default) ===")
    for t in generate_tasks(seed=42, n=5, difficulty="easy"):
        print(f"  {t.task_id}: {t.question}  (gold={t.gold_answer})")
    print()
    print("=== hard ===")
    for t in generate_tasks(seed=42, n=5, difficulty="hard"):
        print(f"  {t.task_id}: {t.question}  (gold={t.gold_answer})")
