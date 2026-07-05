"""Task generator for PaceTest.

Produces a reproducible list of problems, each carrying its own
known correct answer ('gold answer'). Same seed always produces the same
tasks; different seeds produce different problem families.

Three difficulty tiers are supported:

- `difficulty='easy'` (default, Week 4 baseline): two operands drawn from
  [1, 20], one operator drawn from {+, -, *}. Every gold answer is a
  clean integer.

- `difficulty='hard'` (Week 5 Day 6 onward): three operands, two
  operators drawn from {+, -, *, /}, and random parenthesization. Division
  is constrained so the intermediate result is a clean integer, and the
  final answer is also a clean integer; the generator retries a template
  if either produces a non-integer.

- `difficulty='gsm8k'` (Week 6 Day 3 onward): grade-school math word
  problems from the GSM8K benchmark (openai/gsm8k, test split). Each task's
  question is a natural-language word problem; the gold_answer is parsed
  from the `#### <number>` line that terminates the dataset's reference
  solution. Cached under the user's Hugging Face directory on first use.
"""
import random
import re
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


_GSM8K_CACHE = None


def _parse_gsm8k_answer(solution: str) -> float:
    """Parse the numeric answer from a GSM8K reference solution.

    GSM8K answers end with `#### <number>`. The number may contain commas
    (e.g. `1,234`). Returns the answer as a float; downstream code checks
    it is integer-valued.
    """
    if "####" not in solution:
        raise ValueError(f"No `####` line in GSM8K solution: {solution[:80]!r}")
    tail = solution.split("####")[-1].strip()
    tail = tail.replace(",", "")
    return float(tail)


# GSM8K reference solutions annotate every intermediate calculation as
# `<<expr=result>>`. The filter below parses these annotations and verifies
# each one is arithmetically consistent, plus that the terminating
# `#### <final>` matches the last annotation's result. This catches pure
# arithmetic errors and #### mismatches; it does NOT catch semantic bugs
# in which every annotation is internally consistent but uses the wrong
# operands (the `gsm8k_0000` bug documented in Section 3.4).
_GSM8K_ANNOTATION_RE = re.compile(r"<<([^>=]+?)=([^>]+?)>>")
_GSM8K_SAFE_EXPR_RE = re.compile(r"[\d\s\+\-\*\/\.\(\)\,]+")


def check_gsm8k_consistency(solution: str, tolerance: float = 1e-6) -> bool:
    """Return True iff a GSM8K reference solution is arithmetically self-consistent.

    Checks:
      1. Every `<<expr=result>>` annotation satisfies eval(expr) == result.
      2. The terminating `#### <final>` matches the last annotation's result.

    Does NOT catch semantic bugs like using the wrong operand in a sum
    (see gsm8k_0000 at test-split seed 42, which passes this filter but has
    a mismatched final answer relative to the problem's stated arithmetic).
    That class of bug requires reading the problem statement, not just the
    reference solution.

    Args:
        solution: the raw reference solution text.
        tolerance: absolute float tolerance for equality checks.

    Returns:
        True if the solution passes both consistency checks; False otherwise.
    """
    if "####" not in solution:
        return False
    annotations = _GSM8K_ANNOTATION_RE.findall(solution)
    if not annotations:
        return False
    last_result = None
    for raw_expr, raw_result in annotations:
        expr = raw_expr.strip().replace(",", "")
        result_str = raw_result.strip().replace(",", "")
        if not _GSM8K_SAFE_EXPR_RE.fullmatch(expr):
            return False
        try:
            computed = float(eval(expr, {"__builtins__": {}}, {}))
            expected = float(result_str)
        except (ValueError, ZeroDivisionError, SyntaxError):
            return False
        if abs(computed - expected) > tolerance:
            return False
        last_result = expected
    try:
        final = _parse_gsm8k_answer(solution)
    except ValueError:
        return False
    if last_result is None or abs(last_result - final) > tolerance:
        return False
    return True


def _load_gsm8k_pool() -> list[dict]:
    """Return the cached list of GSM8K test-set problems.

    Lazy: import `datasets` and hit the cache only when first invoked, so
    non-GSM8K runs never pay the import cost.
    """
    global _GSM8K_CACHE
    if _GSM8K_CACHE is None:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        _GSM8K_CACHE = [
            {"question": row["question"], "answer": row["answer"]}
            for row in ds
        ]
    return _GSM8K_CACHE


def _generate_gsm8k_tasks(
    rng: random.Random, n: int, filter_bad_rows: bool = True,
) -> list[Task]:
    """Sample n GSM8K problems reproducibly under the given RNG.

    If filter_bad_rows is True (default from Week 8 onward), rows whose
    reference solutions fail `check_gsm8k_consistency` are dropped from
    the pool BEFORE sampling. This changes which rows the RNG selects
    versus filter_bad_rows=False, so reproducing pre-Week-8 GSM8K
    experiments (Sections 4.7, 4.8, 4.9) requires explicit
    filter_bad_rows=False.
    """
    pool = _load_gsm8k_pool()
    if filter_bad_rows:
        pool = [row for row in pool if check_gsm8k_consistency(row["answer"])]
    if n > len(pool):
        raise ValueError(
            f"Requested {n} GSM8K tasks but only {len(pool)} are available."
        )
    sampled = rng.sample(pool, n)
    tasks = []
    for i, row in enumerate(sampled):
        try:
            gold = _parse_gsm8k_answer(row["answer"])
        except ValueError:
            # Skip malformed row and continue; the caller sees fewer than n.
            continue
        tasks.append(Task(
            task_id=f"gsm8k_{i:04d}",
            question=row["question"],
            gold_answer=gold,
            difficulty="gsm8k",
        ))
    return tasks


def generate_tasks(
    seed: int = 42,
    n: int = 20,
    difficulty: str = "easy",
    filter_bad_rows: bool = True,
) -> list[Task]:
    """Generate n seeded tasks at the requested difficulty.

    Args:
        seed: random seed. Same seed always gives the same tasks.
        n: how many tasks to produce.
        difficulty: 'easy' (single-op +/-/*), 'hard' (three-operand
            parenthesised with +/-/*/, integer answers), or 'gsm8k'
            (word problems from the GSM8K benchmark).
        filter_bad_rows: for 'gsm8k' only. If True (default from Week 8
            onward), drops rows whose reference solutions fail
            `check_gsm8k_consistency` before sampling. Ignored for the
            toy tiers. Reproducing Sections 4.7-4.9 requires False.

    Returns:
        A list of Task objects. For 'gsm8k', may return fewer than n if
        some sampled rows have malformed reference answers even after
        pool-level filtering.
    """
    if difficulty not in ("easy", "hard", "gsm8k"):
        raise ValueError(
            f"difficulty must be 'easy', 'hard', or 'gsm8k', got {difficulty!r}"
        )
    rng = random.Random(seed)
    if difficulty == "easy":
        return [_generate_easy_task(rng, i) for i in range(n)]
    if difficulty == "hard":
        return [_generate_hard_task(rng, i) for i in range(n)]
    return _generate_gsm8k_tasks(rng, n, filter_bad_rows=filter_bad_rows)


if __name__ == "__main__":
    print("=== easy (default) ===")
    for t in generate_tasks(seed=42, n=5, difficulty="easy"):
        print(f"  {t.task_id}: {t.question}  (gold={t.gold_answer})")
    print()
    print("=== hard ===")
    for t in generate_tasks(seed=42, n=5, difficulty="hard"):
        print(f"  {t.task_id}: {t.question}  (gold={t.gold_answer})")
    print()
    print("=== gsm8k (first 2, truncated for display) ===")
    for t in generate_tasks(seed=42, n=2, difficulty="gsm8k"):
        q_short = t.question.replace("\n", " ")[:120] + "..."
        print(f"  {t.task_id}: {q_short}  (gold={t.gold_answer})")
