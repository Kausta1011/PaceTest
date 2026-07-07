"""Dump questions and reference solutions for given task ids, for hand verification.

Regenerates the seed=42, n=20, filtered GSM8K pool (the main sweep's pool) and
writes each requested task's question, reference solution, and gold answer to
refs_dump.txt in the repo root.

Run from the repo root:
    python -m scripts.dump_refs gsm8k_0005 gsm8k_0009 gsm8k_0018

Use --seed to dump from a different seed's pool (e.g. --seed 13 once the
seed-13 runs exist).
"""

import argparse
from pathlib import Path

from pacetest.tasks import generate_tasks, _load_gsm8k_pool

OUT_PATH = Path("refs_dump.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="+", help="task ids, e.g. gsm8k_0005")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    tasks = generate_tasks(seed=args.seed, n=args.n, difficulty="gsm8k",
                           filter_bad_rows=True)
    task_map = {t.task_id: t for t in tasks}
    pool = _load_gsm8k_pool()
    solution_by_question = {row["question"]: row["answer"] for row in pool}

    lines = []
    for tid in args.task_ids:
        if tid not in task_map:
            lines.append(f"{tid}: NOT FOUND in seed={args.seed} pool")
            continue
        task = task_map[tid]
        lines.append("=" * 88)
        lines.append(f"{tid} | gold (parsed): {task.gold_answer} | seed {args.seed} pool")
        lines.append("-- QUESTION --")
        lines.append(task.question)
        lines.append("-- REFERENCE SOLUTION --")
        lines.append(solution_by_question.get(task.question, "(question not found in pool)"))
        lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH} with {len(args.task_ids)} requested ids.")


if __name__ == "__main__":
    main()
