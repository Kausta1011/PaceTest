"""Run a scored closed-loop experiment under a given task seed.

Usage (run from the project root):
    python -m scripts.run_scored [--seed S] [--n N] [--rounds K] [--name NAME]

All arguments are optional. Defaults reproduce the Week 4 baseline run
(seed=42, n=20, rounds=20, name=scored_seed42).

This is the general-purpose runner used for multi-seed comparison runs
from Week 4 Day 6 onward. The Week 4 Day 5 baseline run lives in its own
fixed-parameter script (`scripts/run_first_scored.py`) for historical
reference and is no longer the preferred way to launch new experiments.
"""
import argparse

from pacetest.loop import run_loop
from pacetest.tasks import generate_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Run a scored PaceTest closed-loop experiment."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Task seed for generate_tasks. Default: 42.",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of tasks to generate. Default: 20.",
    )
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Number of rounds K. Default: 20.",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name (used in log filename). Default: scored_seed<seed>.",
    )
    args = parser.parse_args()

    run_name = args.name or f"scored_seed{args.seed}"
    tasks = generate_tasks(seed=args.seed, n=args.n)
    print(
        f"Generated {len(tasks)} tasks (seed={args.seed}). "
        f"Starting {args.rounds}-round closed loop as '{run_name}'..."
    )
    print()
    out = run_loop(tasks, num_rounds=args.rounds, run_name=run_name)
    print()
    print(f"Done. Log saved to: {out['log_path']}")
    print(f"Inspect with: python scripts/inspect_log.py {out['log_path']}")


if __name__ == "__main__":
    main()
