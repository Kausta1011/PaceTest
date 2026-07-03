"""Run a 3x3 mini-sweep over feedback_strength x self_judgement_weight.

Usage (run from the project root):
    python -m scripts.run_mini_sweep [--seed S] [--rounds K] [--n N] [--dry-run]

Iterates over the canonical values {0.0, 0.5, 1.0} for each of two knobs
(feedback_strength and self_judgement_weight), holding update_asymmetry
fixed at 0.5. Nine cells total. Writes one JSONL log per cell in
`logs/mini_sweep_fs<X>_sjw<Y>_seed<seed>.jsonl`. Every cell uses the same
task set (generate_tasks with the same seed), so cross-cell differences
are attributable to the knobs and not to task variation.

Wall-clock guide (M1 Pro, qwen3:8b):
    --rounds 3   ~ 10-15 min
    --rounds 5   ~ 15-25 min
    --rounds 20  ~ 1-2 hours

Use --dry-run to preview the cell plan without running anything.
"""
import argparse
import time
from itertools import product

from pacetest.config import LoopConfig
from pacetest.loop import run_loop
from pacetest.tasks import generate_tasks

CANONICAL_VALUES = [0.0, 0.5, 1.0]


def main():
    parser = argparse.ArgumentParser(
        description="Run a mini-sweep over the two primary knobs."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Task seed shared by all cells. Default: 42.",
    )
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Rounds per cell. Default: 5.",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Task-pool size per cell. Default: 20.",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "hard"], default="easy",
        help="Task difficulty tier. Default: easy.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without running.",
    )
    args = parser.parse_args()

    cells = list(product(CANONICAL_VALUES, CANONICAL_VALUES))
    print(f"Mini-sweep plan: {len(cells)} cells, {args.rounds} rounds each.")
    print(f"  seed={args.seed}, n={args.n}, difficulty={args.difficulty}")
    print(f"  fixed: update_asymmetry=0.5")
    print(f"  varying: feedback_strength x self_judgement_weight")
    print()

    if args.dry_run:
        for fs, sjw in cells:
            run_name = f"mini_sweep_{args.difficulty}_fs{fs}_sjw{sjw}_seed{args.seed}"
            print(f"  fs={fs}, sjw={sjw} -> {run_name}")
        print()
        print("Dry run only. No LLM calls made.")
        return

    tasks = generate_tasks(seed=args.seed, n=args.n, difficulty=args.difficulty)
    total_start = time.time()

    for i, (fs, sjw) in enumerate(cells):
        cell_start = time.time()
        config = LoopConfig(
            feedback_strength=fs,
            self_judgement_weight=sjw,
            update_asymmetry=0.5,
        )
        run_name = f"mini_sweep_{args.difficulty}_fs{fs}_sjw{sjw}_seed{args.seed}"
        print(f"[Cell {i + 1}/{len(cells)}] fs={fs}, sjw={sjw} -> {run_name}")
        out = run_loop(
            tasks,
            num_rounds=args.rounds,
            run_name=run_name,
            task_seed=args.seed,
            config=config,
        )
        elapsed = time.time() - cell_start
        print(f"  cell done in {elapsed:.1f}s. log: {out['log_path']}")
        print()

    total_elapsed = time.time() - total_start
    print(f"Mini-sweep complete. Total wall-clock: {total_elapsed / 60:.1f} min.")
    print()
    print("Inspect any cell with:")
    print("  python scripts/inspect_log.py logs/mini_sweep_fs<X>_sjw<Y>_seed<seed>.jsonl")


if __name__ == "__main__":
    main()
