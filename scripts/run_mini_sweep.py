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
from pacetest.prompts import AGENT_PROMPT, GSM8K_AGENT_PROMPT
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
        "--difficulty", choices=["easy", "hard", "gsm8k"], default="easy",
        help="Task difficulty tier. Default: easy.",
    )
    parser.add_argument(
        "--pacemaker", choices=["none", "freeze", "diversity", "gating"],
        default="none",
        help="Pacemaker controller for all cells. Default: none.",
    )
    parser.add_argument(
        "--held-out-n", type=int, default=3, dest="held_out_n",
        help="Held-out task-set size for --pacemaker gating. Default: 3.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without running.",
    )
    args = parser.parse_args()

    pacemaker_arg = None if args.pacemaker == "none" else args.pacemaker
    pm_tag = "" if pacemaker_arg is None else f"_{pacemaker_arg}"

    cells = list(product(CANONICAL_VALUES, CANONICAL_VALUES))
    print(f"Mini-sweep plan: {len(cells)} cells, {args.rounds} rounds each.")
    print(f"  seed={args.seed}, n={args.n}, difficulty={args.difficulty}")
    print(f"  pacemaker={pacemaker_arg}")
    print(f"  fixed: update_asymmetry=0.5")
    print(f"  varying: feedback_strength x self_judgement_weight")
    print()

    if args.dry_run:
        for fs, sjw in cells:
            run_name = (
                f"mini_sweep_{args.difficulty}{pm_tag}"
                f"_fs{fs}_sjw{sjw}_seed{args.seed}"
            )
            print(f"  fs={fs}, sjw={sjw} -> {run_name}")
        print()
        print("Dry run only. No LLM calls made.")
        return

    tasks = generate_tasks(seed=args.seed, n=args.n, difficulty=args.difficulty)
    # Held-out set for oracle-anchored gating: same task family, disjoint
    # seed. Sampled once per sweep so all 9 cells share the same held-out.
    held_out_tasks = None
    if pacemaker_arg == "gating":
        held_out_tasks = generate_tasks(
            seed=args.seed + 1000, n=args.held_out_n, difficulty=args.difficulty,
        )
    # Toy tiers use the arithmetic transcription prompt; gsm8k uses the
    # chain-of-thought word-problem prompt.
    starting_agent_prompt = (
        GSM8K_AGENT_PROMPT if args.difficulty == "gsm8k" else AGENT_PROMPT
    )
    total_start = time.time()

    for i, (fs, sjw) in enumerate(cells):
        cell_start = time.time()
        config = LoopConfig(
            feedback_strength=fs,
            self_judgement_weight=sjw,
            update_asymmetry=0.5,
            pacemaker=pacemaker_arg,
        )
        run_name = (
            f"mini_sweep_{args.difficulty}{pm_tag}"
            f"_fs{fs}_sjw{sjw}_seed{args.seed}"
        )
        print(f"[Cell {i + 1}/{len(cells)}] fs={fs}, sjw={sjw} -> {run_name}")
        out = run_loop(
            tasks,
            num_rounds=args.rounds,
            agent_prompt=starting_agent_prompt,
            run_name=run_name,
            task_seed=args.seed,
            config=config,
            held_out_tasks=held_out_tasks,
        )
        elapsed = time.time() - cell_start
        print(f"  cell done in {elapsed:.1f}s. log: {out['log_path']}")
        print()

    total_elapsed = time.time() - total_start
    print(f"Mini-sweep complete. Total wall-clock: {total_elapsed / 60:.1f} min.")
    print()
    print("Inspect any cell with:")
    print("  python scripts/inspect_log.py logs/mini_sweep_...jsonl")


if __name__ == "__main__":
    main()
