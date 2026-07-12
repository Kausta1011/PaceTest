"""Run a sensitivity strip over update_asymmetry for two knob-cells.

Week 10 sensitivity sweep, Section 4.11. Fixes the two load-bearing
knob-cells from the main sweep and varies update_asymmetry on one seed
and one pacemaker per invocation.

Usage (run from the project root):
    python -m scripts.run_sens_sweep --pacemaker <name> --seed <S> --ua <UA>
        [--rounds K] [--n N] [--dry-run]

Knob-cells (fixed):
    (fs=1.0, sjw=0.5)  sycophancy anchor
    (fs=0.0, sjw=0.5)  abandonment anchor

Filename convention:
    sens_sweep_gsm8k_{condition}_fs{fs}_sjw{sjw}_ua{ua}_seed{seed}.jsonl
    where condition is one of baseline, freeze, diversity, gating.

Refuses to launch if any target log file already exists. Week 8
filename-debt safeguard: the logger opens in append mode, so re-running
a completed cell would silently pollute the existing log. The
collision check turns that failure mode into a hard stop.

Wall-clock guide (M1 Pro, qwen3:8b, K=20, n=20):
    baseline    ~25 min / cell (ua != 0.5), ~30 min (ua == 0.5)
    freeze      ~24 min / cell (ua == 1.0)
    diversity   ~22 min / cell (ua == 1.0)
    gating      ~30 min / cell (ua == 1.0, cache hits reduce cost)
"""
import argparse
import time
from pathlib import Path

from pacetest.config import LoopConfig
from pacetest.loop import run_loop
from pacetest.prompts import GSM8K_AGENT_PROMPT
from pacetest.tasks import generate_tasks

# Fixed load-bearing cells from the main sweep.
# (fs, sjw): sycophancy anchor, abandonment anchor.
KNOB_CELLS = [(1.0, 0.5), (0.0, 0.5)]


def _run_name(pacemaker: str, fs: float, sjw: float, ua: float, seed: int) -> str:
    """Build the sensitivity-sweep filename stem for one cell."""
    return (
        f"sens_sweep_gsm8k_{pacemaker}"
        f"_fs{fs}_sjw{sjw}_ua{ua}_seed{seed}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity sweep over update_asymmetry."
    )
    parser.add_argument(
        "--pacemaker",
        choices=["baseline", "freeze", "diversity", "gating"],
        required=True,
        help="baseline maps to LoopConfig.pacemaker=None.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--ua", type=float, required=True,
        help="update_asymmetry for this invocation (typically 0.0, 0.5, 1.0).",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--held-out-n", type=int, default=3, dest="held_out_n",
        help="Held-out task-set size for --pacemaker gating. Default: 3.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without running.",
    )
    args = parser.parse_args()

    pacemaker_arg = None if args.pacemaker == "baseline" else args.pacemaker

    print(f"Sensitivity strip: {len(KNOB_CELLS)} cells, {args.rounds} rounds each.")
    print(f"  pacemaker={args.pacemaker}  seed={args.seed}  ua={args.ua}")
    print(f"  knob-cells: {KNOB_CELLS}")
    print()

    # Collision check (Week 8 filename-debt safeguard).
    proposed = [
        Path("logs") / (_run_name(args.pacemaker, fs, sjw, args.ua, args.seed) + ".jsonl")
        for fs, sjw in KNOB_CELLS
    ]
    existing = [p for p in proposed if p.exists()]
    if existing:
        print("ERROR: proposed log file(s) already exist. Refusing to overwrite:")
        for p in existing:
            print(f"  {p}")
        print()
        print("Delete the existing file(s) manually if you intend to re-run,")
        print("or change --seed / --ua / --pacemaker.")
        raise SystemExit(1)

    if args.dry_run:
        for fs, sjw in KNOB_CELLS:
            name = _run_name(args.pacemaker, fs, sjw, args.ua, args.seed)
            print(f"  fs={fs}, sjw={sjw} -> {name}")
        print()
        print("Dry run only. No LLM calls made.")
        return

    tasks = generate_tasks(seed=args.seed, n=args.n, difficulty="gsm8k")
    held_out_tasks = None
    if pacemaker_arg == "gating":
        held_out_tasks = generate_tasks(
            seed=args.seed + 1000, n=args.held_out_n, difficulty="gsm8k",
        )

    total_start = time.time()
    for i, (fs, sjw) in enumerate(KNOB_CELLS):
        cell_start = time.time()
        config = LoopConfig(
            feedback_strength=fs,
            self_judgement_weight=sjw,
            update_asymmetry=args.ua,
            pacemaker=pacemaker_arg,
        )
        run_name = _run_name(args.pacemaker, fs, sjw, args.ua, args.seed)
        print(f"[Cell {i + 1}/{len(KNOB_CELLS)}] fs={fs}, sjw={sjw} -> {run_name}")
        out = run_loop(
            tasks,
            num_rounds=args.rounds,
            agent_prompt=GSM8K_AGENT_PROMPT,
            run_name=run_name,
            task_seed=args.seed,
            config=config,
            held_out_tasks=held_out_tasks,
        )
        elapsed = time.time() - cell_start
        print(f"  cell done in {elapsed:.1f}s. log: {out['log_path']}")
        print()

    total_elapsed = time.time() - total_start
    print(f"Sensitivity strip complete. Total wall-clock: {total_elapsed / 60:.1f} min.")


if __name__ == "__main__":
    main()
