"""Cross-model validation sweep on a hosted API (Week 11, proposal Phase 5).

Runs the two load-bearing knob-cells through the unmodified PaceTest
pipeline against a frontier model, to test whether the tool-sycophancy
and tool-abandonment findings of Sections 4.10 and 4.11 are properties of
the closed loop or of qwen3:8b specifically.

Usage (run from the project root, with the key exported):

    export ANTHROPIC_API_KEY='sk-ant-...'

    # 1. See the plan and the cost estimate. Makes NO API calls.
    python -m scripts.run_crossmodel_sweep --pacemaker baseline --seed 42 --dry-run

    # 2. Smallest useful real run: one condition, one seed, both cells.
    python -m scripts.run_crossmodel_sweep --pacemaker baseline --seed 42

    # 3. Full proposal-scale sweep: repeat for each condition and seed.
    #    baseline/freeze/diversity/gating x seeds 42, 13, 77 = 24 cells.

Filename convention (distinct from both earlier sweeps so nothing collides):
    xmodel_gsm8k_{condition}_fs{fs}_sjw{sjw}_ua{ua}_seed{seed}_K{rounds}_{modeltag}.jsonl

Determinism: the Messages API accepts no seed, so these runs are NOT
byte-reproducible in the sense of Section 3.5. Temperature is pinned to
0.0. Each log header records backend, model, and seed_honoured=false so
the distinction is visible to any later reader.
"""
import argparse
import os
import time
from pathlib import Path

from pacetest.config import LoopConfig
from pacetest.loop import run_loop
from pacetest.prompts import GSM8K_AGENT_PROMPT
from pacetest.tasks import generate_tasks

# Same two cells as the Week 10 sensitivity sweep, for direct comparability.
KNOB_CELLS = [(1.0, 0.5), (0.0, 0.5)]

# Calibrated against the first real cross-model run (1 August 2026, seed 42
# baseline, both cells): 116 calls, 108,529 input and 47,239 output tokens,
# so 936 in and 407 out per call. The earlier figures here were derived from
# qwen3:8b logs and understated output by roughly 7x, because the frontier
# model writes far longer reasoning than the local one. Used only for the
# preflight estimate; the billed figure is metered per call at run end.
_EST_IN_TOK_PER_CALL = 940
_EST_OUT_TOK_PER_CALL = 410


def _run_name(pacemaker, fs, sjw, ua, seed, model_tag, rounds):
    """Build the cross-model log stem.

    K is in the name deliberately. The logger appends rather than
    overwrites, so two runs that differ only in trajectory length would
    otherwise share a filename and silently concatenate into one corrupt
    log. That failure caused two incidents in Week 8 under the older
    naming scheme, which encoded condition and seed but not K.
    """
    return (f"xmodel_gsm8k_{pacemaker}_fs{fs}_sjw{sjw}_ua{ua}"
            f"_seed{seed}_K{rounds}_{model_tag}")


def _estimate_calls(pacemaker, rounds, held_out_n):
    """Calls per cell: agent forward passes, rewrites, and gating probes.

    One agent call per round. Two rewrite calls per non-terminal round
    (agent prompt and tool doc). Under gating, the pacemaker additionally
    scores two candidate prompts against every held-out task at each
    consultation, which dominates the total.
    """
    agent = rounds
    rewrites = 2 * (rounds - 1)
    gating = 2 * held_out_n * (rounds - 1) if pacemaker == "gating" else 0
    return agent + rewrites + gating


def main():
    p = argparse.ArgumentParser(description="Cross-model validation sweep.")
    p.add_argument("--pacemaker",
                   choices=["baseline", "freeze", "diversity", "gating"],
                   required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ua", type=float, default=0.5,
                   help="Default 0.5 to match the main sweep of Section 4.10.")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--held-out-n", type=int, default=3, dest="held_out_n")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and cost estimate, make no API calls.")
    args = p.parse_args()

    # Force the hosted backend for this script only, before any pacetest
    # module reads it. Everything downstream picks this up automatically.
    os.environ["PACETEST_BACKEND"] = "anthropic"
    from pacetest.llm_api import (DEFAULT_API_MODEL, PRICE_PER_MTOK,
                                  usage_report)

    model = os.environ.get("PACETEST_API_MODEL", DEFAULT_API_MODEL)
    model_tag = model.split("-2")[0].replace(".", "")
    pacemaker_arg = None if args.pacemaker == "baseline" else args.pacemaker

    calls = _estimate_calls(args.pacemaker, args.rounds, args.held_out_n)
    cells = len(KNOB_CELLS)
    rates = PRICE_PER_MTOK.get(model, {"input": 0.0, "output": 0.0})
    est_cost = (calls * cells * _EST_IN_TOK_PER_CALL / 1e6 * rates["input"]
                + calls * cells * _EST_OUT_TOK_PER_CALL / 1e6 * rates["output"])

    print(f"Cross-model sweep: {args.pacemaker}, seed {args.seed}, ua={args.ua}")
    print(f"  model            {model}")
    print(f"  cells            {cells} (fs/sjw = " +
          ", ".join(f"{fs}/{sjw}" for fs, sjw in KNOB_CELLS) + ")")
    print(f"  calls per cell   ~{calls}")
    print(f"  calls total      ~{calls * cells}")
    print(f"  estimated cost   ~${est_cost:.2f} USD (preflight estimate)")
    print()

    # Collision guard, same policy as the Week 10 runner: the logger opens
    # in append mode, so re-running a finished cell would pollute its log.
    targets = [Path("logs") / (_run_name(args.pacemaker, fs, sjw, args.ua,
                                         args.seed, model_tag, args.rounds)
                               + ".jsonl")
               for fs, sjw in KNOB_CELLS]
    existing = [t for t in targets if t.exists()]
    if existing:
        print("Refusing to launch. These log files already exist:")
        for t in existing:
            print(f"  {t}")
        return

    if args.dry_run:
        for t in targets:
            print(f"  would write {t}")
        print("\nDry run only. No API calls made, nothing spent.")
        return

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY is not set. Export it and re-run.")
        return

    tasks = generate_tasks(seed=args.seed, n=args.n, difficulty="gsm8k")
    held_out_tasks = None
    if pacemaker_arg == "gating":
        held_out_tasks = generate_tasks(seed=args.seed + 1000,
                                        n=args.held_out_n, difficulty="gsm8k")

    start = time.time()
    for i, (fs, sjw) in enumerate(KNOB_CELLS):
        config = LoopConfig(feedback_strength=fs, self_judgement_weight=sjw,
                            update_asymmetry=args.ua, pacemaker=pacemaker_arg)
        run_name = _run_name(args.pacemaker, fs, sjw, args.ua, args.seed,
                             model_tag, args.rounds)
        print(f"[Cell {i + 1}/{cells}] fs={fs}, sjw={sjw} -> {run_name}")
        out = run_loop(tasks, num_rounds=args.rounds,
                       agent_prompt=GSM8K_AGENT_PROMPT, run_name=run_name,
                       task_seed=args.seed, config=config,
                       held_out_tasks=held_out_tasks)
        print(f"  log: {out['log_path']}")
        print(f"  running meter: {usage_report()}")
        print()

    print(f"Sweep complete in {(time.time() - start) / 60:.1f} min.")
    final = usage_report()
    print(f"Billed usage: {final['calls']} calls, "
          f"{final['input_tokens']:,} input tokens, "
          f"{final['output_tokens']:,} output tokens, "
          f"~${final['estimated_cost_usd']} USD.")
    print("Record this figure in the lab notebook for the Section 3.7 entry.")


if __name__ == "__main__":
    main()
