"""Free-form sweep runner for self-directed experiments (Week 11+).

This is deliberately separate from the three runners that produced the
thesis results. Those write `mini_sweep_*`, `sens_sweep_*` and `xmodel_*`
logs; this one writes `custom_*` and nothing else reads that prefix, so
no analysis of the thesis data can accidentally pick up an exploratory
run. Nothing here can overwrite an existing thesis log.

The filename encodes EVERY parameter that can be varied, so two runs that
differ in any respect land in different files:

    custom_{family}_{condition}_fs{fs}_sjw{sjw}_ua{ua}_T{temp}
           _seed{seed}_K{rounds}_n{n}_{backend}-{modeltag}.jsonl

Usage (from the project root):

    # See the plan and the cost estimate. Makes no calls, spends nothing.
    python -m scripts.run_custom_sweep --backend openai --dry-run

    # A single cell, everything else at thesis defaults
    python -m scripts.run_custom_sweep --backend openai \
        --fs 1.0 --sjw 0.5 --ua 0.5 --temperature 0.0 --seed 42

    # Temperature sweep on one cell
    for T in 0.0 0.3 0.7 1.0; do
      python -m scripts.run_custom_sweep --backend openai --temperature $T \
        --fs 1.0 --sjw 0.5 --seed 42
    done

Backends: ollama (free, local, slow), anthropic, openai. Hosted backends
need ANTHROPIC_API_KEY or OPENAI_API_KEY exported in the same shell.
"""
import argparse
import os
import time
from pathlib import Path

from pacetest.config import LoopConfig
from pacetest.loop import run_loop
from pacetest.prompts import GSM8K_AGENT_PROMPT, AGENT_PROMPT
from pacetest.tasks import generate_tasks

# Per-call token estimates for the preflight, measured from the Week 11
# cross-model runs (936 input, 407 output per call on a frontier model).
_EST_IN, _EST_OUT = 940, 410


def _tag(model: str) -> str:
    """Filesystem-safe short form of a model identifier."""
    return model.split("-2")[0].replace(".", "").replace("/", "-").strip("-")


def _run_name(a, model_tag):
    return (f"custom_{a.family}_{a.pacemaker}"
            f"_fs{a.fs}_sjw{a.sjw}_ua{a.ua}_T{a.temperature}"
            f"_seed{a.seed}_K{a.rounds}_n{a.n}_{a.backend}-{model_tag}")


def _estimate_calls(pacemaker, rounds, held_out_n):
    """One agent call per round, two rewrites per non-terminal round, plus
    gating's held-out probes (two candidate prompts x every held-out task)."""
    return rounds + 2 * (rounds - 1) + (
        2 * held_out_n * (rounds - 1) if pacemaker == "gating" else 0)


def main():
    p = argparse.ArgumentParser(description="Free-form PaceTest sweep.")
    p.add_argument("--backend", choices=["ollama", "anthropic", "openai"],
                   default="ollama")
    p.add_argument("--model", default=None,
                   help="Override the backend's default model identifier.")
    p.add_argument("--family", choices=["gsm8k", "easy", "hard"], default="gsm8k")
    p.add_argument("--pacemaker",
                   choices=["baseline", "freeze", "diversity", "gating"],
                   default="baseline")
    p.add_argument("--fs", type=float, default=1.0,
                   help="feedback_strength: fraction of rounds the rewriter "
                        "is shown the oracle's correct/incorrect flag.")
    p.add_argument("--sjw", type=float, default=0.5,
                   help="self_judgement_weight: 0 = rewrite from the oracle, "
                        "1 = rewrite from the agent's own reasoning.")
    p.add_argument("--ua", type=float, default=0.5,
                   help="update_asymmetry: 0 = only the tool doc is rewritten, "
                        "1 = only the agent prompt, 0.5 = both every round.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for every call in the run.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--held-out-n", type=int, default=3, dest="held_out_n")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    # Set the environment before importing anything that reads it.
    os.environ["PACETEST_BACKEND"] = a.backend
    os.environ["PACETEST_TEMPERATURE"] = str(a.temperature)
    if a.model:
        os.environ["PACETEST_API_MODEL"] = a.model
    from pacetest.llm import active_model
    from pacetest.llm_api import PRICE_PER_MTOK, usage_report

    model = active_model()
    model_tag = _tag(model)
    pacemaker_arg = None if a.pacemaker == "baseline" else a.pacemaker
    calls = _estimate_calls(a.pacemaker, a.rounds, a.held_out_n)
    rates = PRICE_PER_MTOK.get(model, {"input": 0.0, "output": 0.0})
    cost = calls * (_EST_IN / 1e6 * rates["input"] + _EST_OUT / 1e6 * rates["output"])

    print(f"Custom sweep")
    print(f"  backend / model  {a.backend} / {model}")
    print(f"  knobs            fs={a.fs}  sjw={a.sjw}  ua={a.ua}  temperature={a.temperature}")
    print(f"  design           {a.family}, {a.pacemaker}, seed {a.seed}, K={a.rounds}, n={a.n}")
    print(f"  calls            ~{calls}")
    print(f"  estimated cost   " +
          (f"~${cost:.3f} USD" if a.backend != "ollama" else "free (local)"))

    target = Path("logs") / (_run_name(a, model_tag) + ".jsonl")
    print(f"  target           {target}")
    print()
    if target.exists():
        print("Refusing to launch: that log already exists. Change a parameter "
              "or move the old file.")
        return
    if a.dry_run:
        print("Dry run only. Nothing was called and nothing was spent.")
        return
    if a.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY is not set."); return
    if a.backend == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        print("OPENAI_API_KEY is not set."); return

    tasks = generate_tasks(seed=a.seed, n=a.n, difficulty=a.family)
    held_out = (generate_tasks(seed=a.seed + 1000, n=a.held_out_n,
                               difficulty=a.family)
                if pacemaker_arg == "gating" else None)
    start = time.time()
    out = run_loop(
        tasks, num_rounds=a.rounds,
        agent_prompt=GSM8K_AGENT_PROMPT if a.family == "gsm8k" else AGENT_PROMPT,
        run_name=_run_name(a, model_tag), task_seed=a.seed,
        config=LoopConfig(feedback_strength=a.fs, self_judgement_weight=a.sjw,
                          update_asymmetry=a.ua, pacemaker=pacemaker_arg),
        held_out_tasks=held_out,
    )
    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Log: {out['log_path']}")
    if a.backend != "ollama":
        print(f"Billed usage: {usage_report()}")


if __name__ == "__main__":
    main()
