"""Run a scored closed-loop experiment under a given task seed and knob config.

Usage (run from the project root):
    python -m scripts.run_scored [--seed S] [--n N] [--rounds K]
                                 [--name NAME]
                                 [--feedback-strength FS]
                                 [--self-judgement-weight SJW]
                                 [--update-asymmetry UA]

All arguments are optional. Defaults reproduce the Week 4 baseline run
(seed=42, n=20, rounds=20, fs=1.0, sjw=0.5, ua=0.5).

This is the general-purpose runner used for multi-seed comparison runs
from Week 4 Day 6 onward and for individual sweep cells from Week 5 Day 4
onward. The Week 4 Day 5 baseline run lives in its own fixed-parameter
script (`scripts/run_first_scored.py`) for historical reference.
"""
import argparse

from pacetest.config import LoopConfig
from pacetest.loop import run_loop
from pacetest.prompts import AGENT_PROMPT, GSM8K_AGENT_PROMPT
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
    parser.add_argument(
        "--feedback-strength", type=float, default=1.0,
        dest="feedback_strength",
        help="Feedback-strength knob in [0.0, 1.0]. Default: 1.0.",
    )
    parser.add_argument(
        "--self-judgement-weight", type=float, default=0.5,
        dest="self_judgement_weight",
        help="Self-judgement-weight knob in [0.0, 1.0]. Default: 0.5.",
    )
    parser.add_argument(
        "--update-asymmetry", type=float, default=0.5,
        dest="update_asymmetry",
        help="Update-asymmetry knob in [0.0, 1.0]. Default: 0.5.",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "hard", "gsm8k"], default="easy",
        help="Task difficulty tier. Default: easy.",
    )
    args = parser.parse_args()

    config = LoopConfig(
        feedback_strength=args.feedback_strength,
        self_judgement_weight=args.self_judgement_weight,
        update_asymmetry=args.update_asymmetry,
    )
    run_name = args.name or f"scored_seed{args.seed}"
    tasks = generate_tasks(seed=args.seed, n=args.n, difficulty=args.difficulty)
    # Select the starting agent prompt appropriate to the task family.
    # Toy tiers ('easy', 'hard') use the arithmetic transcription prompt;
    # 'gsm8k' uses the chain-of-thought word-problem prompt.
    starting_agent_prompt = (
        GSM8K_AGENT_PROMPT if args.difficulty == "gsm8k" else AGENT_PROMPT
    )
    print(
        f"Generated {len(tasks)} tasks (seed={args.seed}, difficulty={args.difficulty}). "
        f"Config: {config.asdict()}. "
        f"Starting {args.rounds}-round closed loop as '{run_name}'..."
    )
    print()
    out = run_loop(
        tasks,
        num_rounds=args.rounds,
        agent_prompt=starting_agent_prompt,
        run_name=run_name,
        task_seed=args.seed,
        config=config,
    )
    print()
    print(f"Done. Log saved to: {out['log_path']}")
    print(f"Inspect with: python scripts/inspect_log.py {out['log_path']}")


if __name__ == "__main__":
    main()
