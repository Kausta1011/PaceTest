"""Inspect a PaceTest JSONL log file.

Usage:
    python scripts/inspect_log.py logs/loop_smoke.jsonl

Prints the metadata header, a one-line summary per round, and aggregate
protocol-compliance and correctness rates. Useful when you want a quick
look at a run without firing up a notebook or jq.
"""
import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    """Read a JSONL log and split it into (header, list_of_rounds).

    A PaceTest log always begins with one 'header' line and then one
    'round' line per round. The header carries reproducibility metadata
    (model, seed, git commit, ollama version, python version, platform).
    """
    header = None
    rounds = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "header":
                header = entry
            elif entry.get("type") == "round":
                rounds.append(entry)
    if header is None:
        raise ValueError(f"No header line found in {path}")
    return header, rounds


def print_header(header: dict) -> None:
    print("=" * 70)
    print(f"Run: {header.get('run_name')}")
    print(f"  model:                {header.get('model')}")
    print(f"  llm_seed:             {header.get('seed')}")
    print(f"  task_seed:            {header.get('task_seed', '(not recorded)')}")
    print(f"  num_tasks:            {header.get('num_tasks', '(not recorded)')}")
    print(f"  num_rounds:           {header.get('num_rounds', '(not recorded)')}")
    print(f"  feedback_strength:    {header.get('feedback_strength', '(not recorded)')}")
    print(f"  self_judgement_wt:    {header.get('self_judgement_weight', '(not recorded)')}")
    print(f"  update_asymmetry:     {header.get('update_asymmetry', '(not recorded)')}")
    print(f"  git_commit:           {str(header.get('git_commit', 'unknown'))[:12]}")
    print(f"  ollama_version:       {header.get('ollama_version')}")
    print(f"  python_version:       {header.get('python_version')}")
    print(f"  platform:             {header.get('platform')}")
    print("=" * 70)


def print_rounds(rounds: list[dict]) -> None:
    if not rounds:
        print("\n(no rounds in this log)")
        return
    print()
    print(f"{'#':<4} {'task_id':<12} {'success':<8} {'correct':<8} {'answer':<10} {'gold':<10}")
    print("-" * 60)
    for r in rounds:
        rid = r.get("round", "?")
        task_id = r.get("task_id", "?")
        success = r.get("success")
        correct = r.get("correct")
        answer = r.get("agent_answer")
        gold = r.get("gold_answer")
        print(f"{rid:<4} {str(task_id):<12} {str(success):<8} {str(correct):<8} {str(answer):<10} {str(gold):<10}")


def print_summary(rounds: list[dict]) -> None:
    n = len(rounds)
    if n == 0:
        return
    n_success = sum(1 for r in rounds if r.get("success") is True)
    n_correct = sum(1 for r in rounds if r.get("correct") is True)
    n_compliant_but_wrong = sum(
        1 for r in rounds
        if r.get("success") is True and r.get("correct") is False
    )
    print()
    print("Summary:")
    print(f"  rounds:                {n}")
    print(f"  protocol-compliant:    {n_success}/{n} ({n_success/n:.0%})")
    print(f"  actually correct:      {n_correct}/{n} ({n_correct/n:.0%})")
    print(f"  compliant-but-wrong:   {n_compliant_but_wrong}/{n} ({n_compliant_but_wrong/n:.0%})")
    print(f"    (this is the protocol-vs-correctness gap; RQ2 lives here)")


def main():
    parser = argparse.ArgumentParser(description="Inspect a PaceTest JSONL log file.")
    parser.add_argument("log_path", type=Path, help="Path to a PaceTest JSONL log file.")
    args = parser.parse_args()

    if not args.log_path.exists():
        raise SystemExit(f"Log file not found: {args.log_path}")

    header, rounds = load_jsonl(args.log_path)
    print_header(header)
    print_rounds(rounds)
    print_summary(rounds)


if __name__ == "__main__":
    main()
