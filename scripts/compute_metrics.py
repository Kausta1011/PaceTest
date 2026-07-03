"""Compute cycling and sycophancy metrics over one or more PaceTest logs.

Usage (run from the project root):
    python -m scripts.compute_metrics [logs/*.jsonl ...]

Or, with no arguments, walks all *.jsonl files under logs/:
    python -m scripts.compute_metrics

For each log file, prints a one-line summary of the two Week 6 Day 1
metrics: sign_flip_rate and sycophancy_decoupling. Also prints the
per-log knob values and difficulty for quick cross-run reading.
"""
import argparse
import json
from pathlib import Path

from pacetest.metrics import sign_flip_rate, sycophancy_decoupling


def load_rounds(path: Path) -> tuple[dict, list[dict], int]:
    """Read a JSONL log, return (header, list_of_round_dicts, skipped_count).

    Skips lines that fail to parse. Legacy logs (some Week 3 files) contain
    entries with raw newlines in string fields, which violates the
    one-object-per-line JSONL contract; the newer logger escapes properly.
    Rather than crash on such files, missing rounds are treated as absent.
    """
    header = None
    rounds = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if entry.get("type") == "header":
                header = entry
            elif entry.get("type") == "round":
                rounds.append(entry)
    return header, rounds, skipped


def main():
    parser = argparse.ArgumentParser(description="Compute PaceTest cycling metrics.")
    parser.add_argument(
        "log_paths", nargs="*", type=Path,
        help="Paths to log files. If empty, walks logs/*.jsonl.",
    )
    args = parser.parse_args()

    if not args.log_paths:
        args.log_paths = sorted(Path("logs").glob("*.jsonl"))

    if not args.log_paths:
        print("No log files found under logs/.")
        return

    print(f"{'run':<45}{'diff':<7}{'fs':<5}{'sjw':<6}{'flip':<7}{'decoup':<8}{'succ':<8}{'corr':<8}")
    print("-" * 94)
    for path in args.log_paths:
        header, rounds, skipped = load_rounds(path)
        if header is None:
            print(f"{path.name:<45}(no header found; {skipped} malformed lines skipped)")
            continue
        if skipped:
            print(f"{'  # ' + path.name + ' had ' + str(skipped) + ' malformed lines (legacy log; skipped safely)':<94}")
        diff = header.get("difficulty", "?")
        fs = header.get("feedback_strength", "?")
        sjw = header.get("self_judgement_weight", "?")
        flip = sign_flip_rate(rounds)
        dec = sycophancy_decoupling(rounds)
        n = len(rounds)
        n_success = sum(1 for r in rounds if r.get("success"))
        n_correct = sum(1 for r in rounds if r.get("correct"))
        # Pre-oracle logs (Weeks 1-3) have no `correct` field. Mark them
        # explicitly so their decoupling numbers are not misread as real.
        has_oracle = any("correct" in r for r in rounds)
        decoup_str = f"{dec['decoupling']:+.2f}" if has_oracle else "  (pre-orc)"
        corr_str = f"{n_correct}/{n}" if has_oracle else "  (n/a)"
        print(
            f"{path.stem:<45}{str(diff):<7}{str(fs):<5}{str(sjw):<6}"
            f"{flip:<7.2f}{decoup_str:<11}"
            f"{n_success}/{n:<6}{corr_str:<8}"
        )


if __name__ == "__main__":
    main()
