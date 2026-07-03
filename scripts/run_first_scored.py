"""First scored 20-round run on the toy task family (Week 4 baseline).

Run:
    python scripts/run_first_scored.py

Generates 20 seeded arithmetic tasks (seed=42) and runs the full closed
loop for 20 rounds, writing the JSONL log to logs/week4_first_scored.jsonl.
Wall-clock: roughly 15 to 25 minutes on an M1 Pro with qwen3:8b.

This is the first run that produces a complete correctness record per round.
Each log entry carries both `success` (protocol compliance) and `correct`
(oracle truth), so the protocol-vs-correctness gap can be measured from a
single log file.
"""
from pacetest.loop import run_loop
from pacetest.tasks import generate_tasks


def main():
    tasks = generate_tasks(seed=42, n=20)
    print(f"Generated {len(tasks)} seeded tasks. Starting 20-round closed loop...")
    print()
    out = run_loop(
        tasks,
        num_rounds=20,
        run_name="week4_first_scored",
        task_seed=42,
    )
    print()
    print(f"Done. Log saved to: {out['log_path']}")
    print(f"Inspect with: python scripts/inspect_log.py {out['log_path']}")


if __name__ == "__main__":
    main()
