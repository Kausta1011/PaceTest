"""Dump questions and reference solutions for the six Cell 8 suspect rounds.

Regenerates the seed=42, n=20, filtered GSM8K pool (same call the main sweep
used), picks the six task ids that produced success=True, correct=False in
logs/mini_sweep_gsm8k_fs1.0_sjw0.5_seed42.jsonl, and writes each question,
reference solution, gold answer, and the agent's logged answer to
cell8_refs.txt in the repo root for hand verification.

Run from the repo root:
    python scripts/dump_cell8_refs.py
"""

import json
from pathlib import Path

from pacetest.tasks import generate_tasks, _load_gsm8k_pool

LOG_PATH = Path("logs/mini_sweep_gsm8k_fs1.0_sjw0.5_seed42.jsonl")
OUT_PATH = Path("cell8_refs.txt")
TARGET_IDS = [
    "gsm8k_0010",
    "gsm8k_0011",
    "gsm8k_0012",
    "gsm8k_0013",
    "gsm8k_0016",
    "gsm8k_0018",
]


def load_agent_answers(log_path):
    """Map task_id -> (agent_answer, gold_answer) from the log."""
    answers = {}
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("task_id")
            if tid in TARGET_IDS:
                answers[tid] = (rec.get("agent_answer"), rec.get("gold_answer"))
    return answers


def main():
    tasks = generate_tasks(seed=42, n=20, difficulty="gsm8k", filter_bad_rows=True)
    task_map = {t.task_id: t for t in tasks}

    pool = _load_gsm8k_pool()
    solution_by_question = {row["question"]: row["answer"] for row in pool}

    agent_answers = load_agent_answers(LOG_PATH)

    lines = []
    for tid in TARGET_IDS:
        task = task_map[tid]
        agent_answer, gold_logged = agent_answers.get(tid, (None, None))
        lines.append("=" * 88)
        lines.append(f"{tid} | gold (parsed): {task.gold_answer} | "
                     f"gold (logged): {gold_logged} | agent answered: {agent_answer}")
        lines.append("-- QUESTION --")
        lines.append(task.question)
        lines.append("-- REFERENCE SOLUTION --")
        lines.append(solution_by_question.get(task.question, "(question not found in pool)"))
        lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH} with {len(TARGET_IDS)} entries.")


if __name__ == "__main__":
    main()
