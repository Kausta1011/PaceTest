# Save as scripts/verify_week2.py (create scripts/ folder if needed)
from pacetest.forward_pass import run_one_task
from pacetest.logger import init_log, log_round

log_path = init_log(run_name="week2_verify")
print(f"Logging to: {log_path}")

tasks = ["What is 5 + 3?", "What is 10 - 4?", "What is 2 * 6?"]
for i, task in enumerate(tasks):
    result = run_one_task(task)
    log_round(log_path, i, result)
    print(f"Round {i}: success={result['success']}, answer={result['agent_answer']}")