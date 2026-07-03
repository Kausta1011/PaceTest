"""Append-only JSONL logger for experiment runs.

Invariant: every log line is a complete JSON object. `json.dumps()`
escapes newlines inside strings (as `\\n`), so multi-line agent
prompts and rewritten artefacts fit on a single line. Some legacy Week 3
logs violate this invariant and readers must therefore be tolerant of
JSON parse errors on individual lines. See `scripts/compute_metrics.py`
and `scripts/inspect_log.py` for the skip-on-error pattern.
"""
import json
import time
import platform
import subprocess
from pathlib import Path

from pacetest.llm import MODEL, DEFAULT_SEED


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _ollama_version() -> str:
    """Return Ollama's reported version, or 'unknown'."""
    try:
        r = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def init_log(
    run_name: str = None,
    log_dir: str = "logs",
    extra_metadata: dict = None,
) -> Path:
    """Create a new JSONL log file with a metadata header.

    Args:
        run_name: Optional identifier. Defaults to 'run_<unix-timestamp>'.
        log_dir: Folder to write into. Created if it doesn't exist.
        extra_metadata: Optional dict of extra fields to merge into the
            header. Use for run-level config that is not otherwise captured
            (task seed, knob values, pacemaker choice, etc.). Fields in
            extra_metadata never override the built-in fields.

    Returns:
        Path to the new log file.
    """
    if extra_metadata is None:
        extra_metadata = {}

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"run_{int(time.time())}"

    path = log_dir_path / f"{run_name}.jsonl"

    header = {
        "type": "header",
        "timestamp": time.time(),
        "run_name": run_name,
        "model": MODEL,
        "seed": DEFAULT_SEED,
        "git_commit": _git_commit(),
        "ollama_version": _ollama_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    # Merge caller-supplied fields without letting them clobber built-ins.
    for key, value in extra_metadata.items():
        if key not in header:
            header[key] = value

    with open(path, "a") as f:
        f.write(json.dumps(header) + "\n")
    return path


def log_round(log_path: Path, round_num: int, result: dict) -> None:
    """Append one round's result dictionary to the log file."""
    entry = {
        "type": "round",
        "round": round_num,
        "timestamp": time.time(),
        **result,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")