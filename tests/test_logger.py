"""Tests for the JSONL logger."""
import json
import os
import tempfile
from pathlib import Path

from pacetest.logger import init_log, log_round, log_run_stats


def test_init_log_creates_file_with_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = init_log(run_name="test_init", log_dir=tmp)
        assert path.exists()
        with open(path) as f:
            header = json.loads(f.readline())
        assert header["type"] == "header"
        assert header["run_name"] == "test_init"
        assert "model" in header
        assert "git_commit" in header


def test_log_round_appends():
    with tempfile.TemporaryDirectory() as tmp:
        path = init_log(run_name="test_append", log_dir=tmp)
        fake_result = {"task": "5+3", "success": True}
        log_round(path, 0, fake_result)
        log_round(path, 1, fake_result)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rounds
        round0 = json.loads(lines[1])
        assert round0["type"] == "round"
        assert round0["round"] == 0
        assert round0["task"] == "5+3"


def test_extra_metadata_appears_in_header():
    """extra_metadata fields must be written to the header."""
    with tempfile.TemporaryDirectory() as tmp:
        path = init_log(
            run_name="test_extra",
            log_dir=tmp,
            extra_metadata={"task_seed": 13, "knob_alpha": 0.5},
        )
        with open(path) as f:
            header = json.loads(f.readline())
        assert header["task_seed"] == 13
        assert header["knob_alpha"] == 0.5


def test_extra_metadata_cannot_clobber_builtins():
    """extra_metadata fields must NOT overwrite built-in header fields.

    A caller that accidentally passes {'model': 'gpt-4'} in extra_metadata
    must not silently corrupt the reproducibility record. Built-ins win.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = init_log(
            run_name="test_no_clobber",
            log_dir=tmp,
            extra_metadata={"model": "gpt-4", "run_name": "hijacked"},
        )
        with open(path) as f:
            header = json.loads(f.readline())
        assert header["model"] != "gpt-4"
        assert header["run_name"] == "test_no_clobber"


# ---- log_run_stats (Week 10 Day 1) ----

def test_log_run_stats_appends_one_correctly_typed_line():
    """log_run_stats appends exactly one JSON object with type=run_stats.

    Verifies both the append semantics (does not touch header or round
    lines) and the type discipline that lets existing readers skip the
    tail record. Also confirms user-supplied fields make it into the
    written entry and that a timestamp is stamped in.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = init_log(run_name="test_run_stats", log_dir=tmp)
        log_round(path, 0, {"task": "5+3", "success": True})
        log_round(path, 1, {"task": "6+2", "success": True})

        stats = {
            "wall_clock_s": 12.5,
            "pacemaker_consultations": 3,
            "verdict_counts": {"accept": 2, "freeze": 1},
            "gating_cache_hits": 4,
            "gating_cache_misses": 2,
        }
        log_run_stats(path, stats)

        with open(path) as f:
            lines = f.readlines()

        # Exactly 4 lines: header + 2 rounds + 1 run_stats.
        assert len(lines) == 4, f"expected 4 lines, got {len(lines)}"

        entry = json.loads(lines[-1])
        assert entry["type"] == "run_stats"
        assert "timestamp" in entry
        # User-supplied stats survive the write intact.
        assert entry["wall_clock_s"] == 12.5
        assert entry["pacemaker_consultations"] == 3
        assert entry["verdict_counts"] == {"accept": 2, "freeze": 1}
        assert entry["gating_cache_hits"] == 4
        assert entry["gating_cache_misses"] == 2

        # Header and rounds untouched by the tail write.
        assert json.loads(lines[0])["type"] == "header"
        assert json.loads(lines[1])["type"] == "round"
        assert json.loads(lines[2])["type"] == "round"


def test_readers_skip_unknown_type_run_stats():
    """Existing readers (inspect_log, compute_metrics) must skip type=run_stats.

    Regression test locking the reader-skip contract. The two consumer
    scripts share the same skip-on-unknown-type pattern; if either grows
    a special case for run_stats or crashes on it, this test fails and
    Section 4.11's log-tail plan breaks.

    Constructs a log with header, one round, and a run_stats tail, then
    invokes both readers' load functions. Expects the reader to expose
    only the header and the one round; the run_stats line must not
    appear in the returned round list and must not raise.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = init_log(run_name="test_reader_skip", log_dir=tmp)
        log_round(path, 0, {
            "task": "5+3",
            "task_id": "toy_0",
            "success": True,
            "correct": True,
            "agent_answer": 8,
            "gold_answer": 8,
        })
        log_run_stats(path, {
            "wall_clock_s": 1.0,
            "pacemaker_consultations": 0,
            "verdict_counts": {},
        })

        # inspect_log.py contract: (header, rounds) with rounds excluding the tail.
        from scripts.inspect_log import load_jsonl
        header, rounds = load_jsonl(path)
        assert header["type"] == "header"
        assert len(rounds) == 1, f"inspect_log surfaced {len(rounds)} rounds, expected 1"
        assert rounds[0]["type"] == "round"

        # compute_metrics.py contract: (header, rounds, skipped) with skipped
        # counting malformed-JSON lines only. A well-formed run_stats line
        # is NOT malformed; skipped must remain 0.
        from scripts.compute_metrics import load_rounds
        header2, rounds2, skipped = load_rounds(path)
        assert header2["type"] == "header"
        assert len(rounds2) == 1, f"compute_metrics surfaced {len(rounds2)} rounds, expected 1"
        assert skipped == 0, (
            f"compute_metrics counted run_stats as malformed (skipped={skipped}); "
            "reader-skip contract violated."
        )


if __name__ == "__main__":
    test_init_log_creates_file_with_header()
    print("test_init_log_creates_file_with_header passed")
    test_log_round_appends()
    print("test_log_round_appends passed")
    test_extra_metadata_appears_in_header()
    print("test_extra_metadata_appears_in_header passed")
    test_extra_metadata_cannot_clobber_builtins()
    print("test_extra_metadata_cannot_clobber_builtins passed")
    test_log_run_stats_appends_one_correctly_typed_line()
    print("test_log_run_stats_appends_one_correctly_typed_line passed")
    test_readers_skip_unknown_type_run_stats()
    print("test_readers_skip_unknown_type_run_stats passed")