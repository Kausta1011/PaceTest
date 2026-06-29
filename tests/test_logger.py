"""Tests for the JSONL logger."""
import json
import os
import tempfile
from pathlib import Path

from pacetest.logger import init_log, log_round


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


if __name__ == "__main__":
    test_init_log_creates_file_with_header()
    print("test_init_log_creates_file_with_header passed")
    test_log_round_appends()
    print("test_log_round_appends passed")