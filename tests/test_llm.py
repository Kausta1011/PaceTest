"""Smoke tests for the llm() helper function"""
from pacetest.llm import llm

def test_llm_returns_string():
    out = llm("Say in one word.")
    assert isinstance(out, str)
    assert len(out) > 0

def test_llm_deterministic():
    """Same prompt + same seed should give same output"""
    a = llm("What is 5 + 3? Reply with just the number")
    b = llm("What is 5 + 3? Reply with just the number")
    assert a == b, f"Expected deterministic output. Got: {a!r} vs {b!r}"

if __name__ == "__main__":
    test_llm_returns_string()
    print("test_llm_returns_string passed")
    test_llm_deterministic
    print("test_llm_deterministic passed")