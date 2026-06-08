"""Tests for the calculator tool"""
from pacetest.tool import calculator

def test_basic_arithmetic():
    assert calculator("5+3") == 8.0
    assert calculator("10-4") == 6.0
    assert calculator("6*2") == 12.0
    assert calculator("15/3") == 5.0    

def test_parantheses_and_precedence():
    assert calculator("(10+5) * 2") == 30.0
    assert calculator("100 / (5+5)") == 10.0
    assert calculator("2+3*4") == 14

def test_rejects_unsafe_input():
    for bad in ["import os", "open(/tmp)", "5+a"]:
        try:
            calculator(bad)
            assert False, f"Should have rejected : {bad!r}"
        except ValueError:
            pass


if __name__ == "__main__":
    test_basic_arithmetic()
    print("test_basic_arithmetic passed")

    test_parantheses_and_precedence()
    print("test_parantheses_and_precedence passed")

    test_rejects_unsafe_input()
    print("test_rejects_unsafe_input")