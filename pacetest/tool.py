"""The calculator tool. Pure python function, no LLM involved"""
import re

def calculator(expr: str) -> float:
    """Evaluate a simple arithmatic operation and return a number
    
    Supports: +, -, *, /, parentheses, decimal numbers.

    Args:
        expr: a string like "5+3" or "(10/2) / 4"

    Returns:
        THe numeric result as float

    Raises:
        ValueError: if the expression contains unsafe characters or fails to evaluate    
    """
    # Whitelist: only digits, operators, parentheses, spaces, decimals

    if not re.fullmatch(r"[\d\s\+\-\*\/\.\(\)]+", expr):
        raise ValueError(f"Invalid characters in expression: {expr!r}")
    
    try:
        # eval with empty globals + locals = no access to Python built-ins
        result = eval(expr, {"__builtins__" : {}},{})
        return float(result)
    except Exception as e:
        raise ValueError(f"Could not evaluate {expr!r}: {e}")
    


if __name__ == "__main__":
    print(calculator("5+3"))
    print(calculator("(10 + 5) * 2"))