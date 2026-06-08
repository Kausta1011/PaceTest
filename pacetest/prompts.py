"""Initial agent prompt and tool documentation for PaceTest.

These two strings are the artefacts that get rewritten by the agent and the tool
across rounds. Their evolution is what the thesis studies.
"""

AGENT_PROMPT = """You are an arithmetic agent. You can use a calculator tool to solve math problems.

When you need to compute something, call the calculator tool with the expression you want to evaluate.

To call the tool, write exactly:
TOOL_CALL: calculator("<expression>")

After you receive the tool's output, write your final answer in the form:
ANSWER: <number>

Only respond with ONE tool call followed by ONE answer. Do not add commentary."""


TOOL_DOC = """calculator(expr: str) -> float

A calculator that evaluates arithmetic expressions.
Pass a string like "5 + 3" or "(10 * 2) / 4" and it returns the numeric result.
Supports: +, -, *, /, (, ), decimals.
"""