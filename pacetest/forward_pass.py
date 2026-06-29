"""Forward pass: run one task through the agent and tool pipeline."""

import re
from pacetest.llm import llm
from pacetest.tool import calculator
from pacetest.prompts import AGENT_PROMPT, TOOL_DOC


def parse_tool_call(text: str):
    """Extract the expression from a 'TOOL_CALL: calculator(...)' line.

    Returns the expression string, or None if no tool call is found.
    """
    match = re.search(r'TOOL_CALL:\s*calculator\s*\(\s*"([^"]*)"\s*\)', text)
    return match.group(1) if match else None


def parse_answer(text: str):
    """Extract the number from an 'ANSWER: <number>' line.

    Returns a float, or None if no answer is found.
    """
    match = re.search(r'ANSWER:\s*(-?\d+\.?\d*)', text)
    return float(match.group(1)) if match else None


def run_one_task(task: str,
                 agent_prompt: str = AGENT_PROMPT,
                 tool_doc: str = TOOL_DOC) -> dict:
    """Run one task end-to-end through the agent and tool.

    Args:
        task: The task description (e.g. "What is 5 + 3?").
        agent_prompt: The agent's instructions.
        tool_doc: The tool's documentation.

    Returns:
        Dictionary with task, agent_response, tool_call_expr, tool_output,
        tool_error, agent_answer, success.
    """
    # 1. Build the full prompt
    full_prompt = (
        f"{agent_prompt}\n\n"
        f"Tool documentation:\n{tool_doc}\n\n"
        f"Task: {task}\n\n"
        f"Your response:"
    )

    # 2. Get the agent's response
    agent_response = llm(full_prompt)

    # 3. Parse the tool call
    tool_call_expr = parse_tool_call(agent_response)

    # 4. Run the calculator if we found a tool call
    tool_output = None
    tool_error = None
    if tool_call_expr is not None:
        try:
            tool_output = calculator(tool_call_expr)
        except ValueError as e:
            tool_error = str(e)

    # 5. Parse the agent's answer
    agent_answer = parse_answer(agent_response)

    # 6. Success means the agent produced both a tool call AND an answer
    success = tool_call_expr is not None and agent_answer is not None

    return {
        "task": task,
        "agent_prompt": agent_prompt,
        "tool_doc": tool_doc,
        "agent_response": agent_response,
        "tool_call_expr": tool_call_expr,
        "tool_output": tool_output,
        "tool_error": tool_error,
        "agent_answer": agent_answer,
        "success": success,
    }


if __name__ == "__main__":
    result = run_one_task("What is 5 + 3?")
    print("Agent response:")
    print(result["agent_response"])
    print()
    print(f"Tool call: {result['tool_call_expr']!r}")
    print(f"Tool output: {result['tool_output']}")
    print(f"Agent answer: {result['agent_answer']}")
    print(f"Success: {result['success']}")