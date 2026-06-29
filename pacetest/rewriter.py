"""Rewriter functions: ask the LLM to rewrite the agent prompt and tool doc."""
from pacetest.llm import llm


def rewrite_agent_prompt(current_prompt: str, recent_result: dict) -> str:
    """Produce a rewritten agent prompt based on the last round's outcome."""
    meta_prompt = f"""You are improving an AI agent's instructions.

CURRENT AGENT PROMPT:
{current_prompt}

LAST ROUND OUTCOME:
Task: {recent_result.get('task')}
Agent's response: {recent_result.get('agent_response')}
Tool call extracted: {recent_result.get('tool_call_expr')}
Tool output: {recent_result.get('tool_output')}
Tool error: {recent_result.get('tool_error')}
Agent's claimed answer: {recent_result.get('agent_answer')}
Success (followed protocol): {recent_result.get('success')}

Rewrite the agent prompt to be more effective on similar tasks. The agent MUST still produce its response in this exact format:

TOOL_CALL: calculator("<expression>")
ANSWER: <number>

Return only the new prompt text, nothing else. Do not add explanation."""

    return llm(meta_prompt, max_tokens=600).strip()


def rewrite_tool_doc(current_doc: str, recent_result: dict) -> str:
    """Produce a rewritten tool documentation based on the last round's outcome."""
    meta_prompt = f"""You are improving a tool's documentation so an AI agent uses it correctly.

CURRENT TOOL DOCUMENTATION:
{current_doc}

LAST ROUND OUTCOME:
Task: {recent_result.get('task')}
Agent's tool call: {recent_result.get('tool_call_expr')}
Tool output: {recent_result.get('tool_output')}
Tool error: {recent_result.get('tool_error')}

Rewrite the tool documentation to be clearer about what the tool does and how to call it. Return only the new documentation, nothing else. Do not add explanation."""

    return llm(meta_prompt, max_tokens=400).strip()