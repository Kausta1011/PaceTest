"""Rewriter functions: ask the LLM to rewrite the agent prompt and tool doc.

Includes a sanity fallback: if the LLM produces an empty or malformed
rewrite, keep the current prompt or doc. This prevents the empty-cascade
pathology observed during Week 3 integration.

From Week 5 onward the rewriter also consumes a `LoopConfig` (Section
3.2.10). The `judgement_emphasis` directive from `pacetest.knobs` is
embedded into the meta-prompt so the rewriter's behaviour tracks the
self_judgement_weight knob. The `correct` field is displayed if present
in the input result dictionary, or shown as "(not available this round)"
if the loop stripped it under a fractional feedback_strength setting.
The rewriter itself does NOT decide whether the oracle signal is
available; that is the loop's responsibility.
"""
from pacetest.config import LoopConfig
from pacetest.knobs import judgement_emphasis
from pacetest.llm import llm


def _build_agent_meta_prompt(
    current_prompt: str, recent_result: dict, config: LoopConfig
) -> str:
    """Assemble the meta-prompt for rewriting the agent prompt."""
    return f"""You are improving an AI agent's instructions.

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
Oracle correctness: {recent_result.get('correct', '(not available this round)')}

REWRITE GUIDANCE:
{judgement_emphasis(config)}

Rewrite the agent prompt to be more effective on similar tasks. The agent MUST still produce its response in this exact format:

TOOL_CALL: calculator("<expression>")
ANSWER: <number>

Return only the new prompt text, nothing else. Do not add explanation."""


def _build_tool_meta_prompt(
    current_doc: str, recent_result: dict, config: LoopConfig
) -> str:
    """Assemble the meta-prompt for rewriting the tool documentation."""
    return f"""You are improving a tool's documentation so an AI agent uses it correctly.

CURRENT TOOL DOCUMENTATION:
{current_doc}

LAST ROUND OUTCOME:
Task: {recent_result.get('task')}
Agent's tool call: {recent_result.get('tool_call_expr')}
Tool output: {recent_result.get('tool_output')}
Tool error: {recent_result.get('tool_error')}
Agent's claimed answer: {recent_result.get('agent_answer')}
Success (followed protocol): {recent_result.get('success')}
Oracle correctness: {recent_result.get('correct', '(not available this round)')}

REWRITE GUIDANCE:
{judgement_emphasis(config)}

Rewrite the tool documentation to be clearer about what the tool does and how to call it. Return only the new documentation, nothing else. Do not add explanation."""


def rewrite_agent_prompt(
    current_prompt: str, recent_result: dict, config: LoopConfig = None
) -> str:
    """Produce a rewritten agent prompt based on the last round's outcome.

    Args:
        current_prompt: the agent prompt in force at the end of last round.
        recent_result: the augmented dictionary from score_round. If the
            loop has stripped `correct` under feedback_strength policy,
            the meta-prompt will note that the oracle signal is not
            available this round.
        config: LoopConfig whose self_judgement_weight shapes the rewriter
            meta-prompt via judgement_emphasis. Defaults to a Week-4-like
            baseline configuration.

    Returns:
        A new agent prompt, or `current_prompt` if the LLM produces an
        empty, too-short, or format-broken rewrite.
    """
    if config is None:
        config = LoopConfig()
    meta_prompt = _build_agent_meta_prompt(current_prompt, recent_result, config)
    new = llm(meta_prompt, max_tokens=600).strip()
    # Sanity fallback: empty / too-short / format-broken rewrites are rejected.
    if len(new) < 50 or "TOOL_CALL:" not in new or "ANSWER:" not in new:
        return current_prompt
    return new


def rewrite_tool_doc(
    current_doc: str, recent_result: dict, config: LoopConfig = None
) -> str:
    """Produce a rewritten tool documentation based on the last round's outcome.

    Args:
        current_doc: the tool doc in force at the end of last round.
        recent_result: the augmented dictionary from score_round.
        config: LoopConfig whose self_judgement_weight shapes the rewriter
            meta-prompt. Defaults to a Week-4-like baseline configuration.

    Returns:
        A new tool documentation string, or `current_doc` if the LLM
        produces a too-short rewrite.
    """
    if config is None:
        config = LoopConfig()
    meta_prompt = _build_tool_meta_prompt(current_doc, recent_result, config)
    new = llm(meta_prompt, max_tokens=800).strip()
    # Sanity fallback: too-short rewrites are rejected.
    if len(new) < 30:
        return current_doc
    return new
