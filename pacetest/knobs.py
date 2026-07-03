"""Apply the three LoopConfig knobs to per-round decisions.

Each knob is expressed as a helper that the loop calls once per round.
The helpers are deterministic: same round number and same config produce
the same decisions across runs. Non-reproducibility is a real risk in
sweep experiments, so the pseudo-random pattern used for fractional
knob values is derived from a hash rather than the global RNG.

Feedback strength decides whether the oracle's `correct` flag is shown
to the rewriter this round. Update asymmetry decides which of the two
artefacts (agent prompt, tool documentation) is rewritten this round.
Self-judgement weight is not a per-round decision; it is a directive
that shapes the rewriter's meta-prompt, produced by `judgement_emphasis`.
"""
import hashlib

from pacetest.config import LoopConfig


def _round_score(round_num: int, salt: str) -> float:
    """Return a deterministic pseudo-random value in [0.0, 1.0) for a round.

    Same round_num and salt always produce the same score. Different
    salts produce independent-looking sequences without needing extra
    seed parameters. This is what makes fractional knob values (e.g.,
    feedback_strength=0.5) reproducible across runs.
    """
    h = hashlib.md5(f"{salt}:{round_num}".encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


def should_include_oracle(round_num: int, config: LoopConfig) -> bool:
    """Should the rewriter see the oracle's `correct` flag this round?

    At `feedback_strength=1.0` always True (Week 4 baseline). At 0.0
    always False. Fractional values produce a deterministic pattern
    where approximately `feedback_strength * K` rounds out of K return
    True over any long-enough K.
    """
    fs = config.feedback_strength
    if fs >= 1.0:
        return True
    if fs <= 0.0:
        return False
    return _round_score(round_num, "feedback") < fs


def should_update_agent(round_num: int, config: LoopConfig) -> bool:
    """Should the agent prompt be rewritten this round?

    Agent-update probability rises linearly from 0 at `update_asymmetry=0.0`
    to 1 at `update_asymmetry=0.5`, then stays at 1 for all values above 0.5.
    This means the Week 4 baseline (`update_asymmetry=0.5`) rewrites the
    agent prompt every round, matching pre-Week-5 behaviour exactly.
    """
    prob = min(1.0, 2.0 * config.update_asymmetry)
    if prob >= 1.0:
        return True
    if prob <= 0.0:
        return False
    return _round_score(round_num, "update_agent") < prob


def should_update_tool(round_num: int, config: LoopConfig) -> bool:
    """Should the tool documentation be rewritten this round?

    Tool-update probability is the mirror of the agent-update probability:
    1.0 at `update_asymmetry=0.0`, falling linearly to 0.0 at 1.0. At the
    Week 4 baseline (0.5) both artefacts are rewritten every round.
    """
    prob = min(1.0, 2.0 * (1.0 - config.update_asymmetry))
    if prob >= 1.0:
        return True
    if prob <= 0.0:
        return False
    return _round_score(round_num, "update_tool") < prob


def judgement_emphasis(config: LoopConfig) -> str:
    """Return a text directive for the rewriter meta-prompt.

    The self_judgement_weight knob is discretised into three regimes:
    oracle-focused (low weight), balanced (mid weight), self-critique-focused
    (high weight). The returned string is embedded verbatim into the
    rewriter's meta-prompt template so the rewriter knows how to weight
    correctness feedback versus the agent's own reasoning.
    """
    w = config.self_judgement_weight
    if w < 0.34:
        return (
            "Base your rewrite primarily on the oracle's correctness assessment "
            "(the `correct` flag). The agent's own reasoning is secondary."
        )
    if w < 0.67:
        return (
            "Weigh the oracle's correctness assessment (the `correct` flag) "
            "and the agent's own reasoning about the task with roughly equal emphasis."
        )
    return (
        "Base your rewrite primarily on how the agent reasoned about the task. "
        "The oracle's correctness assessment is secondary."
    )


if __name__ == "__main__":
    print("=== Week 4 baseline defaults ===")
    default = LoopConfig()
    for r in range(5):
        print(
            f"round {r}: "
            f"oracle={should_include_oracle(r, default)}, "
            f"update_agent={should_update_agent(r, default)}, "
            f"update_tool={should_update_tool(r, default)}"
        )
    print(f"judgement_emphasis: {judgement_emphasis(default)!r}")
    print()

    print("=== Aggressive: fs=0.25, sjw=0.75, ua=1.0 ===")
    aggressive = LoopConfig(
        feedback_strength=0.25,
        self_judgement_weight=0.75,
        update_asymmetry=1.0,
    )
    n_oracle = sum(should_include_oracle(r, aggressive) for r in range(100))
    n_agent = sum(should_update_agent(r, aggressive) for r in range(100))
    n_tool = sum(should_update_tool(r, aggressive) for r in range(100))
    print(f"over 100 rounds: oracle_shown={n_oracle}/100 (target ~25),")
    print(f"                 agent_updated={n_agent}/100 (target 100),")
    print(f"                 tool_updated={n_tool}/100 (target 0)")
    print(f"judgement_emphasis: {judgement_emphasis(aggressive)!r}")
