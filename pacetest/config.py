"""Configuration container for one closed-loop experiment run.

Holds the three controllable knobs defined in the proposal (Section 2.3):
feedback_strength, self_judgement_weight, update_asymmetry.

The knobs will be consumed by the rewriter (Week 5 Days 2-3) and by the
loop (Week 5 Days 3-4) once the plumbing lands. On Day 1 this file only
defines the container and the validation; behaviour changes come next.

Defaults are chosen to reproduce Week 4 baseline behaviour, so existing
runners that do not yet pass a config produce the same log content as
before the knobs were introduced.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class LoopConfig:
    """The three controllable knobs of the closed rewriting loop.

    Fields:
        feedback_strength: fraction of rounds in which the oracle signal
            is provided to the rewriter, in [0.0, 1.0]. 1.0 means the
            rewriter always sees the oracle's `correct` bit; 0.0 means
            it never sees it. Default 1.0 (Week 4 baseline).
        self_judgement_weight: relative weight of the agent's own
            self-critique versus the oracle signal in the rewrite prompt,
            in [0.0, 1.0]. 0.0 means the rewriter uses only oracle
            grounding; 1.0 means the rewriter uses only the agent's
            self-critique. Default 0.5 (balanced emphasis), which most
            closely reproduces the Week 4 rewriter's neutral behaviour
            (the Week 4 meta-prompt referenced neither signal explicitly).
        update_asymmetry: relative update frequency of the agent's prompt
            versus the tool's documentation per round, in [0.0, 1.0].
            0.0 means only the tool doc is rewritten; 1.0 means only the
            agent prompt is rewritten; 0.5 means both are rewritten every
            round. Default 0.5 (Week 4 baseline: symmetric).
    """
    feedback_strength: float = 1.0
    self_judgement_weight: float = 0.5
    update_asymmetry: float = 0.5

    def __post_init__(self):
        """Validate ranges at construction time. Raises ValueError."""
        for name in ("feedback_strength", "self_judgement_weight", "update_asymmetry"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be a number, got {type(value).__name__}"
                )
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {value!r}"
                )

    def asdict(self) -> dict:
        """Return the config as a plain dict, suitable for log metadata."""
        return asdict(self)


if __name__ == "__main__":
    default = LoopConfig()
    print("Default config:")
    for k, v in default.asdict().items():
        print(f"  {k}: {v}")
    print()
    aggressive = LoopConfig(
        feedback_strength=0.25,
        self_judgement_weight=0.75,
        update_asymmetry=1.0,
    )
    print("Aggressive-rewriter config (agent only, mostly self-critique):")
    for k, v in aggressive.asdict().items():
        print(f"  {k}: {v}")
