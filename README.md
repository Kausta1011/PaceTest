# PaceTest

**When do AI agents and their tools deceive each other?**

PaceTest is the experimental codebase for my M.Sc. thesis in Big Data & Artificial Intelligence (SRH Berlin University of Applied Sciences, 2026). It studies *agent-tool co-adaptation*: a closed loop in which an LLM agent's system prompt and its tool's documentation are both rewritten round after round based on feedback. The project asks whether this loop cycles or drifts instead of converging, whether the tool ends up confirming the agent's errors ("tool sycophancy"), and whether simple regulators ("pacemakers") can keep the loop stable.

## Research questions

| | Question |
|---|---|
| **RQ1** | Does co-adaptation cycle, and where in the parameter space does it happen? |
| **RQ2** | Does tool sycophancy emerge, and what does it look like at the artefact level? |
| **RQ3** | Which pacemaker mechanism best prevents degradation, and at what cost? |

## How the loop works

Each round, a local LLM agent solves arithmetic and GSM8K word problems with access to a calculator tool. An oracle scores the answers, and a rewriter updates the agent prompt and the tool doc. Three knobs, each in {0.0, 0.5, 1.0}, shape the dynamics:

- `feedback_strength`: how much oracle feedback drives the rewrite
- `self_judgement_weight`: how much the agent's own judgement counts
- `update_asymmetry`: relative update rate of agent vs. tool

Four pacemaker conditions are compared:

| Condition | Fires on |
|---|---|
| `baseline` | No regulation |
| `freeze` | Within-run geometric spikes in artefact embeddings |
| `diversity` | Stagnation; reseeds from hand-authored prompts |
| `gating` | Correctness drop on held-out tasks (memoised) |

## Headline results

Main sweep: `qwen3:8b`, 3 seeds x 9 knob cells x 4 conditions, K = 20 rounds, n = 20 tasks. Overall, 180 trajectories and 3,680 scored rounds.

| Condition | Correct (of 540) | Degraded cells (of 27) |
|---|---|---|
| Baseline | 380 (70.4%) | concentrated in a few cells |
| **Freeze** | **510 (94.4%)** | **0** |
| Diversity | 349 (64.6%) | 12 (net harm) |
| **Gating** | **505 (93.5%)** | **1** (~1.7x wall-clock) |

- **RQ1:** Cycling is concentrated, not widespread. It is tied to specific knob settings (e.g. `fs=0.0, sjw=0.5` degrades on all three seeds).
- **RQ2:** Tool sycophancy exists conditionally. In one trajectory the agent sends only the first step to the calculator and reports its output verbatim, giving a sycophancy decoupling of +0.70. The mirror failure, *tool abandonment*, also appears. All 205 compliant-but-wrong events were hand-verified against reference solutions.
- **RQ3:** Freeze is the recommended default, with gating as a supplement. Diversity reseeding is harmful as a default.
- **Sensitivity:** The corruption channel is agent-side rewriter drift, not tool-doc drift.
- **Cross-model check** (`claude-haiku-4-5`, `gpt-4o-mini`, total API cost ≈ USD 2.52): correctness degradation and prompt ratcheting reproduce, but the specific sycophancy signature does not. RQ2 conclusions are therefore scoped to Qwen3-8B.

Full methodology, tables, and caveats are in the thesis.

## Repository layout

```
pacetest/        core library
  loop.py          closed-loop driver (forward pass -> oracle -> rewriter)
  forward_pass.py  agent + tool call per task
  rewriter.py      agent-prompt and tool-doc rewriting
  pacemakers.py    freeze / diversity / gating
  knobs.py         frozen LoopConfig and knob handling
  tasks.py         toy arithmetic + GSM8K (with bad-gold filter)
  oracle.py        answer scoring
  metrics.py, embedding_metrics.py   sign-flip rate, sycophancy decoupling,
                                     semantic distance, trajectory variance
  llm.py, llm_api.py                 Ollama / Anthropic / OpenAI backends
  logger.py        JSONL run logs
scripts/         experiment runners and analysis tools
tests/           pytest suite (107 tests)
cell8_refs.txt   append-only verification receipts for hand-checked events
```

## Setup

Requires Python 3.12 and [Ollama](https://ollama.com).

```bash
git clone https://github.com/Kausta1011/PaceTest.git
cd PaceTest
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3:8b
```

Default model settings: `qwen3:8b`, temperature 0.3, seed 42.

Optional, for the cross-model runs:

```bash
export PACETEST_BACKEND=anthropic   # or openai; default is ollama
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY=...
export PACETEST_API_MODEL=...       # optional model override
```

## Running experiments

Run everything from the repo root with `python -m`. Logs are written to `logs/` (gitignored).

```bash
# Single scored run
python -m scripts.run_scored --seed 42 --rounds 20 \
    --feedback-strength 1.0 --self-judgement-weight 0.5 --update-asymmetry 0.5

# 3x3 knob sweep for one seed (add --dry-run to preview)
python -m scripts.run_mini_sweep --seed 42 --rounds 20

# Update-asymmetry sensitivity strip
python -m scripts.run_sens_sweep --pacemaker gating --seed 42 --ua 1.0

# Cross-model validation (use --dry-run first to see the cost estimate)
python -m scripts.run_crossmodel_sweep --backend openai --pacemaker baseline --seed 42 --dry-run

# Metrics over all logs
python -m scripts.compute_metrics

# Dump reference solutions for manual verification
python -m scripts.dump_refs gsm8k_0005 --seed 42
```

The sweep runners refuse to overwrite an existing log, so a completed run cannot be silently polluted.

## Tests

```bash
pytest -q
```

## Reproducibility notes

- Local runs with Ollama are seeded and deterministic.
- Hosted APIs do not honour seeds, so the cross-model runs are not byte-reproducible.
- About 8.5% of sampled GSM8K rows fail an arithmetic-consistency check on their gold answer and are filtered out.

## Author

**Kaustubh Salunke** · M.Sc. Big Data & AI, SRH Berlin
[LinkedIn](https://www.linkedin.com/in/kaustubh-salunke-a4264619a) · [Portfolio](https://kausta1011.github.io)

Supervised by Prof. Dr. Alexander Iliev.
