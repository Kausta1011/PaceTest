"""Embedding-based cycling metrics for PaceTest.

Two metrics defined in Section 3.3.3 of the thesis. Both consume a
list of round dictionaries produced by the loop and return a single
scalar summary of trajectory-level structure.

Requires `sentence-transformers`. Uses the `all-MiniLM-L6-v2` model
(proposal Section 4.1(b)); CPU-only, ~90MB on first use, cached under
the user's Hugging Face directory. The model is loaded lazily on the
first call to any metric function, so importing this module is cheap.
"""
import numpy as np


_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None


def _get_model():
    """Return the cached SentenceTransformer, loading it on first use."""
    global _MODEL
    if _MODEL is None:
        # Lazy import so importing this module never triggers a slow
        # torch/transformers load unless a metric is actually called.
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _embed(texts: list[str]) -> np.ndarray:
    """Return unit-normalised embeddings of the given texts, shape (N, D)."""
    # sentence-transformers rejects the empty string; replace with a
    # single space so pathological rounds do not crash the metric.
    safe = [t if t else " " for t in texts]
    embs = _get_model().encode(
        safe,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embs, dtype=np.float32)


def _mean_consecutive_cosine_distance(texts: list[str]) -> float:
    """Mean cosine distance between consecutive elements of `texts`.

    Distances are clipped to [0.0, 2.0] to absorb floating-point noise
    when two identical prompts are embedded (dot product can compute to
    1.0 plus/minus epsilon).
    """
    if len(texts) < 2:
        return 0.0
    embs = _embed(texts)
    dists = []
    for i in range(1, len(embs)):
        # Unit vectors -> cosine similarity = dot product.
        sim = float(np.dot(embs[i], embs[i - 1]))
        dists.append(max(0.0, 1.0 - sim))
    return float(np.mean(dists))


def semantic_distance_agent(rounds: list[dict]) -> float:
    """Mean cosine distance between consecutive agent prompts.

    A run whose agent prompt never changes has semantic distance 0.
    A run whose agent prompt drifts substantively every round has
    semantic distance close to 1. Fractional values in between capture
    the average per-round drift of the agent artefact.
    """
    prompts = [r.get("agent_prompt", "") for r in rounds]
    return _mean_consecutive_cosine_distance(prompts)


def semantic_distance_tool(rounds: list[dict]) -> float:
    """Mean cosine distance between consecutive tool documentation strings.

    Same interpretation as `semantic_distance_agent`, applied to the
    tool artefact.
    """
    docs = [r.get("tool_doc", "") for r in rounds]
    return _mean_consecutive_cosine_distance(docs)


def joint_trajectory_variance(rounds: list[dict]) -> float:
    """Trace of covariance of joint (agent_prompt, tool_doc) embeddings.

    Concatenates each round's agent prompt and tool doc with a rare
    separator, embeds each concatenated string, and returns the sum of
    per-dimension variances across the K embeddings. Higher = more
    total drift of the joint artefact pair across the trajectory.

    Trace-of-covariance is the natural definition of total variance in
    a high-dimensional embedding space and matches the proposal's
    Section 4.1(f) framing.
    """
    if len(rounds) < 2:
        return 0.0
    joint_texts = [
        (r.get("agent_prompt", "") or " ")
        + " ||| "
        + (r.get("tool_doc", "") or " ")
        for r in rounds
    ]
    embs = _embed(joint_texts)
    # np.var default is population variance (ddof=0); sum across dims.
    per_dim = np.var(embs, axis=0)
    return float(np.sum(per_dim))
