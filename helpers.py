"""Deterministic helpers — registered as @router.skill() so they appear in the
workflow DAG when reasoners call them via app.call(), while declaring intent
that no LLM judgement happens here.

The skill / reasoner distinction:
  - @router.skill()    → deterministic. No LLM. Same data in → same data out.
  - @router.reasoner() → cognitive. May call .ai() / .harness().

Skills are first-class in the control plane (discovery, workflow tracking, VCs)
per `Agent.skill()` docstring at sdk/python/agentfield/agent.py:2169.
"""

from __future__ import annotations

import os

from agentfield import AgentRouter

# Skills router — prefix="" keeps canonical func-name IDs ("load_episode_keyframes"
# stays as `roboscribe-af.load_episode_keyframes`, not `loader_load_episode_keyframes`).
skills_router = AgentRouter(prefix="", tags=["loader"])


@skills_router.skill(tags=["loader"])
async def load_episode_keyframes(
    episode_id: int,
    n_keyframes: int = 4,
    task: str | None = None,
) -> dict:
    """Deterministic: load N keyframes from the configured TaskAdapter.

    Returns ready-to-use multimodal payload:
      {
        episode_id, fps, n_total_frames, n_keyframes,
        keyframe_indices: [int, ...],
        keyframes_b64: ["data:image/png;base64,...", ...],
        task_name, task_description, embodiment,
      }

    The base64 strings are formatted as `data:image/png;base64,...` data URLs,
    which `.ai()` auto-detects as inline images via its positional multimodal
    args (no external fetch required).
    """
    from tasks import get_adapter  # lazy — keeps startup fast

    adapter = get_adapter(task or os.getenv("AF_TASK"))
    frames = adapter.load_keyframes(episode_id, n=n_keyframes)
    return {
        "episode_id": frames.episode_id,
        "fps": frames.fps,
        "n_total_frames": frames.n_total_frames,
        "n_keyframes": len(frames.keyframes_b64),
        "keyframe_indices": frames.keyframe_indices,
        "keyframes_b64": frames.keyframes_b64,
        "task_name": adapter.name,
        "task_description": adapter.description,
        "embodiment": adapter.embodiment,
    }


@skills_router.skill(tags=["loader"])
async def load_episode_actions(
    episode_id: int,
    task: str | None = None,
) -> dict:
    """Deterministic: load the full action + state stream for an episode."""
    from tasks import get_adapter

    adapter = get_adapter(task or os.getenv("AF_TASK"))
    actions = adapter.load_actions(episode_id)
    return {
        "episode_id": actions.episode_id,
        "fps": actions.fps,
        "n_total_frames": actions.n_total_frames,
        "actions": actions.actions,
        "state": actions.state,
        "task_name": adapter.name,
        "embodiment": adapter.embodiment,
    }


# ────────────────────────────────────────────────────────────────────────────
# Trajectory analytics — deterministic skills consumed by the action thread.
# Each produces raw numbers AND a prose summary suitable for LLM consumption
# (LLMs reason over natural language, not 50-element float arrays).
# ────────────────────────────────────────────────────────────────────────────


def _vec_norm(v: list[float]) -> float:
    return sum(x * x for x in v) ** 0.5


@skills_router.skill(tags=["motion"])
async def velocity_profile(
    states: list[list[float]],
    fps: float = 10.0,
) -> dict:
    """Compute velocity + smoothness statistics from a position-state sequence.

    Returns numeric arrays + a prose summary. `states` is a list of N-D position
    vectors (PushT: 2D agent xy). Used by ee_trajectory_phaser to drive the
    trajectory_phase_classifier LLM call.
    """
    if len(states) < 2:
        return {
            "mean_speed": 0.0,
            "max_speed": 0.0,
            "n_pauses": 0,
            "mean_jerk": 0.0,
            "duration_sec": 0.0,
            "prose_summary": (
                f"Trajectory too short to analyse motion (only {len(states)} samples)."
            ),
        }

    dt = 1.0 / fps
    speeds: list[float] = []
    for a, b in zip(states[:-1], states[1:]):
        delta = [bi - ai for ai, bi in zip(a, b)]
        speeds.append(_vec_norm(delta) / dt)

    jerks: list[float] = []
    for a, b in zip(speeds[:-1], speeds[1:]):
        jerks.append(abs(b - a) / dt)

    mean_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    pause_thr = 0.1 * mean_speed if mean_speed > 0 else 0.0
    n_pauses = 0
    run = 0
    for s in speeds:
        if s < pause_thr:
            run += 1
            if run == 3:
                n_pauses += 1
        else:
            run = 0
    mean_jerk = sum(jerks) / len(jerks) if jerks else 0.0
    duration_sec = (len(states) - 1) * dt

    smoothness = (
        "very smooth" if mean_jerk < 5
        else "moderately smooth" if mean_jerk < 20
        else "jerky"
    )
    pause_phrase = (
        "with no detectable pauses" if n_pauses == 0
        else f"with {n_pauses} brief pause{'s' if n_pauses != 1 else ''}"
    )
    speed_var = (max_speed - 0) / (mean_speed + 1e-9)
    consistency = (
        "speed was consistent" if speed_var < 2.0
        else "speed varied considerably (peak much higher than mean)"
    )
    prose = (
        f"Over {duration_sec:.1f} seconds ({len(states)} samples at {fps:.0f} fps), "
        f"the end-effector moved with a mean speed of {mean_speed:.1f} units/sec "
        f"(peak {max_speed:.1f}); {consistency}. The motion was {smoothness} "
        f"(mean jerk {mean_jerk:.1f}) {pause_phrase}."
    )

    return {
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "mean_jerk": mean_jerk,
        "n_pauses": n_pauses,
        "duration_sec": duration_sec,
        "prose_summary": prose,
    }


# ────────────────────────────────────────────────────────────────────────────
# Embedding skill — OpenRouter open-source embedding model
# ────────────────────────────────────────────────────────────────────────────

# Default embedder is BAAI/bge-large-en-v1.5 (MIT-licensed, 1024-dim, strong
# open-source retrieval baseline). Verified live against OpenRouter on 2026-05-16.
# Override per-call via `model=` kwarg if you want a smaller or bigger embedder.
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_DIM = 1024


@skills_router.skill(tags=["embed"])
async def embed_text(
    text: str,
    model: str | None = None,
) -> dict:
    """Embed `text` via OpenRouter's open-source embedding endpoint.

    Returns the vector as a list (1024-dim with the default BAAI model).
    Skill, not reasoner — pure deterministic API call, no LLM judgement.
    """
    import httpx

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": model or DEFAULT_EMBED_MODEL,
        "input": text,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    vec = data["data"][0]["embedding"]
    return {
        "embedding": vec,
        "dim": len(vec),
        "model": payload["model"],
        "tokens": data.get("usage", {}).get("prompt_tokens", 0),
    }
