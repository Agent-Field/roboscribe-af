"""PushT TaskAdapter — synthetic renderer for v1 development.

Why synthetic for v1: lerobot/pusht v3 stores frames as concatenated AV1-encoded
MP4 files (one big video per chunk, episodes need offset lookup). Decoding cleanly
requires the lerobot library + AV1 decoder. Heavy deps just to validate the
reasoner architecture. The synthetic renderer below produces *visually accurate*
PushT scenes (light blue T-block, green target outline, gray circular agent on a
white plane) from procedurally-generated trajectories. Qwen3-VL gets the same
kind of input it would get from the real dataset.

v1.5 follow-up: add `LeRobotPushTAdapter` that uses the `lerobot` library proper
to load real frames. The reasoner DAG does not change — only this adapter.

Layout produced:
  ~10 episodes, each 50 frames at 10 fps
  Each frame: 96x96 RGB
  Trajectories: agent pushes T-block toward target (varied success per episode)
"""

from __future__ import annotations

import base64
import io
import logging
import math
import random
from functools import lru_cache

from PIL import Image, ImageDraw

from .base import EpisodeActions, EpisodeFrames, TaskAdapter
from .registry import register_adapter

log = logging.getLogger("roboscribe-af.tasks.pusht")


PUSHT_DESCRIPTION = (
    "PushT is a 2D simulated manipulation task. A circular agent (the end-effector) "
    "pushes a T-shaped block on a flat plane. The goal is to move the T-block from "
    "its initial random pose into alignment with a fixed green T-shaped target outline "
    "in the centre of the scene. Episodes run at ~10 fps, typically 50–300 frames long. "
    "Observations are top-down 96x96 RGB renders. Actions are 2D target positions for "
    "the agent."
)


# Procedural episode bank — deterministic seeds for reproducibility.
_N_EPISODES = 10
_FRAMES_PER_EPISODE = 50
_IMG_SIZE = 96
_FPS = 10.0

_AGENT_RADIUS = 6
_AGENT_COLOR = (60, 60, 70)
_TBLOCK_COLOR = (130, 175, 240)  # light blue
_TARGET_COLOR = (60, 175, 90)  # green outline
_BG_COLOR = (250, 250, 250)

# T-block dimensions (relative units)
_T_BAR_W = 28
_T_BAR_H = 8
_T_STEM_W = 8
_T_STEM_H = 22


def _t_polygon(cx: float, cy: float, theta: float) -> list[tuple[float, float]]:
    """Return the 8-vertex polygon for a T-shape centred at (cx, cy), rotated by theta (rad)."""
    # T-shape in local coords (origin at centroid).
    # Top bar (width _T_BAR_W, height _T_BAR_H), stem hanging down.
    half_w = _T_BAR_W / 2
    half_stem_w = _T_STEM_W / 2
    local = [
        (-half_w, -_T_BAR_H / 2),
        (+half_w, -_T_BAR_H / 2),
        (+half_w, +_T_BAR_H / 2),
        (+half_stem_w, +_T_BAR_H / 2),
        (+half_stem_w, +_T_BAR_H / 2 + _T_STEM_H),
        (-half_stem_w, +_T_BAR_H / 2 + _T_STEM_H),
        (-half_stem_w, +_T_BAR_H / 2),
        (-half_w, +_T_BAR_H / 2),
    ]
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return [
        (cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t)
        for (x, y) in local
    ]


def _render_frame(
    agent_xy: tuple[float, float],
    block_xy: tuple[float, float],
    block_theta: float,
    target_xy: tuple[float, float],
    target_theta: float = 0.0,
) -> Image.Image:
    img = Image.new("RGB", (_IMG_SIZE, _IMG_SIZE), _BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Target outline (green, hollow)
    draw.polygon(_t_polygon(*target_xy, target_theta), outline=_TARGET_COLOR, width=2)

    # T-block (filled blue)
    draw.polygon(_t_polygon(*block_xy, block_theta), fill=_TBLOCK_COLOR)

    # Agent (gray circle)
    ax, ay = agent_xy
    draw.ellipse(
        (ax - _AGENT_RADIUS, ay - _AGENT_RADIUS, ax + _AGENT_RADIUS, ay + _AGENT_RADIUS),
        fill=_AGENT_COLOR,
    )
    return img


def _episode_trajectory(episode_id: int):
    """Generate a deterministic trajectory for episode `episode_id`.

    Adds per-episode VARIETY so cross-modal disagreements happen on different
    episodes (proves the architecture handles heterogeneity). Episode-mod
    cycles through:
      0: clean success
      1: jerky-but-successful (high acceleration peaks)
      2: hesitation (a pause mid-push)
      3: undershoot (push stops before target)
      4: clean success (different starting positions)
      ... and so on
    """
    rng = random.Random(episode_id * 17 + 3)
    variant = episode_id % 5
    target_xy = (_IMG_SIZE / 2, _IMG_SIZE / 2)

    block_start = (
        target_xy[0] + rng.uniform(-22, 22),
        target_xy[1] + rng.uniform(-22, 22),
    )
    block_start_theta = rng.uniform(-math.pi / 4, math.pi / 4)
    agent_start = (
        block_start[0] + rng.uniform(-30, 30),
        block_start[1] + rng.uniform(-30, 30),
    )

    # End-of-push target — for variant 3 (undershoot) we stop short.
    push_end_xy = target_xy if variant != 3 else (
        block_start[0] * 0.4 + target_xy[0] * 0.6,
        block_start[1] * 0.4 + target_xy[1] * 0.6,
    )

    frames_meta = []
    states = []
    pause_frames = {
        2: range(20, 28),  # variant 2 = hesitation at mid-push
    }.get(variant, range(0, 0))

    for t in range(_FRAMES_PER_EPISODE):
        p = t / max(_FRAMES_PER_EPISODE - 1, 1)

        # Compute base position
        if p < 0.4:
            a = p / 0.4
            agent = (
                agent_start[0] * (1 - a) + block_start[0] * a,
                agent_start[1] * (1 - a) + block_start[1] * a,
            )
            block = block_start
            theta = block_start_theta
        else:
            a = (p - 0.4) / 0.6
            agent = (
                block_start[0] * (1 - a) + push_end_xy[0] * a,
                block_start[1] * (1 - a) + push_end_xy[1] * a,
            )
            block = (
                block_start[0] * (1 - a) + push_end_xy[0] * a,
                block_start[1] * (1 - a) + push_end_xy[1] * a,
            )
            theta = block_start_theta * (1 - a)

        # Variant-specific perturbations
        if t in pause_frames:
            # Hesitation: freeze position (overwrite with previous)
            if states:
                agent = (states[-1][0], states[-1][1])
                block = (block[0], block[1])
        if variant == 1:
            # Jerky: random small kicks
            agent = (agent[0] + rng.uniform(-2.0, 2.0), agent[1] + rng.uniform(-2.0, 2.0))

        agent = (max(8, min(_IMG_SIZE - 8, agent[0])), max(8, min(_IMG_SIZE - 8, agent[1])))
        frames_meta.append((agent, block, theta, target_xy))
        states.append([float(agent[0]), float(agent[1])])

    actions = [list(states[min(i + 1, _FRAMES_PER_EPISODE - 1)]) for i in range(_FRAMES_PER_EPISODE)]
    return frames_meta, actions, states


@lru_cache(maxsize=_N_EPISODES)
def _episode_frames_cached(episode_id: int) -> list[bytes]:
    """Render all frames for `episode_id` to PNG bytes, cached."""
    meta, _actions, _states = _episode_trajectory(episode_id)
    out = []
    for (agent, block, theta, target_xy) in meta:
        img = _render_frame(agent, block, theta, target_xy)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _bytes_to_data_url(png_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"


class PushTAdapter:
    """Synthetic PushT adapter (v1). Renders 96x96 PushT-like scenes on demand."""

    name = "pusht"
    source_dataset_id = "synthetic_pusht_v1"
    description = PUSHT_DESCRIPTION
    embodiment = "2D circular end-effector on plane"
    relevant_scorers = ["smoothness", "task_success", "novelty"]

    def load_keyframes(self, episode_id: int, n: int = 4) -> EpisodeFrames:
        if not (0 <= episode_id < _N_EPISODES):
            raise ValueError(
                f"episode_id={episode_id} out of range [0, {_N_EPISODES})"
            )
        frames = _episode_frames_cached(episode_id)
        total = len(frames)
        if n >= total:
            picks = list(range(total))
        elif n == 1:
            picks = [0]
        else:
            picks = [round(i * (total - 1) / (n - 1)) for i in range(n)]
        return EpisodeFrames(
            episode_id=episode_id,
            fps=_FPS,
            n_total_frames=total,
            keyframe_indices=picks,
            keyframes_b64=[_bytes_to_data_url(frames[i]) for i in picks],
        )

    def load_actions(self, episode_id: int) -> EpisodeActions:
        if not (0 <= episode_id < _N_EPISODES):
            raise ValueError(
                f"episode_id={episode_id} out of range [0, {_N_EPISODES})"
            )
        _meta, actions, states = _episode_trajectory(episode_id)
        return EpisodeActions(
            episode_id=episode_id,
            fps=_FPS,
            n_total_frames=len(states),
            actions=actions,
            state=states,
        )

    def n_episodes(self) -> int:
        return _N_EPISODES


def _build_pusht() -> TaskAdapter:
    return PushTAdapter()


register_adapter("pusht", _build_pusht)
