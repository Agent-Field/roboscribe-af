"""Aloha bimanual TaskAdapter — synthetic v1.

Proves that the OXE-Narrator reasoner DAG is dataset-agnostic. This adapter
implements the same `TaskAdapter` Protocol; switching to it is one env-var
change (`AF_TASK=aloha_transfer`). The reasoner DAG does NOT change.

For v1 we synthesise visually-distinct Aloha-style frames (two arms + a cube
+ a target zone in a 320x180 viewport). For v1.5 the same adapter swaps to
real `lerobot/aloha_sim_transfer_cube_human` data via the `lerobot` library.

Action space is 14-DoF (bimanual ViperX-300, 7 DoF per arm). For synthetic
v1 we model only the 4 end-effector position components (x_left, y_left,
x_right, y_right) and pad the rest with zeros — that's enough to make the
trajectory analytics produce realistic prose summaries.
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

log = logging.getLogger("roboscribe-af.tasks.aloha")


ALOHA_DESCRIPTION = (
    "Aloha (bimanual cube transfer) is a simulated tabletop manipulation task "
    "with two 6-DoF ViperX-300 arms. The right arm picks up a small cube from "
    "the right side of the table and transfers it to the left arm, which then "
    "places it in a target zone on the left. Episodes run at ~10 fps, typically "
    "100–200 frames long. Observations are multi-camera (top + wrist) RGB at "
    "640x480; for synthetic v1 we render a single top-down 320x180 RGB. Actions "
    "are 14-DoF joint targets (7 per arm). Success = cube placed inside the "
    "target zone with neither arm in contact."
)


_N_EPISODES = 10
_FRAMES_PER_EPISODE = 100
_IMG_W = 320
_IMG_H = 180
_FPS = 10.0

# Visual constants
_BG = (240, 240, 235)
_TABLE = (200, 195, 185)
_ARM_LEFT_COLOR = (60, 110, 200)
_ARM_RIGHT_COLOR = (200, 90, 60)
_CUBE_COLOR = (100, 180, 100)
_TARGET_COLOR = (90, 220, 90)


def _render_frame(
    left_arm_xy: tuple[float, float],
    right_arm_xy: tuple[float, float],
    cube_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> Image.Image:
    img = Image.new("RGB", (_IMG_W, _IMG_H), _BG)
    draw = ImageDraw.Draw(img)
    # Table band
    draw.rectangle((0, 60, _IMG_W, _IMG_H - 20), fill=_TABLE)
    # Target zone (left side)
    tx, ty = target_xy
    draw.ellipse((tx - 18, ty - 12, tx + 18, ty + 12), outline=_TARGET_COLOR, width=2)
    # Cube
    cx, cy = cube_xy
    draw.rectangle((cx - 6, cy - 6, cx + 6, cy + 6), fill=_CUBE_COLOR)
    # Left arm — drawn as base + segment + gripper-circle
    lx, ly = left_arm_xy
    draw.line((20, 30, lx, ly), fill=_ARM_LEFT_COLOR, width=3)
    draw.ellipse((lx - 5, ly - 5, lx + 5, ly + 5), fill=_ARM_LEFT_COLOR)
    # Right arm
    rx, ry = right_arm_xy
    draw.line((_IMG_W - 20, 30, rx, ry), fill=_ARM_RIGHT_COLOR, width=3)
    draw.ellipse((rx - 5, ry - 5, rx + 5, ry + 5), fill=_ARM_RIGHT_COLOR)
    return img


def _episode_trajectory(episode_id: int):
    """Synthetic Aloha cube transfer trajectory.

    Phase plan:
      0.00-0.25  : both arms move from rest to working area, right approaches cube
      0.25-0.40  : right grasps cube, lifts
      0.40-0.65  : right hands cube to left (mid-table)
      0.65-0.85  : left transports cube to target zone, places
      0.85-1.00  : both arms retreat
    """
    rng = random.Random(episode_id * 41 + 7)
    variant = episode_id % 4  # 0:clean, 1:slow, 2:fumble, 3:near-miss
    target_xy = (60.0, 110.0)
    cube_start = (240.0 + rng.uniform(-10, 10), 110.0 + rng.uniform(-4, 4))

    rest_left = (40.0, 70.0)
    rest_right = (_IMG_W - 40.0, 70.0)
    handoff = (160.0, 110.0)

    frames_meta = []
    states = []  # we'll store [lx, ly, rx, ry] per frame (4-DoF EE positions)
    cube = cube_start

    for t in range(_FRAMES_PER_EPISODE):
        p = t / max(_FRAMES_PER_EPISODE - 1, 1)

        if p < 0.25:
            a = p / 0.25
            lx, ly = rest_left[0] + (handoff[0] - rest_left[0]) * 0.0, rest_left[1] + (handoff[1] - rest_left[1]) * 0.0
            rx, ry = rest_right[0] * (1 - a) + cube_start[0] * a, rest_right[1] * (1 - a) + cube_start[1] * a
            cube = cube_start
        elif p < 0.40:
            a = (p - 0.25) / 0.15
            lx, ly = rest_left[0] + (handoff[0] - rest_left[0]) * 0.2 * a, rest_left[1] + (handoff[1] - rest_left[1]) * 0.2 * a
            rx, ry = cube_start[0], cube_start[1] - 5 * a  # lift
            cube = (rx, ry)
        elif p < 0.65:
            a = (p - 0.40) / 0.25
            lx, ly = rest_left[0] + (handoff[0] - rest_left[0]) * (0.2 + 0.8 * a), rest_left[1] + (handoff[1] - rest_left[1]) * (0.2 + 0.8 * a)
            rx, ry = cube_start[0] + (handoff[0] - cube_start[0]) * a, cube_start[1] - 5 + (handoff[1] - cube_start[1] + 5) * a
            cube = (rx, ry)
        elif p < 0.85:
            a = (p - 0.65) / 0.20
            lx, ly = handoff[0] + (target_xy[0] - handoff[0]) * a, handoff[1] + (target_xy[1] - handoff[1]) * a
            rx, ry = handoff[0] + (rest_right[0] - handoff[0]) * a, handoff[1] + (rest_right[1] - handoff[1]) * a
            cube = (lx, ly)
        else:
            a = (p - 0.85) / 0.15
            lx, ly = target_xy[0] + (rest_left[0] - target_xy[0]) * a, target_xy[1] + (rest_left[1] - target_xy[1]) * a
            rx, ry = rest_right
            cube = target_xy if variant != 3 else (target_xy[0] + 22, target_xy[1] + 8)

        # Variant perturbations
        if variant == 1:
            lx += rng.uniform(-0.5, 0.5)
            rx += rng.uniform(-0.5, 0.5)
        elif variant == 2 and 0.40 < p < 0.55:
            # Fumble during handoff — arms briefly retreat
            lx += rng.uniform(-6, 6)
            rx += rng.uniform(-6, 6)

        states.append([float(lx), float(ly), float(rx), float(ry)])
        frames_meta.append(((lx, ly), (rx, ry), cube, target_xy))

    actions = [list(states[min(i + 1, _FRAMES_PER_EPISODE - 1)]) for i in range(_FRAMES_PER_EPISODE)]
    return frames_meta, actions, states


@lru_cache(maxsize=_N_EPISODES)
def _episode_frames_cached(episode_id: int) -> list[bytes]:
    meta, _a, _s = _episode_trajectory(episode_id)
    out = []
    for (l_xy, r_xy, cube_xy, target_xy) in meta:
        img = _render_frame(l_xy, r_xy, cube_xy, target_xy)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _bytes_to_data_url(png_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"


class AlohaTransferAdapter:
    """Synthetic Aloha bimanual cube transfer."""

    name = "aloha_transfer"
    source_dataset_id = "synthetic_aloha_transfer_v1"
    description = ALOHA_DESCRIPTION
    embodiment = "bimanual ViperX-300 14-DoF (synthetic 4-DoF EE positions in v1)"
    relevant_scorers = [
        "smoothness",
        "visual_quality",
        "task_success",
        "novelty",
        "edge_case",
    ]

    def load_keyframes(self, episode_id: int, n: int = 4) -> EpisodeFrames:
        if not (0 <= episode_id < _N_EPISODES):
            raise ValueError(f"episode_id={episode_id} out of range [0, {_N_EPISODES})")
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
            raise ValueError(f"episode_id={episode_id} out of range [0, {_N_EPISODES})")
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


register_adapter("aloha_transfer", lambda: AlohaTransferAdapter())
