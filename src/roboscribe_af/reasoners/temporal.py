"""Temporal coordination reasoners — segmentation + meta-spawned per-segment narrators.

After visual_thread + action_thread complete, the temporal layer:

  temporal_segmenter (composer)
  ├── boundary_skill (deterministic — visual_action_aligner finds candidate boundaries
  │   from action velocity changes + keyframe indices)
  └── boundary_judger (.ai — promotes candidates to real sub-task boundaries
      based on visual + action context)

Then per detected segment (meta-spawned at runtime, NOT pre-defined):

  segment_narrator (composer)  [N instances, parallel via asyncio.gather]
  ├── verb_proposer (.ai — what action verb describes THIS segment?)
  └── object_role_assigner (.ai — which detected objects are involved here?)

This is the meta-prompting pattern: the number of narrator instances and
the prompt each one gets depends on what temporal_segmenter found.
"""

from __future__ import annotations

import asyncio
import logging
import os

from agentfield import AgentRouter
from pydantic import BaseModel, Field

from roboscribe_af.schemas import MotionPhase  # reuse phase schema

log = logging.getLogger("roboscribe-af.reasoners.temporal")

temporal_router = AgentRouter(prefix="", tags=["temporal"])

NODE_ID = os.getenv("AGENT_NODE_ID", "roboscribe-af")


# ────────────────────────────────────────────────────────────────────────────
# Schemas — local to temporal layer
# ────────────────────────────────────────────────────────────────────────────


class BoundaryCandidates(BaseModel):
    """Deterministic skill output: candidate frame indices that LOOK like boundaries."""

    candidate_frame_indices: list[int]
    rationale: str
    n_total_frames: int


class TemporalSegment(BaseModel):
    """One sub-task segment within an episode."""

    start_frame: int
    end_frame: int
    label: str = Field(description="Short verb-phrase: 'approach the cube', 'push toward target', etc.")
    phase: str = Field(description="Reuses MotionPhase tags: approach / contact / manipulate / retreat / idle")
    objects_involved: list[str] = Field(description="Lowercase canonical object names that matter for this segment.")
    confident: bool


class SegmentList(BaseModel):
    """`temporal_segmenter` output."""

    segments: list[TemporalSegment]
    n_total_frames: int
    n_segments: int


# ────────────────────────────────────────────────────────────────────────────
# Boundary skill — deterministic candidate proposal from velocity changes
# ────────────────────────────────────────────────────────────────────────────


@temporal_router.skill(tags=["temporal", "skill"])
async def visual_action_aligner(
    states: list[list[float]],
    keyframe_indices: list[int],
    fps: float = 10.0,
) -> dict:
    """Propose candidate boundary frame indices from state-velocity changes.

    Heuristic: a boundary is where the velocity vector direction OR magnitude
    changes substantially. We use a simple speed-acceleration peak detector
    over the state trajectory. Always returns at least 2 candidates (start +
    end) and at most ~6 (cap for v1).
    """
    if len(states) < 4:
        return {
            "candidate_frame_indices": [0, max(0, len(states) - 1)],
            "rationale": "trajectory too short for boundary detection",
            "n_total_frames": len(states),
        }

    # Per-step speed
    dt = 1.0 / fps
    speeds = []
    for a, b in zip(states[:-1], states[1:]):
        d2 = sum((bi - ai) ** 2 for ai, bi in zip(a, b))
        speeds.append((d2 ** 0.5) / dt)

    # Acceleration magnitudes
    accels = [abs(speeds[i + 1] - speeds[i]) / dt for i in range(len(speeds) - 1)]

    # Peak detector: top 4 accel peaks well-separated
    indexed = sorted(enumerate(accels), key=lambda kv: kv[1], reverse=True)
    candidates: list[int] = [0]
    for idx, _val in indexed:
        if all(abs(idx - c) > max(3, len(states) // 8) for c in candidates):
            candidates.append(idx + 1)  # +1 because accel index is one in from speed
        if len(candidates) >= 5:
            break
    candidates.append(len(states) - 1)
    candidates = sorted(set(candidates))
    return {
        "candidate_frame_indices": candidates,
        "rationale": (
            f"detected {len(candidates)} candidate boundaries from acceleration peaks "
            f"over {len(states)} frames (fps={fps:.0f})"
        ),
        "n_total_frames": len(states),
    }


# ────────────────────────────────────────────────────────────────────────────
# Boundary judger — .ai that promotes candidates to real boundaries
# ────────────────────────────────────────────────────────────────────────────


class JudgedBoundaries(BaseModel):
    """`boundary_judger` output: which candidates are real sub-task boundaries."""

    promoted_frame_indices: list[int] = Field(
        description=(
            "Subset of candidate_frame_indices that correspond to real sub-task "
            "transitions. Always include 0 (start) and the final frame index (end). "
            "Typically 2-5 boundaries for a PushT episode."
        )
    )
    rationale: str
    confident: bool


@temporal_router.reasoner(tags=["temporal", "leaf"])
async def boundary_judger(
    candidate_frame_indices: list[int],
    n_total_frames: int,
    visual_motion_summary: str,
    action_trajectory_prose: str,
    visual_phase: str,
    action_phase: str,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Judge which candidate frame indices correspond to real sub-task boundaries."""
    result = await temporal_router.ai(
        (
            f"Below are signals from a {n_total_frames}-frame robotic episode. "
            "A deterministic skill proposed candidate boundary frames where the "
            "trajectory changed sharply. Decide which are REAL sub-task "
            "transitions (the episode's natural phases — approach → contact → "
            "manipulate → retreat) versus noise.\n\n"
            f"Candidate frame indices: {candidate_frame_indices}\n\n"
            f"Visual motion summary: {visual_motion_summary}\n"
            f"Visual-derived motion phase: {visual_phase}\n"
            f"Action trajectory: {action_trajectory_prose}\n"
            f"Action-derived motion phase: {action_phase}\n\n"
            "Return promoted_frame_indices as a sorted ascending list, always "
            "including 0 and the final frame index. Aim for 2-5 boundaries that "
            "feel like real phase transitions."
        ),
        system=(
            "You are a robotics temporal segmentation expert. "
            f"Task context: {task_description}\n\n"
            "Return ONLY the requested schema fields."
        ),
        schema=JudgedBoundaries,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Segment narrator — meta-spawned per detected segment
# ────────────────────────────────────────────────────────────────────────────


@temporal_router.reasoner(tags=["temporal", "narrator"])
async def segment_narrator(
    start_frame: int,
    end_frame: int,
    n_total_frames: int,
    visual_scene_summary: str,
    canonical_objects: list[str],
    visual_phase: str,
    action_phase: str,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Describe ONE sub-task segment.

    Called once per detected segment. Each instance gets its own prompt
    parameterized to the specific frame range and the broader episode context.
    This is the meta-prompting pattern — the SET of narrator calls is
    determined at runtime by temporal_segmenter.
    """
    fraction = (
        f"frames {start_frame}-{end_frame} (of {n_total_frames}, "
        f"~{int(100 * (end_frame - start_frame) / max(n_total_frames, 1))}% of episode)"
    )
    objs = ", ".join(canonical_objects) if canonical_objects else "<none>"
    result = await temporal_router.ai(
        (
            f"Describe ONE sub-task segment of a robotic episode: {fraction}.\n\n"
            f"Episode scene: {visual_scene_summary}\n"
            f"Episode objects: {objs}\n"
            f"Visual-derived episode-level phase: {visual_phase}\n"
            f"Action-derived episode-level phase: {action_phase}\n\n"
            "Output: a short verb-phrase label for THIS segment "
            "('approach the cube', 'push toward target', 'release contact'), "
            "the dominant motion phase tag for the segment "
            "(approach / contact / manipulate / retreat / idle), and the subset "
            "of canonical objects that are actively involved in THIS segment."
        ),
        system=(
            "You are a careful temporal narrator producing training-grade "
            f"sub-task labels. Task context: {task_description}\n\n"
            "Return ONLY the requested schema fields."
        ),
        schema=TemporalSegment,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Composer — temporal_segmenter
# ────────────────────────────────────────────────────────────────────────────


@temporal_router.reasoner(tags=["temporal", "composer"])
async def temporal_segmenter(
    states: list[list[float]],
    keyframe_indices: list[int],
    fps: float,
    visual_scene_summary: str,
    visual_motion_summary: str,
    canonical_objects: list[str],
    visual_phase: str,
    action_phase: str,
    action_trajectory_prose: str,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Detect sub-task boundaries, then META-SPAWN one narrator per segment.

    Topology this reasoner produces:
      temporal_segmenter
      ├── visual_action_aligner (skill)
      ├── boundary_judger (.ai)
      └── segment_narrator × N  (parallel via asyncio.gather — count depends on
                                  what boundary_judger returned at runtime)
    """
    # Step 1: deterministic candidate boundaries.
    candidates_dict = await temporal_router.call(
        f"{NODE_ID}.visual_action_aligner",
        states=states,
        keyframe_indices=keyframe_indices,
        fps=fps,
    )

    # Step 2: judge which candidates are real boundaries.
    judged_dict = await temporal_router.call(
        f"{NODE_ID}.boundary_judger",
        candidate_frame_indices=candidates_dict["candidate_frame_indices"],
        n_total_frames=candidates_dict["n_total_frames"],
        visual_motion_summary=visual_motion_summary,
        action_trajectory_prose=action_trajectory_prose,
        visual_phase=visual_phase,
        action_phase=action_phase,
        task_description=task_description,
        model=model,
    )
    boundaries = sorted(set(judged_dict["promoted_frame_indices"]))
    if not boundaries:
        boundaries = [0, candidates_dict["n_total_frames"] - 1]

    # Step 3: META-SPAWN one narrator per segment, in parallel.
    segment_ranges = [
        (boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
    ]
    n_total = candidates_dict["n_total_frames"]
    narrator_dicts = await asyncio.gather(
        *[
            temporal_router.call(
                f"{NODE_ID}.segment_narrator",
                start_frame=s,
                end_frame=e,
                n_total_frames=n_total,
                visual_scene_summary=visual_scene_summary,
                canonical_objects=canonical_objects,
                visual_phase=visual_phase,
                action_phase=action_phase,
                task_description=task_description,
                model=model,
            )
            for (s, e) in segment_ranges
        ]
    )

    return {
        "segments": narrator_dicts,
        "n_total_frames": n_total,
        "n_segments": len(narrator_dicts),
        "boundaries": boundaries,
    }
