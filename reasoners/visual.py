"""Visual thread reasoners.

Cognitive cascade for the vision side of an episode:

    visual_thread (orchestrator — parallel fan-out)
    ├── scene_describer (composer — fans out object_detector per keyframe + composes)
    │   └── object_detector × N keyframes  (parallel asyncio.gather)
    └── motion_phase_classifier (parallel with scene_describer)

Depth from entry (annotate_episode → visual_thread → scene_describer → object_detector) = 4.
Parallelism at multiple layers: scene_describer || motion_phase_classifier, and
object_detector × N inside scene_describer.

Every reasoner here accepts `model: str | None = None` and threads it through.
"""

from __future__ import annotations

import asyncio
import logging
import os

from agentfield import AgentRouter

from schemas import (
    DetectedObjects,
    MotionPhase,
    SceneDescription,
    VisualThreadResult,
)

log = logging.getLogger("roboscribe-af.reasoners.visual")

# prefix="" → reasoner ids match function names (canonical default)
visual_router = AgentRouter(prefix="", tags=["visual"])

# Read node id from env inside the router file (router.node_id does NOT proxy
# per the SDK surface contract in choosing-primitives.md).
NODE_ID = os.getenv("AGENT_NODE_ID", "roboscribe-af")


# ────────────────────────────────────────────────────────────────────────────
# Layer 1 leaf — object_detector
# Cognitive question: "what concrete, named objects are visible in THIS frame?"
# ────────────────────────────────────────────────────────────────────────────


@visual_router.reasoner(tags=["visual", "leaf"])
async def object_detector(
    keyframe_b64: str,
    frame_index: int,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Detect objects in a single keyframe.

    Input: one image (data URL) + the textual task context so the model knows
    what kind of scene it's looking at (the embodiment + the task semantics).
    Output: flat DetectedObjects schema.
    """
    result = await visual_router.ai(
        (
            f"Frame {frame_index} of a robotic episode. List the concrete, "
            "named visible objects (geometric shapes, robot parts, target markers, "
            "etc.). Use short lowercase phrases without articles. If the visual "
            "signal is ambiguous, set `confident: false`."
        ),
        keyframe_b64,
        system=(
            "You are a precise visual annotator for robotics datasets. "
            f"Task context: {task_description}\n\n"
            "Return ONLY the requested schema fields."
        ),
        schema=DetectedObjects,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Layer 1 leaf — motion_phase_classifier
# Cognitive question: "across THIS sequence of keyframes, what is the dominant
# motion phase?"
# ────────────────────────────────────────────────────────────────────────────


@visual_router.reasoner(tags=["visual", "leaf"])
async def motion_phase_classifier(
    keyframes_b64: list[str],
    keyframe_indices: list[int],
    task_description: str,
    model: str | None = None,
) -> dict:
    """Classify the dominant motion phase across a keyframe sequence."""
    # Multimodal positional args: prompt text + N images, in order.
    user_intro = (
        "Below are keyframes of a robotic episode in temporal order "
        f"(frame indices {keyframe_indices}). Examine how the scene evolves. "
        "Classify the dominant motion phase across the whole sequence: "
        "'approach' (robot reaches toward object), 'contact' (robot engages "
        "object), 'manipulate' (object is being repositioned), 'retreat' (robot "
        "disengages), or 'idle' (no meaningful motion). Pick ONE."
    )
    result = await visual_router.ai(
        user_intro,
        *keyframes_b64,
        system=(
            "You are a robotics motion analyst. "
            f"Task context: {task_description}\n\n"
            "Return ONLY the requested schema fields."
        ),
        schema=MotionPhase,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Single-frame validator — objective ground-truth probe
# Used to quickly benchmark vision quality across models on known frames.
# Each question has a binary or categorical answer we can score against truth.
# ────────────────────────────────────────────────────────────────────────────


from pydantic import BaseModel, Field as _Field  # local import to avoid global churn


class FrameValidation(BaseModel):
    """Objective per-frame ground-truth probe."""

    agent_visible: bool = _Field(description="Is the robot agent (small circular dot end-effector) visible in the frame?")
    agent_color: str = _Field(description="Colour of the agent dot: 'blue' / 'gray' / 'red' / 'other'. Lowercase single word.")
    n_t_shapes: int = _Field(ge=0, le=10, description="Number of distinct T-shaped objects visible (e.g. block + target outline = 2).")
    block_color: str = _Field(description="Colour of the SOLID T-shaped block (not the outline). 'gray' / 'blue' / 'red' / 'other'. Lowercase single word.")
    target_color: str = _Field(description="Colour of the T-shaped target outline. 'green' / 'gray' / 'red' / 'other'. Lowercase single word.")
    block_aligned_with_target: bool = _Field(description="Is the solid block roughly aligned in position AND rotation with the target outline (substantial overlap)? True only if both criteria hold.")
    agent_touching_block: bool = _Field(description="Is the agent dot directly touching or in contact with the edge of the block?")
    confident: bool


# Default scout/expert model pair — both open-source via OpenRouter.
# Env overrides let an operator set their own preferred small/big pair without code changes.
DEFAULT_SCOUT  = os.getenv("SCOUT_MODEL",  "openrouter/qwen/qwen3-vl-8b-instruct")
DEFAULT_EXPERT = os.getenv("EXPERT_MODEL", "openrouter/qwen/qwen3-vl-235b-a22b-instruct")


@visual_router.reasoner(tags=["visual", "validator"])
async def frame_validator(
    keyframe_b64: str,
    model: str | None = None,
) -> dict:
    """Probe a single frame with 7 objective questions. Used for cross-model
    accuracy benchmarking against ground-truth-labelled frames.

    Returns a flat FrameValidation schema. The questions are designed so
    every answer has a clear right-or-wrong vs human-labelled ground truth.
    """
    result = await visual_router.ai(
        (
            "You are looking at a single frame from the PushT 2D robotic "
            "manipulation task. Answer each of the schema's questions about "
            "what is visible in this frame. Be objective and concrete. If "
            "uncertain about any field, set confident=false."
        ),
        keyframe_b64,
        system=(
            "PushT scene contents: a 2D scene with a small circular robot "
            "end-effector (the 'agent'), a solid T-shaped block being pushed, "
            "and a T-shaped target outline marking where the block should end up. "
            "Use single-word lowercase colour names. Return ONLY the schema fields."
        ),
        schema=FrameValidation,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# smart_frame_validator — confidence-gated SCOUT → EXPERT escalation
#
# Cheap small-model scout produces the first pass. If it signals doubt
# (confident=False OR detects fewer T-shapes than expected → likely missed
# an overlapping shape), the result is escalated to the flagship expert.
# Otherwise the scout's answer is returned.
#
# Either model can be None: if scout is None, expert is called directly;
# if expert is None, scout is the only worker (no escalation possible).
# This makes the reasoner robust to single-model environments.
# ────────────────────────────────────────────────────────────────────────────


@visual_router.reasoner(tags=["visual", "routing", "entry"])
async def smart_frame_validator(
    keyframe_b64: str,
    scout_model: str | None = None,
    expert_model: str | None = None,
    expected_n_t_shapes: int = 2,
) -> dict:
    """Two-tier routing: cheap scout first, escalate to expert on doubt signals.

    Semantics (per user requirement):
      • Both `scout_model` AND `expert_model` provided → routed escalation
      • Only `scout_model` provided                    → scout-only (single-tier)
      • Only `expert_model` provided                   → expert-only (single-tier)
      • Neither provided                               → uses DEFAULT_SCOUT/EXPERT
                                                          (i.e. routed escalation
                                                          with the open-source defaults)

    Escalation triggers (any one fires) when routed:
      • scout reports confident=False
      • scout reports n_t_shapes < expected_n_t_shapes (likely missed an overlap)

    Returns the FrameValidation fields plus routing metadata:
      `_routing` ∈ {"scout_only", "expert_only", "expert_escalated"}
      `_scout_result` (when expert_escalated, for audit)
      `_scout_model`, `_expert_model` (whichever ran)
    """
    # Resolve defaults — only when caller said NOTHING about either model.
    # If caller explicitly passed one but not the other, that's a deliberate
    # single-tier request → respect it.
    if scout_model is None and expert_model is None:
        scout_model = DEFAULT_SCOUT
        expert_model = DEFAULT_EXPERT

    # Single-tier mode: expert-only
    if scout_model is None:
        expert_result = await visual_router.call(
            f"{NODE_ID}.frame_validator",
            keyframe_b64=keyframe_b64,
            model=expert_model,
        )
        return {**expert_result, "_routing": "expert_only", "_expert_model": expert_model}

    # Scout always runs first when present
    scout_result = await visual_router.call(
        f"{NODE_ID}.frame_validator",
        keyframe_b64=keyframe_b64,
        model=scout_model,
    )

    # Single-tier mode: scout-only (no expert wired)
    if expert_model is None:
        return {**scout_result, "_routing": "scout_only", "_scout_model": scout_model}

    # Routed escalation decision
    needs_expert = (
        not scout_result.get("confident", False)
        or scout_result.get("n_t_shapes", 0) < expected_n_t_shapes
    )
    if not needs_expert:
        return {**scout_result, "_routing": "scout_only", "_scout_model": scout_model}

    # Escalate to expert
    expert_result = await visual_router.call(
        f"{NODE_ID}.frame_validator",
        keyframe_b64=keyframe_b64,
        model=expert_model,
    )
    return {
        **expert_result,
        "_routing": "expert_escalated",
        "_scout_result": scout_result,
        "_scout_model": scout_model,
        "_expert_model": expert_model,
    }


# ────────────────────────────────────────────────────────────────────────────
# Layer 2 composer — scene_describer
# Cognitive question: "across all keyframes, what is the canonical scene?"
# Internally: fans out object_detector per keyframe, then synthesises.
# ────────────────────────────────────────────────────────────────────────────


@visual_router.reasoner(tags=["visual", "composer"])
async def scene_describer(
    keyframes_b64: list[str],
    keyframe_indices: list[int],
    task_description: str,
    model: str | None = None,
) -> dict:
    """Compose a canonical scene description from per-keyframe object detections.

    This reasoner fans out `object_detector` over each keyframe in parallel,
    then synthesises the per-frame detections into a single SceneDescription.
    Depth: visual_thread → scene_describer → object_detector (3 layers from
    visual_thread, 4 from annotate_episode).
    """
    # Step 1: parallel object detection over every keyframe.
    detection_dicts = await asyncio.gather(
        *[
            visual_router.call(
                f"{NODE_ID}.object_detector",
                keyframe_b64=kb,
                frame_index=fi,
                task_description=task_description,
                model=model,
            )
            for kb, fi in zip(keyframes_b64, keyframe_indices)
        ]
    )
    detections = [DetectedObjects(**d) for d in detection_dicts]

    # Step 2: convert structured detections to prose for the synthesizer's
    # LLM input (archei rule: strings between LLMs, not JSON serialisation).
    bullets = []
    for det, idx in zip(detections, keyframe_indices):
        confidence_tag = "(confident)" if det.confident else "(LOW confidence)"
        objs = ", ".join(det.objects) if det.objects else "<none>"
        bullets.append(f"- frame {idx} {confidence_tag}: {objs}")
    detections_prose = "Per-keyframe object detections:\n" + "\n".join(bullets)

    # Step 3: synthesise into one SceneDescription.
    result = await visual_router.ai(
        (
            "Synthesise the per-frame detections below into a single canonical "
            "scene description for the whole episode. Deduplicate objects, keep "
            "names lowercase and concrete, and summarise the visible activity "
            "in 1-2 short sentences. If multiple frames disagree heavily about "
            "an object, prefer naming what is consistent and set `confident: "
            "false`.\n\n" + detections_prose
        ),
        system=(
            "You are a careful scene composer for robotics datasets. "
            f"Task context: {task_description}\n\n"
            "Return ONLY the requested schema fields."
        ),
        schema=SceneDescription,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Layer 3 orchestrator — visual_thread
# Parallel: scene_describer || motion_phase_classifier.
# Returns the assembled VisualThreadResult — feeds into temporal segmenter
# and the final composer.
# ────────────────────────────────────────────────────────────────────────────


@visual_router.reasoner(tags=["visual", "thread"])
async def visual_thread(
    keyframes_b64: list[str],
    keyframe_indices: list[int],
    task_description: str,
    model: str | None = None,
) -> dict:
    """Run the visual half of an episode analysis.

    Parallel: scene_describer (which itself fans out object_detector × N) and
    motion_phase_classifier. Both consume the same keyframes; they answer
    different cognitive questions.
    """
    scene_dict, motion_dict = await asyncio.gather(
        visual_router.call(
            f"{NODE_ID}.scene_describer",
            keyframes_b64=keyframes_b64,
            keyframe_indices=keyframe_indices,
            task_description=task_description,
            model=model,
        ),
        visual_router.call(
            f"{NODE_ID}.motion_phase_classifier",
            keyframes_b64=keyframes_b64,
            keyframe_indices=keyframe_indices,
            task_description=task_description,
            model=model,
        ),
    )

    scene = SceneDescription(**scene_dict)
    motion = MotionPhase(**motion_dict)

    result = VisualThreadResult(
        scene=scene,
        motion=motion,
        keyframe_indices=keyframe_indices,
        n_frames_analysed=len(keyframes_b64),
    )
    return result.model_dump()
