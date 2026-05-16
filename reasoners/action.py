"""Action thread reasoners — the second modality of the cross-verification.

    action_thread (orchestrator)
    └── ee_trajectory_phaser (composer)
        ├── velocity_profile  (skill — deterministic numeric analysis)
        └── trajectory_phase_classifier (.ai — prose summary → MotionPhase)

Depth from entry: annotate_episode → action_thread → ee_trajectory_phaser →
trajectory_phase_classifier = 4 layers.

`trajectory_phase_classifier` emits a `MotionPhase` schema — the SAME schema
as `visual.motion_phase_classifier`. This is intentional: the cross-modal
verifier downstream compares the two to validate consistency.
"""

from __future__ import annotations

import logging
import os

from agentfield import AgentRouter

from schemas import ActionThreadResult, MotionPhase

log = logging.getLogger("roboscribe-af.reasoners.action")

action_router = AgentRouter(prefix="", tags=["action"])

NODE_ID = os.getenv("AGENT_NODE_ID", "roboscribe-af")


# ────────────────────────────────────────────────────────────────────────────
# Layer 1 leaf — trajectory_phase_classifier
# Cognitive question: "from the action trajectory alone, what motion phase?"
# Input is the PROSE summary produced by the velocity_profile skill upstream.
# ────────────────────────────────────────────────────────────────────────────


@action_router.reasoner(tags=["action", "leaf"])
async def trajectory_phase_classifier(
    trajectory_prose: str,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Classify the dominant motion phase from a prose trajectory summary."""
    result = await action_router.ai(
        (
            "Based on the trajectory summary below, classify the dominant "
            "motion phase of the episode: 'approach' (robot reaches toward "
            "object), 'contact' (engages object), 'manipulate' (object being "
            "repositioned), 'retreat' (disengages), or 'idle' (no meaningful "
            "motion). Pick ONE.\n\n"
            f"Trajectory summary:\n{trajectory_prose}"
        ),
        system=(
            "You are a robotics trajectory analyst. You make judgments from "
            "kinematic summaries alone — without visual input. "
            f"Task context: {task_description}\n\n"
            "Return ONLY the requested schema fields."
        ),
        schema=MotionPhase,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Layer 2 composer — ee_trajectory_phaser
# Calls velocity_profile skill, renders prose, hands to phase classifier.
# ────────────────────────────────────────────────────────────────────────────


@action_router.reasoner(tags=["action", "composer"])
async def ee_trajectory_phaser(
    states: list[list[float]],
    fps: float,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Compose: skill computes trajectory metrics → prose; classifier judges phase."""
    # Step 1: deterministic velocity / jerk / pause analysis.
    vp = await action_router.call(
        f"{NODE_ID}.velocity_profile",
        states=states,
        fps=fps,
    )
    prose = vp["prose_summary"]

    # Step 2: classify phase from prose summary.
    phase_dict = await action_router.call(
        f"{NODE_ID}.trajectory_phase_classifier",
        trajectory_prose=prose,
        task_description=task_description,
        model=model,
    )

    return {
        "phase": phase_dict,
        "trajectory_prose": prose,
    }


# ────────────────────────────────────────────────────────────────────────────
# Layer 3 orchestrator — action_thread
# For PushT (no gripper, no force sensing), just wraps ee_trajectory_phaser.
# For Aloha (bimanual w/ gripper events) the orchestrator would gather()
# additional sub-reasoners here.
# ────────────────────────────────────────────────────────────────────────────


@action_router.reasoner(tags=["action", "thread"])
async def action_thread(
    states: list[list[float]],
    fps: float,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Action-side analysis of an episode. Mirrors visual_thread's shape.

    For PushT v1 there is only one sub-composer (no gripper events to detect).
    For tasks with grippers (Aloha) this orchestrator gathers additional
    composers in parallel via asyncio.gather.
    """
    composer_dict = await action_router.call(
        f"{NODE_ID}.ee_trajectory_phaser",
        states=states,
        fps=fps,
        task_description=task_description,
        model=model,
    )

    phase = MotionPhase(**composer_dict["phase"])
    result = ActionThreadResult(
        phase=phase,
        trajectory_prose=composer_dict["trajectory_prose"],
        n_frames=len(states),
    )
    return result.model_dump()
