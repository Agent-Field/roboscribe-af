"""Composer router — cross-modal verifier + episode goal + final annotation.

Sits above visual_thread and action_thread. Reasoners here consume the
modality outputs and synthesize the final EpisodeAnnotation.

    cross_modal_verifier (leaf .ai)
    episode_goal_synthesizer (leaf .ai)
    annotate_episode (entry — orchestrates everything)
"""

from __future__ import annotations

import asyncio
import logging
import os

from agentfield import AgentRouter

from roboscribe_af.schemas import (
    ActionThreadResult,
    EpisodeAnnotation,
    EpisodeGrain,
    GrainSegment,
    ModalityConsistency,
    Provenance,
    QualitySignals,
    SceneDescription,
    VisualThreadResult,
)

log = logging.getLogger("roboscribe-af.reasoners.composer")

composer_router = AgentRouter(prefix="", tags=["composer"])

NODE_ID = os.getenv("AGENT_NODE_ID", "roboscribe-af")


# ────────────────────────────────────────────────────────────────────────────
# Cross-modal verifier — the heart of the dual-modality cross-check
# ────────────────────────────────────────────────────────────────────────────


@composer_router.reasoner(tags=["composer", "leaf"])
async def cross_modal_verifier(
    visual_phase: str,
    visual_rationale: str,
    action_phase: str,
    action_rationale: str,
    scene_summary: str,
    trajectory_prose: str,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Compare visual-derived and action-derived phase judgments.

    Both threads independently produced a `MotionPhase`. This reasoner judges
    coherence between them and surfaces any anomalies. **Same prompt has access
    to BOTH chains of reasoning** — disagreement here is meaningful signal.
    """
    result = await composer_router.ai(
        (
            "Two independent analyses of the same robotic episode are below.\n\n"
            f"VISUAL analysis (from looking at video keyframes):\n"
            f"  Scene: {scene_summary}\n"
            f"  Motion phase: '{visual_phase}'\n"
            f"  Rationale: {visual_rationale}\n\n"
            f"ACTION analysis (from looking at the trajectory alone):\n"
            f"  Trajectory: {trajectory_prose}\n"
            f"  Motion phase: '{action_phase}'\n"
            f"  Rationale: {action_rationale}\n\n"
            "Are these consistent? Score 0.0 (contradictory) to 1.0 (perfectly "
            "consistent). Same phase = ~1.0; adjacent phases like approach↔contact "
            "= ~0.7; opposing like idle↔manipulate = ~0.2. Set `anomaly_flag` to a "
            "short tag like 'phase_disagreement', 'jerky_with_consistent_phase', "
            "or 'none'."
        ),
        system=(
            "You are a strict cross-modal consistency checker for robotics "
            f"annotation pipelines. Task context: {task_description}\n\n"
            "Return ONLY the requested schema fields."
        ),
        schema=ModalityConsistency,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Episode goal synthesizer — produces grain_episode (the top-grained label)
# ────────────────────────────────────────────────────────────────────────────


@composer_router.reasoner(tags=["composer", "leaf"])
async def episode_goal_synthesizer(
    scene_summary: str,
    canonical_objects: list[str],
    visual_phase: str,
    action_phase: str,
    trajectory_prose: str,
    task_description: str,
    model: str | None = None,
) -> dict:
    """Synthesize a one-sentence goal + outcome tag for the episode."""
    objs = ", ".join(canonical_objects) if canonical_objects else "<none detected>"
    result = await composer_router.ai(
        (
            "Synthesize a top-grained label for this robotic episode.\n\n"
            f"Scene: {scene_summary}\n"
            f"Canonical objects: {objs}\n"
            f"Visual motion phase: '{visual_phase}'\n"
            f"Action motion phase: '{action_phase}'\n"
            f"Trajectory: {trajectory_prose}\n\n"
            "Write: (1) a single concise sentence stating what the robot is "
            "trying to accomplish — the inferred high-level intent. (2) An "
            "outcome tag: 'successful', 'partial', 'failed', or 'inconclusive'."
        ),
        system=(
            "You are a careful annotator producing training-grade labels for "
            f"robotics datasets. Task context: {task_description}\n\n"
            "Return ONLY the requested schema fields."
        ),
        schema=EpisodeGrain,
        model=model,
    )
    return result.model_dump()


# ────────────────────────────────────────────────────────────────────────────
# Entry reasoner — annotate_episode
# Orchestrates everything: loaders, threads, verifier, synthesizer.
# Full parallelism wherever independent.
# ────────────────────────────────────────────────────────────────────────────


REASONERS_TO_TRACK = [
    "load_episode_keyframes",
    "load_episode_actions",
    "object_detector",
    "motion_phase_classifier",
    "scene_describer",
    "visual_thread",
    "velocity_profile",
    "trajectory_phase_classifier",
    "ee_trajectory_phaser",
    "action_thread",
    "cross_modal_verifier",
    "episode_goal_synthesizer",
]


@composer_router.reasoner(tags=["composer", "entry"])
async def annotate_episode(
    episode_id: int,
    n_keyframes: int = 4,
    task: str | None = None,
    model: str | None = None,
    commit_to_deeplake: bool = True,
    annotation_branch: str = "roboscribe-v1",
) -> dict:
    """Produce a hierarchical annotation for one robotics episode.

    Topology:
      1. Parallel load: keyframes (skill) || actions (skill)
      2. Parallel threads: visual_thread || action_thread
         - visual_thread fans out object_detector × N + motion_phase_classifier
         - action_thread chains velocity_profile (skill) → trajectory_phase_classifier
      3. Parallel synthesis: cross_modal_verifier || episode_goal_synthesizer
      4. Assemble EpisodeAnnotation

    Per-request `model` overrides AIConfig default and threads through every
    downstream .ai() call.
    """
    # ── Phase 1: parallel data loading ────────────────────────────────────
    keyframes_dict, actions_dict = await asyncio.gather(
        composer_router.call(
            f"{NODE_ID}.load_episode_keyframes",
            episode_id=episode_id,
            n_keyframes=n_keyframes,
            task=task,
        ),
        composer_router.call(
            f"{NODE_ID}.load_episode_actions",
            episode_id=episode_id,
            task=task,
        ),
    )
    keyframes_b64 = keyframes_dict["keyframes_b64"]
    keyframe_indices = keyframes_dict["keyframe_indices"]
    task_description = keyframes_dict["task_description"]
    embodiment = keyframes_dict["embodiment"]
    task_name = keyframes_dict["task_name"]

    states = actions_dict["state"]
    fps = actions_dict["fps"]

    # ── Phase 2: parallel modality threads ────────────────────────────────
    visual_dict, action_dict = await asyncio.gather(
        composer_router.call(
            f"{NODE_ID}.visual_thread",
            keyframes_b64=keyframes_b64,
            keyframe_indices=keyframe_indices,
            task_description=task_description,
            model=model,
        ),
        composer_router.call(
            f"{NODE_ID}.action_thread",
            states=states,
            fps=fps,
            task_description=task_description,
            model=model,
        ),
    )
    visual = VisualThreadResult(**visual_dict)
    action = ActionThreadResult(**action_dict)

    # ── Phase 2.5: temporal segmenter (depends on both threads) ────────────
    # This fans out per-segment narrators internally via asyncio.gather, so
    # the number of segments is determined at runtime by boundary_judger.
    segmenter_dict = await composer_router.call(
        f"{NODE_ID}.temporal_segmenter",
        states=states,
        keyframe_indices=keyframe_indices,
        fps=fps,
        visual_scene_summary=visual.scene.scene_summary,
        visual_motion_summary=visual.scene.motion_summary,
        canonical_objects=visual.scene.canonical_objects,
        visual_phase=visual.motion.phase,
        action_phase=action.phase.phase,
        action_trajectory_prose=action.trajectory_prose,
        task_description=task_description,
        model=model,
    )

    # ── Phase 3: parallel synthesis (verifier + goal synthesizer) ─────────
    consistency_dict, grain_dict = await asyncio.gather(
        composer_router.call(
            f"{NODE_ID}.cross_modal_verifier",
            visual_phase=visual.motion.phase,
            visual_rationale=visual.motion.one_sentence_rationale,
            action_phase=action.phase.phase,
            action_rationale=action.phase.one_sentence_rationale,
            scene_summary=visual.scene.scene_summary,
            trajectory_prose=action.trajectory_prose,
            task_description=task_description,
            model=model,
        ),
        composer_router.call(
            f"{NODE_ID}.episode_goal_synthesizer",
            scene_summary=visual.scene.scene_summary,
            canonical_objects=visual.scene.canonical_objects,
            visual_phase=visual.motion.phase,
            action_phase=action.phase.phase,
            trajectory_prose=action.trajectory_prose,
            task_description=task_description,
            model=model,
        ),
    )
    consistency = ModalityConsistency(**consistency_dict)
    grain = EpisodeGrain(**grain_dict)

    # ── Phase 4: assemble final annotation ────────────────────────────────
    anomaly_flags = [consistency.anomaly_flag] if consistency.anomaly_flag != "none" else []
    if not visual.scene.confident:
        anomaly_flags.append("low_visual_confidence")
    if not action.phase.confident:
        anomaly_flags.append("low_action_confidence")
    human_review = (
        consistency.consistency_score < 0.6
        or not grain.confident
        or not visual.scene.confident
    )

    grain_segments = [
        GrainSegment(**seg) for seg in segmenter_dict.get("segments", [])
    ]

    annotation = EpisodeAnnotation(
        episode_id=episode_id,
        embodiment=embodiment,
        task_name=task_name,
        grain_episode=grain,
        grain_segments=grain_segments,
        scene=visual.scene,
        visual_motion=visual.motion,
        action_motion=action.phase,
        consistency=consistency,
        quality_signals=QualitySignals(
            visual_action_consistency=consistency.consistency_score,
            anomaly_flags=anomaly_flags,
            human_review_recommended=human_review,
        ),
        provenance=Provenance(
            reasoners_invoked=REASONERS_TO_TRACK
            + ["visual_action_aligner", "boundary_judger", "temporal_segmenter", "segment_narrator"],
            keyframe_indices=keyframe_indices,
            n_action_frames=len(states),
        ),
    )
    annotation_dict = annotation.model_dump()

    # ── Phase 5: optional — embed scene + commit to Deep Lake branch ──────
    # Two skill calls appear in the DAG:
    #   embed_text(scene_summary) → vector
    #   commit_annotation_to_branch(annotation, scene_embedding=vector)
    # Failures are non-fatal — annotation is still returned in the response.
    if commit_to_deeplake:
        try:
            embed_dict = await composer_router.call(
                f"{NODE_ID}.embed_text",
                text=visual.scene.scene_summary,
            )
            scene_vec = embed_dict.get("embedding")
            commit_dict = await composer_router.call(
                f"{NODE_ID}.commit_annotation_to_branch",
                annotation=annotation_dict,
                branch_name=annotation_branch,
                scene_embedding=scene_vec,
            )
            annotation_dict["_deeplake_commit"] = commit_dict
            annotation_dict["_embedding_model"] = embed_dict.get("model")
        except Exception as e:  # noqa: BLE001
            log.warning("Deep Lake commit failed: %s", e)
            annotation_dict["_deeplake_commit_error"] = str(e)[:200]

    return annotation_dict
