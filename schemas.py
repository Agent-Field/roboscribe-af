"""Pydantic schemas for OXE-Narrator reasoner outputs.

Every .ai() call binds to one of these. Discipline (per AgentField skill):
  - Flat schemas, 2-4 attributes max
  - Every .ai() schema includes `confident: bool`
  - Cross-boundary objects re-reconstructed explicitly on the receiver side
    (see choosing-primitives.md — Pydantic does NOT auto-reconstitute over app.call)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────────────
# Layer 1 — leaf reasoner outputs
# ────────────────────────────────────────────────────────────────────────────


class DetectedObjects(BaseModel):
    """`object_detector` output — what objects appear in ONE keyframe."""

    objects: list[str] = Field(
        description=(
            "Short, concrete object names visible in the frame. "
            "Lowercase, no articles. Examples: 'blue t-shaped block', "
            "'green target outline', 'circular gray agent'."
        )
    )
    confident: bool = Field(
        description="Did the visual signal in this frame clearly support these labels?"
    )


class MotionPhase(BaseModel):
    """`motion_phase_classifier` output — dominant motion phase in a sequence."""

    phase: str = Field(
        description=(
            "One of: 'approach' | 'contact' | 'manipulate' | 'retreat' | 'idle'."
        )
    )
    one_sentence_rationale: str = Field(
        description="Evidence-grounded reason for the chosen phase."
    )
    confident: bool


# ────────────────────────────────────────────────────────────────────────────
# Layer 2 — composer outputs
# ────────────────────────────────────────────────────────────────────────────


class SceneDescription(BaseModel):
    """`scene_describer` output — canonical episode-level scene summary."""

    scene_summary: str = Field(
        description=(
            "One paragraph (2-3 sentences) describing what the episode shows: "
            "embodiment, key objects, what the robot appears to be doing."
        )
    )
    canonical_objects: list[str] = Field(
        description=(
            "Deduplicated list of objects that appear across the episode "
            "(lowercase, no articles)."
        )
    )
    motion_summary: str = Field(
        description="One short sentence summarising the dominant motion pattern."
    )
    confident: bool


# ────────────────────────────────────────────────────────────────────────────
# Layer 3 — thread outputs (assembled from composers/leaves)
# ────────────────────────────────────────────────────────────────────────────


class VisualThreadResult(BaseModel):
    """`visual_thread` output — scene + motion + provenance."""

    scene: SceneDescription
    motion: MotionPhase
    keyframe_indices: list[int]
    n_frames_analysed: int


class ActionThreadResult(BaseModel):
    """`action_thread` output — phase derived from the action stream alone.

    The cross-modal verifier downstream will compare this `phase` to the
    `motion.phase` from the visual_thread. Agreement = high confidence;
    disagreement = flag.
    """

    phase: MotionPhase
    trajectory_prose: str = Field(
        description="The prose summary of the trajectory that drove the phase classification."
    )
    n_frames: int


# ────────────────────────────────────────────────────────────────────────────
# Cross-modal verification
# ────────────────────────────────────────────────────────────────────────────


class ModalityConsistency(BaseModel):
    """`cross_modal_verifier` output — does the visual story cohere with action?"""

    consistency_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "0.0 (fully contradictory) to 1.0 (fully consistent). "
            "Same phase label → ~1.0. Adjacent phases (approach↔contact) → ~0.7. "
            "Opposing phases (idle↔manipulate) → ~0.2."
        ),
    )
    rationale: str = Field(
        description="One short sentence explaining the consistency judgment."
    )
    anomaly_flag: str = Field(
        description=(
            "Single short tag describing any anomaly detected (e.g. "
            "'phase_disagreement', 'jerky_with_consistent_phase', 'none')."
        )
    )
    confident: bool


# ────────────────────────────────────────────────────────────────────────────
# Final episode annotation (grain_episode + quality + provenance summary)
# ────────────────────────────────────────────────────────────────────────────


class EpisodeGrain(BaseModel):
    """Top-grained episode-level annotation."""

    goal: str = Field(
        description=(
            "One concise sentence stating what the robot is trying to accomplish "
            "in this episode (the inferred high-level intent)."
        )
    )
    outcome: str = Field(
        description="Single tag: 'successful' | 'partial' | 'failed' | 'inconclusive'."
    )
    confident: bool


class QualitySignals(BaseModel):
    """Quality / consistency signals for downstream filtering."""

    visual_action_consistency: float = Field(ge=0.0, le=1.0)
    anomaly_flags: list[str]
    human_review_recommended: bool


class Provenance(BaseModel):
    """Lightweight provenance summary — full chain is in the VC."""

    reasoners_invoked: list[str]
    keyframe_indices: list[int]
    n_action_frames: int


class GrainSegment(BaseModel):
    """One detected sub-task segment in the temporal hierarchy."""

    start_frame: int
    end_frame: int
    label: str
    phase: str
    objects_involved: list[str]
    confident: bool


class EpisodeAnnotation(BaseModel):
    """`annotate_episode` final output — hierarchical annotation ready to commit."""

    episode_id: int
    annotation_version: str = "v0.1.0"
    embodiment: str
    task_name: str

    # Hierarchical grains: episode → segments
    grain_episode: EpisodeGrain
    grain_segments: list[GrainSegment]

    # Modality outputs
    scene: SceneDescription
    visual_motion: MotionPhase
    action_motion: MotionPhase
    consistency: ModalityConsistency

    quality_signals: QualitySignals
    provenance: Provenance
