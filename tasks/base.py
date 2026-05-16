"""TaskAdapter Protocol — the seam between the reasoner DAG and per-dataset logic.

Every adapter implements this Protocol. The reasoner code calls these methods
without knowing PushT vs Aloha vs anything else.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EpisodeFrames:
    """Materialised frames for one episode, ready to hand to a vision LLM.

    keyframes: list of base64-encoded PNG strings (already shaped for `.ai()`
    positional multimodal use via `f"data:image/png;base64,{b}"`).

    Frame indices index back into the original episode timeline so reasoners
    can cite "at frame 42" → seconds via fps.
    """

    episode_id: int
    fps: float
    n_total_frames: int
    keyframe_indices: list[int]
    keyframes_b64: list[str]


@dataclass(frozen=True)
class EpisodeActions:
    """Action stream for one episode.

    actions: list of action vectors (one per timestep), length == n_total_frames.
    Shape is task-specific (PushT: 2-DoF target pos; Aloha: 14-DoF joints).
    The adapter normalises into a flat list-of-lists so downstream code is uniform.
    """

    episode_id: int
    fps: float
    n_total_frames: int
    actions: list[list[float]]
    state: list[list[float]]  # observed state per frame, may be shorter-DoF than action


class TaskAdapter(Protocol):
    """Per-dataset adapter.

    Names and descriptions feed directly into reasoner prompts — keep them
    natural-language and concrete.
    """

    name: str  # e.g. "pusht"
    source_dataset_id: str  # e.g. "lerobot/pusht"
    description: str  # nat-lang for reasoner prompts
    embodiment: str  # e.g. "2D end-effector", "Franka 7-DoF", "Aloha bimanual 14-DoF"
    relevant_scorers: list[str]  # which quality dimensions apply (currently informational)

    def load_keyframes(self, episode_id: int, n: int = 4) -> EpisodeFrames:
        """Load N keyframes from episode_id, evenly spaced + change-detected.

        Returns frames ready to hand to a vision LLM via .ai() positional args.
        """
        ...

    def load_actions(self, episode_id: int) -> EpisodeActions:
        """Load the full action + state stream for episode_id."""
        ...

    def n_episodes(self) -> int:
        """Total episode count in the source dataset."""
        ...
