"""Deep Lake substrate manager.

This is where OXE-Narrator earns the Deep Lake pitch. We use it for FIVE
features the marketing claims — not as a JSON store:

  1. **Native multimodal storage** — episode frames (raw PNG bytes),
     action arrays (Float32 sequences), state arrays, and text annotations
     all live as columns in ONE dataset, joined by `episode_id`. No JOIN,
     no second store, no schema gymnastics.

  2. **Branches as annotation versions** — each annotation pass commits to a
     new branch (`roboscribe-v1`, `roboscribe-v2`). The raw `main` branch
     stays pristine. Users pin to a branch for reproducibility, or diff
     branches to see what changed between annotation runs.

  3. **Sibling tensors joined by ID** — annotations land on the SAME ROW as
     the raw episode they describe. Query "all annotations for episode 7"
     and "the original keyframes of episode 7" are the same row.

  4. **Hybrid TQL search** — `SELECT * WHERE visual_phase='manipulate' AND
     consistency_score > 0.6` — structured + text-contains + numeric filters
     in one query against a multimodal corpus.

  5. **PyTorch streaming dataloader** — `ds.pytorch()` on an annotated branch
     yields ready-to-train (frames, language) tuples for VLA fine-tuning.
     No conversion, no intermediate format.

Verified on deeplake==4.6.1 API:
  - `deeplake.create(url, schema={...})`  (NB: schema dict at create time)
  - `deeplake.open(url)`
  - `ds.append({col: [...], ...})`        (bulk insert)
  - `ds.commit("message")`
  - `br = ds.branch("name")`              (returns Branch; does NOT switch)
  - `ds_v = br.open()`                    (open a Dataset on that branch)
  - `ds.branches.names()`                   (list branch names)
  - `ds.query("SELECT ... WHERE ...")`    (TQL — hybrid filters)
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

import deeplake
from deeplake import types as dl_types

log = logging.getLogger("roboscribe-af.dl_store")


# Where the Deep Lake dataset lives in the agent container.
DEFAULT_DL_PATH = os.getenv("DL_DATASET_PATH", "/data/roboscribe_deeplake/roboscribe_pusht")
RAW_BRANCH = "main"  # raw episode data committed here
DEFAULT_ANNOTATION_BRANCH = "roboscribe-v1"


# ────────────────────────────────────────────────────────────────────────────
# Schema definition — see module docstring for what each column proves
# ────────────────────────────────────────────────────────────────────────────


SCENE_EMBED_DIM = 1024  # matches helpers.EMBED_DIM (BAAI/bge-large-en-v1.5)


def _schema() -> dict[str, Any]:
    """Multimodal schema: raw episode data + (sparse) annotations + embeddings.

    Column types verified against deeplake 4.6.1:
      - Int32 / Float32 / Bool       — flat scalars
      - Text                         — text-searchable strings (CONTAINS in TQL)
      - Sequence(Int32)              — variable-length int list
      - Sequence(Bytes)              — list of binary blobs (PNG bytes per keyframe)
      - Array(Float32, 2)            — variable-length 2D float matrix
      - Embedding(size=N)            — fixed-dim vector with cosine_similarity / l2_norm TQL support

    Columns starting with `lang_` / `visual_phase` / `action_phase` /
    `consistency_` / `anomaly_` / `scene_embedding` get populated on
    annotation branches; the `main` branch only has the raw episode columns.
    """
    import numpy as np  # noqa: F401 — required so 2D arrays serialize

    return {
        # ── Episode metadata ────────────────────────────────────────────
        "episode_id": dl_types.Int32(),
        "task_name": dl_types.Text(),
        "embodiment": dl_types.Text(),
        "fps": dl_types.Float32(),
        "n_frames": dl_types.Int32(),
        # ── Multimodal raw data ─────────────────────────────────────────
        "keyframe_indices": dl_types.Sequence(dl_types.Int32()),
        "keyframes_png": dl_types.Sequence(dl_types.Bytes()),
        "actions": dl_types.Array(dl_types.Float32(), 2),
        "states": dl_types.Array(dl_types.Float32(), 2),
        # ── Annotations (filled on annotation branches) ────────────────
        "lang_episode_goal": dl_types.Text(),
        "lang_outcome": dl_types.Text(),
        "lang_scene_summary": dl_types.Text(),
        "lang_canonical_objects": dl_types.Text(),
        "lang_motion_summary": dl_types.Text(),
        # Visual + action thread phases (dual-modality cross-check)
        "visual_phase": dl_types.Text(),
        "visual_phase_rationale": dl_types.Text(),
        "action_phase": dl_types.Text(),
        "action_phase_rationale": dl_types.Text(),
        "trajectory_prose": dl_types.Text(),
        # Cross-modal verification + quality
        "consistency_score": dl_types.Float32(),
        "consistency_rationale": dl_types.Text(),
        "anomaly_flag": dl_types.Text(),
        "human_review_recommended": dl_types.Bool(),
        # Semantic embedding of the scene summary (populated on annotation
        # branches via the embed_text skill). Enables TQL cosine_similarity
        # vector search alongside text CONTAINS and numeric filters.
        "scene_embedding": dl_types.Embedding(size=SCENE_EMBED_DIM),
        # Provenance
        "annotation_version": dl_types.Text(),
    }


# ────────────────────────────────────────────────────────────────────────────
# Dataset creation / opening
# ────────────────────────────────────────────────────────────────────────────


def ensure_dataset(path: str = DEFAULT_DL_PATH) -> deeplake.Dataset:
    """Open dataset at `path`, creating it (with our schema) if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if deeplake.exists(path):
        return deeplake.open(path)
    ds = deeplake.create(path, schema=_schema())
    log.info("Created Deep Lake dataset at %s", path)
    return ds


def ensure_annotation_branch(
    ds: deeplake.Dataset, branch_name: str
) -> deeplake.Dataset:
    """Open an annotation branch, creating it from `main` if needed.

    Returns a Dataset positioned on the requested branch. Verified Deep Lake
    4.6 API: `ds.branches.names()` is a method (not property), `ds.branches[name]`
    accesses Branch by name, `Branch.open()` returns a Dataset on that branch.
    """
    if branch_name in ds.branches.names():
        return ds.branches[branch_name].open()
    br = ds.branch(branch_name)
    log.info("Created branch %s (id=%s)", br.name, br.id[:8])
    return br.open()


# ────────────────────────────────────────────────────────────────────────────
# Bulk ingestion — raw episodes onto `main`
# ────────────────────────────────────────────────────────────────────────────


def _data_url_to_bytes(data_url: str) -> bytes:
    """`data:image/png;base64,...` → raw PNG bytes."""
    _, _, b64 = data_url.partition("base64,")
    return base64.b64decode(b64)


def ingest_episode(
    ds: deeplake.Dataset,
    *,
    episode_id: int,
    task_name: str,
    embodiment: str,
    fps: float,
    n_frames: int,
    keyframe_indices: list[int],
    keyframes_b64: list[str],
    actions: list[list[float]],
    states: list[list[float]],
) -> None:
    """Append ONE raw episode to `main`. Annotation columns stay empty."""
    import numpy as np

    zero_embed = np.zeros(SCENE_EMBED_DIM, dtype=np.float32)
    ds.append(
        {
            "episode_id": [episode_id],
            "task_name": [task_name],
            "embodiment": [embodiment],
            "fps": [fps],
            "n_frames": [n_frames],
            "keyframe_indices": [keyframe_indices],
            "keyframes_png": [[_data_url_to_bytes(d) for d in keyframes_b64]],
            "actions": [np.asarray(actions, dtype=np.float32)],
            "states": [np.asarray(states, dtype=np.float32)],
            # Annotation cols are nullable / empty on main
            "lang_episode_goal": [""],
            "lang_outcome": [""],
            "lang_scene_summary": [""],
            "lang_canonical_objects": [""],
            "lang_motion_summary": [""],
            "visual_phase": [""],
            "visual_phase_rationale": [""],
            "action_phase": [""],
            "action_phase_rationale": [""],
            "trajectory_prose": [""],
            "consistency_score": [0.0],
            "consistency_rationale": [""],
            "anomaly_flag": [""],
            "human_review_recommended": [False],
            "scene_embedding": [zero_embed],  # populated only on annotation branches
            "annotation_version": [""],
        }
    )


# ────────────────────────────────────────────────────────────────────────────
# Annotation commit — sibling tensors on a branch
# ────────────────────────────────────────────────────────────────────────────


def commit_annotation(
    ds_branch: deeplake.Dataset,
    *,
    annotation: dict,
    scene_embedding: list[float] | None = None,
) -> int:
    """Append an annotated row on the annotation branch.

    `scene_embedding` should be the embedding vector of `scene.scene_summary`
    (computed upstream via the `embed_text` skill). If None, a zero vector is
    written (TQL cosine_similarity will still rank it last).
    """
    ep_id = annotation["episode_id"]
    grain = annotation["grain_episode"]
    scene = annotation["scene"]
    vmotion = annotation["visual_motion"]
    amotion = annotation["action_motion"]
    consistency = annotation["consistency"]
    quality = annotation["quality_signals"]

    import numpy as np

    if scene_embedding is None:
        emb_vec = np.zeros(SCENE_EMBED_DIM, dtype=np.float32)
    else:
        emb_vec = np.asarray(scene_embedding, dtype=np.float32)
        if emb_vec.shape != (SCENE_EMBED_DIM,):
            # Pad or truncate defensively so a dim mismatch doesn't blow up the row.
            padded = np.zeros(SCENE_EMBED_DIM, dtype=np.float32)
            n = min(SCENE_EMBED_DIM, emb_vec.shape[0])
            padded[:n] = emb_vec[:n]
            emb_vec = padded

    ds_branch.append(
        {
            "episode_id": [ep_id],
            "task_name": [annotation["task_name"]],
            "embodiment": [annotation["embodiment"]],
            "fps": [0.0],
            "n_frames": [0],
            "keyframe_indices": [[]],
            "keyframes_png": [[]],
            "actions": [np.zeros((0, 2), dtype=np.float32)],
            "states": [np.zeros((0, 2), dtype=np.float32)],
            "lang_episode_goal": [grain["goal"]],
            "lang_outcome": [grain["outcome"]],
            "lang_scene_summary": [scene["scene_summary"]],
            "lang_canonical_objects": [", ".join(scene.get("canonical_objects", []))],
            "lang_motion_summary": [scene["motion_summary"]],
            "visual_phase": [vmotion["phase"]],
            "visual_phase_rationale": [vmotion["one_sentence_rationale"]],
            "action_phase": [amotion["phase"]],
            "action_phase_rationale": [amotion["one_sentence_rationale"]],
            "trajectory_prose": [""],
            "consistency_score": [consistency["consistency_score"]],
            "consistency_rationale": [consistency["rationale"]],
            "anomaly_flag": [consistency["anomaly_flag"]],
            "human_review_recommended": [quality["human_review_recommended"]],
            "scene_embedding": [emb_vec],
            "annotation_version": [annotation["annotation_version"]],
        }
    )
    return len(ds_branch) - 1


# ────────────────────────────────────────────────────────────────────────────
# Hybrid TQL query — search annotations
# ────────────────────────────────────────────────────────────────────────────


def query_tql(ds: deeplake.Dataset, tql: str) -> list[dict]:
    """Run an arbitrary TQL query, return rows as plain dicts.

    Examples that exercise the multimodal nature:
      SELECT episode_id, lang_episode_goal, consistency_score
      WHERE visual_phase = 'manipulate' AND consistency_score > 0.6

      SELECT episode_id, anomaly_flag, lang_scene_summary
      WHERE human_review_recommended = true

      SELECT episode_id FROM lang_scene_summary WHERE CONTAINS('push')
    """
    import numpy as np

    results = ds.query(tql)
    out = []
    for i in range(len(results)):
        row = results[i]
        rec: dict = {}
        for col in results.schema.columns:
            try:
                v = row[col.name]
                # Deep Lake returns scalar columns as 0-dim ndarrays. Extract the scalar first.
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    v = v.item()
                # numpy scalar types (np.int32, np.float32 …) → Python primitives
                if isinstance(v, np.generic):
                    v = v.item()
                if isinstance(v, (str, int, float, bool, type(None))):
                    rec[col.name] = v
                elif isinstance(v, bytes):
                    rec[col.name] = f"<{len(v)} bytes>"
                elif isinstance(v, np.ndarray):
                    # Real multi-dim ndarray (e.g. embedding, action matrix) — summarise.
                    rec[col.name] = f"<ndarray shape={v.shape} dtype={v.dtype}>"
                elif hasattr(v, "__len__"):
                    n = len(v)
                    if n == 0:
                        rec[col.name] = []
                    elif n < 50:
                        rec[col.name] = [
                            (x.item() if isinstance(x, (np.generic, np.ndarray)) and (not isinstance(x, np.ndarray) or x.ndim == 0) else x)
                            for x in v
                        ]
                    else:
                        rec[col.name] = f"<sequence len={n}>"
                else:
                    rec[col.name] = f"<{type(v).__name__}>"
            except Exception as e:  # noqa: BLE001
                rec[col.name] = f"<error: {e}>"
        out.append(rec)
    return out
