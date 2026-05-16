"""Deep Lake skills + reasoners — the substrate integration layer.

This router exposes Deep Lake operations to the rest of the cognitive graph:
  - ingest_corpus_to_deeplake  (skill)  — bulk load synthetic episodes
  - commit_annotation_to_branch (skill) — write annotation to a versioned branch
  - search_episodes            (skill)  — TQL hybrid query over annotations
  - corpus_stats               (skill)  — quick "what's in the lake?" summary

Skills (not reasoners) because each is deterministic — no LLM judgment.
Each is callable via app.call so it shows up in the workflow DAG when the
annotation pipeline writes to the lake.
"""

from __future__ import annotations

import logging
import os

from agentfield import AgentRouter

from roboscribe_af.dl_store import (
    DEFAULT_ANNOTATION_BRANCH,
    DEFAULT_DL_PATH,
    commit_annotation,
    ensure_annotation_branch,
    ensure_dataset,
    ingest_episode,
    query_tql,
)

log = logging.getLogger("roboscribe-af.reasoners.dl")

dl_router = AgentRouter(prefix="", tags=["deeplake"])


@dl_router.skill(tags=["deeplake", "ingest"])
async def ingest_corpus_to_deeplake(
    n_episodes: int = 10,
    n_keyframes: int = 4,
    task: str | None = None,
    dl_path: str | None = None,
) -> dict:
    """Ingest N synthetic episodes into the Deep Lake substrate on `main`.

    Idempotent at the row level (re-ingesting will append duplicates — for v1
    we assume a clean dataset; v1.5 will add dedupe by episode_id).
    """
    from roboscribe_af.tasks import get_adapter  # lazy

    adapter = get_adapter(task)
    ds = ensure_dataset(dl_path or DEFAULT_DL_PATH)
    written = []
    for ep in range(n_episodes):
        frames = adapter.load_keyframes(ep, n=n_keyframes)
        actions = adapter.load_actions(ep)
        ingest_episode(
            ds,
            episode_id=ep,
            task_name=adapter.name,
            embodiment=adapter.embodiment,
            fps=frames.fps,
            n_frames=frames.n_total_frames,
            keyframe_indices=frames.keyframe_indices,
            keyframes_b64=frames.keyframes_b64,
            actions=actions.actions,
            states=actions.state,
        )
        written.append(ep)

    ds.commit(f"ingested {n_episodes} synthetic {adapter.name} episodes")
    return {
        "dataset_path": dl_path or DEFAULT_DL_PATH,
        "episodes_ingested": written,
        "n_rows": len(ds),
        "current_branch": str(ds.current_branch),
        "all_branches": list(ds.branches.names()),
    }


@dl_router.skill(tags=["deeplake", "annotation"])
async def commit_annotation_to_branch(
    annotation: dict,
    branch_name: str = DEFAULT_ANNOTATION_BRANCH,
    dl_path: str | None = None,
    scene_embedding: list[float] | None = None,
) -> dict:
    """Write an EpisodeAnnotation onto a Deep Lake branch.

    Creates the branch from `main` if it doesn't exist. The annotation lands
    as sibling tensors joined to the raw episode by `episode_id`. If
    `scene_embedding` is supplied it goes into the vector column; otherwise
    a zero vector is written.
    """
    ds = ensure_dataset(dl_path or DEFAULT_DL_PATH)
    ds_br = ensure_annotation_branch(ds, branch_name)
    row_idx = commit_annotation(
        ds_br,
        annotation=annotation,
        scene_embedding=scene_embedding,
    )
    ds_br.commit(
        f"annotated episode_id={annotation['episode_id']} v={annotation['annotation_version']}"
    )
    return {
        "dataset_path": dl_path or DEFAULT_DL_PATH,
        "branch": branch_name,
        "row_index": row_idx,
        "episode_id": annotation["episode_id"],
        "n_rows_on_branch": len(ds_br),
        "embedding_attached": scene_embedding is not None,
    }


@dl_router.reasoner(tags=["deeplake", "search"])
async def vector_search_episodes(
    query: str,
    branch_name: str = DEFAULT_ANNOTATION_BRANCH,
    top_k: int = 5,
    min_similarity: float = 0.0,
    dl_path: str | None = None,
    model: str | None = None,
) -> dict:
    """Semantic search: embed the query string and rank episodes by
    cosine similarity to their `scene_embedding`.

    Hybrid demo — combine with structured/text filters via TQL:
      SELECT ... WHERE cosine_similarity(scene_embedding, ARRAY[...]) > 0.5
             AND consistency_score > 0.6
             AND CONTAINS(lang_scene_summary, 'block')
    """
    NODE_ID = os.getenv("AGENT_NODE_ID", "roboscribe-af")

    # Step 1: embed the query
    embed_result = await dl_router.call(
        f"{NODE_ID}.embed_text",
        text=query,
        model=model,
    )
    qvec = embed_result["embedding"]
    qvec_str = "ARRAY[" + ", ".join(f"{x:.6f}" for x in qvec) + "]"

    # Step 2: TQL — rank by cosine similarity, threshold optional
    tql = (
        f"SELECT episode_id, lang_episode_goal, lang_scene_summary, "
        f"consistency_score, anomaly_flag "
        f"WHERE cosine_similarity(scene_embedding, {qvec_str}) > {min_similarity:.4f} "
        f"ORDER BY cosine_similarity(scene_embedding, {qvec_str}) DESC "
        f"LIMIT {top_k}"
    )

    search_result = await dl_router.call(
        f"{NODE_ID}.search_episodes",
        tql=tql,
        branch_name=branch_name,
        dl_path=dl_path,
    )
    return {
        "query": query,
        "branch": branch_name,
        "embedding_model": embed_result["model"],
        "embedding_dim": embed_result["dim"],
        "top_k": top_k,
        "n_rows": search_result.get("n_rows", 0),
        "rows": search_result.get("rows", []),
    }


@dl_router.skill(tags=["deeplake", "search"])
async def search_episodes(
    tql: str,
    branch_name: str = DEFAULT_ANNOTATION_BRANCH,
    dl_path: str | None = None,
    limit: int = 50,
) -> dict:
    """Run a TQL query against an annotation branch.

    TQL supports structured filters (numeric, equality), text predicates
    (CONTAINS), and ORDER BY / LIMIT — all in one query over the multimodal
    columns. Examples:

      SELECT episode_id, lang_episode_goal, consistency_score
      WHERE visual_phase = 'manipulate' AND consistency_score > 0.5

      SELECT episode_id, anomaly_flag
      WHERE human_review_recommended = true

      SELECT episode_id, lang_scene_summary
      WHERE CONTAINS(lang_scene_summary, 'push')
    """
    ds = ensure_dataset(dl_path or DEFAULT_DL_PATH)
    if branch_name in ds.branches.names():
        ds = ensure_annotation_branch(ds, branch_name)
    rows = query_tql(ds, tql)
    return {
        "tql": tql,
        "branch": str(ds.current_branch),
        "n_rows": len(rows),
        "rows": rows[:limit],
    }


@dl_router.skill(tags=["deeplake", "stats"])
async def corpus_stats(
    dl_path: str | None = None,
) -> dict:
    """High-level "what's in the lake?" summary across all branches."""
    resolved_path = dl_path or DEFAULT_DL_PATH
    ds = ensure_dataset(resolved_path)
    branch_info = []
    for name in ds.branches.names():
        try:
            ds_b = ds.branches[name].open()
            branch_info.append({"name": name, "n_rows": len(ds_b)})
        except Exception as e:  # noqa: BLE001
            branch_info.append({"name": name, "error": str(e)[:80]})
    columns = [c.name for c in ds.schema.columns]
    return {
        "dataset_path": resolved_path,
        "n_columns": len(columns),
        "columns": columns,
        "branches": branch_info,
    }
