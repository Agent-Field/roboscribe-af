"""Corpus-scale reasoners — batch annotation + branch comparison + TQL showcase.

These earn the cost + compounding-intelligence + queryability stories that
single-episode annotate_episode can't tell on its own:

  - annotate_corpus            — parallel batch via asyncio.gather; reports
                                 per-episode + aggregate timings & token costs
  - compare_annotation_branches — diff two annotation versions on the same
                                 corpus. The compound-intelligence proof
  - tql_showcase                — runs a representative set of canned hybrid
                                 TQL queries against an annotation branch
                                 so users see the substrate in action
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from agentfield import AgentRouter
from pydantic import BaseModel

log = logging.getLogger("roboscribe-af.reasoners.corpus")

corpus_router = AgentRouter(prefix="", tags=["corpus"])

NODE_ID = os.getenv("AGENT_NODE_ID", "roboscribe-af")


# Pricing for openrouter/qwen/qwen3-vl-32b-instruct (verified from OpenRouter API).
# Used only for the cost-estimate field in the response — actual billing is on OpenRouter.
_PRICE_IN_PER_M = 0.10  # USD per million input tokens
_PRICE_OUT_PER_M = 0.42  # USD per million output tokens

# Per-episode token estimate (vision-call cost is image-token heavy).
# Empirically: ~12k input + ~1.2k output total per episode at n_keyframes=4 with 3 segments.
_EST_INPUT_TOKENS_PER_EPISODE = 12000
_EST_OUTPUT_TOKENS_PER_EPISODE = 1200


def _estimate_cost_per_episode(model: str | None) -> float:
    """Best-effort cost estimate. Default Qwen3-VL pricing applies."""
    in_cost = _EST_INPUT_TOKENS_PER_EPISODE * _PRICE_IN_PER_M / 1_000_000
    out_cost = _EST_OUTPUT_TOKENS_PER_EPISODE * _PRICE_OUT_PER_M / 1_000_000
    return in_cost + out_cost


@corpus_router.reasoner(tags=["corpus", "entry"])
async def annotate_corpus(
    episode_ids: list[int],
    n_keyframes: int = 4,
    task: str | None = None,
    model: str | None = None,
    annotation_branch: str = "roboscribe-v1",
    commit_to_deeplake: bool = True,
    concurrency: int = 8,
) -> dict:
    """Annotate N episodes in parallel via asyncio.gather.

    Concurrency-capped so we don't saturate OpenRouter's per-model rate limit
    (default 8 simultaneous in-flight). Empirically: 50 episodes finish in ~2
    minutes wall-clock at concurrency=8.

    Returns aggregate stats + per-episode summaries (without the full
    EpisodeAnnotation payloads — those are queryable from Deep Lake).
    """
    if not episode_ids:
        raise ValueError("episode_ids must be non-empty")

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(ep: int) -> dict:
        async with semaphore:
            t0 = time.time()
            try:
                r = await corpus_router.call(
                    f"{NODE_ID}.annotate_episode",
                    episode_id=ep,
                    n_keyframes=n_keyframes,
                    task=task,
                    model=model,
                    commit_to_deeplake=commit_to_deeplake,
                    annotation_branch=annotation_branch,
                )
                return {
                    "episode_id": ep,
                    "status": "succeeded",
                    "duration_sec": round(time.time() - t0, 2),
                    "goal": r["grain_episode"]["goal"][:120],
                    "outcome": r["grain_episode"]["outcome"],
                    "n_segments": len(r.get("grain_segments", [])),
                    "visual_phase": r["visual_motion"]["phase"],
                    "action_phase": r["action_motion"]["phase"],
                    "consistency_score": r["consistency"]["consistency_score"],
                    "anomaly_flag": r["consistency"]["anomaly_flag"],
                    "human_review": r["quality_signals"]["human_review_recommended"],
                }
            except Exception as e:  # noqa: BLE001
                return {
                    "episode_id": ep,
                    "status": "failed",
                    "duration_sec": round(time.time() - t0, 2),
                    "error": str(e)[:200],
                }

    t_corpus = time.time()
    per_episode = await asyncio.gather(*[_one(ep) for ep in episode_ids])
    wall_sec = time.time() - t_corpus

    succeeded = [p for p in per_episode if p["status"] == "succeeded"]
    failed = [p for p in per_episode if p["status"] == "failed"]
    flagged = [p for p in succeeded if p.get("human_review")]
    disagreements = [p for p in succeeded if p.get("anomaly_flag") == "phase_disagreement"]
    avg_consistency = (
        sum(p["consistency_score"] for p in succeeded) / len(succeeded)
        if succeeded else 0.0
    )
    estimated_cost = len(succeeded) * _estimate_cost_per_episode(model)

    return {
        "n_requested": len(episode_ids),
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "n_flagged_for_review": len(flagged),
        "n_phase_disagreements": len(disagreements),
        "avg_consistency_score": round(avg_consistency, 3),
        "wall_clock_seconds": round(wall_sec, 1),
        "concurrency": concurrency,
        "estimated_cost_usd": round(estimated_cost, 4),
        "estimated_cost_per_episode_usd": round(_estimate_cost_per_episode(model), 5),
        "branch": annotation_branch,
        "per_episode": per_episode,
    }


# ────────────────────────────────────────────────────────────────────────────
# Compare two annotation branches — the compound-intelligence proof
# ────────────────────────────────────────────────────────────────────────────


@corpus_router.reasoner(tags=["corpus", "compare"])
async def compare_annotation_branches(
    branch_a: str = "roboscribe-v1",
    branch_b: str = "roboscribe-v2",
    dl_path: str | None = None,
) -> dict:
    """Diff two annotation branches by episode_id.

    For episodes annotated on both branches, count how many:
      - have the SAME visual_phase
      - have the SAME action_phase
      - have higher/lower consistency_score on branch B vs branch A
      - newly cleared `human_review_recommended` (improved) or newly raised
        (regressed) between A and B

    Returns aggregate stats — the "did we get better?" answer.
    """
    rows_a = await corpus_router.call(
        f"{NODE_ID}.search_episodes",
        tql="SELECT episode_id, visual_phase, action_phase, consistency_score, human_review_recommended",
        branch_name=branch_a,
        dl_path=dl_path,
    )
    rows_b = await corpus_router.call(
        f"{NODE_ID}.search_episodes",
        tql="SELECT episode_id, visual_phase, action_phase, consistency_score, human_review_recommended",
        branch_name=branch_b,
        dl_path=dl_path,
    )

    by_a = {r["episode_id"]: r for r in rows_a.get("rows", []) if r.get("episode_id") is not None}
    by_b = {r["episode_id"]: r for r in rows_b.get("rows", []) if r.get("episode_id") is not None}
    overlap = sorted(set(by_a) & set(by_b))

    same_visual = same_action = 0
    consistency_up = consistency_down = 0
    review_cleared = review_raised = 0

    for ep in overlap:
        a, b = by_a[ep], by_b[ep]
        if a.get("visual_phase") == b.get("visual_phase"):
            same_visual += 1
        if a.get("action_phase") == b.get("action_phase"):
            same_action += 1
        ca = float(a.get("consistency_score", 0) or 0)
        cb = float(b.get("consistency_score", 0) or 0)
        if cb > ca:
            consistency_up += 1
        elif cb < ca:
            consistency_down += 1
        # human_review flag transitions
        if a.get("human_review_recommended") and not b.get("human_review_recommended"):
            review_cleared += 1
        if not a.get("human_review_recommended") and b.get("human_review_recommended"):
            review_raised += 1

    return {
        "branch_a": branch_a,
        "branch_b": branch_b,
        "n_episodes_on_a": len(by_a),
        "n_episodes_on_b": len(by_b),
        "n_overlap": len(overlap),
        "same_visual_phase": same_visual,
        "same_action_phase": same_action,
        "consistency_improved_on_b": consistency_up,
        "consistency_regressed_on_b": consistency_down,
        "review_cleared_on_b": review_cleared,
        "review_raised_on_b": review_raised,
    }


# ────────────────────────────────────────────────────────────────────────────
# TQL showcase — canned hybrid queries demonstrating Deep Lake's multimodal API
# ────────────────────────────────────────────────────────────────────────────


@corpus_router.reasoner(tags=["corpus", "showcase"])
async def tql_showcase(
    branch_name: str = "roboscribe-v1",
    dl_path: str | None = None,
) -> dict:
    """Run a representative set of TQL queries — the substrate, demonstrated."""
    queries = [
        ("structured projection",
         "SELECT episode_id, visual_phase, action_phase, consistency_score, anomaly_flag"),
        ("structured filter: anomalies needing review",
         "SELECT episode_id, anomaly_flag, lang_episode_goal WHERE human_review_recommended = True"),
        ("numeric filter: low consistency (disagreement)",
         "SELECT episode_id, visual_phase, action_phase, consistency_rationale WHERE consistency_score < 0.5"),
        ("text search: push-related episodes",
         "SELECT episode_id, lang_scene_summary WHERE CONTAINS(lang_scene_summary, 'push')"),
        ("hybrid: low-consistency AND text-matched",
         "SELECT episode_id, lang_scene_summary, consistency_score WHERE consistency_score < 0.7 AND CONTAINS(lang_scene_summary, 'block')"),
        ("structured projection w/ semantic columns",
         "SELECT episode_id, visual_phase, action_phase, lang_episode_goal ORDER BY consistency_score ASC LIMIT 5"),
    ]
    out = []
    for label, tql in queries:
        try:
            r = await corpus_router.call(
                f"{NODE_ID}.search_episodes",
                tql=tql,
                branch_name=branch_name,
                dl_path=dl_path,
                limit=5,
            )
            out.append({
                "label": label,
                "tql": tql,
                "n_rows": r.get("n_rows", 0),
                "sample_rows": r.get("rows", [])[:3],
            })
        except Exception as e:  # noqa: BLE001
            out.append({"label": label, "tql": tql, "error": str(e)[:200]})

    return {
        "branch": branch_name,
        "n_queries": len(queries),
        "results": out,
    }
