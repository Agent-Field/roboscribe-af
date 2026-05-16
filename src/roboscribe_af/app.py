"""Roboscribe-AF agent — top-level entry.

A multi-agent annotation system for robotics demonstration data. Built on
AgentField (multi-agent runtime) and Deep Lake (multimodal substrate). Every
episode runs through a 5-phase cognitive cascade fanning out to 200-300
cooperating agents working concurrently.

Layout:

  helpers.py                 Layer 0: deterministic skills (loaders, profilers, embedder)
  schemas.py                 All Pydantic models for every .ai() and the final annotation
  dl_store.py                Deep Lake substrate manager (schema, branches, TQL)
  tasks/                     Pluggable per-dataset adapters (PushT, Aloha bimanual)
  reasoners/
    visual.py                Visual modality thread + frame validator + smart routing
    action.py                Action modality thread (trajectory analysis)
    temporal.py              Sub-task segmentation with meta-spawned narrators
    composer.py              Cross-modal verifier + episode goal synthesizer + entry
    corpus.py                Parallel batch annotation, branch comparison, TQL showcase
    dl.py                    Deep Lake skills and vector search
    smoke.py                 Smoke-test reasoners (kept as safety net)
"""

import os

from agentfield import Agent, AIConfig

# Register adapters at import time. Adding a new dataset = `import roboscribe_af.tasks.<name>`.
from roboscribe_af.tasks import aloha, pusht  # noqa: F401 — side-effect: register_adapter(...)

from roboscribe_af.helpers import skills_router
from roboscribe_af.reasoners import (
    action_router,
    composer_router,
    corpus_router,
    dl_router,
    smoke_router,
    temporal_router,
    visual_router,
)

app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "roboscribe-af"),
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    version="0.1.0",
    description=(
        "Multi-agent annotation system for robotics demonstration data. "
        "Dual-modality (visual + action) cross-verified annotations on PushT, "
        "LeRobot, Open X-Embodiment, or any LeRobot-format dataset."
    ),
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openrouter/qwen/qwen3-vl-32b-instruct"),
    ),
    dev_mode=True,
)

app.include_router(skills_router)
app.include_router(smoke_router)
app.include_router(visual_router)
app.include_router(action_router)
app.include_router(temporal_router)
app.include_router(composer_router)
app.include_router(dl_router)
app.include_router(corpus_router)


def main() -> None:
    """Run the agent server. Entry point referenced by `[project.scripts]`."""
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)


if __name__ == "__main__":
    main()
