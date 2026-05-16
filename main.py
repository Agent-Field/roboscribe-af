"""roboscribe-af — multi-reasoner hierarchical annotator for robotics episodes.

Entry point. The architecture is built bottom-up:

  Layer 0 (helpers.py):       deterministic skills — keyframe loading, action loading
  Layer 1 (reasoners/visual): leaf .ai() reasoners — object_detector, motion_phase_classifier
  Layer 2 (reasoners/visual): composers — scene_describer (calls object_detector per keyframe)
  Layer 3 (reasoners/visual): visual_thread (calls scene_describer + motion_classifier)
  ...
  Layer N (this file):        annotate_episode — top-level entry

Smoke reasoners from the .ai() verification phase still register under the
"smoke" tag so they can be re-tested if real reasoners start misbehaving.
"""

import os

from agentfield import Agent, AIConfig

# Register adapters at import time. Adding a new dataset = `import tasks.<name>`.
from tasks import aloha, pusht  # noqa: F401 — side-effect: register_adapter(...)

from helpers import skills_router
from reasoners import (
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
        "Hierarchical language annotator for robotics datasets. "
        "Dual-modality (visual + action) cross-verified annotations on PushT / "
        "LeRobot / Open X-Embodiment episodes."
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), auto_port=False)
