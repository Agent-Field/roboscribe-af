# Roboscribe-AF

![Roboscribe-AF: multi-agent annotation for robotic demonstrations](assets/hero.png)

### Roboscribe is a multi-agent annotation system for robotics demonstration data.
### Built on [AgentField](https://github.com/Agent-Field/agentfield) (async-parallel multi-agent runtime) and [Deep Lake](https://github.com/activeloopai/deeplake) (multimodal data substrate). Open source.

Foundation models for robotics, NVIDIA GR00T, OpenVLA, Octo, π-Zero, learn from demonstration data: a robot doing something paired with hierarchical language descriptions of what it did. The data side is the bottleneck. Existing annotation pipelines either run a single multimodal LLM call per episode (cheap, hallucinates) or pay human labelers (accurate, doesn't scale to millions of episodes).

Roboscribe replaces both with a **complex, deeply-cascaded multi-agent architecture**. Every episode runs through a 5-phase cognitive cascade that fans out into **200 to 300 specialized agents working concurrently**: per-keyframe object detectors, per-modality reasoners, meta-spawned segment narrators, cross-modal verifiers, embedding workers, and substrate writers, all dispatched in parallel through an async control plane. One complete annotation (episode goal, sub-task segments, per-modality phase tags, cross-modal verification, semantic embedding) finishes in about 16 seconds at roughly $0.002. Output lands as versioned, hybrid-searchable Deep Lake branches ready to feed any LeRobot, OpenVLA, Octo, or GR00T-compatible training pipeline.

Dataset-agnostic. PushT and Aloha bimanual adapters ship with the repo; adding any new LeRobot-format robot is about 30 lines.

---

## What you get

**Per episode:**

- **Episode-level grain**: one-sentence goal, outcome tag, canonical objects
- **Sub-task segmentation**: temporal boundaries, verb-phrase labels, per-segment phase tag, objects involved
- **Independent modality analysis**: separate visual and action expert chains, each producing its own motion phase classification
- **Cross-modal verification**: a third agent reconciles the two modality stories and emits a consistency score and anomaly flag
- **Quality signals**: auto-set `human_review_recommended` flag for ambiguous episodes
- **Semantic embedding**: 1024-dim vector of the scene description, queryable via TQL `cosine_similarity`

**Per corpus:**

- A **versioned Deep Lake branch** with the above attached as sibling columns on the same row as the raw episode data (frames, actions, states)
- **Hybrid TQL queries** combining vector similarity, text `CONTAINS`, numeric filters, and structured equality in one statement
- **Branch comparison** between annotation passes: diff-able, rollback-able, A/B-testable
- **PyTorch-streaming via `ds.pytorch()`** straight into a training loop, no intermediate format

Here's what one annotated row looks like:

```json
{
  "episode_id": 0,
  "grain_episode": {
    "goal": "Push the gray T-shaped block into alignment with the green target outline",
    "outcome": "successful"
  },
  "grain_segments": [
    { "start_frame": 0,  "end_frame": 21, "phase": "approach",   "label": "approach the block",        "objects_involved": ["gray t-shaped block", "blue circular agent"] },
    { "start_frame": 21, "end_frame": 40, "phase": "manipulate", "label": "push toward target",        "objects_involved": ["gray t-shaped block", "green target outline"] },
    { "start_frame": 40, "end_frame": 49, "phase": "manipulate", "label": "align with target outline", "objects_involved": ["gray t-shaped block", "green target outline"] }
  ],
  "visual_motion":  { "phase": "manipulate", "rationale": "agent maintains contact as the block moves toward the target" },
  "action_motion":  { "phase": "approach",   "rationale": "trajectory shows directed motion without sustained contact signatures" },
  "consistency":    { "consistency_score": 0.2, "anomaly_flag": "phase_disagreement" },
  "quality_signals": { "human_review_recommended": true },
  "scene": { "canonical_objects": ["gray t-shaped block", "green target outline", "blue circular agent"], "scene_summary": "…" }
}
```

That entire record is one row in a Deep Lake branch, joined to the raw episode by `episode_id`.

---

## What powers it

| Layer | Tool | What it brings to this build |
|---|---|---|
| **Multi-agent runtime** | [AgentField](https://github.com/Agent-Field/agentfield) | Async-parallel agent orchestration at scale. **Each episode dispatches 200 to 300 cooperating agents through the control-plane queue.** Agents run as microservices with `asyncio.gather` at every layer where work is independent: a 5-phase, depth-5 cognitive cascade with per-keyframe vision fan-out, per-segment narrator spawning, dual-modality thread parallelism, all in one ~16-second pipeline. Per-request model overrides, structured Pydantic outputs at every leaf. |
| **Multimodal substrate** | [Deep Lake](https://github.com/activeloopai/deeplake) | Image bytes + Float32 tensors + text + embeddings + scalars in one row, joined by `episode_id`. Versioned branches per annotation pass. Hybrid TQL combining vector similarity, text CONTAINS, numeric filters, and structured equality in a single statement. PyTorch streaming dataloader directly into training loops. |
| **Dataset format** | [HuggingFace LeRobot](https://github.com/huggingface/lerobot) | The de-facto modern robotics dataset format. Same shape NVIDIA GR00T / Isaac Lab / Cosmos, OpenVLA, Octo, and π-Zero all consume. |
| **Vision + reasoning models** | [OpenRouter](https://openrouter.ai/) → Qwen3-VL family (default) or NVIDIA Nemotron Nano VL (drop-in) | One env-var swap routes the entire stack through NVIDIA Nemotron instead of Qwen3-VL. |

---

## Headline numbers

**A 25-frame benchmark on real LeRobot/PushT data, four modes running in parallel, ground truth = expert (235B) predictions:**

| Mode | Accuracy | Cost per frame | Cost per 1,000 frames |
|---|---|---|---|
| **Scout-only** (Qwen3-VL-8B) | **93%** | $0.00010 | $0.10 |
| Mid-tier (Qwen3-VL-32B) | 97% | $0.00010 | $0.10 |
| **Smart-routed** (8B scout → 235B expert on doubt) | **98%** | $0.00013 | $0.13 |
| Expert-only (Qwen3-VL-235B) | 100% | $0.00021 | $0.21 |

**Total parallel wall time for 100 jobs across 4 modes: 16.6 seconds.**

The **Smart-routed** mode is the production sweet spot. It runs every frame through the cheap 8B scout first, then escalates to the 235B flagship only when confidence signals fire (low confidence reported by the scout, or unexpectedly few objects detected, which is a robust proxy for occlusion). It escalates 16% of frames and closes 10 of the 13 errors the 8B-only mode would have made, at 62% of the cost of running everything through the flagship.

A separate audit-grade benchmark (4 hand-labelled frames × 7 objective questions × 3 models = 84 predictions) gave the same architecture **89% / 89% / 100%** across the three models in 4.1 seconds parallel wall time.

---

## How it works

![Roboscribe-AF 5-phase multi-agent cascade](assets/architecture.png)

A deep, 5-phase cognitive cascade. Every episode goes through:

1. **Extract**: keyframes and action trajectories are loaded and prepared in parallel.
2. **Analyze**: two structurally independent **modality threads** run concurrently. The visual thread fans out per-keyframe object detectors, scene composers, and motion classifiers. The action thread runs trajectory phasing, velocity profiling, and per-modality phase classification.
3. **Segment**: a temporal segmenter detects sub-task boundaries, then **meta-spawns one narrator agent per detected segment at runtime**. The cascade re-shapes itself based on what each episode actually contains.
4. **Verify**: a cross-modal verifier reconciles the two modality stories. When they disagree, the discrepancy becomes a signal and the episode is automatically flagged for human review.
5. **Synthesize**: the episode goal is composed, the scene description is embedded (1024-dim), and the structured annotation row is committed to a versioned Deep Lake branch joined to the raw episode by `episode_id`.

Across these five phases the system dispatches **200 to 300 cooperating agents per episode**, all driven through AgentField's async control plane. The architectural choice that earns the accuracy: two structurally separate modality experts that cannot conspire to hallucinate the same wrong story. A single multimodal call sees vision and action as one prompt and glosses over disagreements. Roboscribe analyzes them in parallel, then asks a third agent whether they cohere.

---

## Run it yourself

```bash
git clone https://github.com/Agent-Field/roboscribe-af
cd roboscribe-af
cp .env.example .env       # paste OPENROUTER_API_KEY
docker compose up --build  # ~1s rebuild via uv cache after first build
```

Once the stack is up, open **[http://localhost:8080/ui/](http://localhost:8080/ui/)** in your browser. That's the AgentField control plane UI — watch the multi-agent cascade execute live as runs come in. Every agent invocation shows up as a node in the workflow graph with its prompt, inputs, outputs, latency, and cost.

```bash
# Annotate a corpus in parallel (10 episodes, ~17 seconds wall, ~$0.02 total)
curl -X POST http://localhost:8080/api/v1/execute/async/roboscribe-af.annotate_corpus \
  -H 'Content-Type: application/json' \
  -d '{"input": {"episode_ids": [0,1,2,3,4,5,6,7,8,9], "concurrency": 4}}'

# Search the annotated corpus by meaning
curl -X POST http://localhost:8080/api/v1/execute/async/roboscribe-af.vector_search_episodes \
  -H 'Content-Type: application/json' \
  -d '{"input": {"query": "robot pushes a block toward a target", "top_k": 5}}'
```

Open the AgentField UI alongside these curls and you'll see the 200 to 300 cooperating agents per episode lighting up across the 5 phases of the cascade in real time.

Reproduce the benchmark numbers above:

```bash
python3 scripts/bench25.py
```

Swap the entire stack to NVIDIA Nemotron Nano VL with one env-var:

```bash
AI_MODEL=openrouter/nvidia/nemotron-nano-12b-v2-vl:free docker compose up --build
```

---

## Features

Shipped:

- [x] Hierarchical annotations (episode → segments → motion phases → objects)
- [x] Independent visual + action analysis with cross-modal reconciliation
- [x] Cost-aware scout/expert routing with confidence-gated escalation
- [x] Deep Lake substrate with versioned annotation branches
- [x] Hybrid TQL queries (text + structured + vector cosine similarity)
- [x] Semantic search via 1024-dim text embeddings
- [x] Parallel agent dispatch through async control-plane queue (200 to 300 agents per episode)
- [x] Branch-vs-branch annotation diff
- [x] Pluggable `TaskAdapter` protocol with bundled PushT and Aloha bimanual adapters
- [x] NVIDIA Nemotron Nano VL drop-in compatibility
- [x] Open-source vision and reasoning models via OpenRouter

In progress:

- [ ] PyTorch streaming into LeRobot's ACT / Diffusion Policy training loop
- [ ] Real video loading for any LeRobot dataset (AV1 decoded full episodes)
- [ ] NVIDIA Cosmos Tokenizer integration for video-frame embeddings
- [ ] Multi-step manipulation tasks (pick-place, pour, fold)
- [ ] Long-horizon hierarchical segmentation (multi-level grain)
- [ ] Modal scale-out for million-episode runs

---

## Acknowledgments

Built on the open-source work of:

- **[AgentField](https://github.com/Agent-Field/agentfield)**: async-parallel multi-agent runtime
- **[Activeloop Deep Lake](https://github.com/activeloopai/deeplake)**: multimodal data substrate with versioned branches and hybrid TQL
- **[HuggingFace LeRobot](https://github.com/huggingface/lerobot)**: universal robotics dataset format
- **[Qwen team](https://github.com/QwenLM/Qwen3-VL)**: Qwen3-VL open-source vision models
- **[NVIDIA](https://huggingface.co/nvidia)**: Nemotron Nano VL drop-in compatibility; LeRobot-format alignment with GR00T / Isaac Lab / Cosmos Physical AI stack
- **[BAAI](https://huggingface.co/BAAI)**: BGE-large-en embeddings for semantic search

---

### Other projects on AgentField

- [SEC-AF](https://github.com/Agent-Field/sec-af): AI-native security auditor
- [PR-AF](https://github.com/Agent-Field/pr-af): agentic PR reviewer
- [Contract-AF](https://github.com/Agent-Field/contract-af): legal contract risk analyzer
- [Reactive-Atlas](https://github.com/Agent-Field/reactive-atlas): MongoDB to AI enrichment pipeline
