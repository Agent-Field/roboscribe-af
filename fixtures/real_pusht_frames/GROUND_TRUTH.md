# Real PushT ground-truth fixture frames

Hand-labelled frames from `lerobot/pusht` (HuggingFace), decoded via AV1 MP4
→ `imageio.imread`. Each frame upscaled from 96×96 to 384×384 (nearest-neighbor)
for visibility. Resolutions and pixel content otherwise unmodified.

Used by `scripts/frame_eval.py` to benchmark vision models on objective
binary/categorical questions where the ground truth is unambiguous.

## Scene contents (PushT 2D)

- **Agent**: small **BLUE** circular dot (the robot end-effector)
- **Block**: solid **GRAY** T-shape (the object being pushed)
- **Target**: **GREEN** T-shape outline (where the block should end up)

## Per-frame labels

### `A_ep0_start.png` — ep0 frame 0 (initial position)
- Agent visible: True (small blue dot, upper area)
- Agent color: blue
- Number of T-shapes: 2
- Block color: gray
- Target color: green
- Block aligned with target: **False** (block rotated and offset)
- Agent touching block: **False** (agent at top, block at bottom)

### `B_ep0_mid.png` — ep0 frame 80 (active manipulation)
- Agent visible: True
- Agent color: blue
- Number of T-shapes: 2
- Block color: gray
- Target color: green
- Block aligned with target: **False** (visibly different rotation)
- Agent touching block: **True** (agent dot adjacent to block edge)

### `C_ep0_end.png` — ep0 last frame (final position, mostly successful)
- Agent visible: True
- Agent color: blue
- Number of T-shapes: 2 (overlapping)
- Block color: gray
- Target color: green
- Block aligned with target: **True** (substantial overlap)
- Agent touching block: **True**

### `D_ep1_start.png` — ep1 frame 0 (different initial pose)
- Agent visible: True
- Agent color: blue
- Number of T-shapes: 2
- Block color: gray
- Target color: green
- Block aligned with target: **False** (target center-top, block at bottom-left)
- Agent touching block: **False**

## Scoring axes (28 predictions per model: 4 frames × 7 questions)

| Axis | Cognitive demand |
|---|---|
| agent_visible, agent_color, block_color, target_color | Basic perception (small models pass easily) |
| n_t_shapes | Object counting (breaks under overlap) |
| block_aligned_with_target | Spatial reasoning (breaks under overlap) |
| agent_touching_block | Contact geometry (breaks under occlusion) |

## Benchmark provenance

```
fixtures/real_pusht_frames/
└── frames pulled from `lerobot/pusht` videos/observation.image/chunk-000/file-000.mp4
   (downloaded via huggingface_hub, decoded via imageio[ffmpeg] AV1)
```

To reproduce on a clean stack:

```bash
docker compose up --build -d
python3 scripts/frame_eval.py     # runs 3 models in parallel, prints scorecard
```
