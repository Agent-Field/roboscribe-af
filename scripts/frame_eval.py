"""Clever cross-model accuracy eval on REAL PushT frames.

4 hand-labelled ground-truth frames × 3 models × 7 objective questions
= 84 binary/categorical predictions. Runs all 12 (frame, model) jobs in parallel.

Total expected runtime: ~10s wall. Cost: ~$0.003.
"""

import base64
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = "http://localhost:8092"
NODE = "roboscribe-af"

MODELS = [
    ("qwen3-vl-8b",   "openrouter/qwen/qwen3-vl-8b-instruct"),
    ("qwen3-vl-32b",  "openrouter/qwen/qwen3-vl-32b-instruct"),
    ("qwen3-vl-235b", "openrouter/qwen/qwen3-vl-235b-a22b-instruct"),
]

# Hand-labelled ground truth from visually inspecting the 4 PNG files in /tmp/real_frames/
GROUND_TRUTH = {
    "A_ep0_start": {
        # Agent (small blue dot) far at top; gray T-block at bottom near target outline but not overlapping
        "agent_visible": True,
        "agent_color": "blue",
        "n_t_shapes": 2,
        "block_color": "gray",
        "target_color": "green",
        "block_aligned_with_target": False,  # block rotated AND offset from target
        "agent_touching_block": False,
    },
    "B_ep0_mid": {
        # Active push: gray block in middle, agent dot beside it, target rotated differently
        "agent_visible": True,
        "agent_color": "blue",
        "n_t_shapes": 2,
        "block_color": "gray",
        "target_color": "green",
        "block_aligned_with_target": False,  # block & target visibly different rotation
        "agent_touching_block": True,  # agent dot adjacent to block edge
    },
    "C_ep0_end": {
        # Successful: block + target visibly overlapping, agent at corner
        "agent_visible": True,
        "agent_color": "blue",
        "n_t_shapes": 2,  # block visibly overlaps target; LLM may say 1 or 2 (we score 2 as truth)
        "block_color": "gray",
        "target_color": "green",
        "block_aligned_with_target": True,
        "agent_touching_block": True,
    },
    "D_ep1_start": {
        # Different initial pose: target center-top (green T), block bottom (gray), agent left
        "agent_visible": True,
        "agent_color": "blue",
        "n_t_shapes": 2,
        "block_color": "gray",
        "target_color": "green",
        "block_aligned_with_target": False,
        "agent_touching_block": False,
    },
}

FRAMES_DIR = Path("/tmp/real_frames")


def png_to_data_url(path: Path) -> str:
    b = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(b).decode('ascii')}"


def call(payload, timeout=120):
    out = subprocess.run(
        ["/usr/bin/curl", "-sS", "-X", "POST",
         f"{BASE}/api/v1/execute/async/{NODE}.frame_validator",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"input": payload})],
        capture_output=True, text=True
    )
    exec_id = json.loads(out.stdout)["execution_id"]
    for _ in range(timeout):
        r = json.loads(subprocess.run(
            ["/usr/bin/curl", "-sS", f"{BASE}/api/v1/executions/{exec_id}"],
            capture_output=True, text=True).stdout)
        if r.get("status") in ("succeeded", "failed"):
            return r
        time.sleep(1)
    return {"status": "timeout"}


def run_one(model_name, model_id, frame_label, data_url):
    t0 = time.time()
    r = call({"keyframe_b64": data_url, "model": model_id})
    dur = time.time() - t0
    return model_name, frame_label, r, dur


# ────────────────────────────────────────────────────────────────────────
# Fire all (frame, model) jobs in parallel
# ────────────────────────────────────────────────────────────────────────
frame_files = sorted(FRAMES_DIR.glob("*.png"))
frame_data_urls = {p.stem: png_to_data_url(p) for p in frame_files}
print(f"\n\033[1;36m==== Cross-model frame validator eval ====\033[0m")
print(f"  frames: {list(frame_data_urls.keys())}")
print(f"  models: {[m[0] for m in MODELS]}")
print(f"  total jobs: {len(frame_data_urls) * len(MODELS)} (parallel)\n", flush=True)

jobs = [
    (mname, mid, flabel, durl)
    for mname, mid in MODELS
    for flabel, durl in frame_data_urls.items()
]

t_start = time.time()
results = {}  # (model, frame) -> response
with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
    futures = [ex.submit(run_one, *j) for j in jobs]
    for fut in concurrent.futures.as_completed(futures):
        m, f, r, dur = fut.result()
        results[(m, f)] = r
        mark = "✓" if r.get("status") == "succeeded" else "✗"
        print(f"  {mark} {m} on {f} in {dur:.1f}s", flush=True)
print(f"\n  total parallel wall: {time.time() - t_start:.1f}s", flush=True)


# ────────────────────────────────────────────────────────────────────────
# Score each model against ground truth
# ────────────────────────────────────────────────────────────────────────
print(f"\n\033[1;36m==== Per-frame predictions (truth | each model) ====\033[0m")
fields = ["agent_visible", "agent_color", "n_t_shapes", "block_color",
          "target_color", "block_aligned_with_target", "agent_touching_block"]

for f_label in sorted(frame_data_urls.keys()):
    truth = GROUND_TRUTH[f_label]
    print(f"\n  --- {f_label} ---")
    print(f"    truth: {json.dumps({k: truth[k] for k in fields})}")
    for mname, _mid in MODELS:
        r = results.get((mname, f_label), {})
        if r.get("status") != "succeeded":
            print(f"    {mname}: FAIL"); continue
        res = r["result"]
        diffs = []
        for k in fields:
            if res.get(k) != truth.get(k):
                diffs.append(f"{k}={res.get(k)}(t={truth[k]})")
        n_correct = len(fields) - len(diffs)
        print(f"    {mname}: {n_correct}/{len(fields)} correct  conf={res.get('confident')}", end="")
        if diffs:
            print(f"  diffs: {', '.join(diffs)}")
        else:
            print()

# Score totals
print(f"\n\033[1;36m==== Scorecard ({len(frame_data_urls)} frames × {len(fields)} questions = {len(frame_data_urls)*len(fields)} predictions per model) ====\033[0m")
for mname, _mid in MODELS:
    total = 0; correct = 0
    per_field = {f: {"correct": 0, "total": 0} for f in fields}
    for f_label in frame_data_urls.keys():
        truth = GROUND_TRUTH[f_label]
        r = results.get((mname, f_label), {})
        if r.get("status") != "succeeded":
            continue
        res = r["result"]
        for k in fields:
            total += 1
            per_field[k]["total"] += 1
            if res.get(k) == truth.get(k):
                correct += 1
                per_field[k]["correct"] += 1
    pct = 100 * correct / max(total, 1)
    print(f"\n  {mname}: {correct}/{total} = {pct:.0f}%")
    for k, s in per_field.items():
        pc = 100 * s["correct"] / max(s["total"], 1)
        print(f"    {k:<32} {s['correct']}/{s['total']}  ({pc:.0f}%)")
