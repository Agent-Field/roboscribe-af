"""Article-grade benchmark: 25 real PushT frames × 4 modes (scout, mid, expert, smart-routed)
× 7 objective questions = 700 predictions. Gold standard = expert (235B) predictions.

Computes per-mode accuracy + cost + parallel wall time.
"""

import base64
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path

BASE = "http://localhost:8092"
NODE = "roboscribe-af"

SCOUT  = "openrouter/qwen/qwen3-vl-8b-instruct"
MID    = "openrouter/qwen/qwen3-vl-32b-instruct"
EXPERT = "openrouter/qwen/qwen3-vl-235b-a22b-instruct"

# Per-token pricing from OpenRouter (USD per token, prompt + completion).
PRICING = {
    SCOUT:  (0.00000008, 0.00000050),
    MID:    (0.00000010, 0.00000042),
    EXPERT: (0.00000020, 0.00000088),
}
# Per-call rough token estimates for one frame_validator call:
#   ~700 input tokens (incl image) + ~80 output tokens (compact flat schema)
EST_IN, EST_OUT = 700, 80


def cost_per_call(model_id: str) -> float:
    p_in, p_out = PRICING[model_id]
    return EST_IN * p_in + EST_OUT * p_out


SCHEMA_FIELDS = [
    "agent_visible", "agent_color", "n_t_shapes",
    "block_color", "target_color",
    "block_aligned_with_target", "agent_touching_block",
]

FRAMES_DIR = Path("/tmp/bench_frames")


def png_to_data_url(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def call(target, payload, timeout=120):
    out = subprocess.run(
        ["/usr/bin/curl", "-sS", "-X", "POST",
         f"{BASE}/api/v1/execute/async/{NODE}.{target}",
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


def run_validator(frame_label, data_url, model_id):
    t0 = time.time()
    r = call("frame_validator", {"keyframe_b64": data_url, "model": model_id})
    return frame_label, model_id, r, time.time() - t0


def run_smart(frame_label, data_url):
    t0 = time.time()
    r = call("smart_frame_validator", {
        "keyframe_b64": data_url,
        "scout_model": SCOUT,
        "expert_model": EXPERT,
    })
    return frame_label, "smart", r, time.time() - t0


def extract(r):
    if r.get("status") != "succeeded":
        return None
    return {k: r["result"].get(k) for k in SCHEMA_FIELDS}


# ────────────────────────────────────────────────────────────────────────
# 1. Load frames
# ────────────────────────────────────────────────────────────────────────
frame_files = sorted(FRAMES_DIR.glob("*.png"))
frame_durls = {p.stem: png_to_data_url(p) for p in frame_files}
print(f"\n\033[1;36m==== Benchmark setup ====\033[0m")
print(f"  frames: {len(frame_durls)}  from real lerobot/pusht (AV1 decoded)")
print(f"  fields per frame: {len(SCHEMA_FIELDS)}  total predictions per mode: {len(frame_durls)*len(SCHEMA_FIELDS)}", flush=True)


# ────────────────────────────────────────────────────────────────────────
# 2. Run all 4 modes in parallel
# ────────────────────────────────────────────────────────────────────────
print(f"\n\033[1;36m==== Dispatching {len(frame_durls)} frames × 4 modes in parallel ====\033[0m", flush=True)

jobs = []
for fl, du in frame_durls.items():
    jobs.append(("scout",   fl, du, SCOUT))
    jobs.append(("mid32b",  fl, du, MID))
    jobs.append(("expert",  fl, du, EXPERT))
    jobs.append(("smart",   fl, du, None))  # smart uses smart_frame_validator

t_start = time.time()
results = {}  # (mode, frame) -> r
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    futures = []
    for mode, fl, du, mid in jobs:
        if mode == "smart":
            futures.append(ex.submit(run_smart, fl, du))
        else:
            futures.append(ex.submit(run_validator, fl, du, mid))
    done_count = 0
    for fut in concurrent.futures.as_completed(futures):
        fl, mid_or_mode, r, dur = fut.result()
        # Figure out which mode this was
        if mid_or_mode == SCOUT:    mode = "scout"
        elif mid_or_mode == MID:    mode = "mid32b"
        elif mid_or_mode == EXPERT: mode = "expert"
        else:                       mode = "smart"
        results[(mode, fl)] = r
        done_count += 1
        if done_count % 10 == 0:
            print(f"  ... {done_count}/{len(futures)} done", flush=True)

total_wall = time.time() - t_start
print(f"\n  total parallel wall: {total_wall:.1f}s", flush=True)


# ────────────────────────────────────────────────────────────────────────
# 3. Use expert (235B) predictions as gold standard
# ────────────────────────────────────────────────────────────────────────
print(f"\n\033[1;36m==== Gold standard = expert (235B) predictions ====\033[0m", flush=True)
gold = {}
for fl in frame_durls:
    r = results.get(("expert", fl), {})
    pred = extract(r)
    if pred is None:
        print(f"  ⚠ missing expert prediction for {fl}", flush=True)
    else:
        gold[fl] = pred
print(f"  gold-standard frames: {len(gold)}", flush=True)


# ────────────────────────────────────────────────────────────────────────
# 4. Per-mode scorecard
# ────────────────────────────────────────────────────────────────────────
print(f"\n\033[1;36m==== Per-mode scorecard (vs 235B gold standard) ====\033[0m", flush=True)

def score_mode(mode):
    total = 0; correct = 0
    per_field = {f: {"correct": 0, "total": 0} for f in SCHEMA_FIELDS}
    failures = 0
    for fl, g in gold.items():
        r = results.get((mode, fl), {})
        pred = extract(r)
        if pred is None:
            failures += 1; continue
        for f in SCHEMA_FIELDS:
            total += 1
            per_field[f]["total"] += 1
            if pred[f] == g[f]:
                correct += 1
                per_field[f]["correct"] += 1
    return total, correct, per_field, failures


print(f"\n  {'mode':<10} {'overall':>9} {'agent_vis':>10} {'agent_col':>10} {'n_t':>5} {'blk_col':>8} {'tgt_col':>8} {'aligned':>8} {'touch':>7}")
print(f"  {'-'*80}")
for mode in ("scout", "mid32b", "expert", "smart"):
    total, correct, per_field, fails = score_mode(mode)
    pct = 100 * correct / max(total, 1)
    field_pcts = []
    for f in SCHEMA_FIELDS:
        s = per_field[f]
        fp = 100 * s["correct"] / max(s["total"], 1)
        field_pcts.append(f"{fp:>6.0f}%")
    row = f"  {mode:<10} {pct:>7.0f}%  {' '.join(f'{x:>9}' for x in field_pcts)}"
    print(row)


# ────────────────────────────────────────────────────────────────────────
# 5. Smart-routing analysis: escalation rate + cost
# ────────────────────────────────────────────────────────────────────────
print(f"\n\033[1;36m==== Smart-routing analysis ====\033[0m", flush=True)
escalations = 0
scout_only = 0
for fl in frame_durls:
    r = results.get(("smart", fl), {})
    if r.get("status") != "succeeded": continue
    routing = r["result"].get("_routing", "?")
    if routing == "expert_escalated":
        escalations += 1
    elif routing == "scout_only":
        scout_only += 1

esc_rate = 100 * escalations / max(escalations + scout_only, 1)
print(f"  total smart calls: {escalations + scout_only}")
print(f"  scout-only:        {scout_only}  ({100 - esc_rate:.0f}%)")
print(f"  expert-escalated:  {escalations}  ({esc_rate:.0f}%)", flush=True)


# ────────────────────────────────────────────────────────────────────────
# 6. Cost comparison (USD)
# ────────────────────────────────────────────────────────────────────────
print(f"\n\033[1;36m==== Cost comparison (USD) ====\033[0m", flush=True)
n = len(frame_durls)

cost_scout    = n * cost_per_call(SCOUT)
cost_mid      = n * cost_per_call(MID)
cost_expert   = n * cost_per_call(EXPERT)
# Smart cost: scout per frame + expert per escalation
cost_smart    = n * cost_per_call(SCOUT) + escalations * cost_per_call(EXPERT)

vs_expert = lambda c: f"({100 * c / cost_expert:.0f}% of expert)"
print(f"\n  {'mode':<10}  {'cost':>10}  {'per-frame':>11}  {'vs expert':>15}")
print(f"  {'-'*55}")
print(f"  {'scout':<10}  ${cost_scout:>8.4f}  ${cost_scout/n:>8.5f}    {vs_expert(cost_scout):>15}")
print(f"  {'mid32b':<10}  ${cost_mid:>8.4f}  ${cost_mid/n:>8.5f}    {vs_expert(cost_mid):>15}")
print(f"  {'smart':<10}  ${cost_smart:>8.4f}  ${cost_smart/n:>8.5f}    {vs_expert(cost_smart):>15}")
print(f"  {'expert':<10}  ${cost_expert:>8.4f}  ${cost_expert/n:>8.5f}    {vs_expert(cost_expert):>15}")

# Per-1000-frame projection
print(f"\n  Per 1,000 frames:")
for mode, c in (("scout", cost_scout), ("mid32b", cost_mid), ("smart", cost_smart), ("expert", cost_expert)):
    print(f"    {mode:<10}  ${c * 1000 / n:.2f}")

print("\n\033[1;32m✓ BENCHMARK COMPLETE\033[0m")
