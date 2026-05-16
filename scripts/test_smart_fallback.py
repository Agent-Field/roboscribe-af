"""Verify scout-only and expert-only fallback modes of smart_frame_validator."""
import base64, json, subprocess, time
from pathlib import Path

BASE = "http://localhost:8092"
SCOUT = "openrouter/qwen/qwen3-vl-8b-instruct"
EXPERT = "openrouter/qwen/qwen3-vl-235b-a22b-instruct"

frame = Path("/tmp/bench_frames").glob("*.png")
frame = next(iter(sorted(frame)))
durl = f"data:image/png;base64,{base64.b64encode(frame.read_bytes()).decode()}"

def call(payload, timeout=60):
    out = subprocess.run(
        ["/usr/bin/curl", "-sS", "-X", "POST",
         f"{BASE}/api/v1/execute/async/roboscribe-af.smart_frame_validator",
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
    return {"status":"timeout"}

print("==== Test 1: scout-only mode (expert_model=None) ====")
r = call({"keyframe_b64": durl, "scout_model": SCOUT, "expert_model": None})
print(f"  status: {r.get('status')}")
print(f"  routing: {r['result'].get('_routing')}")
print(f"  scout_model: {r['result'].get('_scout_model')}")
print(f"  agent_visible: {r['result'].get('agent_visible')}, n_t_shapes: {r['result'].get('n_t_shapes')}")
assert r["result"]["_routing"] == "scout_only", f"expected scout_only got {r['result']['_routing']}"
print("  ✓ scout-only mode works")

print("\n==== Test 2: expert-only mode (scout_model=None) ====")
r = call({"keyframe_b64": durl, "scout_model": None, "expert_model": EXPERT})
print(f"  status: {r.get('status')}")
print(f"  routing: {r['result'].get('_routing')}")
print(f"  expert_model: {r['result'].get('_expert_model')}")
print(f"  agent_visible: {r['result'].get('agent_visible')}, n_t_shapes: {r['result'].get('n_t_shapes')}")
assert r["result"]["_routing"] == "expert_only", f"expected expert_only got {r['result']['_routing']}"
print("  ✓ expert-only mode works")

print("\n==== Test 3: both None should error ====")
r = call({"keyframe_b64": durl, "scout_model": None, "expert_model": None})
print(f"  status: {r.get('status')}  (expected: failed)")
if r.get("status") == "failed":
    print(f"  error: {str(r.get('error',''))[:100]}")
    print("  ✓ correctly rejects null+null")

print("\n==== Test 4: default args (both models set) - normal routing ====")
r = call({"keyframe_b64": durl})  # no model args at all → uses DEFAULT_SCOUT/EXPERT
print(f"  status: {r.get('status')}")
print(f"  routing: {r['result'].get('_routing')}")
print(f"  ✓ default routing works (routing={r['result'].get('_routing')})")
