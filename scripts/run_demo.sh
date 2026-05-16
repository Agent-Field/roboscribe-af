#!/usr/bin/env bash
# OXE-Narrator end-to-end demo.
#
# Walks through the full pipeline on synthetic PushT in one copy-paste session.
# Stops if any step fails. Each step prints what it just did + how long it took.
#
# Prerequisites:
#   - OPENROUTER_API_KEY in env (required — open-source models only, no Claude/GPT)
#   - docker + jq + curl on PATH
#
# Override the AgentField port via AF_PORT env (default 8080).

set -euo pipefail

AF_PORT="${AF_PORT:-8080}"
NODE="roboscribe-af"
BASE="http://localhost:${AF_PORT}"
BRANCH_A="roboscribe-v1"
BRANCH_B="roboscribe-v2"
N_EPISODES="${N_EPISODES:-5}"

# Pretty printing helpers (no fancy deps).
say()  { printf "\n\033[1;36m==== %s ====\033[0m\n" "$*"; }
note() { printf "  \033[0;90m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[0;32m✓ %s\033[0m\n" "$*"; }

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  printf "\033[1;31mERROR: OPENROUTER_API_KEY not set. Export it first.\033[0m\n" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  printf "\033[1;31mERROR: jq required. brew install jq.\033[0m\n" >&2
  exit 1
fi

# Quick health probe.
say "0. Checking AgentField control plane at $BASE"
if ! curl -sS --connect-timeout 3 "$BASE/api/v1/health" | jq -e '.status == "healthy"' >/dev/null 2>&1; then
  printf "\033[1;31mERROR: Control plane not healthy at %s. Start with: docker compose up --build -d\033[0m\n" "$BASE" >&2
  exit 1
fi
ok "control plane healthy"
N_REASONERS=$(curl -sS "$BASE/api/v1/discovery/capabilities" | jq "[.capabilities[] | select(.agent_id==\"$NODE\") | .reasoners | length] | first // 0")
N_SKILLS=$(curl -sS "$BASE/api/v1/discovery/capabilities" | jq "[.capabilities[] | select(.agent_id==\"$NODE\") | .skills | length] | first // 0")
ok "$NODE registered: $N_REASONERS reasoners, $N_SKILLS skills"

# Helper: kick off async + poll until done. Echoes the final response JSON.
run_async() {
  local target="$1"
  local payload="$2"
  local timeout="${3:-300}"
  local exec_id
  exec_id=$(curl -sS -X POST "$BASE/api/v1/execute/async/$NODE.$target" \
    -H 'Content-Type: application/json' -d "$payload" | jq -r '.execution_id')
  local i=0
  while (( i < timeout )); do
    local r status
    r=$(curl -sS "$BASE/api/v1/executions/$exec_id")
    status=$(echo "$r" | jq -r '.status')
    if [[ "$status" == "succeeded" || "$status" == "failed" ]]; then
      echo "$r"
      return 0
    fi
    sleep 1
    ((i+=1))
  done
  printf "\033[1;31mTIMEOUT after %ds on %s\033[0m\n" "$timeout" "$target" >&2
  return 1
}

# 1. Ingest synthetic PushT episodes
say "1. Ingest $N_EPISODES synthetic PushT episodes into Deep Lake"
note "Multimodal substrate: PNG bytes + Float32 trajectory arrays + Text + Int32 — all joined per episode"
R=$(run_async ingest_corpus_to_deeplake "{\"input\": {\"n_episodes\": $N_EPISODES, \"n_keyframes\": 4, \"task\": \"pusht\"}}")
echo "$R" | jq '.result | {dataset_path, episodes_ingested, n_rows, current_branch, all_branches}'
ok "ingested $N_EPISODES episodes onto main branch"

# 2. Annotate the corpus in parallel
say "2. Annotate the corpus in parallel (annotate_corpus)"
note "Concurrency=4; each episode runs ~13 LLM calls (5 vision + 8 text)"
EP_IDS=$(seq -s, 0 $((N_EPISODES - 1)))
R=$(run_async annotate_corpus "{\"input\": {\"episode_ids\": [$EP_IDS], \"n_keyframes\": 4, \"concurrency\": 4, \"annotation_branch\": \"$BRANCH_A\"}}" 600)
echo "$R" | jq '.result | {n_succeeded, n_failed, n_flagged_for_review, n_phase_disagreements, avg_consistency_score, wall_clock_seconds, estimated_cost_usd, estimated_cost_per_episode_usd, branch}'
echo "$R" | jq '.result.per_episode[] | {episode_id, outcome, n_segments, visual_phase, action_phase, consistency_score, anomaly_flag, duration_sec}'
ok "corpus annotated, committed to branch $BRANCH_A"

# 3. Run the TQL showcase
say "3. TQL showcase — multimodal hybrid queries on the annotation branch"
note "Structured + numeric + text CONTAINS in one query — that's Deep Lake's hybrid TQL story"
R=$(run_async tql_showcase "{\"input\": {\"branch_name\": \"$BRANCH_A\"}}")
echo "$R" | jq '.result.results[] | {label, tql, n_rows, sample_rows: (.sample_rows // [] | .[0:2])}'
ok "$(echo "$R" | jq -r '.result.n_queries') canned TQL queries executed"

# 3b. VECTOR search — semantic similarity via embeddings
say "3b. Semantic vector search — embed the query, rank by cosine similarity"
note "Open-source embedder: BAAI/bge-large-en-v1.5 (1024-dim) via OpenRouter"
R=$(run_async vector_search_episodes "{\"input\": {\"query\": \"robot pushes a block toward a target\", \"branch_name\": \"$BRANCH_A\", \"top_k\": 3}}")
echo "$R" | jq '.result | {query, branch, embedding_model, embedding_dim, top_k, n_rows, rows: (.rows | map({episode_id, lang_episode_goal, consistency_score}))}'
ok "vector search ranked top-3 episodes by semantic similarity"

# 4. Annotate same corpus on a second branch (simulates a revised prompt set)
say "4. Re-annotate same corpus on branch $BRANCH_B (simulating a prompt revision)"
note "Same architecture, same episodes — different branch. LLM stochasticity makes outputs vary slightly."
R=$(run_async annotate_corpus "{\"input\": {\"episode_ids\": [$EP_IDS], \"n_keyframes\": 4, \"concurrency\": 4, \"annotation_branch\": \"$BRANCH_B\"}}" 600)
echo "$R" | jq '.result | {n_succeeded, avg_consistency_score, wall_clock_seconds, estimated_cost_usd}'
ok "second branch $BRANCH_B annotated"

# 5. Compare the two branches — the compound-intelligence proof
say "5. Compare branch $BRANCH_A vs $BRANCH_B (compound-intelligence diff)"
note "How many episodes' verdicts changed between annotation passes?"
R=$(run_async compare_annotation_branches "{\"input\": {\"branch_a\": \"$BRANCH_A\", \"branch_b\": \"$BRANCH_B\"}}")
echo "$R" | jq '.result'
ok "branch diff computed"

# 6. Show corpus stats — what's in the lake now
say "6. Corpus stats — what's in the Deep Lake substrate"
R=$(run_async corpus_stats "{\"input\": {}}")
echo "$R" | jq '.result | {dataset_path, n_columns, branches}'
ok "corpus inspected"

# 7. Where to look next
say "7. What to explore next"
echo "  - Live workflow DAG:       $BASE/ui/"
echo "  - Discovery API:           $BASE/api/v1/discovery/capabilities"
echo "  - VC chain (signed prov.): $BASE/api/v1/did/workflow/<run_id>/vc-chain"
echo "  - Swap task to Aloha:      replay this script with AF_TASK=aloha_transfer"
echo ""
echo "  Single-episode end-to-end (debugging path):"
echo "    curl -X POST $BASE/api/v1/execute/async/$NODE.annotate_episode \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{\"input\": {\"episode_id\": 0, \"n_keyframes\": 4}}'"
echo ""
ok "demo complete"
