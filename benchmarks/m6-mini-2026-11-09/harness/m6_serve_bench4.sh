#!/bin/bash
# Start one ANEMLL deployment at a 32K cap from its own source checkout, sample memory every 10 s (wired + the
# server worker's footprint), run the four-way harness, stop the server.
#   m6_serve_bench4.sh <name> <src dir> <bundle> <build> [drafter package]
set -euo pipefail
Q=$HOME/.local/share/inference-server/experiments/qualityup-20261109
E=$HOME/.local/share/inference-server/experiments/anemll-20261108
R=$E/models/anemll-qwen38-v02-440f4e36
B4=$Q/bench4
name=$1 src=$2 bundle=$3 build=$4 draft=${5:-on}
export FORGE_BUNDLE=$bundle BUILD=$build MODEL=$R/model CTX=32768 PI_SYNC=0 PY=$E/venv/bin/python \
  ANEMLL_FORGE_STATE=$B4/state/$name EMBED_NPY=$R/model/embed_tokens_fp16.npy DRAFT=$draft
mkdir -p "$ANEMLL_FORGE_STATE"
wired() { vm_stat | awk '/page size of/ {ps=$8} /Pages wired down/ {gsub("\\.","",$4); printf "%.2f", $4*ps/1e9}'; }
worker_fp() {  # MB, footprint of the server's worker (child of server.pid)
  local p c; p=$(cat "$ANEMLL_FORGE_STATE/server.pid" 2>/dev/null) || return 0
  c=$(pgrep -P "$p" | head -1); [ -n "$c" ] || return 0
  footprint -p "$c" 2>/dev/null | awk '/Footprint:/ {v=$5; u=$6; if (u=="GB") v*=1024; if (u=="KB") v/=1024; print v; exit}'
}
echo "$(date +%T),$name,baseline,$(wired),," >> "$B4/mem.csv"
cd "$src"
START_WAIT_S=7200 scripts/qwen38_server.sh start > "$B4/start_$name.log" 2>&1 || { tail -40 "$B4/start_$name.log"; exit 1; }
curl -s localhost:8765/health > "$B4/health_$name.json"
echo "$(date +%T),$name,loaded,$(wired),$(worker_fp)," >> "$B4/mem.csv"
( while true; do echo "$(date +%T),$name,sample,$(wired),$(worker_fp),$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6}')" >> "$B4/mem.csv"; sleep 10; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
"$PY" "$B4/m6_bench4.py" "$name" anemll "$B4/prompts.json" "$B4/bench4.csv" "$B4/responses.jsonl"
kill $SAMPLER 2>/dev/null || true
cp "$ANEMLL_FORGE_STATE/server.log" "$B4/server_$name.log" 2>/dev/null || true
scripts/qwen38_server.sh stop || true
touch "$B4/$name.done"
