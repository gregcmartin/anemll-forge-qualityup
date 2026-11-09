#!/bin/bash
# Phase B (Splash stopped): original ANEMLL (703131e) and the latest fork (ada23b3) on release 0.2 with its own
# drafter, then qualityup with the body-only drafter. All at a 32K cap, sequential.
set -uo pipefail
Q=$HOME/.local/share/inference-server/experiments/qualityup-20261109
E=$HOME/.local/share/inference-server/experiments/anemll-20261108
R=$E/models/anemll-qwen38-v02-440f4e36
B4=$Q/bench4
ts() { date '+%F %T'; }
for spec in "original|$E/src|$R|$R/coreai" "fork|$E/src-speedup|$R|$R/coreai" "qualityup|$Q/src|$Q/builds|$Q/builds/qualityup_full_kv8"; do
  IFS='|' read -r name src bundle build <<< "$spec"
  echo "$(ts) START $name"
  bash "$B4/m6_serve_bench4.sh" "$name" "$src" "$bundle" "$build"; echo "$(ts) END $name rc=$?"
done
touch "$B4/anemll3.done"
echo "$(ts) all done"
