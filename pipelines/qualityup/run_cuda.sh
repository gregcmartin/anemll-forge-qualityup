#!/usr/bin/env bash
# qualityup weights on one 96 GB CUDA GPU: the release 0.2 recipe (docs/QUANTIZATION_TRAINING.md) with the LUT4
# late-MLP plan. Run after setup_cuda.sh, from any directory:
#
#   bash pipelines/qualityup/run_cuda.sh all              # every stage in order; finished stages are skipped
#   bash pipelines/qualityup/run_cuda.sh gptq teacher     # chosen stages
#   STEPS=5 EVAL_EVERY=5 TAG=smoke bash pipelines/qualityup/run_cuda.sh train   # short QAT trial (reuses the export)
#   TAG=control PLAN=configs/quantization/mix25in_vq3pA.json bash pipelines/qualityup/run_cuda.sh model
#       # the release 0.2 plan on the same data and teacher: the matched control for the plan change
#
# Stages: calib corpus traces gptq gptq_eval teacher check train materialize eval pack
#   "data" = calib corpus traces teacher; "model" = gptq gptq_eval check train materialize eval pack; "all" = both.
# Optional agentic calibration rows: $DATA/calib_pi_ids.npy (48 rows of 1024 tokens, from qwen38_calib_pi.py with
# explicitly selected sessions). Without it, 48 rows spread over the generated corpus (130 tool-use prompts) take
# their place in GPTQ calibration, and the QAT rows are corpus + chat + WikiText.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export WORK="${WORK:-$HOME/qualityup}"
export MODEL="${MODEL:-$WORK/Qwen3.8-27B}"
export DATA="${DATA:-$WORK/data}"
export WIKI="${WIKI:-$WORK/wikitext}"
export DEV_TRACE="$DATA/kl_dev" SEL_TRACE="$DATA/kl_selection" TEACHER="$DATA/qat_teacher"
TAG="${TAG:-qualityup}"
PLAN="${PLAN:-configs/quantization/mix25in_vq3pA_lut4L24-63.json}"
if [[ -z "${GPTQ_TAG:-}" ]]; then          # TAG=smoke trains briefly on qualityup's GPTQ export
  GPTQ_TAG="${TAG}_gptq"; [[ "$TAG" == smoke ]] && GPTQ_TAG=qualityup_gptq
fi
export -n TAG PLAN GPTQ_TAG STEPS EVAL_EVERY 2>/dev/null || true   # read here; set explicitly per call
GPTQ_EXPORT="$WORK/runs/export/$GPTQ_TAG"
QAT_OUT="$WORK/runs/export/${TAG}_qat"
FULL="$WORK/runs/export/${TAG}_full"
PY="${PY:-$ROOT/.venv-cuda/bin/python}"
LOGS="$WORK/logs"
mkdir -p "$DATA" "$WORK/runs" "$LOGS"
PI="$DATA/calib_pi_ids.npy"

if [[ -f "$PI" ]]; then
  CAL_MIX="$DATA/calib_chat_ids.npy:36,$PI:48"
  QAT_TRAIN="$DATA/corpus_ids.npy:meta,$PI:44,$DATA/calib_chat_ids.npy:32"
else
  CAL_MIX="$DATA/calib_chat_ids.npy:36,$DATA/calib_corpus48_ids.npy:48"
  QAT_TRAIN="$DATA/corpus_ids.npy:meta,$DATA/calib_chat_ids.npy:32"
fi
export TRAIN="$QAT_TRAIN" WIKI_ROWS=44 HOLD=4 SEQ=1024 TOPK=256 TEACHER_DTYPE=fp16

run() {  # run <stage> <command...>: log to $LOGS/<tag>.<stage>.log; a .done marker skips it next time
  local stage="$1"; shift
  local key="$TAG.$stage"; [[ "$stage" =~ ^(calib|corpus|traces|teacher)$ ]] && key="data.$stage"
  if [[ -f "$LOGS/$key.done" ]]; then echo "== $key: done earlier (rm $LOGS/$key.done to redo)"; return; fi
  echo "== $key: start $(date -u +%FT%TZ)"
  local t0=$SECONDS
  "$@" 2>&1 | tee "$LOGS/$key.log"
  touch "$LOGS/$key.done"
  echo "== $key: done in $(( (SECONDS - t0) / 60 )) min"
}

s_calib() { CAL_OUT="$DATA/calib_chat_ids.npy" "$PY" scripts/qwen38_calib_gen.py; }
s_corpus() {
  CORPUS_PROMPTS=configs/kl/qat_corpus_prompts.json CAL_OUT="$DATA/corpus_ids.npy" \
  SEQ=1024 MAX_NEW=1792 GEN_BATCH=64 VAL_FRAC=0.05 SEED=1234 "$PY" scripts/qwen38_corpus_gen.py
  "$PY" - <<EOF
import json, numpy as np
r = np.load("$DATA/corpus_ids.npy"); n = json.load(open("$DATA/corpus_ids.json"))["n_train"]
pick = np.linspace(0, n - 1, 48).round().astype(int)          # spread over every prompt category
np.save("$DATA/calib_corpus48_ids.npy", r[pick]); print("calib_corpus48_ids", r[pick].shape, "of", n, "train rows")
EOF
}
s_traces() {
  TRACE="$DEV_TRACE" "$PY" scripts/qwen38_kl.py generate
  # top-512 reference: the same file serves the PyTorch evals here and m6_kl512_eval.py (KL-512) on the M6
  TRACE="$DEV_TRACE" TOPK=512 "$PY" scripts/qwen38_kl.py reference
  TRACE="$SEL_TRACE" PROMPTS_FILE=configs/kl/qualityup_selection_prompts.json "$PY" scripts/qwen38_kl.py generate
  TRACE="$SEL_TRACE" "$PY" scripts/qwen38_kl.py reference
}
s_gptq() {
  OUT="$WORK/runs" TAG="$GPTQ_TAG" EXPORT=1 METHOD=gptq BASIS=online LOAD_DTYPE=bf16 PLAN="$PLAN" \
  MIXER="LUT4 per-tensor + pcs" KV_FMT="INT8 per-channel" HEAD="LUT4 per-tensor + pcs" \
  AW=1 BASELINE=0 ASYM=0 LR_RANK=64 LR_PARTS=gdn,attn LR_ITERS=1 NCAL=128 \
  MIX_ROT=readers+writers KMEANS_SEED=1 KMEANS_THREADS=1 KMEANS_SAMPLE=300000 KMEANS_INIT=3 \
  CAL_MIX="$CAL_MIX" "$PY" scripts/qwen38_gptq_27b.py
}
s_gptq_eval() {
  for tr in "$DEV_TRACE" "$SEL_TRACE"; do
    EXPORT_DIR="$GPTQ_EXPORT" TRACE="$tr" TAG="$GPTQ_TAG" "$PY" scripts/qwen38_kl.py eval
  done
}
s_teacher() { BS=2 "$PY" scripts/qwen38_qat_kl.py teacher; }
s_check() { EXPORT_DIR="$GPTQ_EXPORT" CHECK_BASE="$GPTQ_TAG" CHECK_SEQ=32 "$PY" scripts/qwen38_qat_kl.py check; }
s_train() {
  EXPORT_DIR="$GPTQ_EXPORT" OUT_DIR="$QAT_OUT" TRAIN_PARTS=lut,scale,head,lr STEPS="${STEPS:-600}" BS=2 \
  EVAL_EVERY="${EVAL_EVERY:-50}" WARMUP=20 LR_LUT=2e-4 LR_SCALE=2e-4 LR_LRF=2e-5 LR_HEAD=1e-4 DEV_SEQ=32 \
    "$PY" scripts/qwen38_qat_kl.py train
}
s_materialize() { "$PY" scripts/qwen38_overlay_materialize.py "$QAT_OUT" "$FULL"; }
s_eval() {
  for tr in "$DEV_TRACE" "$SEL_TRACE"; do
    EXPORT_DIR="$FULL" TRACE="$tr" TAG="${TAG}_full" "$PY" scripts/qwen38_kl.py eval
  done
  "$PY" - <<EOF
import json, pathlib
for tr in ("$DEV_TRACE", "$SEL_TRACE"):
    for tag in ("$GPTQ_TAG", "${TAG}_full", "control_gptq", "control_full"):
        f = pathlib.Path(tr) / f"kl_{tag}.json"
        if f.exists():
            r = json.loads(f.read_text())
            print(f"{pathlib.Path(tr).name:13s} {tag:20s}", {k: r[k] for k in r if k in
                  ("mean_kl", "median_kl", "p99_kl", "top1_agree", "ppl", "ref_ppl", "size_gib_incl_lr")})
EOF
}
s_pack() {  # for the M6: materialized export, run logs and the dev trace (no training data)
  tar -C "$WORK/runs/export" -cf "$WORK/${TAG}_full.tar" "$(basename "$FULL")"
  tar -C "$WORK" -rf "$WORK/${TAG}_full.tar" logs
  tar -C "$DATA" -rf "$WORK/${TAG}_full.tar" kl_dev   # BF16 trace + top-512 reference (public prompts) for KL-512
  ls -la "$WORK/${TAG}_full.tar"
}

stages=("$@")
[[ ${#stages[@]} -eq 0 ]] && { sed -n 2,16p "$0"; exit 1; }
expand=()
for s in "${stages[@]}"; do
  case "$s" in
    all) expand+=(calib corpus traces teacher gptq gptq_eval check train materialize eval pack) ;;
    data) expand+=(calib corpus traces teacher) ;;
    model) expand+=(gptq gptq_eval check train materialize eval pack) ;;
    *) expand+=("$s") ;;
  esac
done
for s in "${expand[@]}"; do
  declare -F "s_$s" >/dev/null || { echo "unknown stage: $s" >&2; exit 1; }
  if [[ "$TAG" == smoke && "$s" =~ ^(gptq|gptq_eval|check)$ ]]; then continue; fi
  if [[ "$TAG" == smoke ]]; then rm -rf "$QAT_OUT"; rm -f "$LOGS/smoke.$s.done"; fi
  run "$s" "s_$s"
done
