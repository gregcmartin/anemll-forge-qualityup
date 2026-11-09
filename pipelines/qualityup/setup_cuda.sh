#!/usr/bin/env bash
# One-time setup of a rented Linux CUDA machine for the qualityup weights.
#   bash pipelines/qualityup/setup_cuda.sh
# Machine: 1x RTX PRO 6000 Blackwell (96 GB), >= 128 GB system RAM, >= 400 GB free disk, NVIDIA driver for CUDA 12.8+.
# Creates .venv-cuda (Python 3.11, CUDA PyTorch, the quantization requirements, flash-linear-attention, accelerate),
# downloads the pinned BF16 checkpoint and WikiText-2, and runs the CPU quantization fixtures.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-$HOME/qualityup}"
MODEL="${MODEL:-$WORK/Qwen3.8-27B}"
QWEN_REV="${QWEN_REV:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"   # the revision release 0.2 was fitted from
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
VENV="$ROOT/.venv-cuda"
mkdir -p "$WORK"/{data,runs,logs,wikitext}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -g | head -2
df -h "$WORK" | tail -1

if [[ ! -x "$VENV/bin/python" ]]; then
  if command -v python3.11 >/dev/null; then
    python3.11 -m venv "$VENV"
  elif command -v uv >/dev/null; then
    uv venv --python 3.11 "$VENV"
  else
    echo "need python3.11 or uv on PATH" >&2; exit 1
  fi
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch --index-url "$TORCH_INDEX"
"$PY" -m pip install -q -r "$ROOT/requirements-quantization.txt" accelerate flash-linear-attention \
  "huggingface_hub[cli]" pyarrow
"$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}, {p.name}, {p.total_memory / 2**30:.0f} GiB, sm_{p.major}{p.minor}")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); (x @ x).sum().item()
import fla, transformers  # noqa: F401  (fla: fast DeltaNet forward/backward)
print("transformers", transformers.__version__, "| flash-linear-attention ok")
EOF

HF="$VENV/bin/hf"
if [[ ! -f "$MODEL/config.json" ]]; then
  "$HF" download Qwen/Qwen3.8-27B --revision "$QWEN_REV" --local-dir "$MODEL"
fi
if [[ ! -f "$WORK/wikitext/wiki2_test.txt" ]]; then
  "$HF" download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/*" --local-dir "$WORK/wikitext/raw"
  WORK="$WORK" "$PY" - <<'EOF'
import os, glob, pyarrow.parquet as pq
w = os.path.join(os.environ["WORK"], "wikitext")
for split in ("train", "test"):
    files = sorted(glob.glob(f"{w}/raw/wikitext-2-raw-v1/{split}-*.parquet"))
    text = "".join(t for f in files for t in pq.read_table(f).column("text").to_pylist())
    open(f"{w}/wiki2_{split}.txt", "w").write(text)
    print(split, len(text), "characters")
EOF
fi

cd "$ROOT"
KMEANS_THREADS=1 "$PY" -m unittest discover -s tests -p 'test_quantization*.py' 2>&1 | tail -3
echo "setup done: WORK=$WORK MODEL=$MODEL; next: bash pipelines/qualityup/run_cuda.sh all"
