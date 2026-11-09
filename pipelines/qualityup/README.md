# qualityup: LUT4 late-MLP weights, 32K cap, body-only drafter

These weights follow release 0.2's GPTQ + QAT recipe ([docs/QUANTIZATION_TRAINING.md](../../docs/QUANTIZATION_TRAINING.md)) with one change in the plan. The MLP matrices of layers 24–63 use 4-bit lookup tables (`LUT4 per-tensor + pcs`) instead of 3-bit vector tables. The plan is [`mix25in_vq3pA_lut4L24-63.json`](../../configs/quantization/mix25in_vq3pA_lut4L24-63.json); make others with [`scripts/make_lut4_plan.py`](../../scripts/make_lut4_plan.py).

Why these layers: in the in-domain MLP band sweep (`QUANTIZATION_NOTES.md`), layers 24–63 hold 89% of the MLP quantization error. A LUT4 matrix keeps about a third of a 3-bit table's GPTQ error (`docs/QUANTIZATION.md`).

Expected cost on M6 (estimated from per-layer measurements, not yet measured as a whole build):
- about +3.1 GB of Neural Engine memory;
- +8.8 ms per verify call, about 7% slower decode;
- prefill unchanged.

At a 32K cap with the body-only `all3` drafter, release 0.2.1 measured 19.3 GB total on a 32 GB M6, so the estimate is about 22.4 GB.

**Status: no weights yet.** Quality and speed are unmeasured. Publish weights only after the KL-512 gate below passes.

## 1. Rent a machine

- 1× **RTX PRO 6000 Blackwell (96 GB)**, Linux, NVIDIA driver for CUDA 12.8 or newer
- **≥ 128 GB system RAM**: the training step loads the 54 GB BF16 model into RAM before quantizing it
- **≥ 400 GB free disk**: the checkpoint, corpus, teacher cache and two exports
- SSH access

## 2. Set up (once)

```sh
git clone <this repository> anemll-forge && cd anemll-forge
bash pipelines/qualityup/setup_cuda.sh      # venv, CUDA PyTorch, fla, BF16 checkpoint (pinned revision), WikiText-2
```

**Optional:** add agentic calibration rows from your own Pi sessions. Render them on your Mac with only the sessions you choose; the session text goes to the rented machine. Then copy the result to `$WORK/data/calib_pi_ids.npy`:

```sh
SESSIONS_GLOB="$HOME/.pi/agent/sessions/<chosen project>/*.jsonl" MODEL=<dir with the Qwen tokenizer> \
  MAX_ROWS=48 OUT=calib_pi_ids.npy python scripts/qwen38_calib_pi.py
```

Without this file, 48 rows spread over the generated corpus take their place (it includes 130 tool-use prompts).

## 3. Train

```sh
P=pipelines/qualityup/run_cuda.sh
bash $P data                          # chat calibration, 527-prompt corpus, dev + selection traces, teacher cache
bash $P gptq gptq_eval check          # GPTQ with the plan; the parity check gates QAT
STEPS=5 EVAL_EVERY=5 TAG=smoke bash $P train   # 5-step trial: seconds per step and peak GPU memory
bash $P train materialize eval pack   # 600 QAT steps; KL on both traces; $WORK/qualityup_full.tar
# optional matched control, the release 0.2 plan on the same data and teacher:
TAG=control PLAN=configs/quantization/mix25in_vq3pA.json bash $P model
```

- Finished stages are skipped on a rerun, and each stage logs to `$WORK/logs/`.
- `eval` prints mean, median and p99 KL, top-1 agreement and perplexity for each export on the dev trace (the 64 built-in prompts) and the selection trace ([48 disjoint prompts](../../configs/kl/qualityup_selection_prompts.json)).
- QAT picks its best checkpoint on the selection trace.
- The control isolates the plan change from the new data. The published release 0.2 export isn't distributed, so it can't be scored here.

## 4. Convert, compile and gate on the M6 (macOS 27)

Run these from this repository's checkout. Copy `qualityup_full.tar` to the M6 first.

1. **Convert** with release 0.2.1's runtime settings: INT8 KV cache and contexts up to 32K. Match the attention settings recorded in the 0.2.1 build's `manifest.json`, and start with a single chunk as in [docs/WORKFLOW.md](../../docs/WORKFLOW.md):
   ```sh
   MODEL=<as in docs/WORKFLOW.md step 5> EXPORT_DIR=<untarred>/qualityup_full OUT=<builds> \
     python coreai/qwen38_coreai_build.py all --kv-cache-dtype kv8 --ctx 8192,16384,32768 --pctx 8192,16384,32768
   ```
2. **Drafter:** copy release 0.2.1's body-only drafter into the bundle (`drafter/all3/`):
   ```sh
   hf download anemll/anemll-forge-qwen3.8-27B-192K --include "drafter/all3/*" --local-dir <bundle>
   ```
   It uses the target's own output head, so it pairs with the new head. It was distilled against release 0.2's target, so check its tokens per verify.
3. **Compile and serve:** `python forge.py compile --ctx 32K --build <build>`, then `scripts/qwen38_server.sh start`. In this repository the wrapper defaults to `CTX=32K` and picks `drafter/all3/` when it exists.
4. **Quality gate:** run `scripts/m6_kl512_eval.py run` for the release 0.2 build and for the new build with the same `--ref-dir <untarred>/kl_dev`. Publish only if the new build's KL-512 and top-1 agreement are better.
5. **Memory and speed:** record the process and Neural Engine memory at 32K, and decode speed against release 0.2 with the same drafter.

Then upload the bundle following [docs/HUGGING_FACE.md](../../docs/HUGGING_FACE.md) and replace the estimates in this file with the measured numbers.
