# qualityup: LUT4 late-MLP weights, 32K cap, body-only drafter

These weights follow release 0.2's GPTQ + QAT recipe ([docs/QUANTIZATION_TRAINING.md](../../docs/QUANTIZATION_TRAINING.md)) with one change in the plan. The MLP matrices of layers 24–63 use 4-bit lookup tables (`LUT4 per-tensor + pcs`) instead of 3-bit vector tables. The plan is [`mix25in_vq3pA_lut4L24-63.json`](../../configs/quantization/mix25in_vq3pA_lut4L24-63.json); make others with [`scripts/make_lut4_plan.py`](../../scripts/make_lut4_plan.py).

Why these layers: in the in-domain MLP band sweep (`QUANTIZATION_NOTES.md`), layers 24–63 hold 89% of the MLP quantization error. A LUT4 matrix keeps about a third of a 3-bit table's GPTQ error (`docs/QUANTIZATION.md`).

Measured cost on M6 (whole build, against release 0.2 with the same drafter and prompts; [details below](#m6-results-9-november-2026)):
- +8.5 ms per verify call at 2K context (+5.2 ms at 31.5K), against the +8.8 ms estimated from per-layer measurements;
- prefill 3–4% slower, where no change was expected;
- 21.6 GB total at the 32K cap with the body-only `all3` drafter. The per-layer estimate was 22.4 GB, from release 0.2.1's 19.3 GB.

**Status: trained, converted, gated and published on 9 November 2026.** The M6 KL-512 gate passed: mean KL 0.0500 against 0.0515 and top-1 agreement 91.75% against 91.42%. Only the top-1 gain is statistically significant. Weights: [gregcmartin/anemll-forge-qwen3.8-27B-qualityup](https://huggingface.co/gregcmartin/anemll-forge-qwen3.8-27B-qualityup), revision `2bcb0b930f4188a21897f5b4005acc223d148a54`, with the quantized export under `export/qualityup_full/`.

## Training run, 9 November 2026

Hardware: 1× RTX PRO 6000 Blackwell Workstation (96 GB), rented on Vast.ai at $1.72/h. Calibration used no Pi session rows; 48 corpus rows took their place.

| Stage | Wall time |
| --- | ---: |
| Data: chat calibration 5 min, corpus 37 min, traces 9 min, teacher cache 3 min | 54 min |
| GPTQ with the plan | 58 min |
| GPTQ eval | 15 min |
| Parity check (0.39% relative, tolerance 2%) | 3 min |
| QAT, 600 steps: 13 s per step, 45.3 GiB peak GPU memory, evaluations every 50 steps included | 199 min |
| Final eval and pack | 15 min |

The pipeline took about 6 hours. The whole rental, including setup and download, took about 7 hours and cost $12.85.

KL against BF16 Qwen3.8-27B, measured on CUDA with top-512 reference traces. This is **not** the M6 KL-512 gate:

| Export | Trace | Mean KL | p99 KL | Top-1 agreement |
| --- | --- | ---: | ---: | ---: |
| GPTQ only | dev (64 prompts, 40,090 tokens) | 0.0472 | 0.736 | 93.0% |
| GPTQ + QAT | dev | **0.0297** | **0.277** | **93.8%** |
| GPTQ only | selection (48 prompts, 33,250 tokens) | 0.0451 | 0.693 | 93.2% |
| GPTQ + QAT | selection | **0.0289** | **0.276** | **94.0%** |

- QAT lowered mean KL by 37% and p99 KL by 62% relative to GPTQ alone.
- QAT kept step 550, which had the lowest selection-trace KL: 0.02897, against 0.02912 at step 600.
- The quantized weights total 10.69 GiB, or 10.98 GiB with the low-rank factors.
- These numbers can't be compared directly with release 0.2. Its PyTorch export isn't distributed, so it can't be scored on these traces. The M6 gate in step 4 below makes that comparison.

## M6 results, 9 November 2026

Mac mini M6, 32 GB, macOS 27. Conversion settings: release 0.2.1's graph (kv8 cache, 8-bit attention with FP8 softmax and PV, transposed keys), contexts 8K/16K/32K. The conversion took 24 minutes; compile overlapped it, at about 22 s per chunk.

**KL-512 gate.** Both builds were scored on the CUDA run's BF16 top-512 dev trace: 64 prompts, 40,090 positions, at the 8K entry. Release 0.2 is the published package, with its V8 cache. The paired bootstrap resamples the 64 prompts 20,000 times.

| | qualityup | release 0.2 | qualityup − release 0.2, paired 95% CI |
| --- | ---: | ---: | --- |
| Mean KL | **0.0500** | 0.0515 | −0.0044 to +0.0013 (not significant) |
| Median KL | 0.0093 | 0.0093 | |
| p99 KL | **0.445** | 0.513 | |
| Top-1 agreement | **91.75%** | 91.42% | +0.06 to +0.60 points (significant) |
| Perplexity (BF16: 2.149) | 2.141 | 2.144 | |

- Qualityup had lower KL on only 30 of the 64 prompts. Mean KL between the two builds is 0.073, with 89.8% top-1 agreement.
- On CUDA, QAT took this export to 0.030. On the M6, the graph's own approximations (tanh SiLU, INT8 attention, quantized cache) add error that both weight sets share. That shared error narrows the gap.

**Speed.** Method:
- 3 prompts each of prose (this repository's Markdown) and code (its Python), at about 2K and 31.5K tokens;
- identical prompts for both models (same SHA-256);
- greedy decoding, 256 output tokens, thinking off;
- no prompt-cache reuse, 32K cap, the body-only `all3` drafter.

Rates come from the server's own log.

| | qualityup | release 0.2 |
| --- | ---: | ---: |
| Prefill, 2K prompts | 291 tok/s | 300 tok/s |
| Prefill, 31.5K prompts | 269 tok/s | 280 tok/s |
| Verify call, 2K context | 114.5 ms | 106.0 ms |
| Verify call, 31.5K context | 125.0 ms | 119.8 ms |
| Decode, 2K (tokens per verify) | 27.8 tok/s (3.18) | 30.9 tok/s (3.27) |
| Decode, 31.5K (tokens per verify) | 24.7 tok/s (3.09) | 24.6 tok/s (2.95) |

Decode speed moves with acceptance, which depends on the text each model writes. The verify-call time is the like-for-like cost.

**Memory** at the 32K cap. Total = wired memory above the 2.2 GB no-model baseline, plus the server footprint (release 0.2.1's method).

| | qualityup (kv8) | release 0.2 (V8) |
| --- | ---: | ---: |
| Wired after load | 18.6 GB | 16.0 GB |
| Peak wired, 31.5K prompts | 21.3 GB | 22.5 GB |
| Peak server footprint | 2.5 GB | 3.1 GB |
| **Total at peak** | **21.6 GB** | 23.4 GB |

Release 0.2's V8 cache keeps keys in FP16, which is why its long-context peak is higher. Release 0.2.1 (release 0.2's weights with kv8) reports 19.3 GB, so qualityup costs about 2.3 GB more at equal cache format; the two were not measured side by side. Swap did not grow.

## 1. Rent a machine

- 1× **RTX PRO 6000 Blackwell (96 GB)**, Linux, NVIDIA driver for CUDA 12.8 or newer
- **≥ 128 GB system RAM**: the training step loads the 54 GB BF16 model into RAM before quantizing it
- **≥ 400 GB free disk**: the checkpoint, corpus, teacher cache and two exports
- SSH access

## 2. Set up (once)

```sh
git clone https://github.com/gregcmartin/anemll-forge-qualityup.git anemll-forge && cd anemll-forge
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
   The published build used release 0.2's numerics and 8-bit attention settings, from `release/huggingface-export/EXPORT.md`, without the M5 function set. Set these in the environment of the command above:
   ```sh
   SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 QCONV_INT8=0 \
   ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 ATT_INT8MM=s8,s8b,sm8,pvf8 KV_KEYS_T=1
   ```
   `MODEL` only needs the small per-layer tensors and the final norm, plus `config.json`. The quantized matrices and head come from the export.
2. **Drafter:** copy release 0.2.1's body-only drafter into the bundle (`drafter/all3/`):
   ```sh
   hf download anemll/anemll-forge-qwen3.8-27B-192K --include "drafter/all3/*" --local-dir <bundle>
   ```
   It uses the target's own output head, so it pairs with the new head. It was distilled against release 0.2's target, so check its tokens per verify.
3. **Compile and serve:** `python forge.py compile --ctx 32K --build <build>`, then `scripts/qwen38_server.sh start`. In this repository the wrapper defaults to `CTX=32K` and picks `drafter/all3/` when it exists.
4. **Quality gate:** run `scripts/m6_kl512_eval.py run` for the release 0.2 build and for the new build with the same `--ref-dir <untarred>/kl_dev`. Publish only if the new build's KL-512 and top-1 agreement are better.
5. **Memory and speed:** record the process and Neural Engine memory at 32K, and decode speed against release 0.2 with the same drafter.

Then stage and upload the bundle following [docs/HUGGING_FACE.md](../../docs/HUGGING_FACE.md), with the body-only drafter at `drafter/all3/` and no other drafter files. `release-manifest` records it with `head_export: null` and `head_norm`.
