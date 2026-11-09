# Speed and quality review for a 32 GB M6 Mac mini (8 November 2026)

This review covers Qwen3.8-27B on the release 0.2 / 0.2.1 Core AI packages. It has three parts: where prefill and decode time goes, what this change set does about it, what is left, and why benchmark scores trail Q4 and Q6 quantizations.

**How the initial review was checked.** The analysis below was done on an M5 Max running macOS 26.6. That OS has no `CoreAI.framework` at runtime, so no model ran during the initial review. ANE times came from the repository's earlier M6 measurements, cited below. Host-side (CPU) times were measured on the M5 Max with the release tensor shapes: vocab 248,320, hidden 5,120, 16 attention layers, a V8 cache with transposed keys, T=8, TP=64. The speed changes were subsequently [measured end to end on an M6 Mac mini](../README.md#latest-m6-mac-mini-benchmark), with three trials of each 2K/32K prose/coding workload. The estimates in this document remain estimates; the measured medians and raw numeric records are linked from the README. The A/B commands are under [Measure on the M6](#measure-on-the-m6).

## Where the time goes on M6

These figures are from `TODO.md` #6, `docs/verifier_len.md`, `docs/research/M6_COMPUTE_ACCELERATION_2026-10-03.md` and `DFLASH2_ANE_PLAN.md`.

| Step | Time |
| --- | --- |
| Verify call (16 chunks + head, 8 rows), default graph | 91.5 / 97.4 / 108.1 / 119.5 / 131.5 ms at 8K / 16K / 32K / 48K / 64K |
| 64-row prefill call | 202.8 / 230.9 / 285.1 / 338.1 / 394.1 ms (316 → 162 rows/s) |
| Output head (h8, 0.63 GB LUT4) | about 6 ms (drafter with head 15.3–15.9 ms, body only 9.5 ms) |
| Drafter call | 15–17 ms per cycle |
| Decode cycle at 8K | 114 ms = drafter 15.3 + verify 96.6 + sampling 1.9 + host 0.6 |
| Decode cycle, owner's coding session | verify + 22.7 ms (drafter 16.7 including the 3 ms gap, sampling 3.8, host 1.8, bookkeeping 0.3) |
| Acceptance, agent traffic | 3.31 tokens per verify (JSON 7.75, thinking 5.26, code 3.88, prose 2.24) |
| Core AI overhead per call | 0.22 ms; 17 calls per verify, 1–2 per draft |

Decode is about 98% serial ANE work. Any speedup has to do one of three things: remove ANE calls, make each cycle yield more tokens, or remove host time that falls between ANE calls. Prefill is compute-bound on the ANE: the MLP runs at about 24 TFLOPS at 64 rows, and 128-row prefill was 21–45% slower per row.

## Changes in this set

| Change | Where | Saving (measured on host / estimated on M6) |
| --- | --- | --- |
| **Prefill runs the output head only for the prompt's last block** | `qwen38_coreai_model.py` `feed` / `prefill_block` / `call` (`logits=`); `qwen38_server.py` `prefill` | One head call (~6 ms ANE) per 64-row block. Estimated **~3% prefill at 8K** (202.8 → ~197 ms per block), ~1.5% at 64K, ~1–2% for the 32-row 128K/192K entries. Earlier blocks' logits were computed and discarded. |
| **Prompt-lookup (n-gram) drafts next to DFlash2**, on by default (`--ngram-draft 6`, `NGRAM_DRAFT=0` turns it off) | new `scripts/ngram_draft.py`; `qwen38_server.py` `draft_cycle` | A lookup cycle skips the drafter call (15–17 ms) and the configured dispatch gap. While the output copies context (edit old text, file paths, re-emitted code, tool JSON), it verifies drafts that are often fully accepted (8 tokens per verify). Searching costs ≤ 0.035 ms at 192K tokens. Lookup candidates are verified by the target, but the matched M6 run found only 7/12 full greedy responses byte-identical. The speed contribution of lookup itself has not been isolated; see the [M6 results](../README.md#latest-m6-mac-mini-benchmark). |
| **Sampling top-k on FP16 rows** (`torch.topk`, one thread) | `qwen38_server.py` `top_indices`, `dist` | 8 rows: **10.1 → 1.9 ms**. The server evaluates rows up to the first rejection plus the bonus row (~4 rows), so the sampling phase drops from about 3.8 to about 1 ms per cycle. The 3 ms drafter gap hides part of that. |
| **Greedy argmax on FP32-widened rows** | `argmax` | 8 rows: 2.1 → 0.7 ms (exact, same first-maximum rule). |
| **Incremental detokenization** | `IncrementalDetokenizer`, `_generate` | The whole reply was decoded again for every token: 0.68 ms per token at 8K output tokens and 1.33 ms at 16K, which is 2–4.4 ms per cycle on long thinking replies. Now it is **0.003 ms per token**. The concatenated deltas equal `tok.decode(all)` (302/302 checks with the real tokenizer). It also fixes a streaming bug: a multi-byte character split over two tokens used to be streamed as U+FFFD and never corrected. Stop strings are searched only in the new text. |
| **Drafter top-k on the raw FP16 head output** | `dflash2_coreai_drafter.py` `propose` | No 7 × 248K FP32 widening and multiply per cycle (1.56 → 1.27 ms). Same candidates and unary scores, because the row factor is positive. Full logits are built only when a caller reads `info["logits"]` (research scripts, `forge.py quick-test`). |
| **Lazy drafter context queue** | `add_context(..., flush=False)`, `flush()` | Needed for lookups. Rows committed by lookup cycles go to the drafter in full 64-row context calls, respecting the configured `DRAFT_GAP` after a verify. In normal DFlash2 cycles nothing changes. |

### What prompt-lookup drafts can do

At 8K a DFlash2 cycle is about 117 ms for 3.31 tokens (28 tok/s). A lookup cycle is a verify plus host time, about 100 ms. It breaks even at about 1.8 accepted drafts and reaches 80 tok/s when all 7 are accepted. Every lookup draft is still verified by the target, so a wrong lookup costs only the difference between its acceptance and DFlash2's. A lookup is used only when the context's suffix (at least 6 tokens, ending at the anchor) occurred earlier. The required match length doubles after a lookup with fewer than 2 accepted tokens and halves after a fully accepted one. This is the same idea as prompt-lookup decoding and SuffixDecoding, which report their largest gains on agentic and code-editing traces. Expect little change on prose and fresh reasoning, and the most on file edits and repeated structures.

**Replay estimate.** A coding-agent transcript (45K assistant tokens, mostly tool calls, no thinking text) was re-tokenized with the Qwen3.8 tokenizer and replayed with a 32K context window. The replay assumed the target writes the recorded text, and DFlash2 cycles were credited the 3.31 tokens per verify measured on agent traffic. The 8K cycle costs above were used. Lookups covered 24% of tool-call tokens at 6.25 tokens per verify, and 0.5% of prose tokens. That gives **28.9 → 33.1 tok/s (+15%) on tool-call output** if DFlash2 accepts only its average on the copied spans. If DFlash2 copies those spans as well as the lookup does, the gain is **+1%**, which is only the skipped drafter calls. Prose came out at −1%. A dedicated M6 test on repeated tool output is still needed to measure this effect. Compare `lookup ... tok/verify` with the request's overall `tok/verify`.

**Host savings and the drafter gap.** In the original 3 ms setting, host work after a verify (sampling, detokenization, bookkeeping) overlapped the wait before the next DFlash2 call. Host savings below that floor therefore did not shorten a DFlash2 cycle. With sampling, the post-verify work was about 6 ms (the owner's session measured 5.9) and is now about 2 ms. That saves about 3 ms, plus 0.3 ms of drafter top-k, so ≈ **+3% decode** in that earlier setting. The measured 1 ms M6 setting follows below.

## Verification

- New tests: `tests/test_ngram_draft.py` checks the lookup against brute force on 400 random sequences, plus the history and the adaptive threshold. `tests/test_server_decode.py` covers the detokenizer (valid, split and invalid UTF-8), top-k and argmax against the previous NumPy forms, `dist` against the previous implementation, the drafter's lazy queue, lazy `info["logits"]` (including `.get`), FP16 top-k equivalence, and whole greedy and sampled generations with a fake target and drafter. Lookup cycles produced the same output with zero drafter calls and context flushed in full blocks. `tests/test_prefill_rows.py` checks that only the last block runs the head.
- Full suite: 312 tests. The only failures are the 4 that also fail without these changes: three need `sklearn`, a quantization-only dependency, and one is `test_server_wrapper`.
- Ties: with FP16 logits, two tokens can tie at the k-th position. `torch.topk` and `np.argpartition` may then keep different tied tokens. Both are a valid top-k.
- The Swift bridge is unchanged.

## Measure on the M6

```sh
# Same bundle and context, then compare the request log lines and /health between runs
NGRAM_DRAFT=0 CTX=16K scripts/qwen38_server.sh restart    # this change set without lookups
CTX=16K scripts/qwen38_server.sh restart                  # with lookups (default)
# Prefill: compare "prefill: N tok in Xs (Y tok/s)" with the upstream commit 703131e on the same prompt.
# Decode: "tok/verify", "ms/cycle (draft .. verify .. sample .. ctx .. host)" and "lookup n/cycles".
```

Use a recorded Pi session or `scripts/m6_server_bench.py` traffic. Synthetic repeated prompts overstate lookup gains.

## Remaining opportunities (not implemented), ranked

1. **Pick the bundle and context for 32 GB.** Up to 64K, release 0.2 prefills about 15% faster than the 0.2.1 192K packages at short contexts (`TODO.md` #6). Verify costs 0.71 ms and prefill 3.4 ms per 1K of entry capacity (not of position), so a lower `CTX` is faster at every position. Release 0.2 at 64K wires about 25.4 of 32 GB with the drafter and is close to swapping. 16K–32K is the speed sweet spot.
2. **Ladder gap entries** (`TODO.md` #5). Calls cost by entry capacity, so a 24K entry saves about 2.6 s on a 32K cold prefill, and 40K/56K entries would help decode at those lengths. The cost is compile time and scratch memory.
3. **DFlash2 dispatch gap, now measured.** On M6, six paired 2K requests (three coding and three prose) at 3 versus 1 ms returned byte-identical outputs and the same verify-cycle counts. Median decode rose 2.65% for coding and 3.04% for prose; median complete request time fell 0.63% and 1.69%. One 32K prose pair gained 2.33% decode but full request time changed by only 0.07%, because prefill dominates. Zero versus 1 ms added under 1% decode and slightly increased complete request time in six paired short runs. The server now defaults to 1 ms on M6 and retains 3 ms on other chips; `DRAFT_GAP_MS` overrides either. The earlier 3 ms rule was from a different greedy run and remains relevant when testing other models or runtimes.
4. **GPU drafter, measured and rejected.** With the same DFlash2 package and ANE target, a direct M6 GPU run took about 240 ms per drafter cycle versus about 15 ms on the ANE. Short coding/prose decode fell to 13.5/8.0 tok/s from a paired ANE control at 42.2/25.1 tok/s. Prefill also fell from about 308 to 286 tok/s. The GPU specialization was removed from this drafter's cache and the ANE package recompiled; the server continues to force ANE drafting.
5. **Multi-draft verify** (`TODO.md` #8). It costs about 2 ms per extra row. Tree verification is hard with Gated DeltaNet because state updates are sequential along a path. Two independent 8-row branches would need a batched GDN graph.
6. **Short prompt tails.** A final block of 9–16 tokens is cheaper as two 8-row verify calls (2 × 91.5 ms) than one 64-row call (202.8 ms) at 8K; at 64K the same holds up to 24 tokens. The saving is small, once per request.
7. **Already ruled out** in this repository's research; don't redo: reduced draft vocabularies (lose more acceptance than they save), T=4/3 verifiers, 128-row prefill, 8/12-layer chunks, KV writes inside the graph, W8A8 MLP, INT8 attention matmuls beyond what C2T ships, Kronecker Hadamard, other softmax forms.

The six-token prompt-lookup threshold also beat a four-token A/B on the same six short prompts: coding was effectively unchanged and prose decode was about 0.7% slower at four tokens. Keep six by default. For the current 32K test prompts, the next substantial prefill improvement requires a new intermediate context entry; the published release 0.2 bundle has 8K, 16K, 32K, 48K and 64K entries, and the prompt plus output crosses the 32K boundary. This is a graph build and compile project, not a runtime flag.

## Would a Rust port help?

Inference runs from Python, but Python only orchestrates. The 16 chunk programs, the head and the drafter run on the ANE through Core AI, called from Python with `ctypes` through the Swift bridge (`coreai/swift_bridge`). Buffers are IOSurfaces bound once, so no tensors are copied per call. With this change set, host time is about 1–3 ms of a 100–150 ms cycle, and part of it already hides inside the 3 ms drafter gap. Per prefill block it is about 2 ms of 200–400 ms. A Rust port could win back at most that share: about 1–2% of decode and under 1% of prefill. Core AI is a Swift-only API, so Rust would still call it through a Swift shim. The per-call overhead (0.22 ms × 17–19 calls per cycle) sits inside Core AI and would not change. A port is not worth it for speed.

## Why benchmark scores trail Q4 and Q6 quantizations

**The quantization is the main cause.** The other ANE-specific parts are not.

| | Release 0.2 (ANE) | UD-Q5_K_XL GGUF | UD-IQ3_XXS GGUF (3.16 bpw) |
| --- | ---: | ---: | ---: |
| Average bits per weight | **3.17** (9.44 GiB) | about 5–6 (not recorded) | 3.16 |
| KL-512 vs BF16 (same 64-chat trace) | **0.0518** | 0.0033 | 0.1625 |
| Top-1 agreement with BF16 | **91.9%** | 97.7% | 89.4% |

Sources: `docs/QUANTIZATION.md` (release 0.2 results) and `docs/research/M6_COMPUTE_ACCELERATION_2026-10-03.md` (the GGUF comparison). A Q4_K_M or Q6_K was not measured on this trace.

- **Bits.** The MLP (17.1B weights, 67% of the model) is a 3-bit vector LUT in all 64 layers. The DeltaNet and attention projections are 2-bit in layers 0–23 and 4-bit in layers 24–63, with rank-64 FP16 corrections. The head is LUT4. K/V projections are INT8. Q4_K_M uses about 4.8 bpw (~15 GB) and Q6_K about 6.6 (~21 GB). Release 0.2 stores about two-thirds of Q4_K_M's bits and about half of Q6_K's.
- **Everything else is small.** These are direct KL measurements between builds: V8 vs FP16 cache 7e-5, kv8 vs V8 7e-5, 8-bit attention (C2/C2T) 4e-4, faster graph 3e-5. After the DeltaNet numerics fixes (`ANE_DELTANET_NUMERICS.md`), ANE fp16 numerics match PyTorch on the same weights: 0.0518 vs 0.0515.
- **For its size the model is good.** At equal bits it beats UD-IQ3_XXS on chat KL by 3x (0.052 vs 0.163) and on the long tail (p99 0.50 vs 3.34). UD-IQ3_XXS wins on WikiText KL (0.051 vs 0.071), and long-context KL (16K windows) is about 10x the chat KL for every model.
- **No task benchmarks exist yet.** `docs/BENCHMARK_PLAN.md` is a plan (IFEval, HumanEval+, MATH-500, Terminal-Bench) with no runs. Check your benchmark harness against the server defaults. These are not properties of the weights:
  - **Thinking is capped at 2,048 tokens** when `max_tokens` is the default 4096, then force-closed.
  - **`presence_penalty` defaults to 1.5 without thinking** (the docs said 0; now corrected).
  - The loop guard stops at 3 repeats over 1,536 tokens, and thinking is off unless requested.
  - Greedy runs (`temperature: 0`) are unaffected by the penalties but still get the thinking cap. See the benchmarking note in `docs/SERVER.md`.

### Quality levers that fit a 32 GB M6

Each lever needs GPTQ + QAT (about 4 h on one 96 GB GPU, `docs/QUANTIZATION_TRAINING.md`), a package rebuild and a 25-minute ANE compile.

1. **Late MLP layers at LUT4 (4 bpw).** Most of the KL is in the MLP and grows with depth: the 2-bit band sweep rises from 0.008 in layers 0–7 to 0.069 in layers 56–63. `QUANTIZATION.md` names late-MLP bytes as the strongest remaining lever. On the first release's formats, `u15` (all 15 late two-bit MLP layers at LUT4) reached KL 0.0521, against 0.0897 for the same recipe without them (`n1`). Its packages needed 15.0 GB of ANE memory, against 13.3 GB. Release 0.2's 3-bit MLP reached the same KL (0.052) without the extra memory, so the next step is LUT4 on top of it. On 0.2 weights, all 64 MLP layers at LUT4 would add about 2.1 GB (8.6 vs 6.4 GB). On M6, LUT4 costs 0.365 ms per MLP layer per verify vs 0.145 for the 3-bit table, so the last 32 layers at LUT4 cost about +7 ms per verify (~7% at 8K) and all 64 about +14 ms. It fits at 16K–32K on 32 GB; at 64K it does not (`QUANTIZATION.md`).
2. **Mixers in layers 0–23 from 2-bit to LUT4.** About +0.5 GB. KL effect not measured.
3. **Longer QAT.** Free at inference. The best checkpoint was the last one (step 600 of 600), and only one seed was run.
4. **A per-tensor mix of 2x16 / 2x64 / LUT4 chosen by sensitivity.** Not yet tried.

Q6-level fidelity (about 21 GB of weights) does not fit next to the ANE scratch, the drafter and the OS on a 32 GB machine. A middle setting such as late-MLP LUT4 is the realistic target, at a decode cost of about 7–15%.
