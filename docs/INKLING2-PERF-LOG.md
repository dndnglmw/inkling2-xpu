# Inkling 2 Mini on 2x Arc Pro B65 — performance log

Written 2026-11-22. Companions: `SETUP/INKLING2-GO-LIVE.md` (operations), `SETUP/INKLING2-DECISIONS.md`
(decisions and rejected ideas). Numbers are recorded as measured; nothing is rounded to improve a story.
Where two figures disagree, both are given with their sources.

## 0. Conventions and caveats

* Hardware: 2x Arc Pro B65, tensor parallel 2. GPU0 is on PCIe Gen5 x16 (46-57 GB/s measured), GPU1 on a PCIe Gen4
  x4 link (7 GB/s measured) (`INKLING2_XPU_HANDOFF.md` Session 3).
* Model: native NVFP4 experts (layers 0-19) and MXFP8 (layers 20-23), rest bf16; weights 8.9 GiB per card.
* "Clean server" = container `inkling2-tp2` from `/home/operator/work/serve-noprof.sh`: max-model-len 262144,
  max-num-seqs 8, 2048-token chunks, gpu-memory-utilization 0.85, prefix caching on, XPU decode graphs
  (sizes 1,2,4,8), `--kv-cache-dtype fp8`, DSpark depth 3. Deviations are stated per table.
* Dates: Nov 21 = the daytime work (up to and including the 1M-context pass and the far-region fold);
  Nov 21 night / Nov 22 = the prefill, 4096-chunk and fusion work. Exact timestamps of individual runs were not
  recorded; the 4096-chunk server's own log line is `11-22 00:17:54`.
* **Per-launch nondeterminism:** two launches of the same build gave different acceptance for the same prompt
  (tokens/step 3.11 / 2.42 / 3.64 vs 3.17 / 2.59 / 3.65 for code / prose / quote). Inside one launch a repeated
  request reproduces to two decimals. Step-time comparisons across launches therefore carry about +-2 ms; the
  kernel profiler and bitwise unit tests are the deterministic evidence.
* **PROF-mode end-to-end tok/s is not representative:** `INKLING_PROF=1` adds roughly 0.9 s per chunk outside the
  forward pass (850 tok/s measured under PROF vs 1316 on the clean server for the same stage). Per-chunk
  `forward_wall` from PROF is the usable figure.
* Prefill benchmark: `agents/prefill_bench.py`, one cold request, `max_tokens=1`, text = concatenated Python source
  files (diverse routing; repetitive filler gives unrealistically narrow expert routing). Token counts are what the
  server reported as `prompt_tokens`; the labels 32K/128K/241K are the nominal targets.

## 1. Prefill, end to end (tok/s)

| Stage | Config | ~32K | ~128K | ~241K |
|---|---|---|---|---|
| **Baseline, diverse text** (Nov 21) | clean server, random corpus order | **1316** (33,584 tokens, 25.5 s) | **1283** (127,688 tokens, 99.5 s) | **1139** (240,962 tokens, 211.5 s) |
| After controller-GEMM, sconv tile, draft fold; fixed corpus (Nov 21 night) | clean server | 1725 (36,923 tokens, 21.4 s) | 1449 (124,669 tokens, 86.0 s) | 1370 (229,712 tokens, 167.6 s) |
| **After Option A** (Nov 22) | + multi-pointer mix, tiered-MoE glue, attention tile (128,128,32); fixed corpus | **1931** (36,923 tokens, 19.1 s) | **1714** (124,669 tokens, 72.7 s) | **1480** (229,712 tokens, 155.2 s) |
| 4096-token chunks (Nov 22) | `serve-4096.sh`, 4096 chunks, same corpus | **2177** (17.0 s) | **1943** (64.2 s) | **1703** (134.9 s) |

Earlier figures on repetitive **filler** text (not comparable with the table, which uses diverse text):
820 tok/s for a 210,038-token prompt (256 s) and 742 tok/s for 518,738 tokens (699 s) on the 1M-config server
(Nov 21), and 1496 tok/s for 258,338 tokens in 173 s on the 262144 server with the first ~37% of the prompt
prefix-cached (about 940 tok/s on the uncached remainder — derived, not measured separately).

Notes on the table:

* The baseline used a corpus shuffled by time seed; later rows use a fixed shuffle (seed 1234) with only a salt line
  differing, so rows 3-5 are comparable with each other, row 2 only approximately.
* First request after a launch pays JIT compilation: 907 tok/s (40.7 s) for the 36,923-token prompt versus the
  steady 1725 before Option A; 1826 (20.2 s) before 1931 after Option A. Steady values are the repeat runs.
* After Option A vs the row before it: +12% (1725 → 1931), +18% (1449 → 1714), +8% (1370 → 1480).
* 4096 vs Option A: +13% (1931 → 2177), +13% (1714 → 1943), +15% (1480 → 1703).
* **4096-chunk KV pool: 329,284 tokens, reported concurrency 1.26x at 262144 per request** (329,284 / 262,144 =
  1.256), against 641,028 tokens / 2.45x at 2048 chunks (641,028 / 262,144 = 2.445). The guard (pool must hold one
  262144-token request) passed; the concurrency halving is why 4096 is not the production setting.
* Early long runs (1M-config server, Nov 21): a 493,538-token prompt took 499 s (990 tok/s average); 1255 tok/s at
  100K (handoff Session 3).
* The user accepted 1480 tok/s at 241K and 1931 at 32K and closed prefill work.

## 2. Prefill chunk profile (`INKLING_PROF=1`, T=2048, eager, ms per chunk)

| Phase | Before (~30K depth) | After controller GEMM | After sconv tile | After, ~256K depth |
|---|---|---|---|---|
| `draft.ctx_kv` | 5 | 5 | 5 | 5 |
| `pre_attn` | 202 | 68 | 68-69 | 68 |
| `a.proj` | 28 | 27-28 | 27-28 | 27 |
| `a.kvconv` | 26 | 26-27 | 11-12 | 11 |
| `a.norms` | 11 | 11 | 11-12 | 11-12 |
| `a.core` | 240 (239-246) | 257-258 | 277-278 | **953-960** |
| `a.wo` | 11 | 11-12 | 11-12 | 12 |
| `attention` (sum) | 318 | 336-338 | 340-344 | 1018-1024 |
| `attn_allreduce` | 96 | 93-97 | 93-100 | 96-101 |
| `attn_sconv` | 53 | 53 | 11-12 | 11 |
| `pre_mlp` | 98 | 54 | 54-55 | 54 |
| `moe` (includes the MoE all-reduce) | 651 | 646-648 | 640-642 | 583 |
| `mlp_sconv` | 54 | 54-55 | 14 | 14 |
| **`forward_wall`** | **1499** | **1335** | **1253** | **1877** |

At ~256K depth attention core is 960 of 1877 ms (51%).

**The ~550 ms that first looked unattributed** (measured chunk time against MoE + dense + attention + all-reduce
micro-benchmarks) resolved to: loop-controller / attention-residual mixes `pre_attn` 202 + `pre_mlp` 98 = **300 ms**,
short convolutions 53 + 54 = **107 ms**, all-reduce **about 100 ms** (the attention all-reduce, 93-97 ms; the MoE
all-reduce sits inside the `moe` phase). Nothing remained unexplained once the in-engine phase timers were available.

Component cost model used before the PROF split (per 2048-token chunk): tiered MoE 546 ms (11.37 ms x 48 visits);
dense GEMMs 37 ms (qkvr 0.58 + wo 0.19 = 0.77 ms per visit, 80 TFLOP/s); all-reduce 1.84 ms per call for 9.4 MB, x96 =
177 ms (decode-size all-reduce 0.02 ms); attention at context 16,000 / 105,000 / 210,000: local layer
3.56 / 2.61 / 2.60 ms, global layer 12.99 / 44.88 / 85.11 ms (40 local + 8 global visits per chunk = 246 / 463 /
785 ms). Measured chunk times (2048 / tok/s): 1556 ms at 32K, 1596 ms at 128K, 1798 ms at 241K.

## 3. Component micro-benchmarks

**Prefill kernels (T=2048, real shapes):**

| Item | Before | After |
|---|---|---|
| Controller blend, 3 streams | 7.022 ms (stacked per-token kernels) | 1.536 ms (bmm + fused blend/norm, rel err 8.8e-5) |
| Controller blend, 3 streams, expanded dest | 6.956 | 1.386 |
| Controller blend, 1 stream | 2.337 | 0.493 |
| Controller blend, 3 streams, T=300 | 1.033 | 0.226 |
| Controller blend, 1 stream, T=70 | 0.089 | 0.147 (slower, hence the threshold of 128 tokens) |
| Short conv, attention shape | 1.971 ms | 0.246 ms (bandwidth floor 0.047) |
| Short conv, MLP shape (addend + fp32 residual) | 2.013 ms | 0.318 ms |
| Mix op, 3 queries x 5 states + norm (stacked vs multi-pointer, 1024 columns, 4 warps) | 1.864 | 1.540 |
| Mix op, 1 query x 5 states + norm | 1.628 | 0.943 |
| Mix op, 4 queries x 3 states | 1.194 | 0.691 |
| Mix op, MLP shape | 1.569 | 0.544 |
| Prefill attention, local / global layer, ctx 16K, tile (64,64,16) -> (128,128,32) | 3.53 / 12.23 ms | 3.35 / 11.05 ms |

Mix-op timings are noisy (the stacked reference varied 1.16-2.4 ms between runs of the same shape).

**Prefill MoE per visit, real routing, T=2048, slower EP rank (48-visit mean):**

| Path | Mean per visit | x48 per chunk |
|---|---|---|
| tiered dequant + bmm (before glue restructure) | 11.37 ms | 546 ms |
| tiered, after the single global slot table | **10.40 ms** | 499 ms |
| old capacity bucket + Triton overflow | 14.20 ms (14.21 after) | 682 ms |
| pure Triton in-register dequant | 13.39 ms (13.40 after) | 643 ms |

Worst visit: tiered 12.15 ms (11.08 after), old 19.23, pure Triton 16.86. Device-time split of one tiered visit
(profiler): bmm 3.737 ms, bf16 dequant 3.574 ms, output-tier kernel 0.657, SITU 0.468, reduce 0.394, fills 0.255,
zeros 0.244 (about 9.7 ms of device time). Pure-Triton path per kernel: gate/up 6.266 ms, down 3.326 ms (about
14 TFLOP/s). Pure-Triton config sweep (`INKLING_MOE_CFG`): default stages 16.32 ms; `128,64,64,16,2` 11.78;
`128,64,64,8,2` **10.66** (vs 11.12 tiered in the same run); `128,64,128,16,2` 11.43; the rest 13.2-27.1 ms.
Single-visit timings are noisy: later reruns on the same code gave tiered 12.52 / 12.01 ms and the same
`128,64,64,8,2` config 15.85 / 13.93 ms, which is why the 48-visit means above are the figures to use.

**Matmul ceiling check (E=64, cap=128, K=2304, N=2304, bf16 weights, no dequant):** oneDNN bmm 1.79 ms =
**48.5 TFLOP/s** (679 MB weights); Triton `tl.dot` best of seven configs 4.66 ms = **18.7 TFLOP/s**
(`128,64,64,w16,s3`); the others 4.98-8.10 ms (10.7-17.5 TFLOP/s).

**Vendor grouped GEMM, MXFP4 weights (not usable, see DECISIONS §4):** gate/up 1.94 ms = 44.9 TFLOP/s, down
1.38 ms = 31.6 TFLOP/s (3.3 ms per visit versus 7.3 ms for dequant + bmm); on real counts with fp32 output 2.85 ms.

**FP4 decoder variants:** bit-trick e2m1/e4m3 decoders are exact over all 16 / 256 inputs; 48-visit means tiered
10.45 (reference) vs 10.52 (fast), pure Triton 13.41 vs 13.49 — no speedup (repeat: 10.43 vs 10.45, 13.33 vs 13.39).

**Decode-size MoE per visit (ms; random top-8 routing, `agents/decode/bench_expert_dtype.py`):**

| | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| TP2 bf16 / NVFP4 / MXFP8 | 0.323 / 0.192 / 0.217 | 0.413 / 0.169 / 0.188 | 0.519 / 0.262 / 0.305 | 0.946 / 0.516 / 0.578 |
| TP2 bf16 / NVFP4 | 1.68x | 2.43x | 1.98x | 1.83x |
| TP1 bf16 / NVFP4 / MXFP8 | 0.288 / 0.165 / 0.185 | 0.524 / 0.262 / 0.310 | 0.928 / 0.539 / 0.580 | 1.609 / 0.978 / 1.065 |
| TP1 bf16 / NVFP4 | 1.75x | 2.00x | 1.72x | 1.64x |

CORRECTION (Nov 22, micro-bench `agents/moe_micro.py`): the GB/s figures above counted w13 only and skipped
w2 — with w2 included, the NVFP4 decode kernel already runs at ~75-95% of the card's achievable rate
(~454-575 GB/s physical at TP2 T=4 against the 608 GB/s card), matching the bf16 machinery's efficiency;
the dequant costs only ~5% over a raw-load variant (0.249 vs 0.236 ms per visit). The decode MoE is near
its practical floor: logical per-pair re-reads (32 pairs over ~25 unique local experts, L2 deduping ~21%)
put the realistic floor at ~10-11 ms per step vs 12.4 measured — <=2 ms upside. Split-K config sweep
(`INKLING_MOEQ_SPLIT`) at T=2/4/8: default 0.172 / 0.292 / 0.490 ms; no config was better.

## 4. Decode history (single stream unless stated)

| tok/s | What | Source |
|---|---|---|
| 2.0 | eager, bf16 experts, first working port | handoff Session 2b |
| ~9 | eager, native 4-bit experts | Session 2c |
| 26 | XPU full-decode graphs | Session 2d |
| 40.4 short / 23.5 at ~494K | 500K server, bf16 KV, graphs, no speculation | Session 3, Nov 21 |
| 31.4, 41.6 | supplied for this document as the short-context start and an intermediate step; the underlying runs are not in the repo or the handoff log | — |
| 49.3 | plain decode, TP2 graphs, bf16 KV, later in the Nov 21 session (after the Session 3 changes: fp32 LM head, vendor far-key attention, split-K MoE GEMV) | session record |
| 184 aggregate / 8 concurrent streams | plain decode, same stack | session record |
| 22 (TP1) vs 31 (TP2) | decode is not bandwidth/comm bound (31 may be the "31.4" above; not confirmed) | Session 3 |
| **73.9 mean** | DSpark depth 3, bf16 KV, max-model-len 65,536, `spec_k_bench` 4 classes | Nov 21 |
| 72.1 mean | same, depth 5 | Nov 21 |
| **67.3 mean** | depth 3, fp8 KV, max-model-len 1,048,576 server, FP8 dense off | Nov 21 |
| **66.9 mean** | depth 3, fp8 KV, 262144 server, 2048 chunks | Nov 22 |

Per-class short-context results (`spec_k_bench`, 384-token completions, 2 prompts per class; tok/s, tokens per
step, step ms):

| Run | prose | facts | chat | code | mean |
|---|---|---|---|---|---|
| depth 3, bf16 KV, 64K (Nov 21) | 52.5 | 59.4 | 87.3 | 96.2 | 73.9 |
| depth 5 | | | | | 72.1 |
| depth 3, fp8 KV, 1M-config, FP8 dense off | 48.7 / 1.88 / 38.6 | 53.7 / 2.08 / 38.6 | 78.9 / 3.08 / 39.1 | 87.8 / 3.52 / 40.1 | 67.3 |
| same + FP8 dense, run 1 | 49.4 / 1.81 / 36.6 | 55.7 / 2.04 / 36.6 | 81.6 / 3.00 / 36.7 | 73.5 / 3.22 / 43.8 (jitter) | 65.0 |
| same + FP8 dense, repeats | 49.5 / 36.5, 49.3 / 36.6 | 55.7 / 36.6, 55.6 / 36.7 | 81.7 / 36.7, 81.4 / 36.8 | 84.5 / 38.1, 85.0 / 37.9 | 67.8, 67.8 |
| 262144 server, 2048 chunks (Nov 22) | 48.3 / 1.88 / 38.9 | 53.3 / 2.08 / 38.9 | 78.3 / 3.08 / 39.4 | 87.5 / 3.52 / 40.2 | 66.9 |

* The 73.9 (bf16 KV, 65,536 max) vs 67.3 (fp8 KV, 1M-config) difference was never isolated to a cause.
* The `spec_k_bench` code class showed 62.8 ms and 64.2 ms steps on the first run after two fresh launches and
  40.2 ms afterwards (and 43.8 ms once); run alone or in other orders code measured 81.5-94.9 tok/s at
  39.7-40.5 ms. Treated as first-run warm-up (DECISIONS §8).
* Per-class order test (same server): code 81.9 tok/s (3.25 tok/step, 39.7 ms), code2 94.9 (3.84, 40.5), prose 44.6
  (1.73, 38.8), chat 77.2 (3.03, 39.3).
* Step-cost model measured by concurrent decode (TP2, graphs; ms for T tokens): T=1 20.3, T=2 23.9, T=4 30.7,
  T=8 43.5; draft cost about 4.8 ms. Fixed depth 3 sits at the oracle depth derived from per-position acceptance.

## 5. Long-context decode on repetitive filler text (spec depth 3)

**The tokens-per-step figures in this section come from a repeated filler sentence and overstate real text.**
The 3.98 tokens/step below was measured on that filler; real documents give about 3.2 (code) and 2.5 (prose),
section 7.

| Context (tokens) | Build | tok/s | tokens/step | step | Needle |
|---|---|---|---|---|---|
| 210,038 | before far-region fold (1M-config, Nov 21) | 15.2 | 3.81 | **249.6 ms** | found |
| 518,738 | before far-region fold | 4.1 | 2.09 | **515.1 ms** | found |
| 67,210 | after far fold (Nov 21) | 87.4 | 3.91 | 44.8 ms | found at 67,238 |
| 134,410 | after far fold | 79.0 | 3.98 | 50.3 ms | found at 134,438 |
| 220,510 | after far fold | 71.6 | 3.98 | **55.5 ms** | |
| 258,310 | after far fold | 70.0 | 3.93 | 56.2 ms | found at 258,338 |
| 258,310 | after draft fold | 73.9 | 3.97 | **53.7 ms** | |

A 940,000-token needle was started and never completed (the client process died); 1M is ruled out regardless.
Prefix hit rate in the decode calls was 1.000 on every rung and cache-hit overhead 0.3-0.7 s, so the 250 ms was
neither cache misses nor a non-flash fallback.

**Root cause of 250 ms (found Nov 21):** the vendor flash-attention call for the far region splits the KV sweep only when
q_len = 1. At 210,000 tokens (215 MB of fp8 KV per layer per card): q_len 1 = 616 us (349 GB/s); q_len 2, 3, 4, 5, 8, 16
all ~22.5 ms (about 10 GB/s, flat in q_len, so rows do share the sweep but it runs almost serially).
Fix: fold a request's verify rows into the GQA group dimension (one q_len-1 sweep). T=2: 22,598 → 613 us; T=4:
22,610 → 845 us (per-row expansion: 1,204 us); T=8: 22,614 → 1,323 us (expansion 2,259 us). Output error vs the old
path 3.5e-3, LSE error 2.9e-6. Depth gate against an fp32 reference (contexts 64K / 134K / 258K, T=2 and 4):
fold 2.70e-3 to 2.84e-3, old path 2.28e-3 to 2.37e-3, LSE error 9.5e-7 (fold) vs 1.9e-6 to 3.8e-6 (old).
Env switch `INKLING_XPU_FAR_FOLD` (default 1).

Context-dependent cost split at the time of the fix (kernel micro-benchmarks): target global
attention (8 visits) 0.8 / 2.6 / 5.0 ms and drafter global layer 2.3 / 4.8 / 7.9 ms at 67K / 134K / 220K, against a
measured step growth of 5.8 / 11.3 / 16.5 ms over the ~39 ms short-context step.

## 6. Drafter block attention (global draft layer, bf16 draft KV, one layer)

| Context | Original | Group-fold | Retuned fold |
|---|---|---|---|
| 220K | 8.036 ms | 3.442 ms | **2.083 ms** |

Retune (Nov 22) = BN 128, 16 warps, 3 stages, 512 target programs, 128 max splits (`INKLING_DRAFT_*`). Sweep at
220K: the previous default 3.446 ms; bn 128 / 8 warps 6.222; bn 64 / 4 warps 11.770; bn 128 / 16 warps / 2 stages /
256 programs 2.124; with 512 programs 2.092; bn 128 / 32 warps 3.563; bn 256 / 16 warps 6.172; bn 256 / 32 warps
3.195. Correctness test (`test_triton_block_attention.py`) rel error 0.00199-0.00232.
Effect on the 258K filler step: 56.2 → 53.7 ms (70.0 → 73.9 tok/s) — about half the ~5 ms predicted from the kernel timing.
The drafter has 3 layers; layers 0 and 2 are local (1024 window), layer 1 reads the full context.

## 7. Acceptance and step time on diverse documents (the numbers that matter)

Context = Python source-file corpus (`agents/deep4.py`); tails: code continuation, prose summary, verbatim quote.
tokens/step | tok/s | step ms (marginal timing, 256 vs 96 tokens).

| Run | ctx | code | prose | quote |
|---|---|---|---|---|
| 4096-chunk server, random salt (Nov 22) | ~1.5K | 3.23 / 78.1 / 41.4 | 2.43 / 65.3 / 37.2 | 3.69 / 94.5 / 39.0 |
| same | 250,6xx | **3.21 / 63.8 / 50.3** | **2.48 / 54.6 / 45.4** | **2.15 / 41.7 / 51.7** |
| 2048 server, pre-fusion, random salt | ~1.5K | 3.13 / 80.9 / 38.7 | 2.56 / 80.2 / 31.9 | 3.57 / 94.7 / 37.7 |
| same | 250,6xx | 3.46 / 63.9 / 54.1 | 2.63 / 58.2 / 45.2 | 2.15 / 40.8 / 52.8 |
| 2048 server, pre-fusion, `SALT=fx1` | ~1.5K | 3.25 / 91.6 / 35.5 | 2.57 / 76.8 / 33.5 | 3.59 / 90.0 / 39.9 |
| same | 250,675 | 3.26 / 66.6 / 48.9 | 2.47 / 57.6 / 42.9 | 1.63 / 29.6 / 55.1 |
| fused build, `SALT=fx1`, launch 1 | ~1.5K | 3.10 / 82.1 / 37.7 | 2.61 / 84.6 / 30.8 | 3.71 / 95.1 / 39.0 |
| same | 250,675 | 3.27 / 75.0 / 43.6 | 2.55 / 67.2 / 37.9 | 1.80 / 36.8 / 48.8 |
| fused build, launch 2 | ~1.5K | 3.17 / 81.8 / 38.8 | 2.59 / 77.8 / 33.3 | 3.65 / 97.6 / 37.4 |
| fused build, launch 3 | ~1.5K | 3.11 / 82.1 / 37.8 | 2.42 / 78.1 / 31.1 | 3.64 / 97.8 / 37.2 |

Ranges across runs: short context code 3.10-3.25, prose 2.42-2.61, quote 3.57-3.71; at ~251K code 3.21-3.46, prose
2.47-2.63, quote 1.63-2.15. Eager profiling server at 251K reproduced 3.21 / 2.47 / 2.15. **Acceptance holds with
depth for code and prose; it collapses for verbatim quotation of far-back text (3.69 → 2.15 in the first pair).**
Hundred-tok/s needs a step of 32.1 ms or less at 3.2 tokens/step (24.8 ms or less at 2.5); the 251K steps above are
43.6-54.1 ms (code) and 37.9-45.4 ms (prose).

## 8. Quality measurements

| Check | Result | Date |
|---|---|---|
| llama.cpp 3-pass oracle, 12 varied prompts (`quality_vs_llamacpp.py`) | 12/12 top-1, mean KL 0.0013 (Session 2c); KL 0.0015 (Session 3) | Nov 21 |
| Old (original build) vs final, dense FP8 off, fp8 KV | top-1 12/12, mean KL **0.001241**, max 0.005985; greedy prompt 0 identical for 4/24 tokens | Nov 21 |
| Old vs final with FP8 dense | top-1 12/12, mean KL **0.002394**, max 0.013785 | Nov 21 |
| FP8 dense vs dense-off on the same config | mean KL 0.002436, max 0.013207 | Nov 21 |
| Earlier-build dump (before the FP8-dense step) vs final dense-on | mean KL 0.002638, max 0.014145 | Nov 21 |
| Fold build vs pre-fold build | mean KL 0.000000, greedy identical 24/24 | Nov 21 |
| After Option A vs original | mean KL 0.001243, max 0.005985 | Nov 22 |
| Fused-kernel build vs after-A build | mean KL 0.000000, top-1 12/12, greedy identical 24/24 (twice, incl. after the final routing fix) | Nov 22 |
| KV quantisation simulation (`kv_sim.py`) | TurboQuant-style 4-bit KL 0.0055; naive uniform int4 0.0595; fp8 0.0020 | Nov 21 |
| Needles through the new prefill kernels (diverse text) | found at 31,865 and 123,306 tokens; after Option A at 31,865 | Nov 21-22 |

The 12 gate prompts are under 128 tokens, so they exercise only decode-size kernels; the fold and the prefill
kernels were additionally gated by depth needles and kernel-level fp32 reference tests.

## 9. Decode-step kernel profile and dispatch gap (Nov 22)

Method: one-shot torch-profiler capture of a real 4-token verify forward on the eager (graphs off) profiling server
(`prof.tprof_call`, flag file `agents/tprof_target.flag`); device time of kernels only. One capture per launch.
Graph-mode step is about 39-41 ms; eager wall for the same forward is 110-111 ms (T=4) / 122 ms (T=8), i.e. launch bound.

| Capture | Positions | Kernels | Device total |
|---|---|---|---|
| Before fusion, short | 151-154 | 2342 | **27.94 ms** |
| Before fusion, depth | 250690-250693 | 2342 | **36.05 ms** |
| After fusion, short | 155-158 | **1762** | **26.38 ms** |
| After fusion, depth | 250688-250691 | **1762** | **35.44 ms** |
| (not comparable) 1.5K context | 1559-1562 | 1762 | 28.56 ms |
| (not comparable) first step | 26-29 | 1762 | 23.95 ms (near kernel 1.224 ms: only 26 keys of context) |
| (discarded) T=8 warm-up forward | | 2342 | 41.68 ms |

Largest items, before fusion short (ms, count): `_qg_split` (MoE gate/up) 7.041 x48; dense `gemm` 4.320 x218;
`_qd_split` (MoE down) 3.962 x48; `_inkling_attn` (near attention) 3.254 x48; all-reduce 1.442 x96; bf16 copy
kernels 1.052 x194; `_mix` 0.959 x119; `_ctrl_hidden` 0.848; `_sconv` 0.799 x192; `_ctrl_blend` 0.487; router
scores 0.392; fp32 copies 0.370 x53; `_rows` 0.360 x192; reduce 0.359; router top-k 0.332; `_qg_combine` 0.328;
concat 0.310 x112; `_wnorm` 0.270; `_sconv_write` 0.255 x192; Memcpy 0.163 x124; `_kv_write` 0.138 x48.
After fusion short: `_qg_split` 6.827; gemm 4.345; `_qd_split` 3.847; near attention 3.276; all-reduce 1.402;
`_mix` 0.963; `_ctrl_hidden` 0.851; `_sconv` 0.842; `_ctrl_blend` 0.488; fp32 copies 0.370 x53; concat 0.334 x120;
bf16 copies **0.312 x50**; `_prep` **0.232 x48**; Memcpy 0.137 x104; `_sconv_write` and `_rows` gone.

Fusion changes (exact; see DECISIONS §10): conv-state write folded into the conv kernel (192 launches, 0.255 ms);
four `rows_op` + `kv_write` (240 launches, 0.498 ms) → one `_prep` kernel x48 (0.232 ms); bf16 copy kernels 194 → 50
(1.052 → 0.312 ms). MoE differs by about -0.3 ms at short context and +0.7 ms at depth between captures (routing).

**Context cost in the target forward (device):** before 36.05 - 27.94 = 8.11 ms; after 35.44 - 26.38 = 9.06 ms
(MoE routing differs between captures). Far attention (vendor fmha, 8 visits) 7.390 + 0.175 ms before and 7.375 +
0.167 ms after (0.95 ms per visit, about 270 GB/s). Near attention kernel 3.254 → 4.493 ms (before) and 3.276 →
4.495 ms (after): **+1.2 ms growth with context that fusion does not change**; cause not investigated (suspected
page-walk cost on far-allocated cache pages).

**Dispatch gap, measured (`agents/graph_gap_bench.py`, `ar_graph_bench.py`):**

| Measurement | Result |
|---|---|
| XPU-graph replay of N tiny dependent kernels, N = 100 / 500 / 2000 / 4000 | 1.04 / 0.93 / 0.90 / 0.90 us per kernel (slope 0.90 us) |
| Same kernels eager | 4.43 / 4.13 / 4.13 / 4.11 us per kernel |
| 96 decode-size (4 x 2304 bf16) all-reduces in a graph, 2 ranks | 1.615 ms with `VLLM_XPU_ALLREDUCE_HOST_WAIT=1`, 1.625 ms with 0 (eager 1.932 / 1.953 ms) |
| 96 all-reduces + 960 tiny kernels in a graph | 3.578 ms (HOST_WAIT=1), 3.545 ms (0): about 2.06 us per interleaved tiny kernel |

Implied total dispatch overhead for ~2300 kernels: about 2-4.5 ms; the 580 kernels removed by fusion are worth about
0.5-1.2 ms of gaps on top of the 1.56 ms of device time. The ~8 ms gap suspected before this measurement was wrong.
The 96 all-reduces cost 1.6 ms in a graph (17 us each); the full-chunk all-reduce at T=2048 costs 1.84 ms each.
Graph-step wall-clock before/after fusion could not be resolved (pooled short-context step means 37.2 vs 35.9 ms,
within the +-2 ms launch noise).

Decode-step composition at short context (target forward 27.94 ms device): MoE 12.4 (gate/up 7.041, down 3.962,
combine 0.328, router 0.724, reduce 0.359) = 44%; dense GEMM 4.3 = 15%; near attention + KV write + vendor far
3.6 = 13%; loop mixes/controllers 2.6 = 9%; casts/concat/copies ~2.0 = 7%; all-reduce 1.4 = 5%; conv + rows 1.4 =
5%. The remaining ~11-13 ms of the ~39-41 ms step is the drafter (re-measured Nov 22 via INKLING_PROF=1 ph() over 16 draft forwards at the T=4 verify
shape: 0.4 ms/step total - ctx_kv 0.1 + forward 0.3; the earlier 1.43 was a single tprof capture at a
T=8 warm-up shape), LM head and sampling, and dispatch gaps; this remainder
was not profiled directly.

## 10. Other numbers worth keeping

* One-shot all-reduce (L0 IPC): 18 vs 39 us at 4.6 KB, bit-identical, off by default (`VLLM_XPU_ONESHOT_AR=1`).
* Vendor flash attention for far keys: prefill 2.6-2.9x (40 TFLOP/s); decode 4-5x at 131K-491K (2.0 ms per layer call
  at 491K) — handoff Session 3.
* Fixed per-request KV overhead about 3.4 GiB per rank (conv-state / SWA caches sized for `max_num_batched_tokens`);
  KV 16 KiB per token per rank in bf16, 8 KiB in fp8.
* Host memory with the dev server up at this cap: 63 GB used of 91 GB total, swap 7 GB (`free -g`, Nov 21).

## 11. Speculative-depth sweep and KV-cache dtype A/B (Nov 22)

Method: `agents/deep4.py` on fixed prompts (corpus from the vLLM source tree, `SALT=7`, byte-identical across
cells and runs), 3 server launches per cell, marginal tok/s over the 96-to-256-token window, tokens/step from
the live `vllm:spec_decode_*` metrics. Contexts 8K and ~234K (`ctx` column = actual prompt tokens). Tables quote
runs 2-3 (steady state); a cell's first 234K run after the cold prefill shows ~8-9 ms lower step times on some
tails (cache/pool not yet resident) and 8K verbatim run-1 shows a one-off ~260-300 tok/s warmup burst.
Quality: `agents/ab_quality.py` (12 diverse prompts, greedy 48 tokens with top-20 logprobs per position +
needle retrieval at ~61K/124K/243K), compared offline by `agents/ab_compare.py`.

### 11.1 DSpark depth sweep (FP8 KV)

tok/s (tok/step, step ms), mean of steady runs:

| tail | ctx | no spec | depth 2 | depth 3 (prod) | depth 4 |
|---|---|---|---|---|---|
| code | 8K | 43.8 (1.00, 22.9) | 69.7 (2.57, 36.9) | 84.5 (3.33, 39.4) | 87.1 (3.77, 43.2) |
| prose | 8K | 44.0 (1.00, 22.8) | 65.2 (2.36, 36.3) | 73.2 (2.72, 37.2) | 75.1 (2.97, 39.6) |
| verbatim | 8K | 43.9 (1.00, 22.8) | 81.6 (2.77, 34.0) | 96.0 (3.54, 36.9) | 100.3 (3.96, 39.5) |
| code | 234K | 36.3 (1.00, 27.5) | 58.7 (2.65, 45.2) | 68.5 (3.29, 48.0) | 76.0 (3.85, 50.6) |
| prose | 234K | 36.3 (1.00, 27.5) | 59.9 (2.32, 38.7) | 66.7 (2.74, 41.1) | 51.1 (2.48, 48.6) |
| verbatim | 234K | 36.2 (1.00, 27.6) | 42.2 (1.63, 38.7) | 38.7 (1.80, 46.3) | 41.4 (1.76, 42.4) |

Findings: depth 3 confirmed for production - depth 4 wins code (+11% at 234K) but collapses prose (51.1 vs
66.7 tok/s; per-step acceptance 2.48 vs 2.74 while step time grows to ~49 ms), reproducible across all 3 runs.
Note: no-spec step at 234K is 27.5 ms (22.8 ms at 8K) - the "base step 37-41 ms" quoted in earlier sections
was the WITH-spec verify step, not the autoregressive base. Depth 4's prose acceptance falling BELOW depth 3's
on identical prompts is an open observation (draft feedback through verification changes the chain context).
Method:"mtp" is not wired for Inkling2 - both probes fail at startup with
`Value error, The Inkling2 checkpoint has no mtp_config` (the draft-config loader rewrites any
inkling2/inkling2_loop draft into an Inkling2DSparkConfig unconditionally; there is no chain-MTP draft
implementation for this family). The planned DSpark-vs-MTP A/B therefore became this depth sweep.

### 11.2 KV-cache dtype: FP8 vs BF16 (DSpark depth 3)

Quality (12 short prompts, 576 greedy positions; two independent capture pairs gave byte-identical numbers):

| metric | FP8 KV vs BF16 KV (reference) | gate |
|---|---|---|
| top-1 agreement | 443/576 (76.9%) | 12/12-class |
| mean KL, shared top-20 mass | 0.31800 | <= 0.002 (~160x over) |
| max KL | 16.58 | <= 0.014 |
| greedy 48-token text identical | 7/12 | - |
| needle recall 61K / 124K / 243K | FOUND on both dtypes; 243K answer identical | - |

Launch-noise control: same dtype across separate launches = 576/576 top-1, KL 0.00000, 12/12 text (both BF16
and FP8 pairs) - the XPU serving path is deterministic at temperature 0, so the shift above is entirely the
FP8 KV cache. The kv_sim prediction (fp8 0.0020, Nov 21) missed the live error by ~160x.

Performance (dspark depth 3, mean of steady runs):

| | FP8 KV | BF16 KV |
|---|---|---|
| 8K code / prose / verbatim tok/s | 84.5 / 73.2 / 96.0 | 94.1 / 93.1 / 114.3 |
| 234K code / prose / verbatim tok/s | 68.5 / 66.7 / 38.7 | 77.2 / 58.8 / 45.6 |
| 234K code step | 48.0 ms | 44.4 ms |
| ~250K prefill (needle call) | 151.7-156 s | 140.8-146 s |
| KV pool / concurrency @262144 | 640,939 / 2.45x | 439,959 / 1.68x |

BF16 KV is faster everywhere except prose at 234K (-12%: the drafter's acceptance shifts from 2.74 to 2.35
tok/step under the changed numerics - the DSpark head was effectively tuned while FP8-KV noise was present).
The "fp8 halves far-attention bytes" advantage did not materialise: the fp8 path's dequant/staging overhead
exceeds its bandwidth saving on this port.

**Production switched to BF16 KV on Nov 22** (unit `EXTRA_ARGS` no longer sets `--kv-cache-dtype`; verified
pool 439,959 tokens, 1.68x concurrency). Sole cost: concurrency 2.45x -> 1.68x.

### 11.3 TurboQuant tq4 live, and the eager-vs-graph control (Nov 22)

The Nov-21 simulation had TurboQuant-style tq4 (Hadamard rotation + Lloyd-Max) at KL 0.0055 vs naive int4
0.0595. Measured live (`INKLING_KV_SIM=tq4`, BF16 storage, eager server, same ab_quality suite, compared
against a matched eager BF16 capture): **mean KL 0.34235, max 16.58, top-1 agreement 396/576 (69%), greedy
text 4/12** - statistically the same damage as the live FP8 KV cache (0.318). Needles still found at
61K/124K/243K on both. The tq4 simulation missed the live error by ~62x, exactly like the fp8 simulation
(~160x): **the kv_sim roundtrip hook does not model what the real cache path does on this stack, and
simulated KV-quantisation quality claims are void until the gap is root-caused** (prime suspects: the
checkpoint's clip/L2/value-norm transforms (`qk_l2_clip_threshold`, `rel_l2_clip_threshold`,
`use_value_norm_no_gamma`) interacting with the loop architecture's per-pass cache writes, and the TP2 cache
layout - the offline sim numbers were taken TP1/eager/20K).

Control: **eager and graphed servers are numerically identical** - BF16 eager vs BF16 graphed = 576/576
top-1, KL 0.00000, 12/12 greedy text. Quality captures may therefore always run eager (which the sim hook
requires: it crashes under the graphed server with `UR_RESULT_ERROR_UNSUPPORTED_FEATURE` on its codebook
H2D copy mid-forward; isolated repros pass, so it is an engine-context effect, likely the profile-run pool
teardown), and the numbers transfer 1:1 to the graphed production config.

## 12. Host RAM and the GPU mirror (Nov 22)

Question: why does the host show ~70 GiB of 91 GiB RAM used while every process sums to only 16.9 GiB?

Answer: `/proc/meminfo` on this kernel (Ubuntu 26.04, xe driver) carries a `GPUActive` field - **58.4 GiB of
system RAM is accounted as GPU-active**, and `free` folds it into "used". It is static while a server runs
(identical readings 90 minutes apart) because the engine pre-allocates its full working set at startup
(`gpu_memory_utilization` 0.85): per card, ~8.9 GiB weights + ~7.0 GiB BF16 KV pool + activations/graph pools
= ~27-30 GiB on-card (`torch.xpu.mem_get_info`: 30.1/31.9 and 28.9/31.9 used), and the kernel accounts the
same working set against host RAM. The books close exactly: AnonPages 12.6 + Cached 12.3 + slab/tables ~1 +
GPUActive 58.4 = MemTotal 91.1. Process RSS sum 16.9 GiB (api_server 2.2 + EngineCore 1.9 + two workers
5.9/5.9 + opencode/dashboard/desktop). Swap 764 MiB = desktop apps pushed out by the GPU mirror; harmless.

So "high RAM usage" is **not a leak** - it is the driver's GPU memory accounting, and it scales with the GPU
working set. Two consequences worth recording:

* **The 1M-context hang is now explained mechanically**: raising `MAX_MODEL_LEN` to 1M grows the planned KV
  pool, GPUActive grows past host RAM, and the host thrashes (DECISIONS section 8).
* **Real headroom = MemAvailable (20 GiB at this operating point), not free**. Before raising the cap, pool
  size, or gpu_util, check that MemAvailable stays comfortably positive; the GPU mirror grows one-for-one
  with on-card allocations.

Levers if host headroom is ever needed: smaller `GPU_UTIL` (smaller pools, less GPUActive), fewer concurrent
full-context seqs (same), or an fp8 KV pool (halves the KV share of the mirror - rejected on quality grounds,
section 11.2). No xe module parameter exposes allocation placement; ReBAR is enabled (32 GB BAR 2 on both
cards, asserted in `01-sudo-system-setup.sh`), so the mirror is not a small-BAR workaround - it is how this
kernel accounts the driver's memory.

Stop-test (Nov 22, later): with the server stopped, GPUActive drops 58.4 GiB -> 21 MiB, MemFree 10.5 -> 81.5
GiB, MemAvailable 21.4 -> 92.3 GiB (real DRAM: 98.0 GiB, 49 x 2 GiB blocks), AnonPages 12.6 -> 0.68 GiB.
The entire mirror releases cleanly on stop - no leak, no permanent pinning. Idle host usage ~12 GiB.
While serving, the floor is set by the weights mirror (17.8 GiB for both cards) plus the minimum viable
pool: even a single-262144-request configuration (GPU_UTIL ~0.77) lands at ~61 GiB used. Implication: the
~24 GiB host-RAM target is only reachable with the server stopped; on-demand serving is the lever.

Revert (Nov 22, later still): production back to `GPU_UTIL` 0.85 per user decision - pool 439,855 tokens,
1.68x concurrency at the cap (verified serving). The 0.78 trim above stays recorded as the measured trade.
The RAM targets discussed in this exchange were for OTHER machine specs, not this box.

Machine-sizing rule for spec'ing similar 2x-B65 boxes (the practical output of this exchange): the xe
driver mirrors the GPUs' on-card working set 1:1 into host DRAM (`GPUActive`), so host RAM must cover

    sum over cards of (weights + KV pool + activation/graph/conv-state overheads)  +  ~15 GiB host

At this box's 262144 operating point that is ~59 GiB mirror + ~13 GiB host => a 96 GiB kit is the minimum
comfortable size; 64 GiB would thrash under load, and a 1M-context pool hung a 91-GiB-visible host.
With the server stopped the same host idles at ~12 GiB used. Lower `GPU_UTIL` trims the mirror
proportionally (0.78 measured above) at a direct concurrency cost.


## 13. One-shot AR on BF16: REJECTED; long-context launch noise (Nov 22)

One-shot P2P all-reduce (`VLLM_XPU_ONESHOT_AR=1`) is REJECTED on the current BF16 build. The first
retest was a measurement failure: the A/B launcher's env passthrough never reached the container (sed
anchor mismatch after an earlier edit), so both arms ran the identical production config - the
"bit-identical 576/576, speed neutral" result was production-vs-production. Retried hours later with
the env verified inside the container (`VLLM_XPU_ONESHOT_AR=1` confirmed via docker exec): AR FAILS
the exactness gate - top-1 agreement 554/576 vs the bf16 reference, shared-top-20 KL mean 0.069 /
max 11.5 (gate: KL <= 0.002) - and showed no speed win (the tok/s benches were confounded by an
accidental double-driver race, but the exactness gate alone rejects it). The FP8-era "+1.2%,
bit-identical" claim does not transfer to the current build. Production stays OFF; the knob is not
recommended for any workload.

Protocol rules banked: (a) verify every passthrough env inside the container before trusting a cell's
data; (b) never let two driver scripts share one log/server - concurrent request streams change
batching and confound tok/s benches (single-request exactness captures are immune).

**Long-context acceptance is launch-noisy**: prose-summary tok/step at ~234K measured 2.35, 2.56 and 2.79
across three same-config launches (tok/s 58.8-74.6, about +-11%), while short-prompt logits are
bit-identical across launches (section 11.2). The far-attention reduction order evidently depends on
per-boot page layout, so long-context A/B verdicts need same-boot paired runs or n>=5 repeats; short-context
verdicts remain single-run trustworthy. (This also revises section 8's per-launch nondeterminism note:
+-0.1-0.17 tokens/step applies to LONG contexts; short contexts are deterministic.)

The suspect mechanism for the drafter's sensitivity: the markov head consumes target hidden states whose
bit patterns shift with any numeric-path change (KV dtype, reduction order), and prose acceptance moves
-0.39 / +0.44 tokens/step across such shifts. Not fixable without drafter retraining; verification keeps
outputs correct regardless (spec decoding is correctness-preserving; only speed moves).

Decode remainder, re-attributed (same day): with the drafter measured at 0.4 ms/step (ph() accumulation
over 16 draft forwards at the T=4 verify shape: ctx_kv 0.1 + forward 0.3), the ~11-13 ms wall-vs-device
gap is LM head (estimated ~2 ms from its 0.93 GB/step weight read), sampling (~1 ms, estimate) and
roughly 8-9 ms of scheduler/dispatch gaps - all still unprofiled at the step level (the tprof wrappers
cover model forwards only; their flag is consumed on first capture, so concurrent drivers starve each
other). Next lever: a whole-step profiler capture.


## 14. Iteration-elapsed instrumentation: the decode remainder is inside the bracket (Nov 22)

Enabled `--enable-logging-iteration-details` (GPU-event timing of the whole step, from execute_model
start to sample_tokens end; loggers.py emits "iteration elapsed time" per Iteration line). One gate fix
was required: the fork's `gpu_iteration_timing` property required `device_type == "cuda"`, so the timer
silently never armed on XPU - patched to `in ("cuda", "xpu")` (vllm/config/vllm.py,
instrumentation-only, flag-gated, default off). Result at 8K decode iterations: elapsed p50 34.4 ms,
p90 34.8 (n=363; the 16.4 s max is the prefill iteration). Same-run deep4 walls: code 35.9 /
prose 32.4 / verbatim 31.6 ms per step.

Attribution: the event bracket covers essentially the whole wall step (within ~1.5 ms slop). The
"~8-13 ms unattributed scheduler/dispatch remainder" from the earlier tprof-based model DOES NOT EXIST:
tprof's 27.9 ms body figure is a sync-perturbed lower bound (torch.xpu.synchronize around each phase
removes overlap), and LM head (~1-2 ms), draft (0.4 ms) and sampling/accept all run inside the bracket.
The decode step is device-work-bound end to end; wall - elapsed leaves no large host-side gap.

Killed by this: the "attack the ~8-9 ms scheduler/dispatch gap" lever from the Nov 22 consult (~20% of
the step, the last big suspected item). Remaining decode levers are small: graph coverage for LM head +
sampling (~1-2 ms), the near-attention +1.2 ms growth (~1 ms), drafter quality (retraining, out of scope).

Also this round: immediate command lists (`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` /
`UR_L0_USE_IMMEDIATE_COMMANDLISTS=1`) tested: bit-identical (576/576 top-1, KL 0.00000) but
speed-neutral (step times +-1 ms in mixed directions; the prose tok/s jump was acceptance noise,
tok/step 2.61 -> 3.01 with unchanged step time). Not adopted. Caveat: bit-identical output is also
consistent with the env var not being honored; disambiguating via SYCL_UR_TRACE is not worth it absent
a speed win. Verified LM head is vocab-sharded under TP2 (ParallelLMHead, one bmm read for all verify
rows) - the "shard the head" lever does not exist. The consult's host-sync suspicion was refuted for
the forward/accept path (device-side sampling kernels); the one per-step sync cluster
(async_utils.py:193 copy_event.synchronize + tolist) runs on the async output path, and wall ~= elapsed
shows it is not costing visible time.


## 15. Remaining-lever audit: scatter, FP4, throughput, ceiling, Qwen first pass (Nov 22)

**Near-attention scatter: closed.** Controlled same-boot A/B at a 134K context (GPU-event elapsed,
TIMING=1): ARM A fresh pool (contiguous block ids, p50 40.0 / min 39.4 / p90 40.2, n=61) vs ARM B pool
churned past full turnover (scattered ids via four distinct ~120K prefills; final-run p50 39.9 /
min 39.4 / p90 40.2, n=64 - churn-decode lines excluded). Block-id scatter has ZERO effect on the
near-attention step cost. The earlier "+1.2 ms on full VRAM" anomaly did not reproduce and falls within
the long-context launch-noise band (section 13); it was a cross-boot comparison, the protocol that
finding invalidates. Free-queue-sorting is not worth pursuing. (Note: arm stats must be computed on the
final run's lines only - churn decodes contaminate the raw capture.)

**Vendor FP4 GEMM: capability confirmed, parked.** Bundled oneDNN is v3.13.3; strings on
libtorch_xpu.so show 2827 f4-type + 24 nvfp4 symbol hits - the primitives physically exist in the XPU
build. But there is no Python path, and the kernel's custom layout (u8 packed nibbles with per-16 scale
bytes interleaved) does not match oneDNN's separate-scale-tensor model: a repack plus a C++/SYCL binding
for at most ~2 ms. Parked with the evidence recorded.

**Throughput headroom (for future reference):** two concurrent deep4 streams at 8K aggregate
1.6-1.7x single-stream tok/s (code 152 vs 90, prose 158 vs 95, verbatim 172 vs 108), per-stream latency
-10-35%. Matches the KV pool's 1.68x concurrency design point. Single-stream latency remains the
production goal; this documents the mode switch if needs change.

**Drafter-retraining ceiling (if ever on the table):** at ~3.75 tok/step the same step cost yields
+14% code / +38% prose / +5% verbatim tok/s; a perfect 4.0 yields +20/45/10%. Prose has the most
headroom (2.72 of 4.0). Model-training work, not serving work.

**Qwen head-to-head, first pass:** identical deep4 prompts (SALT-pinned, raw /v1/completions,
temperature 0, 256 tokens) against production Inkling (2x B65 TP2) and a Qwen3-4B-Instruct-2507 Q4_K_M
llama.cpp CPU oracle (4 threads, :8099). Quality: code-continue parity (both produce correct
ConfigLoader continuations); verbatim-quote both accurate (Inkling's output format cleaner);
prose-summary both partial-but-accurate over the 3-file corpus window - Inkling summarized the oauthlib
tokens.py file (third in window, corpus check confirmed the module is really there), Qwen the first
punycode file; neither hallucinated. No cutover signal. Speed is not comparable (GPU+spec-decode vs
CPU oracle): Inkling 36.6-69.2 tok/s on cached-prompt 256-token requests vs Qwen 6.6. Caveats: CPU
oracle, raw completions on a chat-tuned model, 3 prompts - a broader suite would be needed to claim
parity beyond the signature tasks. Qwen gguf downloaded from the ungated unsloth mirror (the official
Qwen GGUF repo path returns 401 ungated).


## 16. Tail-gap instrument incident, and campaign closure (Nov 22)

**Incident.** The last open item (eager-tail gap measurement gating LM-head+sampling graph coverage)
used a worker-level torch.profiler capture: a flag-armed decorator pair on gpu_worker.py
(Worker.execute_model / Worker.sample_tokens), profiler entered at execute_model call 50 and exported
after 3 sample_tokens returns. Outcome: attempt 1's verification step crashed AFTER the edit landed
(py_compile writing into a root-owned __pycache__; fixed to an in-memory compile check) leaving the
patch in place silently; attempts 2-3 hit an HTTP 500 on the second same-prompt decode request in
patched windows 2/2 (never in unpatched windows, 2/2 clean the same day), with only ~41-45
execute_model calls per worker and no server-side traceback found in filtered greps; the subsequent
restoreprod restarted production ON THE PATCHED FILE (restoreprod does not revert source) with the
leftover flag present - production's first request then hung until timeout. Mechanism (per consult,
plausible not proven): the leftover flag armed the profiler inside production at call 50; a stuck XPU
profiler sync blocks one rank while the other waits in a collective - a silent hang. Emergency revert
(pristine file + flag deletion + restart) restored service; deep4 bands normal (94.9/89.8/106.2 tok/s
at 8K). The 500's root cause remains unexplained: decorator-introspection, stdout-print and flag-read
hypotheses were knocked down by source reads; the patch correlation stands (2/2 vs 0/2). Declared not
worth further production windows.

**Closure.** The campaign closes on measured arithmetic: decode bracket p50 34.4 ms ~= wall;
draft 0.4 ms; LM-head bmm floor 0.790 ms/rank at 586 GB/s weight-stream (measured at the exact
production shape); tail (head + logits all-gather + sampler + accept + gaps) <= ~2.6 ms worst case,
likely 1.5-2 ms. The graph-coverage idea is PARKED per the pre-set 0.5 ms gap gate - the gap component
could not be measured without re-running the risky instrument, and the bounded upside cannot change any
decision. If ever revisited: a bounded `with profile(...)` around ONE execute_model+sample_tokens pair
in an off-production window (separate port/container), never `__enter__` across steps, unfiltered logs,
or the pti/unitrace route instead.

**Process lessons (extending section 13's rules):** (a) source patches SURVIVE production restarts -
verify SOURCE STATE (cmp against backup / grep), not just container env, before trusting any window;
(b) verification that writes to root-owned paths fails after the edit has landed - verify in memory or
before writing; (c) one-off patch scripts must refuse to run when a backup already exists (guard added
to patch_xtrace.py). Post-incident checklist executed clean: gpu_worker.py byte-identical to pristine,
zero flag/trace leftovers, the intentional config gate intact (config/vllm.py:645), no leftover
containers, fork working tree clean apart from the intentional gate line.


## 17. System DRAM accounting deep-dive (Nov 22, Claude agent diagnostic)

Follow-up to section 12's mirror finding: a full read-only diagnostic (Claude agent with Bash, no state
changes) reconciled the complete DRAM picture while serving. MemTotal 91.14 GiB: used 75.1 (the mirror
counts inside "used" and is NOT reclaimable - GPUReclaim 0, MemAvailable excludes it), MemFree 3.87,
MemAvailable 16.06, GPUActive mirror 57.7 (stable vs 58.4 earlier), Cached 12.7 (model files,
reclaimable), AnonPages 14.5, Slab 0.8. Reconciliation: free+cached+anon+slab ~= 32 GiB + 57.7 mirror
= total; used minus mirror ~= 17.4 GiB = AnonPages + kernel + ~2 GiB unattributed residual (not
alarming). Per-process top: TP workers 5.77 + 5.75, llama oracle 3.18 (2.3 of it reclaimable GGUF mmap),
api_server 2.11, EngineCore 1.87 - the mirror (57.7) is ~3x all process anon memory combined and no
process owns it (kernel-side xe/TTM pages).

Levers inspected (none adopted): xe module params expose no GTT/mirror-size knob (wedged_mode is debug;
the rest risk driver stability); vm.overcommit_memory=0 is fine and mode 2 would break vLLM; no zram;
8 GiB swap file (2.8 used, 5.2 free, swappiness 10) absorbs anon only, never the mirror; docker cgroup
has NO memory cap (Memory=0) - a cap would convert VRAM spikes into container kills, correctly absent.

Risk verdict: expected-per-the-mirror, steady state safe (PSI memory 0.00 across all windows, swap
mostly cold). OOM failure mode under real pressure = the OOM killer picks the largest anon process =
a vLLM worker = production loss; headroom is 16 GiB available + 5.2 swap. A 234K prefill spike adds
mirror-bounded host DRAM (activations on-card) - empirically fine all day (multiple 234K runs, zero
pressure events), though GPUActive was not logged DURING a spike. Insurance options if ever wanted
(maintenance-window changes, not done): larger swap file or zram for the anon side. The mirror itself
remains the documented cost of the ~61 GiB sizing rule - not wasteful, not fixable, not dangerous.


## 18. The mirror is a software regression: root cause found, fix verified (Nov 22)

Section 17's "expected-per-the-mirror, not fixable" verdict is SUPERSEDED. The 57.7 GiB GPUActive
mirror is a **userspace regression in Intel compute-runtime** (upstream issue
intel/compute-runtime#953, GSD-13038): since 26.09+, a "defer backing for BMG" change makes every
large device allocation in a multi-GPU-visible process get an eagerly-pinned host-RAM mirror on its
peer registration. Single-GPU-visible processes do not pin. This explains every observation of the
day: the bench container ran with ZE_AFFINITY_MASK=0 (one GPU visible) and showed no mirror for a
plain 4 GiB device allocation, while production (ZE_AFFINITY_MASK=0,1) mirrored ~1:1; the config
bisect (drafter off / graphs off / both off - four cells, boot GPUActive 51.1-51.5 GiB and post-8K
60.5-60.6 in all four) showed no serving-stack toggle matters, because the peer registration exists
for TP2 base allocations regardless. GPUActive itself is the kernel's TTM system-page counter for
GPU objects (per-node GPU counters, mid-2025), and Resizable BAR was already active (32 GiB
apertures on both cards) - small-BAR staging was refuted as the mechanism.

**Fix verified on this stack**: `NEOReadDebugKeys=1 EnableDeferBacking=0` (the debug-key workaround
confirmed by Intel in the issue). Production config with the env, verified in-container:
- boot: MemAvailable 74.55 GiB / GPUActive 0.57 GiB (baseline cell was MemAvailable 26.46 / GPUActive 51.10 GiB)
- after the full 8K deep4 suite: MemAvailable 74.57 GiB / GPUActive 0.57 GiB - the mirror does not grow with load
- deep4 bands: 89.8 / 84.2 / 111.2 tok/s - normal, no regression
- exactness vs bf16 reference: top-1 576/576, KL(defer0||bf16) over shared top-20: mean 0.00000  max 0.00000  (576 positions) (gate capture saved)

Net effect: ~50 GiB of host RAM returned at boot, the ~9 GiB per-load growth eliminated; MemAvailable
under full production goes from ~17.5 GiB to ~74.6 GiB. This also retroactively explains the
1M-token hang (mirror exhaustion during pool sizing) and matches the vLLM-TP2 host-OOM reports in
the upstream issue.

**Caveats, on the record**: Intel states debug keys are "strictly for debugging" and recommends the
proper fix; the proper fix shipped in 26.35.39758.10 and was confirmed on 2x B70 (kernel 7.1.8) -
but a 2x B60 reporter on kernel 7.0.0-31 + 26.35 still hit host OOM in the PRIME/dma-buf path
(xe_ttm_tt_populate), fixed only by dropping to kernel 6.17, and one reporter still saw the
behavior on 26.40 (Nov 13). This box is kernel 7.0.0-44, so the runtime upgrade alone may not
suffice here. The env is therefore the pragmatic, verified fix with upstream tracking; the
passthrough in run-ab.sh is inert unless NEOReadDebugKeys=1 is set (all other windows and boots ran
with it off).
