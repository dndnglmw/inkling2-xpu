# Inkling 2 Mini — decisions, rejected ideas, limitations, what is not done

Written 2026-11-22. Companions: `SETUP/INKLING2-GO-LIVE.md` (operations), `SETUP/INKLING2-PERF-LOG.md` (all
numbers, referred to below as PERF-LOG §n). Each entry states the decision, the reason, and the evidence.

## 1. Adaptive speculation depth — not built

* **Decision:** keep a fixed depth of 3 (`num_speculative_tokens=3`).
* **Reason:** a step-cost model (measured verify cost 20.3 / 23.9 / 30.7 / 43.5 ms for 1 / 2 / 4 / 8 tokens plus about
  4.8 ms of drafting) combined with per-position acceptance per prompt class (`agents/decode/spec_positions.py`) puts
  the best fixed depth at the per-class oracle, so a controller had no headroom to buy.
* **Evidence:** depth 3 mean 73.9 tok/s vs depth 5 mean 72.1 on the same four prompt classes (PERF-LOG §4). The
  user's target for such a controller (a step of about 30 ms with 3-4 verified tokens) matches depth 3.
* Drafter depth is bounded by the drafter's block size; the XPU drafter accepts `num_speculative_tokens` from 1 to
  `max_draft_block_size - 1` and raises otherwise.

## 2. FP8 dense (attention) weights — rejected

* **What:** per-output-channel RTN e4m3 weights for the attention `qkvr` and `wo` projections through
  `fp8_gemm_w8a16`, behind `INKLING_XPU_FP8_DENSE`. Kernel microbenchmark at TP2: qkvr 33.6 → 12.3 us, wo 16.9 →
  9.6 us per call.
* **Result:** the step fell by about 2-3 ms (36.5-38.1 ms across the four classes vs 38.6-40.1 ms) but acceptance fell (tokens/step code
  3.22 vs 3.52, prose 1.81 vs 1.88, chat 3.00 vs 3.08), so the mean over the four classes moved only from 67.3 to
  67.8 tok/s (+0.7%; a first run read 65.0, with a 43.8 ms code-class jitter that did not reproduce).
* **Quality gate:** mean KL against the original build **0.002394** (max 0.013785) with FP8 dense, **0.001241**
  (max 0.005985) without; FP8 dense vs dense-off 0.002436. The bar was 0.002 (PERF-LOG §8). Top-1 stayed 12/12.
* **Decision:** switch stays in the code, default off. The LM head was deliberately not converted (its fp8 path
  returns bf16 logits).

## 3. Fused FP4 expert GEMM (Triton) — dead

* **Goal:** stop materialising a bf16 copy of the experts (about 3.6 ms of each ~9.7 ms prefill MoE visit is the bf16
  dequant write, plus the read-back by the matmul). The XPU has no native FP4 ALUs, so the aim was memory traffic,
  not faster FP4 arithmetic.
* **Evidence:** the existing in-register-dequant Triton kernel runs at about 14 TFLOP/s (gate/up 6.266 ms + down
  3.326 ms for ~130 GFLOP). A plain bf16 grouped GEMM in Triton, no dequant at all, on the same shape
  (E=64, cap=128, K=N=2304) tops out at **18.7 TFLOP/s**; oneDNN bmm does the same product at **48.5 TFLOP/s**
  (1.79 ms vs 4.66 ms). The ceiling is Triton's `tl.dot`, not the dequantisation, so no fused Triton kernel can beat
  dequant + oneDNN bmm. Config sweeps of the fused kernel reached at best 10.66 ms per visit vs 11.12 ms tiered (the
  same run), within single-visit noise (PERF-LOG §3).

## 4. Vendor `fp4_gemm` / grouped GEMM route — rejected (quality)

* **Speed:** the vendor CUTLASS grouped GEMM in MXFP4 runs both expert GEMMs in 1.94 ms (gate/up, 44.9 TFLOP/s) +
  1.38 ms (down, 31.6 TFLOP/s) = 3.3 ms per visit, against 7.3 ms for dequant + bmm.
* **Why it fails the gate:** (a) our layers 0-19 are NVFP4 (e4m3 scale per 16 values plus a per-expert outer scale);
  the kernel wants MXFP4 (power-of-two scale per 32). Converting layer 0 gave **11.2% relative weight error**
  (w13 0.1118, w2 0.1119) on top of NVFP4's own roughly 10% quantisation noise. (b) On this architecture (xe2) the
  library's `mxfp4` recipe also quantises the **activations** to MXFP4 (W4A4); per the library's own reference-
  implementation docstring and code it takes quantised activations with block scales. A bf16-activation FP4-weight
  kernel is not exposed. (The same docstring lists activation quantisation for the `mxfp8` recipe, so the MXFP8
  layers 20-23 would not be exact through it either.)
* A first known-answer test failed (relative error 1.23), most likely for reason (b): bf16 activations were passed
  where the kernel expects packed FP4 ones. The MoE-output error of the converted weights was therefore never
  measured end to end; the 11.2% weight error alone was judged enough to reject the route.
* For decode the route that remains is a faster decode-side dequant feeding the existing GEMM, not a format change.

## 5. Bit-trick FP4 decoders — exact, not faster, left off

* e2m1 / e4m3 decoders built from integer operations and bitcasts, verified bit-identical to the reference over all
  16 nibbles and all 256 scale bytes (`test_fp4_decode_fast.py`).
* No speedup: 48-visit tiered mean 10.45 vs 10.52 ms and pure Triton 13.41 vs 13.49 ms (repeat 10.43 / 10.45, 13.33 /
  13.39). The prefill dequant is memory-bound, and the decode kernel (`_qdot16`) already decodes through fp16 bit
  patterns. Opt-in via `INKLING_MOE_FAST_DECODE`, default 0.

## 6. 4096-token prefill chunks — rejected for production; answer-room trade-off

| Chunk | KV pool | Concurrency at 262144 | Prefill (36.9K / 124.7K / 229.7K tokens) |
|---|---|---|---|
| 2048 (production) | 641,028 tokens | 2.45x (641,028 / 262,144 = 2.445) | 1931 / 1714 / 1480 tok/s |
| 4096 | 329,284 tokens | 1.26x (329,284 / 262,144 = 1.256) | 2177 / 1943 / 1703 tok/s |

* **Decision (user):** stay at 2048. Two concurrent 256K sessions is the stated operating point and 4096 halves
  concurrency; the +13% to +15% prefill is worth less than that capacity, and 1480 tok/s at 241K was already accepted.
  The guard passed (the 4096 pool still holds one 262144 request), so this is a capacity decision, not a failure.
  `serve-4096.sh` stays in place for a future "maximum prefill over capacity" choice.
* **Answer room under the 262144 cap:** a 258,338-token prompt leaves 3,806 tokens (the filler runs used 256 of
  them). Options computed but **not adopted** (the cap stays at 262144): keep 262,144 and limit prompts to 245,760
  tokens for 16,384 tokens of room (2.45x), or raise the cap to 278,528 for a 262,144-token prompt plus 16,384 of
  output (641,028 / 278,528 = 2.30x). Both still allow two full-length requests at once.

## 7. Context cap 262144 fixed; 1M ruled out

* **Cap:** 262144 tokens per request, set in `serve-noprof.sh` and every launcher derived from it.
* **1M is unsafe until proven otherwise.** The 1,048,576-token server was live when the host hung: swap was full and
  the machine stopped answering Tailscale pings, and it had to be power-cycled (reported by the user). From then
  on no launcher may use `MAX_MODEL_LEN=1048576`. The 1M-config server had fitted with fp8 KV and passed a needle at
  518,738 tokens; a 940,000-token needle was started and never completed (its client died), so nothing was proven
  at 1M and nothing further was attempted.
* Host memory with the 262144 server up: 63 GB used of 91 GB, swap 7 GB (PERF-LOG §10).
* KV memory model per rank: `M(L) = F + s * L` with `F` about 3.4 GiB (conv-state and sliding-window caches sized
  for `max_num_batched_tokens`) and `s` = 16 KiB per token in bf16, 8 KiB in fp8. **No decision in this project trades
  KV memory for speed.**

## 8. Known limitations

* **Verbatim-quote acceptance collapses at depth:** 3.69 → 2.15 tokens/step in the first pair (range across runs:
  3.57-3.71 at ~1.5K context, 1.63-2.15 at ~251K). The drafter copies poorly from text hundreds of thousands of
  tokens back. This is a draft-quality limit, not a kernel problem, and is deliberately not being fixed with kernel
  work.
* **Near-attention kernel grows 1.2 ms with context** (3.28 → 4.50 ms per forward, unchanged by fusion). Suspected
  page-walk / TLB cost on far-allocated cache pages. Not investigated.
* **`spec_k_bench` warm-up artifact:** the code class read 62.8 ms and 64.2 ms steps on the first benchmark run after a
  fresh launch (43.8 ms once) and 40.2 ms from then on, 39.7-40.5 ms when run in other orders. Treated as first-run
  warm-up; the cause was not found.
* **Concurrency ceiling 2.45x at 262144** (server-reported; 641,028 / 262,144 = 2.445) with 2048 chunks.
* **Host memory:** 63 GB used of 91 GB at this cap with the dev server up.
* **Not batch-invariant:** the same prompt can differ slightly between solo and concurrent runs and with or without a
  prefix-cache hit (`inkling2-switch.md`).
* **Per-launch nondeterminism:** acceptance differs by +-0.1-0.17 tokens/step between launches of the same build;
  graph-step comparisons across launches carry about +-2 ms.
* **GPU1 sits on a PCIe Gen4 x4 link** (7 GB/s) against GPU0's x16 Gen5 (46-57 GB/s).
* **Profiler:** one capture per launch (a second capture killed the engine twice).
* **First request after a launch pays JIT compilation** (up to ~19 s on a 36,923-token prompt).
* **Honest decode ceiling at 258K:** about 85-90 tok/s for code-like text and 75-80 for prose in the best case, about
  70-75 and 60-65 with moderate effort. 100 tok/s needs a step of 32.1 ms or less at 3.2 tokens/step (24.8 ms or less
  at 2.5); the best-case levers stop near 36 ms. The 100 tok/s seen earlier was a repetitive-filler artifact (3.98
  tokens/step).

## 9. What is NOT done

* **Inkling-vs-Qwen quality and verbosity comparison.** Only Inkling-vs-llama.cpp (12/12 top-1, mean KL 0.0013 /
  0.0015) and old-vs-new build comparisons exist. No head-to-head against the Qwen3.8 production model, and no
  measurement of answer length/verbosity differences.
* **`switch-stack.sh` rewrite.** It still knows only the OVMS / Qwen TP2 pair, its `do_tp2` rewrites the
  `tp2-prod.service` drop-in, and the dashboard's server-side `POST /api/stack` has no Inkling guard (GO-LIVE §4).
* **Upstream list never tried:** vLLM #68303 was never evaluated for prefill. Not attempted either: bounded 16K draft
  window (acceptance risk unmeasured), a redesigned decode MoE kernel: REVISED Nov 22 - the micro-bench (`agents/moe_micro.py`) shows the NVFP4
  decode kernel already at ~75-95% of achievable bandwidth (the earlier 261-vs-468 GB/s table miscounted
  bytes, skipping w2; the dequant costs ~5% over raw loads), and per-pair re-reads leave only ~2 ms of the
  12.4 against the unique-expert floor, so the "4-5 ms" estimate was optimistic and the item is
  deprioritized; untried angle remains a vendor bf16-activation FP4 GEMM. Still open: investigating the
  near-attention growth, a real TurboQuant-style KV format (only simulated:
  4-bit KL 0.0055 vs uniform int4 0.0595 vs fp8 0.0020), a bf16-activation FP4 kernel in a vendor/SYCL library, and
  larger prefill chunks without the capacity loss.
* **Power efficiency analysis** (section 12): tokens-per-second-per-watt and the serialisation cost are not recorded.
* The handoff log `INKLING2_XPU_HANDOFF.md` stops at Session 3; this document set replaces it for later work.
* Items in the OpenCode provider block and `inkling2-prod.service` still need to be filled in (GO-LIVE §1.1, §3).

## 10. Kernel fusion (Nov 22) and the exactness evidence

Scope: the tiny kernels in the decode step — casts, concatenation, conv-state write, row norms. Requirement: bit-exact
numerics; a fusion that could not be made bit-exact was not taken. Result: kernels 2342 → 1762, device time
27.94 → 26.38 ms (short context) and 36.05 → 35.44 ms (250K) (PERF-LOG §9).

Taken (all verified bitwise):

* conv-state write folded into the decode conv kernel; strided K/V projections read in place (no `.contiguous()`);
  conv output emitted as bf16-rounded fp32 where the next consumer wanted fp32;
* one `_prep` kernel for the q/k/v/r row operations and the KV-cache write;
* per-call `rel_proj.float()` hoisted into a persistent fp32 copy.

Rejected: a decode-size mix kernel without the stacked copy of the block bank. It was not bit-identical to the
stacked kernel (10 / 24 / 10 / 16 / 17 / 22 mismatching cases for T = 1 / 100 / 2 / 4 / 8 / 37; fp32-level differences
from a different reduction layout), and the stack is only one launch per call, so the code was removed.

Caught during verification: making the attention-conv output fp32 would have routed four mixes per prefill chunk
through the multi-pointer mix kernel (rel error 2e-5 against the stacked kernel, accepted earlier for prefill) where
bf16 input had forced the stacked kernel. `mix_op` now takes `allow_mp`, and the `first`-visit calls pass
`allow_mp=False`, which keeps the previous numerics.

Evidence: bitwise unit tests (conv: strided input, fp32 output, fused cache write, 4 exact cases plus the 12 existing
cases; prep: 20 cases over T = 1 / 4 / 8 / 37 / 300, bf16 and fp8 caches, tau on and off, a padded slot);
the 12-prompt dump identical to the pre-fusion build (mean KL 0.000000, top-1 12/12, greedy identical 24/24 — twice,
including after the routing fix); a cold 32K needle found; and on 2026-11-22, after the final build, the kernel
tests `test_triton_sconv`, `test_prep_fused`, `test_mix_mp`, `test_ctrl_gemm`, `test_triton_attention`,
`test_triton_block_attention` and `test_moe_tiered` all pass. Tokens per step is **not** an exactness witness
across launches (section 8).

## 11. Corrections to earlier beliefs

| Earlier belief | What measurement showed |
|---|---|
| ~550 ms per prefill chunk unattributed | `pre_attn` 202 + `pre_mlp` 98 = 300 ms (loop controllers and mixes), short convs 107 ms (53 + 54), attention all-reduce ~100 ms; tiered-MoE capacity buckets were already the default (PERF-LOG §2) |
| Per-kernel dispatch gap ~8 ms per decode step | graph replay 0.90 us per tiny kernel, ~2 us interleaved with collectives; 96 all-reduces 1.6 ms in a graph; total gap 2-4.5 ms |
| Verify rows each sweep the whole KV (Nx cost) | they share one sweep, but the vendor kernel runs q_len > 1 almost serially (about 10 GB/s vs 349 GB/s); 22.5 ms vs 0.6-0.85 ms per layer call at 210K |
| Far attention ~8.5 ms (initial estimate) | 5.0 ms in the kernel microbench at 220K; 7.57 ms in the in-engine capture at ~250K (0.95 ms per visit, 8 visits) |
| Draft pass context-proportional guess ("~6 ms unattributed") | confirmed: drafter global layer 7.9 ms at 220K, 3.4 ms after the group fold, 2.08 ms after the retune |
| MoE is ~25 ms of the decode step | 0.262 ms per visit at the verify size = 12.4 ms (25 ms was the T=8 figure) |
| ~100 tok/s at depth with 3.98 tokens/step | 3.98 was repetitive filler; real text 3.2 (code) / 2.5 (prose) |
| Decode phase timers under eager PROF show where time goes | eager wall is 110-122 ms vs ~39 ms in a graph (launch bound); per-kernel device time from the profiler is the usable measure |
| First profiler capture (T=8) was a real decode step | it was a warm-up shape (41.68 ms); the real T=4 verify forward is 27.94 ms |

## 12. Rejected ideas and observations

* **bf16 experts for decode (hypothesis: no dequant, so faster).** Rejected by measurement. At the verify shapes the
  bf16 kernel is slower than NVFP4 by 1.64x-2.43x across T = 1-8 on TP1 and TP2 (TP2 T=4: 0.519 vs 0.262 ms per
  visit, 1.98x; T=8: 0.946 vs 0.516 ms, 1.83x), even though it reaches 468-535 GB/s against NVFP4's 261-276 GB/s,
  because it reads about 3.5x more bytes. Footprint: the dequantised bf16 checkpoint (`models/inkling-bf16`) is
  49.2 GiB against the 18.19 GiB NVFP4 checkpoint (8.9 GiB per card). *(The request for this document quoted
  "1.6-2.1x"; the measured range above is the recorded one.)*
* **Hot-expert residency / dequantised-expert cache:** negative for decode; the dequant cache default budget is 0.
  (Figures for this were not preserved in the record.)
* **bf16 far KV in the far-attention path (Option C):** rejected by the user — it doubles KV memory and gives up the
  2x-256K concurrency operating point to speed up one phase.
* **Larger prefill chunks (4096):** section 6.
* **Power observation (session record, 2026-11-22; board-level readings under decode load, not re-measured
  for this document):** 160 W / 140 W for Inkling against 195 W / 170 W for Qwen (the two cards). Combined
  with the decode throughput in PERF-LOG, Inkling lands at roughly 1.4x the tokens-per-watt of the Qwen
  stack at the 262144 cap. The residual inefficiency is attributed to the serialized draft/verify phases
  (the DSpark draft pass runs serially with the target pass), which raise per-token energy over a single
  fused pass.

## 13. Speculative A/B and the KV-cache dtype call (Nov 22)

* **method:"mtp" is not wired for Inkling2 - rejected as an A/B arm.** Both `--speculative-config
  {"method":"mtp"}` probes (with and without a draft model path) fail at startup:
  `Value error, The Inkling2 checkpoint has no mtp_config`. Mechanism: the draft-config loader rewrites any
  `inkling2`/`inkling2_loop` draft config into an `Inkling2DSparkConfig` unconditionally (vllm/config/speculative.py),
  and the fork's classic-MTP path expects chain weights (e_proj/h_proj/enorm/hnorm) that this checkpoint family
  does not ship - its `mtp.safetensors` is a DSpark-stage head (main_proj/markov_head/confidence_head). Only
  Inkling-1 has an MTP implementation (`inkling_mtp`). Wiring plain MTP for Inkling2 would mean writing a new
  chain-MTP draft model, not a config change. The planned DSpark-vs-MTP A/B became the depth sweep below.
* **Speculative depth: keep 3.** Depth 4 wins code (+11% at 234K) but collapses prose (51.1 vs 66.7 tok/s,
  acceptance 2.48 vs 2.74 tok/step, ~49 ms steps); depth 2 loses 10-20% across the board. Data in PERF-LOG
  section 11.1. Open observation: depth 4's per-step acceptance falling below depth 3's on identical prompts.
* **FP8 KV cache: rejected for production; production switched to BF16 KV.** Live FP8-KV vs BF16-KV (deterministic,
  reproduced byte-identically across two capture pairs): mean KL 0.318 over shared top-20 mass (gate <= 0.002,
  ~160x over), top-1 agreement 443/576, greedy text 7/12, max position KL 16.58. Needle retrieval unaffected
  (FOUND at 61K/124K/243K on both). The kv_sim fp8 prediction (0.0020) missed by ~160x - sim numbers do not
  validate the real path on this stack. BF16 KV is also faster (234K code 77.2 vs 68.5 tok/s, step 44.4 vs
  48.0 ms, prefill ~7% faster); its only cost is the KV pool: 640,939 -> 439,959 tokens (2.45x -> 1.68x
  concurrency at the 262144 cap). Accepted. (The earlier "bf16 far KV = Option C rejected" decision concerned
  doubling the FAR-path KV only, at the old 2x-256K operating point; this decision is the full-cache dtype.)
* **FP8-KV fix plan: live-validate TurboQuant, then implement packed tq4 KV if it passes.** The sim (Nov 21)
  had TurboQuant-style tq4 at KL 0.0055 vs naive int4 0.0595; today's lesson is that only live numbers count.
  Suspected root cause of the FP8 failure: unit-scale e4m3 on unnormalized values (`use_value_norm_no_gamma`);
  a Hadamard rotation + Lloyd-Max codebook addresses exactly that. If live tq4 quality holds, implement a real
  packed 4-bit KV (fused Hadamard+codebook in the attention kernel) for ~2x the FP8 pool at near-reference
  quality - owned end-to-end in this port's Triton kernels. A cheaper intermediate: per-head (calibrated)
  FP8 scales instead of the unit-scale clip.
* **Acceptance is launch-deterministic on this stack:** same dtype across launches = 576/576 top-1 agreement,
  KL 0.00000, 12/12 greedy text (both dtypes). Cross-dtype or cross-depth differences are therefore real
  effects, not noise.

* **TurboQuant tq4: live-rejected too (Nov 22).** Simulated 0.0055; live 0.342 mean KL, top-1 69%,
  greedy 4/12 - the same damage as live FP8 KV. The FP8-KV fix plan is revised accordingly: no quantised-KV
  engineering (per-head fp8 scales, packed tq4, anything) until the kv_sim-vs-live gap (~60-160x on both
  schemes) is root-caused - the sim harness does not model the real cache path (suspects: clip / value-norm
  transforms x the loop architecture's per-pass cache writes; TP1-vs-TP2 layout). Live validation via
  ab_quality is mandatory for any KV experiment from now on; eager captures are valid for the graphed
  production config (proven 576/576, KL 0.00000).


## 14. One-shot P2P all-reduce: rejected (Nov 22)

`VLLM_XPU_ONESHOT_AR=1` fails the exactness gate on the current BF16 build: 554/576 top-1 agreement vs
the bf16 reference, shared-top-20 KL mean 0.069 / max 11.5 against the KL <= 0.002 gate, and no speed
win measured. The FP8-era "+1.2%, bit-identical" result does not transfer (and the first retest of it
was itself an env-passthrough no-op - both arms were the production config). The knob stays OFF in
production and is not recommended for any workload. Detail: PERF-LOG section 13.
