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
  the machine stopped answering [redacted] pings, and it had to be power-cycled (reported by the user). From then
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
  deprioritized; untried angle remains a vendor bf16-activation FP4 GEMM. the near-attention growth was CLOSED Nov 22 - a controlled scatter A/B (fresh vs fully-churned
  free-queue, same 134K context) showed zero effect and the original anomaly did not reproduce
  (launch noise, PERF-LOG section 15), a real TurboQuant-style KV format (only simulated:
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


## 15. Decode "scheduler/dispatch gap" lever: does not exist (Nov 22)

The ~8-13 ms unattributed remainder in the decode-step model was an artifact of a sync-perturbed tprof
body measurement (27.9 ms was a lower bound). GPU-event iteration timing - after patching the CUDA-only
gate in vllm/config/vllm.py to include XPU - shows the step bracket at p50 34.4 ms against a 32-36 ms
wall: the device bracket covers the whole step and there is no large host-side gap to reclaim. Decode
is device-work-bound end to end; what remains is small kernel work (LM head + sampling graph coverage,
~1-2 ms) and the near-attention growth (~1 ms). Detail: PERF-LOG section 14.


## 16. Remaining-lever audit closures (Nov 22)

- Near-attention scatter fix: closed, zero measured effect (see PERF-LOG section 15).
- Vendor oneDNN FP4 GEMM: capability confirmed in the bundled build (v3.13.3, nvfp4 symbols in
  libtorch_xpu.so), still parked - no Python path, layout repack + C++ binding for <=~2 ms.
- Free-queue sorting / block-size changes: not pursued (scatter result makes them pointless).
- Decode-latency campaign status: every millisecond of the 34.4 ms step bracket is attributed; remaining
  open item is only the eager-tail gap measurement gating LM-head+sampling graph coverage (~1-2 ms best
  case). The larger real levers are outside serving: drafter retraining (+14-38%) and throughput mode
  (1.6-1.7x at 2 streams).


## 17. LM-head + sampling graph coverage: PARKED (Nov 22)

The last candidate lever (~1-2 ms) is parked. Arithmetic bounds the entire eager tail (LM head +
logits all-gather + sampler + accept + gaps) at <= ~2.6 ms worst case against the 34.4 ms bracket
(draft 0.4, head floor 0.790 measured at 586 GB/s), and the gap component - the only part graphing
removes - could not be measured: the profiling instrument caused an incident (two HTTP 500s in patched
windows, one production hang via a leftover flag + persistent source patch) and is documented in
PERF-LOG section 16 with a safe redesign (bounded with-block, one step-pair, off-production window, or
pti/unitrace). The expected gain does not justify another production-affecting window. The decode
optimization campaign is CLOSED: every millisecond of the step is attributed, every lever measured,
adopted, or parked with recorded evidence.


## 18. System DRAM: leave as is (Nov 22)

Full accounting (PERF-LOG section 17) confirms the mirror finding: GPUActive 57.7 GiB is kernel-side,
unreclaimable, and the dominant term; process anon memory is ~17 GiB; headroom 16 GiB available + 5 GiB
swap with PSI at zero. No legitimate xe/driver knob reduces the mirror; overcommit mode 2 would break
vLLM; a container memory cap would be harmful. Verdict: expected-per-the-mirror, not wasteful, not
dangerous - no change. Optional insurance (bigger swap or zram) noted for a maintenance window only.


## 19. Host-DRAM mirror: fixable, workaround verified (Nov 22)

Supersedes section 18's "leave as is". The 57.7 GiB mirror is compute-runtime regression #953
(defer-backing for BMG, 26.09+), not a hardware or driver cost. Verified fix on this stack:
NEOReadDebugKeys=1 EnableDeferBacking=0 - GPUActive 51.10 -> 0.57 GiB at boot, MemAvailable
26.5 -> 74.5 GiB, no load growth, deep4 bands normal, exactness gate 576/576 (KL(defer0||bf16) over shared top-20: mean 0.00000  max 0.00000  (576 positions)).
Adoption decision (debug-key caveat + kernel-7.0 residual upstream) recorded with the data in
PERF-LOG section 18.


## 20. Production config: 3072 chunks + defer-backing off (ADOPTED Nov 22)

Adopted after gate-verified verification (bit-identical outputs on both changes, decode bands
unchanged, prefill +20.2% at 241K / +13.8% at 32K, MemAvailable 78 GiB under production). The unit
file is now version-controlled at SETUP/inkling2-prod.service. Rollback documented in PERF-LOG
section 19. Prefill remains open on two kernel projects: split-KV global attention and tiered-MoE
grouped-GEMM rework (next campaign).


## 21. 3072 chunks reverted to 2048 (Nov 22, user decision)

The 3072-chunk adoption lasted one session: the user preferred 1.66x concurrency (434K-token bf16 pool)
over the +20.2% prefill at 241K. Defer-backing stays. The chunk-size option remains documented
(+13.8% at 32K / +20.2% at 241K, bit-identical, 1.23x concurrency) and can be re-adopted by a one-line
unit edit if the concurrency trade is ever acceptable. The verify pass also confirmed the 641K pool
record was FP8-era and not comparable to any bf16 figure (PERF-LOG section 19 addendum).


## 22. fp8-dequant MoE project: CLOSED by probe (Nov 22)

The proposed fp8 tiered-MoE path (dequant to fp8 + fp8 GEMM to halve the ~192 GB/chunk dequant
traffic) is dead on this stack, measured: torch._scaled_mm exists and supports per-tensor and rowwise
scales but REJECTS e8m0/MX-style block scales ("Invalid scaling configuration") - the NVFP4 per-16
scales cannot be represented, so the GEMM would not be exact; and batched fp8 torch.bmm works but is
SLOWER than bf16 at the tier shape (3.99 vs 2.90 ms; 170 vs 469 GB/s weight stream) - the fp8 path
does not reach the fast XMX rate. The dequant bandwidth cost of the bmm+oneDNN design is structural.
Remaining prefill items are small polish (allreduce overlap, fusions, local tile sweep; ~5-8% total).


## 23. Cross-GPU P2P: unavailable on this topology (verified Nov 22)

Both B65s are CPU-direct (separate Raphael root ports, same host bridge), but xe/L0 refuses
cross-root-port P2PDMA on BMG (upstream #935) - measured: cross-device copy at 6.6-6.7 GB/s through
host-staged GTT vs 215.7 GB/s same-device. All TP2 collective and EP traffic is host-mediated; the
~6.7 GB/s ceiling explains the allreduce timings. Not user-space-fixable; a future kernel enabling BMG
P2PDMA is the lever (PERF-LOG section 19 addendum for numbers). The defer-backing fix remains
mandatory on this topology.


## 24. DSpark depth: content-adaptive, not length-adaptive (A/B Nov 22)

The bundled drafter is an 8-row block (mtp_config.block_size=8) and accepts deep positions when
run at full depth: k=7 beats k=3 by +26%/+14% (code-continue 8K/250K) and +18% (verbatim 8K), but
LOSES -12% (prose) and -23% (verbatim 250K, flat deep acceptance). The depth optimum tracks
drafter confidence (content predictability), not context length. Static k=7 is not adoptable;
production stays k=3 until adaptive verification (enable_adaptive_verification, in-fork,
DSpark-only, config-only) is A/B'd - expected best-of-both ~+11% blended. F3zz1k's k=7 result
explained: full-block drafter + native draft attention. Kernel-port order settled: MoE NVFP4
grouped GEMM first (~2x/visit), attention second, int8 last (gate). Full table in PERF-LOG.


## 25. Depth map: no dominant static config; ladder semantics verified; adaptive parked as L

Four-config depth A/B (k3 / shrunken-k5 / ladder-k5 native-block / k7, deep4 x3 each): blends
78.6 / 85.0 / 84.4 / 82.9 tok/s, but per-workload optima differ (code->k7, prose-250K->shrunken
k5, verbatim-8K->ladder-k5 record 139.0, prose-8K/verbatim-250K->k3). Block-shrink distortion
proven: native-block drafts accept 2.74 where shrunken-4-row accepts 1.71 on verbatim-250K.
Production stays k=3 pending a traffic-mix call: shrunken-k5 (+8.2% blended, worst cell -15% on
verbatim-250K) vs ladder-k5 (+7.3% blended, worst cell -15% on 250K prose) vs k7 for code-heavy
mixes. Adaptive verification = the measured +18.2%-blended prize but L-size on our XPU backend
(host-side metadata planning; silent-corruption risk) - parked as a campaign; estimator-alone
(needs no confidence head) confirmed. Ladder config verified working on dspark/XPU for the
concurrency axis.


## 26. Gate policy: bit-exactness measures decode shape, not quality (raised Nov 22)

Quality-gate captures across decode shapes: k7-spec and the 2-entry ladder deviate identically
from the k3 reference (525/576 top-1, KL 0.067 mean); NO-SPEC deviates more (450/576, KL 0.212).
The reference is the 4-row-shape trajectory; the decode forward is shape-sensitive and greedy
trajectories diverge chaotically per shape. The gate as written blocks every depth change, the
adaptive-verification path, and the F3zz1k kernel port, while production's own concurrent traffic
already deviates from the bs=1 reference. Production stays k=3 under the standing rule. Operator
decision raised: keep bit-exactness (k3 permanent) / tolerance gate for shape noise / fund
fixed-shape padded decode for true shape-invariance. Evidence: PERF-LOG gate-investigation
section (this date); captures ab-quality-k2e/k7q/nospec.json vs ab-quality-bf16.json.

### RESOLVED Nov 22: bit-exactness gate retained; k=3 permanent (operator decision)

The operator chose option 1: the bit-exactness gate stands as written, k=3 is the permanent
production spec depth. Consequences, stated plainly: (a) the depth menu (k7/k5/ladder-k5,
blended +5.5..+8.2%, best-per-workload +18.2%) is closed; (b) adaptive verification is closed
(it was L-size anyway); (c) the F3zz1k MoE NVFP4 kernel port is closed as an output-changing
lever - a different GEMM reduction structure at the same shape perturbs outputs exactly like a
shape change does, so it cannot pass this gate; (d) fusion/tile/overlap levers (allreduce
overlap, pre_attn/pre_mlp fusion, attention tile sweep) are likewise gated out - they all
reorder reductions. What SURVIVES the gate: output-preserving levers only - the DFlash2 drafter
training project (a better drafter changes drafts, not verification: at num_spec=3 the verify
shape stays the 4-row production shape, so the trajectory and outputs are unchanged while
tok/step rises; estimated +10-14% decode, multi-day distillation project), memory-side tuning
(pool/utilization), and the upstream kernel P2P unlock watch (prefill lever, environment-level).
This is the endstate: quality is bit-pinned to the 4-row production trajectory; speed
improvements must come from draft quality or memory, never from touching target-side numerics.

## 27. Vision tower port adopted - gate-clean via requires_raw_input_tokens (Nov 23)

The checkpoint ships a full PE-Lang-style ViT tower (46 tensors, 0.593 GiB) plus a complete
image processor; the fork served text-only with an explicit stub. Ported the tower from the
community reference implementation (F3zz1k's vision.py copied verbatim, 315 lines: a
plain-torch eager encoder, no new kernels, runs once per image), wired embed_multimodal and
the load_weights visual routing. Three boot bugs fixed on the way: a duplicated-def splice
from the first patch, the loaded-names prefix (encoder params register as visual.*, not
model.visual.*), and a second stub - the top-level embed_input_ids raised on multimodal
inputs; wired the standard _merge_multimodal_embeddings.

First gate run on an image-enabled boot failed 495/576 (KL mean 0.386, bit-stable across
boots) while vision itself answered correctly ("a red circle"). Ruled out, in order: request
poisoning (a suite-first run reproduced the deviation digit-for-digit, so it was config, not
state); KV pool size (an image-0 boot pinned via --kv-cache-memory-bytes to the mm boot's
exact 412,829-token pool scored 576/576 bit-exact - pool size is NOT a trajectory axis); the
tower's resident weights (a forced-tower image-0 boot: 576/576 bit-exact); resolved engine
configs (byte-identical between image-0 and image-1 boots); and text tokenization
(short-prompt ptoks identical across all boots; the needle-prompt ptoks differences are a
harness artifact - random filler per run). Root cause found in the boot log: with mm active
the speculator warns "draft model does not support external multimodal embeddings" and the
runner switches to uses_inputs_embeds mode - embeddings are built by
encoder_runner.get_inputs_embeds OUTSIDE the captured decode graph, so the in-graph embedding
(which graph-pass fusion treats, e.g. fuse_norm_quant) is replaced by an unfused eager
variant, and every decode forward diverges bit-wise from position 0. The drafter is
unaffected: it receives text-only draft inputs on both boot types (supports_mm_inputs is
false either way).

Fix: requires_raw_input_tokens = True on the model class. The runner keeps input_ids; the
model re-embeds internally (identical ops, inside the graph - captured decode graphs are
byte-identical to text-only boots) and overlays the runner-merged embeddings only at image
placeholder rows (token id 200054), inert for text rows. The overlay is guarded by
torch.cuda.is_current_stream_capturing(): host syncs (.any()) are illegal during level-zero
capture (UR_RESULT_ERROR_UNSUPPORTED_FEATURE 44 - two boots burned on this; the traceback
pointed at the .any() line, initially misread as a broadcast-torch.where limitation), and
decode rows never carry image embeddings anyway (image tokens live in the prompt KV cache),
so skipping the overlay during capture is semantically exact while eager prefill applies it.
The broadcast-where was also replaced with an expand+contiguous same-shape where as a
precaution.

Verified on the image-enabled boot: text gate 576/576 KL 0.0 (bit-exact), image smoke answers
correctly with finish=stop. Vision is a one-flag switch (--limit-mm-per-prompt image:1);
production stays image:0 by default - the mm boot costs ~0.6 GiB/card of KV pool (434,084 ->
412,587 tokens, 1.66x -> 1.57x concurrency) for the tower and encoder cache, and vision has
not been asked for in production. The text-only trajectory is bit-identical under both
configs (proven on this boot), so flipping the flag carries no re-verification burden beyond
one gate run.

Nov 23, later: production switched to image:1. The gate re-verified on the production boot
path itself (576/576 KL 0.0, greedy 12/12), a fresh unseen image answered correctly
(finish=stop), and the pool trade is accepted: 434,084 -> 412,587 tokens (1.66x -> 1.57x
concurrency) in exchange for full model capability. Rollback = the .bak-image0 serve script
plus a restart.

## 28. Determinism crisis: root-caused to torch-XPU batched matmul; fixed at the logits head (Nov 23)

After the vision/u-0.88 changes settled, fresh-boot first gate runs began deviating (552/576,
KL ~0.11) while later runs on the same boot were bit-exact - first seen at the 200K-cap boot
and surviving the revert to 262144. A Claude-consulted investigation (evidence package
preserved under work/evidence-bistability.md; tools in SETUP/inkling-det-tools/) localized it
with a hash-based determinism probe:

- The split path (per-row matmuls) is deterministic: 0 flips in ~1,200 executions, both cards.
- The BATCHED matmul is bi-modal per process era: with hash-verified-stable inputs, whole eras
  of execution return systematically different bits (card 1: 200/200 iterations deviant in one
  process); era selection is timing/state random. Our Triton kernels, the oneDNN F.linear path,
  and GPU memory (input hashes stable 0/200) are all exonerated.
- The serving path's exposure is the LOGITS HEAD: model.py computed the 201K-vocab logits with
  a batched torch.bmm(out_dtype=fp32) every decode step - the gate measures exactly those
  logits. The tiered-MoE bmm fallback and the drafter's fp32 matmuls are the same family; at
  the gate's shapes the Triton path covers the MoE, and the drafter's exposure is a queued
  follow-up check.

FIX (gate-clean by construction): the logits head now computes the sampler rows (<=8 per step)
with per-row torch.bmm(out_dtype=fp32) - measured bit-identical to the batched form's stable
mode (equiv_probe: identical digests on both cards, 0 flips in 50 executions) and flake-free.
Cost: <=8 small bmms per step instead of 1 (negligible at serving sizes).

Verified: three consecutive fresh-boot gates 576/576 KL 0.000 (run 1 was the reliably-bad case
before the fix), live canary PASS, vision answers correctly.

PERMANENT GUARD: the boot canary (SETUP/inkling-det-tools/canary/, runtime copy under
work/inkling-det/canary/) - after every restart, 4 fixed prompts x48 positions including the
known-deviant short[1] and short[4] are checked bit-exact (tokens + top-20 logprobs) against
canary_ref.json with a zero-prefix-cache-hit assertion; traffic is admitted only on a pass,
max 2 restarts, then flagged. restart-prod-canary.sh is the restart path. A
CANARY_ALLOW_WARM=1 Mode B (warm-rerun admission) exists as a temporary escape and is to be
retired once upstream ships a fix.

UPSTREAM: the defect is in Intel's torch-XPU batched matmul (torch 2.15.0+xpu, SYCL runtime
2026.1.2, both B65 cards affected; F3zz1k's 2.14.0 stack uses the standard LogitsProcessor
matmul and is plausibly exposed too). det_probe is the minimal repro: cross-process digest
flips plus within-process era flips on plain bf16 matmul with hash-stable inputs. Issue to be
filed on intel/torch-xpu-ops.

Also recorded this day: the 200K cap was rejected (552/576; pool 441,203 -> 386,815 - the
planner quantizes worse at 200K, so 262144 strictly dominates) and max_model_len joins
max-model-len-shape as a trajectory axis in the config-freeze list; u=0.88 adopted (pool
restored to ~441.6K, 1.68x; 0.95 rejected on OOM math - the 250K headroom would go to about
-1.7 GiB per card).

## 29. Chunk-16 fp32 logits head: exact and 15x cheaper at 32 rows (Nov 24)

The fp32 head previously ran one `bmm` per row (adopted for the batched-matmul
mode-B defect). A row sweep on this stack (torch 2.15.0+xpu, SYCL 2026.1.2)
showed oneMKL switching GEMM kernels at M=26: rows <=25 are bit-identical to
the per-row result, rows 26-32 are not. Chunking at 16 rows keeps every GEMM
inside the exact regime and reads the bf16 weight once per chunk instead of
once per row: 49.5 -> 3.4 ms at 32 rows, 24.8 -> 1.7 ms at 16. Adopted after
three fresh-boot gates (576/576, KL 0.000). `INKLING2_HEAD_PER_ROW=1` restores
the old path for one release. Re-run the row sweep on any torch/SYCL/driver
upgrade; do not pre-transpose the weight (changes the last bits, 2.15e-6) and
do not materialise a repeated weight (696 ms).

## 30. All-reduce simple threshold at 4 MiB (Nov 24)

`CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD` was 4 GiB, which sent every TP2
all-reduce down the latency-heavy path. At 4 MiB: prefill 8K 2,017 -> 2,114
(+4.8%), 128K 1,771 -> 1,857 (+4.9%), decode unchanged. A second 64K sample read
15.7% higher; with one sample each, neither 64K figure is quoted. The alternative oneCCL transports tried here did not initialise in this container
(they need a launch model or device access the server does not have), so this
threshold is the only all-reduce win available without restructuring the
launcher.

## 31. Exact-tail ring prefix cache: opt-in, single sequence (Nov 24)

Problem: a cached resend can resume at a block-aligned boundary and recompute
its final chunk at a different size than the original run. On this stack the
result is close but not bit-identical, and the recurrent tail (sliding-window
conv cache, local ring, drafter state) is not what vLLM's block-hash cache
restores.

Design: the worker captures the last `slack` rows of recurrent state plus the
final-chunk hidden rows and the paged KV rows at the exact frontier, and
replays them when a resend ends exactly there. The scheduler admits only hits
the retained frontier can cover; the worker re-raises if it ever sees an
uncovered position, so state is never silently reused. 75K and 128K resends
are bit-identical at 0.32 / 0.44 s.

Status: opt-in (`INK2_RING_PREFIX_CACHE=1`), single sequence only
(`--max-num-seqs 1`), with prefix caching on. State rows for hits inside the
retained slack window are rewritten into the request's own slots; a hit whose
rows cannot be written raises instead of being served. The guard covers eager
forwards -- graph-mode decode neither reads nor mutates the markers. What is
missing for concurrency is per-request capture and a scheduler that admits only
boundaries captured for that request. Measurement on Nov 24 puts a harder limit
upstream of the cache: at eight concurrent sequences (batched matmul off, with
and without the drafter), two identical cold runs mismatched in 13-15 of 40
comparisons, nearly all at the first generated token, while cache-hit runs were
lower (6-9 of 40). The cause is likely batch composition or scheduling and was
not isolated, so an exact cache is necessary but not sufficient. Separately,
two cold runs of one long prompt at a single sequence can also flip at a
near-tie (0/4 at ~1.4K characters, 3-4/4 at 2-3.4K), so token identity is not
a property the stack holds universally; the gate is a KL bound. See PERF-LOG.

## 35. Two profiles, and what is parked (Nov 24)

**Profile split.** `SPEC=1` (default) runs the DSpark drafter; `SPEC=0` is the
exactness profile. They do not produce the same answers: on the 12 gate prompts
the drafter changes 126 of 576 top-1 positions with a median old margin of 1.99
logprob units, and 7 of 12 greedy texts. Each profile therefore needs its own
reference capture; a gate captured under one cannot judge the other.

**Exactness profile is reproducible.** Two idle sequential captures agree on
576/576 (KL 0.00000); 8 identical prompts in one batch give 0/7 flips over three
repeats. Captures must not overlap with other traffic: a concurrent capture
measured 558/576.

**Parked, with reasons:**
- *Router GEMM, long single prompts.* The fp32 gate GEMM is bit-stable at
  T<=256 and not at T>=512. Deterministic algorithms fix it (0/4 flips at every
  length) but change the short-prompt bits, so it needs a one-time re-baseline
  rather than a code change.
- *Attention row geometry.* Identical rows enter the attention kernel and
  differ on exit (attn_in 0/6475 vs attn_part 3231/6475, 1045 with the
  batch-invariant wrapper). The likely cause is tiles aligned to the chunk start
  rather than absolute positions; an independent implementation feeds relative
  position keyed to absolute KV, which is the direction to take.
- *Speculative decoding at depth 8.* At k=7 outputs change and throughput drops
  (432/576, 54.6 tok/s against 92 at k=3), so the verification path is not
  strictly output-preserving at that depth. Fixing it would turn the same
  hardware into more throughput, but it is separate from the profiles above.
- *Alternative drafter* (a retrained DSpark head, ~606 MB, claim 2.98x / accept
  4.36). Rejected at both depths: k=7 gave 432/576 and 54.6 tok/s, k=3 gave
  445/576 agreement (131 positions moved) and 72 tok/s, against 576/576 and
  92 tok/s for the shipped drafter. It is slower in our stack, not faster.

## 36. Acceptance, not attention: what actually moves tokens (Nov 25)

Following the profile split in §35, a six-arm experiment pins down *where* speculative
decoding changes a greedy continuation. Each arm captures the same 12 prompts x 48
tokens (576 positions) and is compared against the frozen SPEC=0 reference. The
reference reproduces bit-exactly across boots and days: an anchor captured two days
earlier still matches at 576/576, KL 0.

| arm | configuration | top-1 | prompts diverging |
| --- | --- | ---: | ---: |
| A | SPEC=0 reference | 576/576 | - |
| B | drafter resident, zero proposals | 576/576 | 0/12 |
| C | reject every draft, verify at k+1 rows | 575/576 | 1/12 |
| D | accept at most the first draft | 424/576 | 8/12 |
| E | as D, but later drafts zeroed instead of inverted | 423/576 | 8/12 |
| H | k=1, natural acceptance | 401/576 | 8/12 |
| - | drafter fully active | 450/576 | 5/12 |

D and E separate later-draft content from the accept path and differ on one prompt
only; their first divergences are the same list. H accepts a draft and never rejects a
tail, and diverges with the same onset positions as D. The trigger is acceptance
itself: neither the number of drafts nor tail rejection matters, and a drafter that is
merely resident changes nothing.

**The attention row geometry is not the cause at these lengths.** An isolated kernel
probe (synthetic tensors, production block-table width) compares one row computed alone
against the same row inside a k+1-row block:

| sequence length | sliding-window layer | full-attention layer |
| --- | --- | --- |
| <= 512 | bitwise equal, every row | bitwise equal, every row |
| 1100, 2048 | rows 1..k differ | all rows differ |

Divergence appears only once the sequence passes the relative-bias extent (1024),
which the 576-position gate never reaches. A per-row verify path that is bit-identical
by construction was implemented and measured end to end: it changes nothing (424/576,
identical to arm D), because at these lengths there is nothing to change.

**What does the work.** Rows after the first read KV that the current forward wrote
itself, from hidden states computed at k+1 rows. A sequential decode would have written
those entries from a one-row forward. One-row and k+1-row forwards are not bit-identical
- dense GEMMs, the MoE router and grouped experts, and the norms all re-tile with row
count - so an accepted token hands the next row a key/value that differs in its last
bits. The first row reads only KV written by earlier steps, which is why rejecting
every draft stays clean and every accepting arm diverges.

**Consequence (revised in §37).** Exact speculative decoding here needs more than a
kernel-local fix; §37 identifies what actually differs. Until then the
two profiles stand as in §35: SPEC=0 for bit-exactness, the drafter for speed, with the
divergence confined to near-ties (median margin 0.133) and no measurable quality
difference on the checks in the README.

## 37. Campaign close-out: why speculative decoding changes greedy tokens (Nov 25)

This supersedes the closing paragraph of §36.

**Two independent effects, both measured.**

**E1 - input divergence.** When a draft token differs from the greedy path, the verify block
computes different per-expert group sizes and a different token path. Per-prompt onsets under
natural drafts appear as early as token 2. This survives perfectly deterministic kernels:
wrong drafts change the computation itself.

**E2 - verify-path-intrinsic accumulation.** With the token path held identical (drafts
replaced by the no-drafter reference stream), the k+1-row verify path still diverges from the
one-row decode path at token 39 (captured) and 41 (eager). The onset position is prompt- and
boot-dependent; the flip lands at the first near-tie the accumulated drift crosses. The two
are separate captured graphs of the same operations.

**Mechanism, all three layers measured.**

| layer | evidence |
| --- | --- |
| origin | fp32 reduction-order differences of about 4e-6, measured in the fp32 cross-pass readout that feeds the loop controllers (a batched matmul over the previous pass's outputs) |
| crossing | bf16 layer outputs hide them: a 4e-6 difference sits well below the bf16 grid (about 4e-3 relative), so roughly one component in 3000 survives a crossing |
| observable | the boundary row's top-20 rotates 0.1-0.2 nats before any argmax moves (20/20 and 19/19 coordinates above 1e-5, 18/19 above 1e-2, max 0.19); the two competing tokens move in opposite directions for a relative swing of about 0.29 nats; the flip lands at token 41 |

**Consequences.**

- The LM head is the **last carrier, not the origin**. A deterministic logits reduction would
  make its own contribution deterministic, which is 100-1000x below the measured shift, and
  would not remove an inherited 0.15-nat rotation.
- **No single operation is incorrect.** The verify and decode captures are separate
  compilations of the same operations, and the end-to-end divergence on identical tokens shows
  their frozen kernel selections do not agree row-for-row. Every operation probed in isolation
  is row-invariant (paged attention, router, both MoE paths, short conv, the loop controllers'
  gated blend at production shape, tensor-parallel all-reduce, bf16 dense). The one exception
  is the cross-pass readout above, which is run-to-run nondeterministic in eager execution even
  at a single row.
- Fix surface, honestly ordered: accept; capture-side per-row verify (bit-exact by
  construction - it replays the sequential computation, one row at a time, inside one graph
  replay - priced at roughly four step-equivalents against acceptance 1.6-3.0 and therefore
  net-negative as a production mode); or a hybrid once a repaired per-position map names the
  differing operations.

**Production call unchanged:** the drafter for speed, the no-drafter profile when bit-exactness
is required. Task-level quality is unchanged in measurement: identical benchmark scores in both
arms, and task outcomes differing by at most one near-tie task in the probe.

One open residue, recorded rather than glossed: the 0.1-0.2 nat rotations imply more bf16
survivors per layer than the one-in-3000 estimate suggests, so per-layer amplification is not
fully quantified.
