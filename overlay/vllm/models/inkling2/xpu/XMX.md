# Using the XMX matrix engines (Intel Arc Pro B65) in the Inkling 2 XPU port

The B65 (Xe2 / Battlemage) has XMX systolic matrix engines, driven by `DPAS`
instructions. All large matmuls in the Inkling 2 XPU kernels are written with
Triton's `tl.dot` on **bf16 tiles** so that the Intel Triton backend lowers them to
DPAS. Everything that is a matrix-vector product stays on the vector ALUs because
there it is memory-bound and XMX cannot help.

## Where XMX is used

| Kernel (file) | `tl.dot` | Notes |
|---|---|---|
| Attention QK^T and P·V (`triton_attention.py`) | yes (both) | bf16 Q/K/V tiles, fp32 accumulate; P is cast to bf16 before P·V |
| Grouped MoE gate/up and down GEMMs (`triton_moe_grouped.py`) | yes | NVFP4 / MXFP8 weights are dequantised **in the tile load** to exact bf16, then DPAS; outer scale applied in fp32 on the accumulator |
| Dense projections (qkvr, wo, router, lm_head) | via `torch.nn.functional.linear` (XPU GEMM backend) | not inspected; assumed to use XMX but **not verified** here |
| Decode MoE (`triton_moe.py`, `triton_moe_q.py`) | **no** | M=1 GEMV per (token, expert) pair, memory-bound (~460 GB/s achieved on BF16); XMX would not help |
| Short conv, attn-res mix, controller blend, clip/norm (`triton_sconv.py`, `triton_loop.py`, `triton_prep.py`) | no | elementwise / reductions, launch-latency bound |

## How to confirm a kernel really lowers to DPAS

Compile it and inspect the Triton GPU IR. The XMX encoding shows up as `#ttig.dpas`
with `opsPerChan = 2` for 16-bit inputs, and every `tt.dot` uses it:

```python
k = triton_attention._inkling_attn_kernel
for cache in k.device_caches.values():
    for compiled in cache[0].values():
        ttgir = compiled.asm["ttgir"]
        print([l for l in ttgir.splitlines() if "dpas" in l.lower()][:1])
        print("tt.dot:", ttgir.count("tt.dot"))
```

Observed for the attention kernel:
`#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, ...}>`
with `tt.dot` count 2 (QK and PV). (This is a structural check of the compiled IR, not a
hardware-counter measurement of XMX utilisation.)

## Rules that matter for XMX throughput

1. **Use 16-bit operands.** bf16 tiles go to DPAS (`opsPerChan = 2`). fp32 `tl.dot` inputs
   would fall back to a slow path, so P is cast to bf16 and dequantised weights are
   produced as exact bf16 (an e2m1 value times an e4m3 block scale has <= 5 mantissa
   bits, so it is exact in bf16).
2. **Fill the arrays with enough warps.** This was the largest lever. With the Triton default
   of 4 warps the XMX units were starved. Measured sweeps (see below) peaked at **16 warps**
   for 64x64 / 128x64 tiles. More is worse (register spills), fewer is much worse.
3. **Tile dims that are multiples of the DPAS shape** (M multiples of 8 / 32 per warp
   cluster, K multiple of 16, N multiple of 16/32). We use BM in {16, 64, 128},
   BN = 64, BK = 64, head_dim 128.
4. **Skip work the matrix engine would waste.** For attention, key blocks entirely beyond
   the relative-bias extent (dist >= 1024) skip the bias gather, and fully visible
   blocks skip the mask arithmetic; this took the kernel from 1.1 to 2.0 TFLOP/s before
   the warp tuning.
5. **Keep reductions off the critical path:** softmax statistics are fp32 on the vector
   units; only the two GEMMs per key block are on XMX.

## Measured results (single B65, TP2 per-rank shapes: 12 q heads, 4 kv heads, D=128)

Attention, prefill chunk of 2048 queries (`bench_attention_prefill.py`):

| config (BM, BN, warps) | ctx 16K | TFLOP/s |
|---|---|---|
| 64, 64, 4 (initial) | 193 ms | 1.1 |
| + bias/mask block skipping, 64,64,4 | 109 ms | 2.0 |
| 64, 64, 8 | 59 ms | 3.7 |
| 64, 32, 8 | 36 ms | 6.0 |
| 128, 64, 16 | 34 ms | 6.4 |
| **64, 64, 16 (default)** | **34 ms** | **6.5** |
| 64, 64, 32 | 60 ms | 3.7 |

Grouped MoE (`test_triton_moe_grouped.py`, 16384 valid pairs, NVFP4):

| (BM, BN, BK, warps) | time |
|---|---|
| 64, 64, 64, 4 | 27.9 ms |
| 64, 64, 64, 8 | 21.1 ms |
| 128, 64, 64, 16 | 16.2 ms |
| **128, 128, 64, 32** | **15.5 ms** |
| default >= 1024 tokens: 128, 64, 64, 16; smaller: 64, 64, 64, 8 | |

End-to-end (real model, TP2): prefill went 590 -> ~1200 tok/s at 16-32K context after
the attention work + MoE grouped GEMM + XMX tuning.

Override for experiments: `INKLING_ATTN_CFG=BM,BN,warps`, `INKLING_MOE_CFG=BM,BN,BK,warps`.

## Not yet using XMX / ideas

* Batched decode (2-8 tokens per step, e.g. 8 concurrent streams or speculative
  verification) still uses the per-pair GEMV kernel. Padding the token dimension to an
  M=8/16 DPAS tile would let the expert weights be read once per expert and reused across
  tokens, using XMX instead of re-streaming per pair.
* FP8/int8 DPAS: the MXFP8 layers dequantise to bf16 before the dot. Native fp8 dot
  (if exposed by the Triton Intel backend for e4m3) would halve the dequant work.
* 6.5 TFLOP/s is most likely well below the B65's XMX peak (not measured here; check the
  vendor spec before quoting a number). The attention kernel is still limited by the block-wise softmax and the rel-bias gather in
  the near-diagonal blocks, and by single-head programs (no GQA sharing of K/V tiles).
