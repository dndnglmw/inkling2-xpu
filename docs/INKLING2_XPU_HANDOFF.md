# Inkling 2 Mini on Intel XPU — hand-off note

Date: 2026-11-21 (session 1)
Working tree: `/home/operator/work/vllm` (branch `pr-inkling2-mini`, HEAD `4c61621b4`)
Docker container: `inkling-xpu` (persistent, running)
Host service `tp2-prod.service` (Qwen3.8 TP2 prod server on :8000) is **running
again** (it was stopped during the XPU tests and restarted at hand-off). To free
the GPUs for the next test round:

```bash
systemctl --user stop tp2-prod.service
# ... run tests ...
systemctl --user start tp2-prod.service
```

Note: `docker stop qwen38-tp2` alone is not enough — the systemd unit recreates it.

---

## 1. Goal

Serve **Inkling 2 Mini** (24 layers, 48 loop slots, 26.4B total / 3.4B active,
1M context, 128-expert MoE top-8, 4-bit experts) with
**vLLM on 2× Intel Arc Pro B65 (32 GB) using TP2**, later with **MTP/DSpark**
speculative decoding. Stock vLLM has no XPU path for this model; we port
`vllm/models/inkling2/nvidia/` to a new `vllm/models/inkling2/xpu/` package with
our own kernels where the CUDA ones are not portable.

## 2. What is already done and validated

### 2.1 Correctness oracle (the big win)

`vllm/models/inkling2/xpu/reference.py` — a self-contained pure-torch,
stateful implementation of the **looped** model (prefill + decode), written from
the HF `inkling2` code plus llama.cpp's loop semantics. Validated against
llama.cpp on CPU (Q8_0 GGUF `models/inkling-gguf/Inkling-2-Mini-Q8_0.gguf`):

| check | result |
|---|---|
| 1-pass (loop disabled) step-0 distribution vs llama.cpp-1pass (patched GGUF) | **corr 0.9985** over top-128, identical top-8 |
| 3-pass step-0 distribution vs llama.cpp | **corr 0.9995**, same top-1 (Δ 0.02 nats) |
| 16-token greedy vs llama.cpp | **16/16 top-1, 126/128 top-8** |

Reproduce:

```bash
cd /home/operator/work
.venv/bin/python vllm/scripts/patch_gguf_loop.py models/inkling-gguf/Inkling-2-Mini-Q8_0.gguf \
    --copy-to models/inkling-gguf/Inkling-2-Mini-Q8_0-1pass.gguf --count 1 --varying 1
./llama.cpp/build/bin/llama-server -m models/inkling-gguf/Inkling-2-Mini-Q8_0.gguf -c 2048 -t 6 --port 8099 &
.venv/bin/python vllm/tests/models/inkling2/xpu/compare_step0.py --passes 3 --port 8099
.venv/bin/python vllm/tests/models/inkling2/xpu/compare_llamacpp.py      # 16-token greedy
```

Both llama.cpp servers from session 1 may still be running on host ports
**8098** (1-pass copy) and **8099** (3-pass). Kill with `pkill -f 'bin/llama-serve[r]'`.

### 2.2 Dequantization

`scripts/inkling2_dequant.py` — NVFP4 (layers 0–19) and MXFP8 (layers 20–23)
expert weights → BF16. Output `models/inkling-bf16/` (11 shards, 49.2 GiB) with
config/tokenizer copied in. **Verified against the GGUF**: `down_proj` cos =
0.9997; `gate_up` is a pure row permutation of the checkpoint (see gotcha #1).

### 2.3 Key semantics (all validated)

- Block k = stream entering layer 6k. At a boundary the pre-boundary stream joins
  the bank and the **post-attention conv output starts the new stream** (no
  residual add). MLP input at a boundary mixes **all blocks including the new
  one** plus the conv output.
- Loop reset per pass: `bank = [blocks[prelude+1:], stream]` (2 entries),
  `blocks = prelude_blocks`, `stream = prelude_stream`.
  Messages = `attn_res_mix(bank[:-1], bank[-1], message_queries[layer-core_start])`,
  query order **(attn, k, v, mlp)**, pre-scaled by `D**-0.5`.
- Controller: `pair = rms([dest, message]); h = pair_bf16 @ W_down.T;
  out = (1-tanh(h W_forget.T))·dest + tanh(h W_gain.T)·message`; norm applied after.
- Router: `scores = sigmoid(x W_gate.T)`; selection uses `scores + gate.bias`,
  weights gathered from **raw** `scores`, renormalised, × 8.
- Activation = SITU with **beta=4.0, linear_beta=25.0** — exactly
  `SituAndMul`, already available on XPU via `torch.ops._C.situ_and_mul`
  (out, in, beta, linear_beta).
- q/r L2 clips 22.6274 / 8.0; V unweighted RMSNorm; log-scaling τ = 1 + 0.1·log(max(1,(p+1)/8192))
  applied to q and r of full-attention layers (τ≡1 below 8192 tokens).
- SWA window = 1024 (mask `dist < window`); rel-bias extent = 1024 for both
  layer types; rel-bias is **zero** outside `0 ≤ dist < extent`; sink is a
  constant logit in the softmax denominator.

### 2.4 KV/cache math (for capacity planning)

`[num_blocks, num_kv_heads, block_size, head_dim]` per layer, K+V adjacent in
the last dim (`[B,H,N,2D]`). Full-attention slots: 8 (layers 5, 11×3, 17×3, 23)
⇒ 4 KB/token/layer ⇒ **32 GiB at 1M**, 8 GiB at 256K, 4 GiB at 128K.
SWA layers are 1024-token rings (~160 MiB total). Conv-state cache ≈ 2 KB/token
per layer slot (tiny, window = 4).

## 3. The vLLM XPU port (current state)

New package `vllm/models/inkling2/xpu/`:

| file | contents |
|---|---|
| `model.py` | `Inkling2XPUModel`, `Inkling2XPULoopModel`, `Inkling2XPUForConditionalGeneration` (multimodal registration with image limit 0, `load_weights` reusing the NVIDIA name map), norms, AttnRes queries |
| `attention.py` | `InklingXPUAttention(AttentionLayerBase)`, per-pass KV caches, metadata builder/backend (`INKLING_XPU`), torch `_paged_attention` (rel bias + sink + gate) and `_dense_attention` (profile fallback) |
| `sconv.py` | `Inkling2XPUShortConv` + torch `short_conv` (static + dynamic taps, fp32, bf16 round) reading/writing the paged conv-state cache |
| `sconv_state.py` | `InklingConvState` (sliding-window cache owner, `AttentionLayerBase`), dummy backend + metadata builder (torch `seq_idx`/`query_start`) |
| `moe.py` | fp32 router, expert-sharded BF16 experts, interleave→packed w13 permutation, SITU via `torch.ops._C.situ_and_mul` with torch fallback |
| `loop.py` | `LoopControllers` (4 per core layer, checkpoint names), `attn_res_mix`, `rms_norm` |
| `reference.py` | the validated oracle (§2.1) |
| `__init__.py` | exports; package `vllm/models/inkling2/__init__.py` now dispatches `current_platform.is_xpu()` → `.xpu.model` |

Tests/scripts under `tests/models/inkling2/xpu/`:
`test_reference_tiny.py` (synthetic invariants, passing),
`compare_step0.py`, `compare_llamacpp.py`, `trace_reference.py` (1-pass op trace
vs HF), `hf_oracle.py` (transformers 1-pass oracle), `check_dequant_vs_gguf.py`,
`smoke_tiny.py` (tiny random-weight vLLM XPU run).

### 3.1 Environment

Docker image `vllm/vllm-openai-xpu:latest` = vLLM 0.34.0, torch 2.15.0+xpu,
triton 3.9.0, vllm_xpu_kernels 0.1.17. Container:

```bash
docker run -d --name inkling-xpu --device /dev/dri --shm-size=32g --ipc=host \
  -v /home/operator/work:/work --entrypoint sleep vllm/vllm-openai-xpu:latest infinity
docker exec -it inkling-xpu bash
# inside:
pip install 'transformers==5.23.0'      # inkling2 needs >=5.23 (image ships 5.22)
git config --global --add safe.directory /work/vllm
cd /work/vllm && VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e .
```

GPU visibility check:
`docker exec inkling-xpu python3 -c "import torch; print(torch.xpu.device_count(), torch.xpu.mem_get_info())"`

### 3.2 Current blocker

`python3 tests/models/inkling2/xpu/smoke_tiny.py` (tiny 4-layer looped model,
dummy weights, TP1, `--enforce-eager`) now **passes engine init including the
profile run**, then dies during the first `generate` with a **device-side
indexing assertion**:

```
.../ATen/native/xpu/sycl/Indexing.h:623: operator():
Assertion `index >= -sizes_[i] && index < sizes_[i] && "index out of bounds"` failed.
```

Full log: `/tmp/smoke.log` inside the container (`docker exec inkling-xpu less /tmp/smoke.log`).

Already fixed along the way (all real issues, keep them fixed):
- `RowParallelLinear(..., return_bias=False)` — this branch defaults to `True`
  and returns `(out, None)`, which broke `wo_ud`.
- `AttentionMetadataBuilder` subclass needs an explicit `__init__` calling
  `super().__init__(kv_cache_spec, layer_names, vllm_config, device)`.
- Profile runs have `attn_metadata` not-a-dict and un-bound KV caches →
  `InklingXPUAttention` falls back to `_dense_attention`; conv falls back to
  "skip conv, keep residual".
- KV cache logical layout is `[num_blocks, num_kv_heads, block_size, 2*head_dim]`
  (head dim 1, block pos dim 2) — index it 4-D, never `.view(-1, H, C)`.
- SWA/conv block tables are ring rows: clamp `pos // block_size` to
  `block_table.shape[1] - 1` (same as the NVIDIA kernels) and clamp ids ≥ 0.

#### Next debugging steps (in order)

1. Get a Python stack for the failing index op:
   - `docker exec -e PYTORCH_XPU_LAUNCH_BLOCKING=1 -e ZE_LAUNCH_BLOCKING=1 inkling-xpu ...`
   - or wrap the suspicious ops (`xf[row_idx]`, `cache[blk, :, pos, sl]`,
     `kv_cache[blocks, :, pos, :]`) with `torch.xpu.synchronize()` to localize.
2. Reproduce outside vLLM: build a tiny synthetic `short_conv` + `_paged_attention`
   test in plain torch-XPU with a fake `InklingXPUSconvMetadata` / attention
   metadata and compare against CPU. This isolates indexing vs. engine
   integration.
3. If it is the advanced-index write `cache[blk, :, pos, sl] = ...` (XPU
   `index_put`/`index` kernels can be picky), replace it with
   `cache.index_put_((blk, torch.arange(H)[None], pos[:,None], slice), ...)` or
   a small `@triton.jit` scatter/gather kernel (that is the intended "own
   kernel" anyway — see §4).
4. Once the tiny model generates: compare against the pure-torch reference
   (`reference.py`) on CPU with the same tiny config/weights for numeric parity,
   then run the real checkpoint (below).

### 3.3 Running the real model (when the tiny test is green)

```bash
docker exec -it inkling-xpu bash -lc '
cd /work/vllm && VLLM_TARGET_DEVICE=xpu python3 -m vllm.entrypoints.openai.api_server \
  --model /work/models/inkling-bf16 --served-model-name inkling2-mini \
  --quantization none --dtype bfloat16 --enforce-eager \
  --tensor-parallel-size 1 --max-model-len 4096 \
  --limit-mm-per-prompt "{\"image\": 0}" --port 8001'
```

Expected greedy from `"The capital of France is"` (llama.cpp / reference):
` Paris` `.` `\n\n` `A` `.` ` ` `Noun` ` clause` `\n\n` `B` `.` ` ` `Adject` `ive` ` clause` `\n\n`
Then repeat with `--tensor-parallel-size 2`.

## 4. Remaining work (prioritized)

1. **Green tiny test + real-checkpoint parity** (blocker above).
2. **Custom Triton-XPU attention kernel** (the main "own kernel"): paged
   prefill + decode with rel bias (gather from a `[T,H,extent]` fp32 table),
   per-(token,head) sink in the denominator, sigmoid gate, SWA mask. Torch path
   is O(S) gathers per request and unusable past a few K tokens. Start from
   `vllm/models/inkling2/nvidia/ops/attention_sm90.py` (portable Triton) minus
   PTX/PDL.
3. **Triton-XPU fused conv** (static + dynamic taps, paged state, fused
   residual/scatter): port `vllm/models/inkling/nvidia/ops/sconv.py` minus PDL.
4. **4-bit experts**: requant BF16 → MXFP4 (group 32, E8M0 scales, packed
   nibbles) or int4-WNA16. Note the XPU fast path `XpuFusedMoe` does **not**
   support SITU, so either (a) patch `XPUExperts._supports_activation` + the
   Intel kernel, or (b) keep our own MoE and add a Triton-XPU grouped GEMM.
   `FusedMoEFactory(activation="situ", activation_situ_beta=4.0,
   activation_situ_linear_beta=25.0)` resolves to TritonExperts on XPU (works,
   slower) — a good intermediate.
5. **TP2**: all-reduces already go through `tensor_model_parallel_all_reduce`
   (xccl); conv/KV caches are TP-shaped (`num_kv_heads // tp`), attn/mlp conv
   streams are full-width (`replicate_hidden=True`). Needs an end-to-end run.
6. **DSpark/MTP**: not ported. NVIDIA implementation is
   `vllm/models/inkling2/nvidia/dspark.py` (589 lines) + `speculator.py`
   (shared). Needs torch RMSNorm (replace flashinfer), our paged attention for
   the draft's block-decode, markov head, `mtp.safetensors` loading.
   `vllm/models/inkling2/__init__.py` currently raises `NotImplementedError`
   for XPU DSpark.
7. **Long context**: 256K fits one card (8 GiB KV); 1M needs both (~32 GiB KV
   + ~50 GB weights at Q4) — requires the 4-bit experts + TP2.

## 5. Gotchas / lessons (do not re-learn)

1. **`w13` is interleaved** `(gate_i, up_i)` in the checkpoint; HF's
   `chunk(2, -1)` is *wrong* for this checkpoint (it matches llama.cpp only
   after interleaving). The XPU MoE permutes at load to packed `[gates; ups]`.
2. `log_scaling_n_floor = 8192` for Mini (not 128000 — that was the Small config).
3. Static conv weight layout is Conv1d `[C,1,K]` with **oldest tap first**
   (flip before use); dynamic `static_w` is `[K,C]` with **tap 0 = current token**.
4. `local_layer_ids` = all except `{5, 11, 17, 23}`; `block_size` = 6 ⇒
   boundaries at 0, 6, 12, 18 (different from the global-attention layers).
5. `qkvr` fused weight row layout per rank:
   `[q, r(H·16), gate, sink(H), pad to /8, k, v]` — `q_side_splits` includes the
   pad; split the fused GEMM output with `split_sizes` (pad appended last).
6. Attention metadata is a **dict keyed by layer name** in the forward context;
   the model's `get_kv_cache_config` + `get_attn_backend` are the only hooks the
   runner needs (`AttentionLayerBase`), no registry entry for backends.
7. `torch.ops._C.situ_and_mul(out, input, beta, linear_beta)` — out first.
8. llama.cpp is the only working reference for the loop; transformers' inkling2
   has no loop support (and its expert chunk is inconsistent with the checkpoint).

---

# Session 2 (2026-11-21) — port is numerically working end to end

## Fixed
1. **Tiny smoke test indexing assert**: `short_conv` (`xpu/sconv.py`) mixed `query_start` (batch row) with
   positions. Previous-token taps now use `arange(T) - lag` vs `query_start`.
2. **Stale quantization config**: `models/inkling-bf16/config.json` had the NVFP4 `quantization_config`
   (XPU rejects it). Removed; original saved as `config.json.orig`.
3. **qkvr pad shard**: fused qkvr has a trailing alignment-pad shard (shard 6) absent from the checkpoint.
   `load_weights` now loads zeros into it (loader's missing-shard tracking).
4. **Wrong K/V offsets in `_project`** (`xpu/attention.py`): used `sum(q_side_splits)` (includes pad) as the
   K row offset; correct is `sum(split_sizes[:4])` (pad is last in the fused weight). This broke every
   full-attention layer whenever the Q-side size % 8 != 0 (tiny model, and the real model at TP2).
   Tiny parity corr 0.41 -> 0.993; real model now answers " Paris.\n\nA. ..." correctly.

## Results
- Real checkpoint, TP2, BF16: runs; prefill distribution at "...is Paris.\n\nA." matches llama.cpp
  (top: ' ' -2.196 vs -2.189). Greedy 16 tokens matches llama.cpp for the first 7 tokens, then picks
  ' Simple' instead of ' ' + 'Noun' (flat distribution, top-2 within ~0.3 nats; decode-vs-prefill noise).
- `tests/models/inkling2/xpu/parity_tiny.py`: tiny random checkpoint, reference vs vLLM XPU. Compares
  full-vocab logprobs, per-layer streams (prefill), per-layer streams for the first decode token, and
  vLLM decode vs vLLM prefill. Local layers are bit-exact between decode and prefill; global ~1e-3.
  Residual disagreements are MoE router ties (random tiny weights), not bugs.
  Run: `docker exec -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e ZE_AFFINITY_MASK=0 -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -w /work/vllm/tests/models/inkling2/xpu inkling-xpu-tp2 python3 parity_tiny.py`
- Debug capture hook: `xpu/model.py:DEBUG_DUMP` (list, None = off).

## TP2 launch (container `inkling-xpu-tp2`, image snapshot `inkling-xpu-snap`)
xccl needs `--cap-add SYS_PTRACE --group-add 990 --group-add 44 --ipc=host` and the prod CCL env
(`CCL_ATL_TRANSPORT=ofi FI_PROVIDER=tcp FI_TCP_IFACE=lo CCL_ZE_IPC_EXCHANGE=pidfd CCL_SEND=direct
CCL_RECV=direct CCL_TOPO_P2P_ACCESS=1 ZE_AFFINITY_MASK=0,1`). The original `inkling-xpu` container lacks these.
Server: `python3 -m vllm.entrypoints.openai.api_server --model /work/models/inkling-bf16 --served-model-name
inkling2-mini --dtype bfloat16 --enforce-eager --tensor-parallel-size 2 --max-model-len 1536
--max-num-batched-tokens 1024 --gpu-memory-utilization 0.95 --limit-mm-per-prompt '{"image": 0}' --port 8001`.
Weights = 24.3 GiB/rank; KV is tight (needs the 4-bit experts for longer context).

## Next
Priority list in §4 stands, now minus item 1 (done): Triton attention kernel, fused conv, 4-bit experts
(KV headroom), then DSpark/MTP. `tp2-prod.service` must be restarted after GPU tests.

## Session 2b — Triton XPU attention kernel (done)
- `xpu/triton_attention.py`: paged attention (rel-bias gather from `[T,H,extent]`, sink in denominator, fused
  sigmoid gate, SWA via per-tile key bound, GQA) with split-KV (flash-decoding) + combine kernel. Wired into
  `_paged_attention`; `INKLING_XPU_TORCH_ATTN=1` falls back to the old torch loop.
- Tests: `tests/.../test_triton_attention.py` (vs torch paged / exact dense ref: all pass, rel err ~2e-3),
  `bench_attention.py`. Per full-attn layer call at TP2 shape (12 heads, 4 kv heads, D=128):
  decode 1K ctx 0.25 ms, 32K 1.8 ms, 128K 6.2 ms (was 40 ms before split-KV); prefill 512 tok 1.5 ms.
- NOTE: the torch `_paged_attention` reads only keys >= seqlen-window for *every* row, so SWA prefill chunks
  longer than the window lose keys for early rows; the Triton kernel bounds keys per tile (correct).
- Real model (TP2) output unchanged ("Paris.\n\nA. Simple..."), but end-to-end decode is only ~2 tok/s eager:
  the bottleneck is now elsewhere — per-expert Python loop + host syncs in `moe.py`, and the torch
  `short_conv` (4 lags x gathers per stream per layer). Next: fused Triton conv, then a grouped-GEMM MoE
  (and 4-bit experts).

## Session 2c — speed + native 4-bit experts (see `vllm/models/inkling2/xpu/XMX.md` for the XMX notes)
Decode (single stream, TP2) 2.0 -> ~9 tok/s; prefill 90 -> ~1200 tok/s at 16-32K; weights 24.3 -> 8.9 GiB/rank.
New kernels (all tested in `tests/models/inkling2/xpu/test_triton_*.py`): `triton_attention.py` (XMX, split-KV decode),
`triton_sconv.py`, `triton_prep.py`, `triton_loop.py` (fused mix/controllers), `triton_moe.py` (BF16 decode GEMV),
`triton_moe_q.py` (NVFP4/MXFP8 decode GEMV + dequant), `triton_moe_grouped.py` (XMX grouped GEMM for prefill).
Native 4-bit checkpoint dir: `models/inkling-xpu4` (symlinks into `inkling-hf`, config without quantization_config,
`inkling_expert_quant.json` = hf_quant_config, named differently so vLLM doesn't auto-detect ModelOpt).
Quality gate: `quality_vs_llamacpp.py` -> 12/12 top-1 agreement vs llama.cpp, mean KL 0.0013.
Dev server: `/work/serve_inkling_dev.sh` (MODEL, MAXLEN, BATCHED, SEQS, UTIL env). `INKLING_PROF=1` prints phase times.
Escape hatches: INKLING_XPU_TORCH_{ATTN,CONV,MOE,LOOP}=1 fall back to the torch reference paths.

## Session 2d — XPU graphs (decode 9 -> 26 tok/s)
Decode forward is graph-safe: no host syncs (tau no longer checks `.all()`), decode attention tiles computed in-kernel
(`DECODE_TILES`), persistent metadata buffers, builders declare `UNIFORM_SINGLE_TOKEN_DECODE`.
Run with `GRAPH=1 /work/serve_inkling_dev.sh` (FULL_DECODE_ONLY, sizes 1,2,4,8; env VLLM_XPU_ENABLE_XPU_GRAPH=1,
VLLM_XPU_ALLREDUCE_HOST_WAIT=1). Quality/concurrency/long-context unchanged. Prod script: SETUP/serve-inkling2-tp2.sh,
switch/rollback: SETUP/inkling2-switch.md. Next big lever: speculative decoding (MTP/DSpark port).

## Session 3 — long-context + decode push (2026-11-21)
Measured (TP2, NVFP4/MXFP8 experts, bf16 KV, XPU graphs, 500K server): decode 40.4 tok/s short ctx (was 26),
23.5 tok/s at ~494K ctx; prefill 1255 tok/s @100K, 990 tok/s avg over a 493,538-token prompt (499 s); needle OK at 8K..494K.
Quality gate unchanged (12/12, KL 0.0015).
Changes: (1) fp32 LM head via bmm(out_dtype=fp32) (-8 ms/token; vLLM's cast path re-converted a 1.85 GB weight per step);
(2) vendor `vllm_xpu_kernels` flash attention for the bias-free FAR keys of global layers + my kernel for the near window,
merged in the combine kernel (LSE partial; empty far region masked by per-token far length because the vendor LSE is only -inf when
all q_len==1): prefill 2.6-2.9x (40 TFLOP/s), decode 4-5x at 131K-491K (2.0 ms/layer-call @491K); env INKLING_XPU_VENDOR_FAR=0 disables;
(3) agent C's split-K uint16 fp4/fp8 decode MoE GEMV (1.1-1.75x; ALU-bound ~0.2 ms/visit floor); (4) one-shot P2P all-reduce
(L0 IPC) ported into xpu_communicator.py, OFF by default (VLLM_XPU_ONESHOT_AR=1, lib /work/oneshot_ar/liboneshot_ar.so): 18 vs 39 us @4.6KB, bit-identical;
(5) DSpark drafter runs on XPU (xpu/dspark.py): 2.3-4.1 tokens/step, but step cost (8-token verify MoE) not yet a win.
Findings: GPU1 sits on a PCIe4 x4 link (7 GB/s) vs GPU0 x16 Gen5 (46-57 GB/s); fixed per-request KV overhead ~3.4 GiB/rank
(conv-state/SWA caches sized for max_num_batched_tokens); TurboQuant sim (kv_sim.py, INKLING_KV_SIM): tq4 KL 0.0055 vs uniform int4 0.0595,
fp8 0.0020; TP1 gives 22 tok/s vs TP2 31 (decode is not bandwidth/comm bound). Real-routing MoE prefill is ~3x slower than the synthetic uniform benchmark.
Tools: agents/ (GLM briefs/reports), tests bench_*.py, kv_quality.py, profile_decode.py (kernel profile at TP1).
