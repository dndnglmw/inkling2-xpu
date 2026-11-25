# Inkling 2 Mini on Intel Arc — vLLM XPU port

A from-scratch Intel XPU inference path for **Inkling 2 Mini** (looped hybrid
MoE: 24 layers, 48 loop slots, 26.4B total / 3.4B active, 128-expert MoE
top-8, native NVFP4/MXFP8 experts), running on **2× Intel Arc Pro B65 (32 GB)
with tensor parallel 2**.

Upstream vLLM added Inkling 2 Mini in [vllm-project/vllm#69028](https://github.com/vllm-project/vllm/pull/69028)
for NVIDIA silicon only (Blackwell/Hopper/DGX Spark); the vLLM XPU supported-models
list has no Inkling entry. This port is, as far as we know, the first Inkling 2
inference path on Intel GPUs — its own Triton kernel suite, per-pass KV/conv-state
caches, TP2 collectives, and a **DSpark speculative-decoding drafter**, against
the *original* 4-bit checkpoint (no requantisation).

## Highlights (measured, TP2)

| | |
|---|---|
| Prefill | 2,114 / 1,857 tok/s at 8K / 128K-token prompts (diverse text; the 64K point was measured twice with a wide spread and is not quoted) |
| Decode | DSpark depth 3, 92 tok/s single-stream, 109 tok/s at 8 concurrent; 77 (code) / 59 (prose) at 234K |
| Context (Nov 23 build, not re-run after the threshold change) | 262,144-token cap; BF16 KV (live FP8 KV rejected: KL 0.318 vs 0.002 gate); needle retrieved at 243K; KV pool ≈1.68× the context length |
| Weights | 8.9 GiB per card — native NVFP4 (layers 0–19) + MXFP8 (20–23), dequantised in-kernel |
| Quality | 12/12 top-1 agreement with a llama.cpp 3-pass oracle, mean KL 0.0013 |
| Graphs | XPU full-decode graphs (sizes 1, 2, 4, 8) incl. the DSpark verify step |

All numbers, decisions and rejected ideas are in [`docs/`](docs/) — this repo
documents the setup as actually operated, including what is *not* done.

## Nov 24 update

The head and threshold changes were checked against the 576-position
bit-exactness gate (576/576 top-1, KL 0.000 against the frozen reference),
and the head was additionally confirmed over three fresh boots. The ring is
default-off; its evidence is bit-identical 75K/128K resends.

| Change | Effect | Status |
|---|---|---|
| Chunk-16 fp32 logits head | oneMKL on this stack switches GEMM kernels at M=26; chunking keeps every head GEMM in the exact regime and reads the bf16 weight once per 16 rows. Head 49.5 → 3.4 ms at 32 rows; 8-concurrent decode 89.9 → 109.4 tok/s (+21.7%); single-stream 92.3 vs 91.1 (within noise) | default |
| 4 MiB all-reduce simple threshold | prefill 8K 2,017 → 2,114 (+4.8%), 128K 1,771 → 1,857 (+4.9%); decode unchanged | default (`CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4194304`) |
| Exact-tail ring prefix cache | a resend whose prefill ends exactly at the retained frontier replays the original final chunk (hidden states + paged KV rows) instead of recomputing at a different chunk size. 75K/128K resends bit-identical, 0.32/0.44 s | **opt-in**, single sequence only: `--max-num-seqs 1 --enable-prefix-caching` and `INK2_RING_PREFIX_CACHE=1` |

The ring retains the last `slack` tokens of recurrent state (sliding-window
conv cache, local ring) plus the final-chunk hidden rows, keyed to the exact
position it was captured at. It fails closed for eager forwards: a hit whose
state rows cannot be written, or that lands outside the retained window,
raises instead of being served. It is ported from
[SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
with the attribution kept in the source files.

The decode and prefill rows above are from the default profile, which runs
with the ring off; the ring is a single-sequence opt-in and is not part of
those numbers.

Concurrency is not covered. In two boots at eight concurrent sequences with
batched matmul off (one with the DSpark drafter, one without), two identical
**cold** runs mismatched in 15/40 and 13/40 comparisons (eight prompts x five
repeats), nearly all at the first generated token. The matching cache-hit
comparisons were lower in both boots (6/40, 9/40). The cause is likely
scheduling or batch composition, which this test did not isolate: an exact
cache removes one source of variation, not every one.

## Repo layout

| Path | Contents |
|---|---|
| `patches/` | The whole port as `git am`-able commit(s) on top of vLLM base commit `4c61621b4`. `0001` now also carries the vision and determinism changes first published as `0002`/`0003`; `0002` is the KV-manager hook for the ring |
| `overlay/` | The same files laid out for manual copy into a vLLM checkout (`vllm/models/inkling2/…`, `tests/…`, `scripts/…`) |
| `docs/INKLING2-GO-LIVE.md` | Operations record: what runs, launchers, env switches, rollback |
| `docs/INKLING2-DECISIONS.md` | Decisions, rejected optimisations, limitations, open items |
| `docs/INKLING2-PERF-LOG.md` | Every measured number (prefill/decode ladder, kernel profiles) |
| `docs/serve-inkling2-tp2.sh` | The docker launcher (TP2, CCL env, GPU wait loop) |
| `docs/inkling2-tp2.service.example` | Draft systemd user unit |

## Getting started

1. **Environment.** Build/run against the Intel XPU vLLM image
   (`vllm/vllm-openai-xpu`, vLLM 0.34 era: torch 2.15.0+xpu, triton 3.9.0,
   `vllm_xpu_kernels` 0.1.17). The image only carries the Python env — the
   patched tree is bind-mounted into it.
2. **Apply the port.**
   ```bash
   git clone https://github.com/vllm-project/vllm.git && cd vllm
   git checkout 4c61621b4
   git am /path/to/inkling2-xpu/patches/*.patch     # or: cp -a overlay/{vllm,tests,scripts}/. .
   VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e .
   ```
3. **Checkpoint.** Serve the original HF checkpoint unmodified via a directory
   of symlinks into it, with `quantization_config` removed from `config.json`
   (the XPU path rejects it) and the `hf_quant_config` content stored as
   `inkling_expert_quant.json` so vLLM does not auto-detect a ModelOpt path.
   Layers 0–19 are NVFP4 experts, 20–23 MXFP8; everything else bf16. See
   `vllm/models/inkling2/xpu/XMX.md` in the overlay for the load-time details,
   and `overlay/scripts/inkling2_dequant.py` if you want a bf16 copy instead.
4. **Launch.** Edit paths in `docs/serve-inkling2-tp2.sh` and run it, or use it
   as a reference for a bare `vllm serve` — the flags that matter:
   `--tensor-parallel-size 2 --max-model-len 262144 --kv-cache-dtype fp8
   --max-num-batched-tokens 2048 --reasoning-parser inkling
   --tool-call-parser inkling --limit-mm-per-prompt '{"image": 0}'` plus the
   DSpark `--speculative-config` (drafter keeps `mtp_config` + `mtp.safetensors`).
   The script also sets the CCL env (`CCL_TOPO_P2P_ACCESS=1` etc.) that TP2
   needs — don't drop it.
5. **Verify.** `pytest tests/models/inkling2/xpu/` inside the container (kernel
   gates vs torch references), `tests/models/inkling2/xpu/quality_vs_llamacpp.py`
   for the llama.cpp oracle comparison.

## Caveats (measured, see `docs/INKLING2-DECISIONS.md` §8)

- Research port run in production by its author; not batch-invariant (same
  prompt can differ slightly between solo and concurrent runs / prefix-cache
  hits).
- 1M context is **ruled out**: a 1M server hung the 91 GB host; 262,144 is the
  production cap.
- Prefill runs eager (graphs are decode-only); image support ported (text stays bit-exact on image-enabled boots - docs/INKLING2-DECISIONS.md section 27).
- Long-context acceptance drops for verbatim quotes from far-back text.
- Numbers depend on topology (one card sits on a PCIe Gen4 x4 link).

## Upstream status

The local branch `pr-inkling2-mini` is based on `4c61621b4`; the only upstream
deltas since (PR #69028 merge fixes) touch NVIDIA csrc paths, so the XPU
package rebases cleanly. Upstreaming `inkling2/xpu` into vLLM is the intended
next step.

## License

Apache-2.0, same as vLLM — this is a derivative work of
[vllm-project/vllm](https://github.com/vllm-project/vllm) at commit
`4c61621b4` (see `LICENSE` and `NOTICE`).

## Experiments & ops tooling

`experiments/` holds the measurement scripts behind the perf log (acceptance/step
benches `deep4`/`deep5`, needle and KV-quality probes, prefill chunking and
attention/allreduce sweeps, expert-dtype benches). `ops/` adds the experiment
server launchers (clean, 4096-chunk, profiling, A/B variants) and the installed
production unit (`ops/inkling2-prod.service`) alongside the draft example in
`docs/`. `docs/inkling2-switch.md`, `INKLING2_XPU_HANDOFF.md` and
`INKLING2_XPU_QUALITY.md` are the pre-cutover switch note and the working
handoff/quality documents.
