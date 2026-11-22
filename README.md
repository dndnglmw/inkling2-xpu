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
| Prefill | 1931 / 1714 / 1480 tok/s at 37K / 125K / 230K-token prompts (diverse text) |
| Decode | DSpark depth 3, ~74 tok/s mean short context; ~64 (code) / 55 (prose) at 250K |
| Context | 262,144-token cap; fp8 KV; needle retrieved at 258K; KV pool 641K tokens (2.45× concurrency) |
| Weights | 8.9 GiB per card — native NVFP4 (layers 0–19) + MXFP8 (20–23), dequantised in-kernel |
| Quality | 12/12 top-1 agreement with a llama.cpp 3-pass oracle, mean KL 0.0013 |
| Graphs | XPU full-decode graphs (sizes 1, 2, 4, 8) incl. the DSpark verify step |

All numbers, decisions and rejected ideas are in [`docs/`](docs/) — this repo
documents the setup as actually operated, including what is *not* done.

## Repo layout

| Path | Contents |
|---|---|
| `patches/` | The whole port as `git am`-able commit(s) on top of vLLM base commit `4c61621b4` |
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
- Prefill runs eager (graphs are decode-only); images disabled.
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
