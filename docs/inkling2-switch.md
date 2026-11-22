# Switching production from Qwen3.8 to Inkling 2 Mini (TP2, 2x Arc Pro B65)

Status as of 2026-11-21. Nothing here is activated; `tp2-prod.service` is unchanged.

## What is verified
| Item | Result |
|---|---|
| Quality vs llama.cpp (3-pass oracle, 12 varied prompts) | 12/12 top-1 agree, mean KL 0.0013 (`tests/models/inkling2/xpu/quality_vs_llamacpp.py`) |
| Native 4-bit experts (NVFP4 L0-19, MXFP8 L20-23), no requantisation | weights 8.9 GiB/card |
| Context | 200K tokens served; needle retrieved at 4K, 8K, 16K, 33K, 105K, 199.5K, no errors |
| API | /v1/completions, /v1/chat/completions; `--reasoning-parser inkling` splits reasoning/answer; `--tool-call-parser inkling` returns structured tool calls |
| Concurrency | 4 concurrent requests OK (not bit-identical to solo runs, see caveats) |
| Prefix caching | enabled; cached vs uncached logprobs agree to ~0.02 nats |

## Performance (TP2, eager) vs the current Qwen3.8 prod
| | Inkling 2 Mini (this port) | Qwen3.8 prod (MTP depth 4, per SETUP docs) |
|---|---|---|
| single-stream decode | **~26 tok/s** (XPU full-decode graphs; ~9 tok/s eager) | roughly 30-55 tok/s in the repo's own reports |
| prefill | ~1200 tok/s at 16-32K depth; 105K in 109 s; 199.5K in 276 s | not compared here |
| max context | 200K (KV room for ~513K tokens) | 200K |

Decode runs under XPU full-decode graphs (sizes 1,2,4,8), which removed the launch overhead
that capped eager mode at ~9 tok/s. There is still no speculative decoding (MTP/DSpark not
ported), which is where Qwen's extra speed comes from, so single-stream decode is slower than
Qwen+MTP but in the same range. Prefill and any batch >8 tokens run eager.

## Caveats
* Not batch-invariant: the same prompt can differ slightly (near-tie tokens) between solo and
  concurrent runs and with/without a prefix-cache hit. Prod Qwen runs with batch-invariance flags.
* Prefill is eager (graphs are decode-only). No speculative decoding. Images disabled (`--limit-mm-per-prompt image=0`).
* The code lives in `/home/operator/work/vllm` (branch `pr-inkling2-mini`, uncommitted work) and is
  bind-mounted into the container; the image `inkling2-xpu:dev` only carries the Python env.
* Native checkpoint dir: `~/work/models/inkling-xpu4` (symlinks into `inkling-hf`).

## Switch (manual, reversible)
```bash
# 1. stop the current prod (frees :8000 and both GPUs)
systemctl --user stop tp2-prod.service            # (tp2-prod-r304.service if that is the active unit)
# 2. start Inkling on :8000 (foreground test first)
"/home/operator/Documents/Default Project/SETUP/serve-inkling2-tp2.sh"
#    ready when:  curl -sf localhost:8000/health
#    smoke:       curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
#                   -d '{"model":"inkling2-mini","messages":[{"role":"user","content":"hi"}]}'
```
Clients must use model name `inkling2-mini` (or set `SERVED_NAME=qwen38` when launching, to keep the
old name). To make it persistent, copy `inkling2-tp2.service.example` to
`~/.config/systemd/user/inkling2-tp2.service`, `systemctl --user daemon-reload`, and
`systemctl --user enable --now inkling2-tp2.service` (after disabling `tp2-prod.service`).

## Rollback
```bash
systemctl --user stop inkling2-tp2.service   # or Ctrl-C the foreground script
docker rm -f inkling2-tp2 2>/dev/null
systemctl --user start tp2-prod.service
```

## Tuning knobs (env on the serve script)
`MAX_MODEL_LEN` (262144), `MAX_NUM_SEQS` (8), `MAX_BATCHED_TOKENS` (2048), `GPU_UTIL` (0.85),
`PREFIX_CACHING` (1), `GRAPH` (1 = decode graphs, 0 = eager), `SERVED_NAME`, `PORT`, `EXTRA_ARGS`.
Debug: `INKLING_PROF=1` prints per-phase timings; `INKLING_XPU_TORCH_{ATTN,CONV,MOE,LOOP}=1` select the torch reference paths.
