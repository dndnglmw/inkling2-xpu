# Inkling 2 Mini on :8000 — operations (go-live record)

Written 2026-11-22 from the session record. Companion documents:
`SETUP/INKLING2-PERF-LOG.md` (every measured number) and `SETUP/INKLING2-DECISIONS.md`
(decisions, rejected ideas, limitations, what is not done).
The pre-cutover switch note (written 2026-11-21, before anything was activated) compared against the
previous production model and is not published here; this file is the operational record.

> **Read this first — what this document could and could not verify.**
> The author of this pass had read access to the repo and `~/work/` only. The installed
> `inkling2-prod.service`, the OpenCode config, and the power measurements live outside that.
> Everything below marked **NEEDS INPUT** was not available and is deliberately left unfilled
> rather than reconstructed from memory. Everything not so marked was read from a file or measured.

## 1. What is running

| Item | Value | Source |
|---|---|---|
| Model | Inkling 2 Mini, served name `inkling2-mini` | `SETUP/serve-inkling2-tp2.sh` (`SERVED_NAME`) |
| Hardware | 2x Intel Arc Pro B65, tensor parallel 2 | |
| Checkpoint served | `/work/models/inkling-xpu4` (= `~/work/models/inkling-xpu4`): native NVFP4 experts (layers 0-19), MXFP8 experts (layers 20-23), everything else bf16; weights 8.9 GiB per card | `INKLING2_XPU_HANDOFF.md` Session 2c |
| Drafter | DSpark, `/work/models/inkling-xpu4-spec` (keeps `mtp_config` + `mtp.safetensors`), `num_speculative_tokens=3` | `serve-noprof.sh` |
| Code | `~/work/vllm`, branch `pr-inkling2-mini`, bind-mounted at `/work/vllm`; the image `inkling2-xpu:dev` carries only the Python environment | `SETUP/serve-inkling2-tp2.sh` |
| Container name (dev/clean launcher) | `inkling2-tp2` | `serve-inkling2-tp2.sh` (`NAME`) |
| Context cap | `--max-model-len 262144` (fixed; 1M is ruled out, see DECISIONS §7) | |
| KV cache | `--kv-cache-dtype fp8` (unit-scale e4m3) | |
| Prefix caching | on | |
| Graphs | XPU full-decode graphs, `FULL_DECODE_ONLY`, capture sizes 1,2,4,8 | |
| Chunk size | `--max-num-batched-tokens 2048` (4096 was measured and rejected for production, DECISIONS §6) | |
| GPU memory utilisation | 0.85 in the clean/profiling launchers (the underlying script defaults to 0.92) | `serve-noprof.sh` |
| KV pool at that config | 641,028 tokens, reported concurrency 2.45x at 262144 tokens per request (641,028 / 262,144 = 2.445; an earlier dev server reported 641,084) | reported at server start, 2026-11-22 |
| API | OpenAI-compatible on :8000; `--reasoning-parser inkling`, `--tool-call-parser inkling`, `--enable-auto-tool-choice`, images disabled (`--limit-mm-per-prompt {"image": 0}`) | `serve-inkling2-tp2.sh` |

### 1.1 `inkling2-prod.service`

**NEEDS INPUT.** The installed unit is not in the repo (`SETUP/systemd-user/` holds the previous
production stack's units, but no `inkling2-prod.service`) and is outside the directories the author
could read. To complete this section, copy the installed unit into the repo and paste
it here verbatim, including every `Environment=`, `ExecStart=`, `ExecStartPre=`, `Conflicts=` line and any
drop-ins (`systemctl --user cat inkling2-prod.service`).

What the repo does contain, so the reader can compare:

* `SETUP/inkling2-tp2.service.example` — the draft unit written before the cutover
  (`Conflicts=` with the units that own :8000, `Restart=always`, `RestartSec=15`,
  `ExecStart=/bin/bash "…/SETUP/serve-inkling2-tp2.sh"`, log appended to `/tmp/inkling2-tp2.log`).
  Whether the installed `inkling2-prod.service` matches it is **not verified**.
* The environment the launcher script itself sets inside the container (this is what any unit that calls
  `serve-inkling2-tp2.sh` inherits; read from the script, not from the installed unit):

| Variable | Value | Meaning |
|---|---|---|
| `ZE_AFFINITY_MASK` | `0,1` | both GPUs visible |
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:0,1` | |
| `VLLM_TARGET_DEVICE` | `xpu` | |
| `PYTHONHASHSEED` | `0` | stable prefix-cache hashing |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | `$GRAPH` (1 by default) | XPU graphs on/off |
| `VLLM_XPU_ALLREDUCE_HOST_WAIT` | `1` | measured to make no difference to in-graph all-reduce cost (PERF-LOG §9) |
| `CCL_ATL_TRANSPORT` | `ofi` | |
| `FI_PROVIDER` | `tcp` | |
| `FI_TCP_IFACE` | `lo` | |
| `CCL_ZE_IPC_EXCHANGE` | `pidfd` | |
| `CCL_SEND` / `CCL_RECV` | `direct` | |
| `CCL_TOPO_P2P_ACCESS` | `1` | GPU-to-GPU direct, no host-memory staging |
| `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD` | `4294967296` | |
| `CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD` | `4294967296` | |
| `CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD` | `4294967296` | |

Container flags from the same script: `--ipc=host --shm-size=32g --cap-add SYS_PTRACE --ulimit core=0`,
`--device /dev/dri` plus the render and video groups, `--volume $WORK_DIR:/work`,
`--workdir /work/vllm`, `--publish ${PORT}:8000`.

Launcher knobs (environment of `serve-inkling2-tp2.sh`): `PORT` (8000), `NAME`, `IMAGE`
(`inkling2-xpu:dev`), `MODEL_DIR`, `CODE_DIR`, `WORK_DIR`, `SERVED_NAME` (`inkling2-mini`), `MAX_MODEL_LEN`
(262144), `MAX_NUM_SEQS` (8), `MAX_BATCHED_TOKENS` (2048), `GPU_UTIL` (0.92 default),
`PREFIX_CACHING` (1), `GRAPH` (1), `EXTRA_ARGS`. The clean launcher sets
`EXTRA_ARGS='--kv-cache-dtype fp8 --speculative-config {"method":"dspark","num_speculative_tokens":3,"model":"/work/models/inkling-xpu4-spec"}'`.

Optional behaviour switches read by the model code (all default as listed; none is needed to run):

| Variable | Default | Effect |
|---|---|---|
| `INKLING_XPU_FP8_DENSE` | 0 | per-channel FP8 attention projections. **Leave off** (fails the quality gate, DECISIONS §2) |
| `INKLING_XPU_FAR_FOLD` | 1 | fold verify rows into the GQA group dim for the far-region flash call (PERF-LOG §5) |
| `INKLING_XPU_DRAFT_FOLD` | 1 | same fold for the drafter's global block attention |
| `INKLING_DRAFT_BN` / `_NW` / `_STAGES` / `_PROGS` / `_MAX_SPLITS` | 128 / 16 / 3 / 512 / 128 | drafter block-attention tiling |
| `INKLING_XPU_FUSED_PREP` | 1 | one launch for q/k/v/r row ops + KV write |
| `INKLING_CTRL_GEMM_MIN_T` | 128 | prefill controller blend uses bmm from this token count |
| `INKLING_SCONV_TILE_MIN_T` | 128 | prefill short-conv tile kernel from this token count |
| `INKLING_MIX_MP_MIN_T` | 128 | multi-pointer mix kernel from this token count |
| `INKLING_MOE_TIERED` | 1 | tiered capacity-bucket MoE for prefill |
| `INKLING_MOE_FAST_DECODE` | 0 | bit-exact integer FP4 decoders for the prefill dequant tile (no speedup measured) |
| `INKLING_PROF` | unset | per-phase timers / one-shot kernel profiler (profiling servers only) |
| `INKLING_XPU_TORCH_{ATTN,CONV,MOE,LOOP}` | 0 | select the torch reference paths |

## 2. Rollback to the previous production stack

The previous production stack (model, launcher, systemd unit) is internal to the author's machine and not
part of this repo; only the mechanics that concern Inkling are recorded here. **The Inkling unit name must
be confirmed against the installed unit (NEEDS INPUT, §1.1).**

```bash
# 1. stop Inkling (frees :8000 and both GPUs)
systemctl --user stop inkling2-prod.service        # confirm the real unit name
docker rm -f inkling2-tp2 2>/dev/null              # the container the launcher creates
# 2. start whatever unit normally owns :8000 on this host
# 3. check
curl -sf localhost:8000/health && curl -s localhost:8000/v1/models
```

Notes that matter during a rollback:

* Clients moved to model name `inkling2-mini` must be moved back to the previous server's model name.
* The GPUs need time to return VRAM after the Inkling container exits; the Inkling launcher itself waits
  up to 5 minutes for 29.5 GiB free on both cards. Start another :8000 server only after `docker ps`
  shows no Inkling container.
* Wait for the other server's own readiness; do not infer it from the port opening.

## 3. OpenCode provider (`inkling-local/inkling2-mini`)

**NEEDS INPUT.** The OpenCode config is outside the readable directories. Paste the provider block verbatim
here (path normally `~/.config/opencode/opencode.json`).

**NEEDS INPUT — rationale for output 16384 rather than 32768.** The reason was stated by the user at the
cutover and is not recorded anywhere the author could read, so it is not reconstructed here. Facts that bear on
it and that *are* measured: the context cap is 262144 tokens (prompt plus output), and at a 258K prompt only about
4K tokens of output room remain (PERF-LOG §5); the answer-room trade-off table is in DECISIONS §6.

## 4. Dashboard

The author's local monitoring dashboard was edited so it describes whatever owns :8000 instead of assuming
the previous model: the TP2 collector accepts the `inkling2-tp2` container (any `-tp2` name), reads the
served model name from `/v1/models` and labels it `Inkling-2-Mini` / `NVFP4 / INT4` / `DSPARK`; a service
row falls back to probing `/health` when :8000 does not answer the router's `/_router` endpoint; and when
an Inkling served model is detected the stack-switch controls are hidden. The dashboard code and the
legacy stack-switch script behind its switch panel target the internal previous stack and are not part of
this repo.

### WARNING — stack-switch actions are not guarded server-side

The switch panel's server-side endpoint runs the legacy stack-switch script, whose protections are
loopback-only access, a required header, and a rate limit — **there is no check for Inkling**. **Do not
fire a stack-switch action while Inkling owns :8000:** the legacy actions compete for :8000 and both GPUs.
A rewrite of the switch tooling is on the not-done list (§6).

## 5. Where everything lives

All paths under `~/work/` unless stated. Several of these scripts have CRLF line endings; the shell
prints a harmless `$'\r': command not found` on the last line and the launch still happens.

| Script | What it does |
|---|---|
| `serve-noprof.sh` | **The clean server.** Sets `PORT=8000 MAX_MODEL_LEN=262144 MAX_NUM_SEQS=8 GPU_UTIL=0.85 PREFIX_CACHING=1 GRAPH=1` and the fp8-KV + DSpark depth-3 `EXTRA_ARGS`, removes container `inkling2-tp2`, runs `SETUP/serve-inkling2-tp2.sh` in the background, log `/tmp/inkling-serve2.log` |
| `serve-4096.sh` | identical but `MAX_BATCHED_TOKENS=4096`; log `/tmp/inkling-serve4096.log`. Kept for if maximum prefill is ever wanted over capacity |
| `serve-inkling-prof.sh` | copy of the launcher with `NAME` default `inkling2-dev` and `--env INKLING_PROF=1` |
| `serve-prof-nograph.sh` | profiling server, container `inkling2-prof`, `GRAPH=0` (eager), 2048 chunks, 262144 cap, fp8 KV, depth 3, 0.85; calls `serve-inkling-prof.sh`; log `/tmp/inkling-prof-nograph.log` |
| `restart-prof.sh` | stops and removes `inkling2-dev`, relaunches it through `serve-inkling-prof.sh` (graphs on, 2048 chunks); log `/tmp/inkling-dev.log` |
| `serve_inkling_dev.sh` | older launcher meant to run **inside** the container (`MODEL`, `MAXLEN`, `BATCHED`, `SEQS`, `UTIL`, `PORT`, `GRAPH`, `TP`, `KVDTYPE`) |
| `serve-noprof-b4096.sh` | superseded by `serve-4096.sh`; never run |
| **`run_inkling_dev_container.sh`** | **BROKEN — do not use.** It mounts `/workspace/vllm` rather than the working tree and dies with `Transformers does not recognize inkling2_loop`: stock transformers cannot load this architecture, only the tree at `/work/vllm` can. It has no `--workdir /work/vllm`, which is what `serve-inkling2-tp2.sh` sets |
| `SETUP/serve-inkling2-tp2.sh` (repo) | the real launcher; every script above ends here |

Profiling servers: exactly one profiler capture per launch is safe; a second capture in the same launch killed the
engine (RPC timeout) both times it was tried. Capture is armed by a flag file `agents/tprof_<tag>.flag` whose
content is the wanted token count (0 = any); stream `docker logs -f` into a file under `agents/` because the
launcher logs under `/tmp` may not be readable from every session.

Code and data:

| What | Where |
|---|---|
| Model code | `~/work/vllm/vllm/models/inkling2/xpu/` (`model.py`, `attention.py`, `triton_*.py`, `moe.py`, `dspark.py`, `prof.py`, `XMX.md`) |
| Tests | `~/work/vllm/tests/models/inkling2/xpu/` (`test_triton_*`, `test_moe_tiered.py`, `test_ctrl_gemm.py`, `test_mix_mp.py`, `test_prep_fused.py`, `test_fp4_decode_fast.py`, `quality_vs_llamacpp.py`, ...) |
| Benchmarks and run output | `~/work/agents/` (`prefill_bench.py`, `deep4.py`, `needle_corpus.py`, `decode/spec_k_bench.py`, `decode/quality_dump_http.py`, `routing/`, `graph_gap_bench.py`, `ar_graph_bench.py`, ...). Do not run Python scripts from `agents/decode/`: `concurrent.py` there shadows the standard library |
| Engineering log | `~/work/INKLING2_XPU_HANDOFF.md` (sessions 2-3 only; later work is recorded in these three documents) |
| Checkpoints | `/work/models/inkling-xpu4` (served), `inkling-xpu4-spec` (drafter), `inkling-hf` (original), `inkling-bf16` (49.2 GiB dequantised, `config.json.orig` keeps the original config) |

## 6. Not done at go-live

See DECISIONS §9. In short: the quality/verbosity comparison against the previous production model, the stack-switch
tooling rewrite, and the items from the upstream list that were never tried.
