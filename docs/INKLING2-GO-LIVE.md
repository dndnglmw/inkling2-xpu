# Inkling 2 Mini on :8000 — operations (go-live record)

Written 2026-11-22 from the session record. Companion documents:
`SETUP/INKLING2-PERF-LOG.md` (every measured number) and `SETUP/INKLING2-DECISIONS.md`
(decisions, rejected ideas, limitations, what is not done).
`SETUP/inkling2-switch.md` is the **pre-cutover** switch note (written 2026-11-21 when nothing was
activated); where it disagrees with this file, this file wins.

> **Revision 2026-11-22 (later the same day): production switched from FP8 KV to BF16 KV.** The unit's
> `EXTRA_ARGS` no longer contains `--kv-cache-dtype fp8` (description now says BF16 KV). Drivers, all measured
> (PERF-LOG section 11): live FP8 KV failed the quality gate by ~160x (mean KL 0.318 vs <= 0.002, top-1
> 443/576, greedy text 7/12, deterministic across launches) AND BF16 was faster (234K code 77.2 vs 68.5 tok/s,
> step 44.4 vs 48.0 ms, prefill ~7% faster); the sole cost is the KV pool, 640,939 -> 439,959 tokens
> (2.45x -> 1.68x concurrency). Sections below describe the FP8-era configuration where not updated.>
> **Revision 2026-11-22 (same day, later): a `GPU_UTIL` trim was tried and REVERTED.** 0.78 ran briefly
> (pool 373,153 tokens / 1.42x concurrency, GPUActive 46.7 GiB, used 59 GiB, MemAvailable 33.4 GiB) to cut
> the host RAM mirror; reverted to 0.85 the same day (pool 439,855 tokens / 1.68x, verified serving) - the
> RAM targets under discussion were for other machine specs. Numbers, the stop-test proving the mirror
> releases fully, and the machine-sizing rule: PERF-LOG section 12.

> **Read this first — what this document could and could not verify.**
> The author of this pass had read access to the repo and `/home/operator/work/` only. The installed
> `inkling2-prod.service`, the OpenCode config, and the power measurements live outside that.
> Items originally marked **NEEDS INPUT** (installed unit, OpenCode block, power figures) were filled in
> on 2026-11-22 from the live files and the session record. The output-16384 rationale was stated by the
> user at the cutover and is not recorded verbatim anywhere, so it stays out. Everything not so marked
> was read from a file or measured.

## 1. What is running

| Item | Value | Source |
|---|---|---|
| Model | Inkling 2 Mini, served name `inkling2-mini` | `SETUP/serve-inkling2-tp2.sh` (`SERVED_NAME`) |
| Hardware | 2x Intel Arc Pro B65, tensor parallel 2 | |
| Checkpoint served | `/work/models/inkling-xpu4` (= `~/work/models/inkling-xpu4`): native NVFP4 experts (layers 0-19), MXFP8 experts (layers 20-23), everything else bf16; weights 8.9 GiB per card | `INKLING2_XPU_HANDOFF.md` Session 2c |
| Drafter | DSpark, `/work/models/inkling-xpu4-spec` (keeps `mtp_config` + `mtp.safetensors`), `num_speculative_tokens=3` | `serve-noprof.sh` |
| Code | `/home/operator/work/vllm`, branch `pr-inkling2-mini`, bind-mounted at `/work/vllm`; the image `inkling2-xpu:dev` carries only the Python environment | `inkling2-switch.md` |
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

Filled in 2026-11-22: the installed unit was read directly and a copy is committed at
`SETUP/systemd-user/inkling2-prod.service`. Verbatim as installed at `~/.config/systemd/user/inkling2-prod.service`:

```ini
[Unit]
Description=Inkling 2 Mini production server (TP2, 262144 cap, FP8 KV, dspark-3) - owns :8000
After=docker.service

[Service]
Type=simple
Environment=PORT=8000
Environment=MAX_MODEL_LEN=262144
Environment=MAX_NUM_SEQS=8
Environment=MAX_BATCHED_TOKENS=2048
Environment=GPU_UTIL=0.85
Environment=PREFIX_CACHING=1
Environment=GRAPH=1
Environment="EXTRA_ARGS=--kv-cache-dtype fp8 --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":3,\"model\":\"/work/models/inkling-xpu4-spec\"}"
ExecStartPre=/bin/bash -c 'for i in $(seq 1 60); do docker info >/dev/null 2>&1 && exit 0; sleep 2; done; echo "dockerd not ready" >&2; exit 1'
ExecStart=/bin/bash "/home/operator/Documents/Default Project/SETUP/serve-inkling2-tp2.sh"
Restart=no

[Install]
WantedBy=default.target
```

Deltas from the pre-cutover draft `SETUP/inkling2-tp2.service.example`: the installed unit has **no
`Conflicts=`** (nothing stops tp2-prod automatically - keep it disabled while Inkling owns :8000) and
**`Restart=no`**: if the container dies the unit stays down and :8000 must be brought back by hand
(`systemctl --user restart inkling2-prod.service`); it does not self-heal. The `Environment=` lines select
the serve-script knobs documented in section 1.

What the repo does contain, so the reader can compare:

* `SETUP/inkling2-tp2.service.example` — the draft unit written before the cutover
  (`Conflicts=tp2-prod.service tp2-prod-r304.service`, `Restart=always`, `RestartSec=15`,
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

## 2. Rollback to tp2-prod (Qwen3.8)

Derived from `SETUP/inkling2-switch.md` and `SETUP/systemd-user/tp2-prod.service`; the Inkling unit name is
confirmed against the installed unit (§1.1, filled 2026-11-22).

```bash
# 1. stop Inkling (frees :8000 and both GPUs)
systemctl --user stop inkling2-prod.service        # unit name confirmed (section 1.1)
docker rm -f inkling2-tp2 2>/dev/null              # the container the launcher creates
# 2. bring Qwen back
systemctl --user start tp2-prod.service            # tp2-prod-r304.service if that was the active Qwen unit
# 3. check
curl -sf localhost:8000/health && curl -s localhost:8000/v1/models
```

Notes that matter during a rollback:

* `tp2-prod.service` is the boot default for the Qwen stack and owns :8000 (`SETUP/systemd-user/tp2-prod.service`).
  It runs `SETUP/serve-b70lab-tp2.sh` with `W4A8=1` and the two `W4A8_*` paths in its `Environment=` lines;
  `W4A8=0` rolls that stack back to the W4A16 baseline.
* Clients that were moved to model name `inkling2-mini` must be moved back to the Qwen served name.
* The GPUs need time to return VRAM after the Inkling container exits; the Inkling launcher itself waits up to
  5 minutes for 29.5 GiB free on both cards. Start Qwen only after `docker ps` shows no Inkling container.
* Wait for the Qwen server's own readiness; do not infer it from the port opening.

## 3. OpenCode provider (`inkling-local/inkling2-mini`)

Filled in 2026-11-22 from the live laptop config (`~/.config/opencode/opencode.json`); the
top-level default `"model"` was switched to `"inkling-local/inkling2-mini"` at the cutover (the old default
would 404). The provider block verbatim:

```json
    "inkling-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Inkling 2 Mini (local B65)",
      "options": {
        "baseURL": "http://<TAILNET-IP>:8000/v1",
        "apiKey": "local"
      },
      "models": {
        "inkling2-mini": {
          "name": "Inkling 2 Mini (TP2, 262144, FP8 KV)",
          "tool_call": true,
          "temperature": true,
          "reasoning": true,
          "attachment": false,
          "modalities": {
            "input": ["text"],
            "output": ["text"]
          },
          "cost": {
            "input": 0.0,
            "output": 0.0
          },
          "limit": {
            "context": 262144,
            "output": 16384
          }
        }
      }
    },
```

**NEEDS INPUT — rationale for output 16384 rather than 32768.** The reason was stated by the user at the
cutover and is not recorded anywhere the author could read, so it is not reconstructed here. Facts that bear on
it and that *are* measured: the context cap is 262144 tokens (prompt plus output), and at a 258K prompt only about
4K tokens of output room remain (PERF-LOG §5); the answer-room trade-off table is in DECISIONS §6.

## 4. Dashboard (`SETUP/10-dashboard.py`)

The dashboard (`llm-dashboard.service`) was edited so it describes whatever owns :8000 instead of assuming Qwen.
The file in the working tree differs from `HEAD` by +1604/−521 lines because the same working copy also contains
the overview redesign (KPI cards, charts) and NVMe sensors; the Inkling-specific changes are:

| Change | Where | Behaviour |
|---|---|---|
| Hard-coded container name | `_tp2()` (line ~852) | was `"qwen38-tp2" in docker ps`; now true if any running container is `qwen38-tp2`, `inkling2-tp2` or ends in `-tp2` |
| Served-model detection | `_tp2()` (~966-979) | reads `/v1/models` and records `served_model`; keeps the last served name while the container is down |
| `/api/stack` live overlay | `do_GET` (~1314-1335) | `switch-stack.sh status` only knows the OVMS/tp2-prod pair; if the collector sees a running TP2 container the response is overlaid with `tp2: true`, `tp2_ready`, and, when the script said `idle`, `detail: "serving: vLLM TP2 (inkling2-mini)"` |
| Router fallback | `_services()` (~757-767) | if `/_router` does not answer the :8000 row is probed on `/health` and labelled `vLLM` / `OpenAI API` instead of `router` / `front door` |
| OVMS rows | `_services()` (~769-772) | a backend row is shown only if its `ovms-card<N>` container is running |
| Labels | JS `modelInfo()` (~1673-1706), page title `Inkling-2-Mini · 2× Arc Pro B65` | a served model name containing `inkling` shows `Inkling-2-Mini`, `NVFP4 / INT4`, `DSPARK`; otherwise `Qwen3.8-27B` labels as before |
| Decode chart | `_tp2()` | adds `decode_tps_short` (≈12 s window) and `requests_done_total` so the overview card follows live tokens/s |
| UI suppression | JS (~2206-2219, ~2787-2790) | when Inkling is detected the OVMS/TP2 switch buttons and the depth selector are hidden and `doSwitch()` shows an alert |

Backups (untracked, same directory):

* `SETUP/10-dashboard.py.bak-20261121-183432` — 125,258 bytes, 2026-11-21 18:34
* `SETUP/10-dashboard.py.bak-stack-183529` — 125,974 bytes, 2026-11-21 18:35 (taken before the `/api/stack` overlay edit)

The edited file is 132,205 bytes (2026-11-22 05:59).

### WARNING — the switch panel still drives `switch-stack.sh`

`POST /api/stack` runs `SETUP/switch-stack.sh start ovms|tp2` (`_post_stack`). Its protection is: loopback-only
unless `--allow-remote-switch`, the `X-Dash-Switch: 1` header, and a rate limit. **There is no check for Inkling
on the server side;** the hiding of the buttons is a client-side convenience, and it only engages when the
collector currently reports an Inkling served model. `switch-stack.sh` knows only the Qwen stacks: `do_tp2`
rewrites the `tp2-prod.service` drop-in from its own defaults and starts the Qwen TP2 container, and the `ovms`
action starts OVMS. **Do not press the switch buttons, and do not POST to `/api/stack`, while Inkling owns :8000.**
Both actions compete for :8000 and both GPUs. A rewrite of `switch-stack.sh` is on the not-done list.

## 5. Where everything lives

All paths under `/home/operator/work/` unless stated. Several of these scripts have CRLF line endings; the shell
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
| Model code | `/home/operator/work/vllm/vllm/models/inkling2/xpu/` (`model.py`, `attention.py`, `triton_*.py`, `moe.py`, `dspark.py`, `prof.py`, `XMX.md`) |
| Tests | `/home/operator/work/vllm/tests/models/inkling2/xpu/` (`test_triton_*`, `test_moe_tiered.py`, `test_ctrl_gemm.py`, `test_mix_mp.py`, `test_prep_fused.py`, `test_fp4_decode_fast.py`, `quality_vs_llamacpp.py`, ...) |
| Benchmarks and run output | `/home/operator/work/agents/` (`prefill_bench.py`, `deep4.py`, `needle_corpus.py`, `decode/spec_k_bench.py`, `decode/quality_dump_http.py`, `routing/`, `graph_gap_bench.py`, `ar_graph_bench.py`, ...). Do not run Python scripts from `agents/decode/`: `concurrent.py` there shadows the standard library |
| Engineering log | `/home/operator/work/INKLING2_XPU_HANDOFF.md` (sessions 2-3 only; later work is recorded in these three documents) |
| Checkpoints | `/work/models/inkling-xpu4` (served), `inkling-xpu4-spec` (drafter), `inkling-hf` (original), `inkling-bf16` (49.2 GiB dequantised, `config.json.orig` keeps the original config) |

## 6. Not done at go-live

See DECISIONS §9. In short: the Inkling-vs-Qwen quality and verbosity comparison, the `switch-stack.sh` rewrite,
and the items from the upstream list that were never tried.

Vision (Nov 23): the tower is ported and gate-clean; enable with --limit-mm-per-prompt '{"image": 1}' - text stays bit-exact (DECISIONS section 27).
