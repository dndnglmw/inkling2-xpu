#!/bin/bash
# Serve Inkling 2 Mini (native NVFP4/MXFP8 experts) on 2x Arc Pro B65, TP2, via the
# custom vLLM XPU port in $CODE_DIR (see vllm/models/inkling2/xpu/XMX.md and the
# ops docs in this repo). Sets the TP2 comms env, waits for both GPUs, and serves
# the OpenAI API on :8000.
set -u
PORT="${PORT:-${1:-8000}}"
NAME="${NAME:-inkling2-tp2}"
IMAGE="${IMAGE:-inkling2-xpu:dev}"
MODEL_DIR="${MODEL_DIR:-$HOME/work/models/inkling-xpu4}"
CODE_DIR="${CODE_DIR:-$HOME/work/vllm}"
WORK_DIR="${WORK_DIR:-$HOME/work}"                 # mounted at /work (model symlinks resolve inside it)
SERVED_NAME="${SERVED_NAME:-inkling2-mini}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
GPU_UTIL="${GPU_UTIL:-0.92}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"               # 1 = --enable-prefix-caching
GRAPH="${GRAPH:-1}"                                 # 1 = XPU full-decode graphs (default), 0 = eager
EXTRA_ARGS="${EXTRA_ARGS:-}"

PREFIX_ARG=(--no-enable-prefix-caching)
[ "$PREFIX_CACHING" = "1" ] && PREFIX_ARG=(--enable-prefix-caching)
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS})

# SPEC profile: 1 = keep whatever --speculative-config EXTRA_ARGS carries
# (default), 0 = exactness profile, which removes that flag and its JSON.
# Measured: without the drafter, eight identical prompts in one batch are
# bit-identical across rows; with it they are not.
SPEC="${SPEC:-1}"
if [ "$SPEC" = "0" ]; then
  _spec_filtered=()
  _spec_skip=0
  for _a in "${EXTRA[@]}"; do
    if [ "$_spec_skip" = "1" ]; then _spec_skip=0; continue; fi
    if [ "$_a" = "--speculative-config" ]; then _spec_skip=1; continue; fi
    _spec_filtered+=("$_a")
  done
  EXTRA=("${_spec_filtered[@]}")
fi
if [ "$GRAPH" = "1" ]; then
  MODE_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}')
else
  MODE_ARGS=(--enforce-eager)
fi

# Model dir symlinks point into ../inkling-hf; with $WORK_DIR mounted at /work they resolve.
case "$MODEL_DIR" in "$WORK_DIR"/*) CONTAINER_MODEL="/work/${MODEL_DIR#"$WORK_DIR"/}" ;; *) echo "MODEL_DIR must be under $WORK_DIR" >&2; exit 1 ;; esac
case "$CODE_DIR" in "$WORK_DIR"/*) ;; *) echo "CODE_DIR must be under $WORK_DIR" >&2; exit 1 ;; esac
CONTAINER_CODE="/work/${CODE_DIR#"$WORK_DIR"/}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
RGD="$(stat -c '%g' /dev/dri/renderD* | sort -u | head -1)"
VGD="$(getent group video | cut -d: -f3)"
# Wait for the driver to reclaim VRAM from any killed predecessor (best-effort, 5 min).
for _ in $(seq 1 60); do
    _free="$(docker run --rm --device /dev/dri --group-add "$RGD" --group-add "$VGD" \
        --entrypoint python3 "$IMAGE" -c \
        "import torch; print(min(torch.xpu.mem_get_info(0)[0], torch.xpu.mem_get_info(1)[0]) / 2**30)" 2>/dev/null | tail -1)"
    [ -z "$_free" ] && break
    awk -v f="$_free" 'BEGIN {exit !(f+0 >= 29.5)}' && break
    sleep 5
done

# ring is opt-in: setting INK2_RING_PREFIX_CACHE=1 also needs
# MAX_NUM_SEQS=1 and PREFIX_CACHING=1, or the model raises at boot.
exec docker run --rm --name "$NAME" --ulimit core=0 \
  --device /dev/dri --group-add "$RGD" --group-add "$VGD" \
  --cap-add SYS_PTRACE --ipc=host --shm-size=32g \
  --publish "${PORT}:8000" \
  --volume "${WORK_DIR}:/work" \
  --workdir "$CONTAINER_CODE" \
  --env ZE_AFFINITY_MASK=0,1 --env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  --env VLLM_TARGET_DEVICE=xpu --env PYTHONHASHSEED=0 \
  --env VLLM_XPU_ENABLE_XPU_GRAPH="$GRAPH" --env VLLM_XPU_ALLREDUCE_HOST_WAIT=1 \
  --env CCL_ATL_TRANSPORT=ofi --env FI_PROVIDER=tcp --env FI_TCP_IFACE=lo \
  --env CCL_ZE_IPC_EXCHANGE=pidfd --env CCL_SEND=direct --env CCL_RECV=direct --env CCL_TOPO_P2P_ACCESS=1 \
  --env CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4194304 \
  --env CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  --env CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  --env INK2_RING_PREFIX_CACHE="${INK2_RING_PREFIX_CACHE:-${RING_PREFIX_CACHE:-0}}" \
  --entrypoint python3 "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model "$CONTAINER_MODEL" --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --dtype bfloat16 "${MODE_ARGS[@]}" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  "${PREFIX_ARG[@]}" \
  --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image": 0}' \
  "${EXTRA[@]}"
