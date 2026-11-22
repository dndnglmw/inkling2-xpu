#!/bin/bash
# Parameterized launcher for the DSpark-vs-MTP and KV-dtype experiments.
#   serve <method> <kv> <depth> [mbt] [graph] [gpu_util]
#     method: dspark|mtp|mtpm|none   (mtpm = mtp + draft model path)
#     kv: fp8|bf16                    graph: 1|0 (FULL_DECODE_ONLY vs eager)
#   stop | status | llama-up | llama-down
set -u
NAME=inkling2-tp2
IMAGE=inkling2-xpu:dev
WORK=/home/operator/work
cd "/home/operator/Documents/Default Project" || exit 1
cmd="${1:-status}"

case "$cmd" in
  stop)
    docker rm -f $NAME >/dev/null 2>&1 || true
    echo "inkling container removed"
    ;;
  llama-up)
    pkill -f 'llama-server.*--port 8099' 2>/dev/null || true
    nohup "$WORK/llama.cpp/build/bin/llama-server" \
      -m "$WORK/models/inkling-gguf/Inkling-2-Mini-Q8_0.gguf" \
      -c 2048 -t 6 --port 8099 > /tmp/llama-ref.log 2>&1 &
    echo "llama.cpp oracle starting on :8099 (pid $!)"
    ;;
  llama-down)
    pkill -f 'llama-server.*--port 8099' 2>/dev/null || true
    echo "llama.cpp oracle stopped"
    ;;
  serve)
    method="${2:-dspark}"; kv="${3:-fp8}"; depth="${4:-3}"; mbt="${5:-2048}"; graph="${6:-1}"; gu="${7:-0.85}"
    case "$method" in
      dspark) SPEC="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":$depth,\"model\":\"/work/models/inkling-xpu4-spec\"}" ;;
      mtp)    SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$depth}" ;;
      mtpm)   SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$depth,\"model\":\"/work/models/inkling-xpu4-spec\"}" ;;
      none)   SPEC="" ;;
      *) echo "unknown method: $method" >&2; exit 2 ;;
    esac
    KVD="--kv-cache-dtype fp8"
    [ "$kv" = "bf16" ] && KVD=""
    GRAPH_ENV=1
    MODE='--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'
    if [ "$graph" = "0" ]; then GRAPH_ENV=0; MODE='--enforce-eager'; fi

    RGD="$(stat -c '%g' /dev/dri/renderD* | sort -u | head -1)"
    VGD="$(getent group video | cut -d: -f3)"
    docker rm -f $NAME >/dev/null 2>&1 || true
    # wait for the driver to reclaim VRAM (same 29.5 GiB gate as the serve script)
    for _ in $(seq 1 60); do
      free="$(docker run --rm --device /dev/dri --group-add "$RGD" --group-add "$VGD" \
        --entrypoint python3 "$IMAGE" -c \
        "import torch; print(min(torch.xpu.mem_get_info(0)[0], torch.xpu.mem_get_info(1)[0]) / 2**30)" 2>/dev/null | tail -1)"
      [ -z "$free" ] && break
      awk -v f="$free" 'BEGIN {exit !(f+0 >= 29.5)}' && break
      sleep 5
    done
    docker run -d --rm --name $NAME --ulimit core=0 \
      --device /dev/dri --group-add "$RGD" --group-add "$VGD" \
      --cap-add SYS_PTRACE --ipc=host --shm-size=32g \
      --publish 8000:8000 --volume "$WORK:/work" \
      --workdir /work/vllm \
      --env ZE_AFFINITY_MASK=0,1 --env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
      --env VLLM_TARGET_DEVICE=xpu --env PYTHONHASHSEED=0 \
      --env VLLM_XPU_ENABLE_XPU_GRAPH=$GRAPH_ENV --env VLLM_XPU_ALLREDUCE_HOST_WAIT=1 \
      --env CCL_ATL_TRANSPORT=ofi --env FI_PROVIDER=tcp --env FI_TCP_IFACE=lo \
      --env CCL_ZE_IPC_EXCHANGE=pidfd --env CCL_SEND=direct --env CCL_RECV=direct --env CCL_TOPO_P2P_ACCESS=1 \
      --env CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
      --env CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
      --env CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
      --entrypoint python3 $IMAGE \
      -m vllm.entrypoints.openai.api_server \
      --model /work/models/inkling-xpu4 --served-model-name inkling2-mini --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 2 --dtype bfloat16 $MODE \
      --gpu-memory-utilization "$gu" --max-model-len 262144 \
      --max-num-seqs 8 --max-num-batched-tokens "$mbt" \
      --enable-prefix-caching \
      --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
      --limit-mm-per-prompt '{"image": 0}' \
      $SPEC $KVD \
      > /tmp/inkling-ab-serve.log 2>&1
    echo "started: method=$method kv=$kv depth=$depth mbt=$mbt graph=$graph gu=$gu  (log /tmp/inkling-ab-serve.log)"
    ;;
  status)
    docker ps -a --format '{{.Names}} | {{.Status}}' | grep inkling || echo "no inkling container"
    curl -s -m 4 -o /dev/null -w 'api=%{http_code}\n' http://127.0.0.1:8000/v1/models || true
    ;;
  *)
    echo "usage: run-ab.sh {serve <method> <kv> <depth> [mbt] [graph] [gu] | stop | status | llama-up | llama-down}"
    ;;
esac