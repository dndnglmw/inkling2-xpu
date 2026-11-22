#!/bin/bash
# Instrumented server with XPU graphs OFF (eager), so per-phase timers are visible.
set -u
cd "/home/operator/Documents/Default Project" || exit 1
export PORT=8000
export NAME=inkling2-prof
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=8
export GPU_UTIL=0.85
export PREFIX_CACHING=1
export GRAPH=0
export MAX_BATCHED_TOKENS=2048
export EXTRA_ARGS='--kv-cache-dtype fp8 --speculative-config {"method":"dspark","num_speculative_tokens":3,"model":"/work/models/inkling-xpu4-spec"}'
docker rm -f inkling2-prof >/dev/null 2>&1 || true
nohup bash /home/operator/work/serve-inkling-prof.sh > /tmp/inkling-prof-nograph.log 2>&1 &
echo "serve pid $! -> /tmp/inkling-prof-nograph.log"

