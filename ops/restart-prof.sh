#!/bin/bash
# Restart the instrumented Inkling dev server with a fixed, known config.
set -u
docker stop inkling2-dev >/dev/null 2>&1 || true
docker rm -f inkling2-dev >/dev/null 2>&1 || true
cd "/home/operator/Documents/Default Project" || exit 1
export PORT=8000
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=8
export GPU_UTIL=0.85
export PREFIX_CACHING=1
export GRAPH=1
export EXTRA_ARGS='--kv-cache-dtype fp8 --speculative-config {"method":"dspark","num_speculative_tokens":3,"model":"/work/models/inkling-xpu4-spec"}'
nohup bash /home/operator/work/serve-inkling-prof.sh > /tmp/inkling-dev.log 2>&1 &
echo "serve pid $! -> log /tmp/inkling-dev.log"

