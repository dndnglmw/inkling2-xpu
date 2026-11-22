#!/bin/bash
# Same server config as serve-noprof.sh but with 4096-token prefill chunks.
set -u
cd "/home/operator/Documents/Default Project" || exit 1
export PORT=8000
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=8
export GPU_UTIL=0.85
export PREFIX_CACHING=1
export GRAPH=1
export MAX_BATCHED_TOKENS=4096
export EXTRA_ARGS='--kv-cache-dtype fp8 --speculative-config {"method":"dspark","num_speculative_tokens":3,"model":"/work/models/inkling-xpu4-spec"}'
docker rm -f inkling2-tp2 >/dev/null 2>&1 || true
nohup bash SETUP/serve-inkling2-tp2.sh > /tmp/inkling-serve4096.log 2>&1 &
echo "serve pid $! -> /tmp/inkling-serve4096.log"

