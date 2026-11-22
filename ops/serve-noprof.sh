#!/bin/bash
set -u
cd "/home/operator/Documents/Default Project" || exit 1
export PORT=8000 MAX_MODEL_LEN=262144 MAX_NUM_SEQS=8 GPU_UTIL=0.85 PREFIX_CACHING=1 GRAPH=1
export EXTRA_ARGS='--kv-cache-dtype fp8 --speculative-config {"method":"dspark","num_speculative_tokens":3,"model":"/work/models/inkling-xpu4-spec"}'
docker rm -f inkling2-tp2 >/dev/null 2>&1 || true
nohup bash SETUP/serve-inkling2-tp2.sh > /tmp/inkling-serve2.log 2>&1 &
echo "serve pid $! -> /tmp/inkling-serve2.log"
