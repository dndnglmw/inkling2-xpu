#!/bin/bash
# Production restart with the bit-exact boot canary as the admission gate.
# Usage: restart-prod-canary.sh   (env CANARY_ALLOW_WARM=1 = temporary Mode B)
set -u
D=/home/operator/work/inkling-det
C="$D/canary/boot_canary.py"

attempt() {
  docker rm -f inkling2-tp2 >/dev/null 2>&1 || true
  systemctl --user restart inkling2-prod.service
  for _ in $(seq 1 60); do
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null)
    [ "$code" = "200" ] && break
    sleep 5
  done
  [ "$code" = "200" ] || { echo "CANARY-ABORT: server did not come up"; exit 2; }
  python3 "$C"
}

echo "[canary] restart + cold pass (attempt 1)"
if attempt; then echo "[canary] ADMITTED (cold pass)"; exit 0; fi
echo "[canary] COLD FAIL on attempt 1"

if [ "${CANARY_ALLOW_WARM:-0}" = "1" ]; then
  echo "[canary] Mode B: warm rerun (temporary - retire after the root-cause fix)"
  if python3 "$C"; then
    echo "[canary] ADMITTED (MODE-B WARM) - deviant cold branch was served to the canary only"
    exit 0
  fi
fi

echo "[canary] restart + cold pass (attempt 2)"
if attempt; then echo "[canary] ADMITTED (cold pass, attempt 2)"; exit 0; fi

echo "[canary] COLD FAIL on attempt 2 - NOT ADMITTING CLEAN: server left UP but flagged"
echo "[canary] investigate: /home/operator/work/inkling-det/logs/"
exit 1
