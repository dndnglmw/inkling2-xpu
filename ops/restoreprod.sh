#!/bin/bash
# Return :8000 to the production Inkling unit after the A/B experiments.
set -u
docker rm -f inkling2-tp2 >/dev/null 2>&1 || true
systemctl --user restart inkling2-prod.service
for _ in $(seq 1 40); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 5
done
systemctl --user is-active inkling2-prod.service
curl -s -m 4 -o /dev/null -w 'api=%{http_code}\n' http://127.0.0.1:8000/v1/models