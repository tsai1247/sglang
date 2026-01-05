#!/bin/bash

set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-12470}
BASE_URL="http://${HOST}:${PORT}"

WARMUP_REQUESTS=${WARMUP_REQUESTS:-3}
WARMUP_MAX_NEW_TOKENS=${WARMUP_MAX_NEW_TOKENS:-32}
WARMUP_PROMPT=${WARMUP_PROMPT:-"Warm up nano-pearl."}
WARMUP_STREAM=${WARMUP_STREAM:-0}

payload_nonstream='{"text":"'"${WARMUP_PROMPT}"'","sampling_params":{"max_new_tokens":'"${WARMUP_MAX_NEW_TOKENS}"',"temperature":0.0}}'
payload_stream='{"text":"'"${WARMUP_PROMPT}"'","sampling_params":{"max_new_tokens":'"${WARMUP_MAX_NEW_TOKENS}"',"temperature":0.0},"stream":true}'

deadline=$((SECONDS + 120))
while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health_generate" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  if [ $SECONDS -ge $deadline ]; then
    echo "health_generate not ready after 120s" >&2
    exit 1
  fi
  sleep 1
done

for _ in $(seq 1 "${WARMUP_REQUESTS}"); do
  curl -sS -o /dev/null \
    -H "Content-Type: application/json" \
    -X POST "${BASE_URL}/generate" \
    -d "${payload_nonstream}"
done

if [ "${WARMUP_STREAM}" = "1" ]; then
  curl -sS -N -o /dev/null \
    -H "Content-Type: application/json" \
    -X POST "${BASE_URL}/generate" \
    -d "${payload_stream}"
fi
