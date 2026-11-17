#!/bin/sh
# Release smoke test (server-release.yml): a server started on the committed
# format-1 model (tests/data/itemknn-format1) answers one request as
# tests/test_serve.py says. Usage: smoke.sh [BASE_URL]
set -eu
url=${1:-http://127.0.0.1:8080}
for _ in $(seq 50); do
    curl -fsS "$url/health" >/dev/null 2>&1 && break
    sleep 0.2
done
curl -fsS -X POST "$url/v1/recommend" -H 'content-type: application/json' \
    -d '{"ids": [20, 40], "k": 3}' | tee /dev/stderr | jq -e '[.items[].id] == [30, 10, 50]' >/dev/null
echo "smoke ok: $url"
