# similaripy-server changelog

Experimental: within 1.x the `/v1` contract (JSON fields, the gRPC proto,
config keys) only grows, but while the label is on a change that is not
additive can come in a minor version, with a migration note here. Server N.x
reads models saved by similaripy N.x. User docs:
[Running the server](https://bogliosimone.github.io/similaripy/server/).

## 1.0.0 (unreleased)

First release, beside similaripy 1.0.0. Reads models saved by similaripy 1.x
(saved format 1).

- `POST /v1/recommend` (JSON) and gRPC `similaripy.v1.Recommender/Recommend`
  on one port: same answers as Python's `SparseKNN.recommend` (ids and
  `float32` scores), plus the request's context (model name, request id, k,
  sorted, view, history counts).
- `x-request-id` in and out (generated if not sent); `Server-Timing` with the
  compute time (JSON).
- Config: a TOML file, `SIMILARIPY_<SECTION>__<KEY>` env vars and
  `--section.key` flags; `--print-config`, `GET /config`; an unknown key is a
  start-up error.
- Limits: `k_max`, `max_history` with `history_policy` truncate or reject,
  `max_body_bytes` (default 1 MiB, JSON and gRPC).
- `defaults.k`, `defaults.sorted` (`false` for candidate generation) and
  `defaults.view`.
- Errors as JSON `{"error": {"code", "message", "request_id"}}` (413 and 408
  included), `code` a fixed string (`bad_json`, `k_over_max`,
  `history_over_max`, `unknown_view`, `bad_weights`, `bad_body`,
  `body_timeout`, `body_too_large`, `internal`); gRPC status codes with the
  same `code` in the `x-error-code` metadata. One stderr line per server
  error.
- Named views from id files (`[views.NAME]` include / exclude files), picked
  per request, ANDed with the request's include / exclude; `GET /v1/views`.
- `GET /metrics` (Prometheus: `similaripy_server_info{version, model}`,
  request / compute histograms, counters, `process_start_time_seconds`,
  `process_resident_memory_bytes`), `GET /health` (liveness), `GET /ready`
  (readiness: 503 while draining).
- Graceful shutdown on SIGTERM / SIGINT: `timeouts.drain_delay`, then
  `timeouts.shutdown_grace`; `timeouts.header_read` against slow clients.
- Release binaries for Linux x86_64 / arm64 (glibc ≥ 2.28) and macOS arm64;
  multi-arch image `ghcr.io/bogliosimone/similaripy-server` (`:1.0.0`,
  `:1.0`).
- No auth, no TLS: private networks only.
