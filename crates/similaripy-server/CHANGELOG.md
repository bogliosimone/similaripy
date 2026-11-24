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
  sorted, view, views, history counts).
- `POST /v1/score` (JSON) and gRPC `Score`: the scores of given items for
  one history (a ranker's feature), the same bits as Python's
  `SparseKNN.score`; at most `limits.max_items` items (default 10000).
- An OpenAPI 3.1 spec of the JSON endpoints (`openapi.json`, in the release
  archive beside `recommender.proto`) and `server.example.toml` (every config
  key with its default, commented).
- `x-request-id` in and out (generated if not sent); `Server-Timing` with the
  compute time (JSON).
- Config: a TOML file, `SIMILARIPY_<SECTION>__<KEY>` env vars and
  `--section.key` flags; `--print-config`, `GET /config`; an unknown key is a
  start-up error.
- Limits: `k_max`, `max_history` with `history_policy` truncate or reject,
  `max_body_bytes` (default 1 MiB, JSON and gRPC), `max_connections`
  (default 1024; more wait in the OS backlog); at most 64 HTTP/2 streams per
  connection and 16 KB of headers; a request body not received within
  `timeouts.header_read` is a 408 (JSON) or `CANCELLED` (gRPC), `body_timeout`;
  weights over 1e30 in magnitude are `bad_weights`; error messages cut to 256
  bytes.
- `defaults.k`, `defaults.sorted` (default `false`: candidate generation,
  unlike Python's `recommend`; `true` = best first).
- Errors as JSON `{"error": {"code", "message", "request_id"}}` (413 and 408
  included), `code` a fixed string (`bad_json`, `k_over_max`,
  `items_over_max`, `history_over_max`, `unknown_view`, `too_many_views`, `bad_weights`,
  `bad_body`, `body_timeout`, `body_too_large`, `internal`); gRPC status codes with the
  same `code` in the `x-error-code` metadata. One stderr line per server
  error.
- Named views from one folder, `views.dir`: `NAME.include.txt` and / or
  `NAME.exclude.txt`, one id per line, full-line `#` comments (the format is
  stable within 1.x and lives in the core: `similaripy_core::serve::view_file`
  reads it and writes it atomically; dot-files and other names ignored,
  symlinks followed, `NAME.meta.json` reserved, at most `views.max_views`,
  default 1024). Picked per request (`view`, plus up to 8 more in `views`,
  all ANDed), ANDed with the request's include / exclude; a request without
  `view` uses the whole model (no default view); `GET /v1/views`.
- Views follow the folder without a restart: it is scanned every
  `views.reload_interval` seconds (default 10, 0 = off): a new name is
  added, a changed file rebuilds its view (a rename or a swapped symlink, as
  in a Kubernetes ConfigMap, counts as a change; unchanged views are
  shared, not copied), a view whose files are gone two scans in a row is
  removed. A file that fails to read keeps its view (at start: listed with
  its error, not served, never a start error). Metrics `view_reloads_total`,
  `view_reload_errors_total{view}`, `view_last_reload_timestamp_seconds`;
  `GET /v1/views` shows each file's `modified` time and a view's `error`.
- `model.path` is resolved once at start: a symlink flipped during the load
  can't mix two model versions.
- `GET /metrics` (Prometheus: `similaripy_server_info{version, model}`,
  request / compute histograms per endpoint and format, counters,
  `process_start_time_seconds`,
  `process_resident_memory_bytes`), `GET /health` (liveness), `GET /ready`
  (readiness: 503 while draining).
- Graceful shutdown on SIGTERM / SIGINT: `timeouts.drain_delay`, then
  `timeouts.shutdown_grace`; `timeouts.header_read` against slow clients.
- Release binaries for Linux x86_64 / arm64 (glibc ≥ 2.28) and macOS arm64;
  multi-arch image `ghcr.io/bogliosimone/similaripy-server` (`:1.0.0`,
  `:1.0`).
- No auth, no TLS: private networks only.
