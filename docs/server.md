# 🖥️ Running the server

`similaripy-server` serves a model saved with
[`SparseKNN.save`](serving.md#saving-and-loading) ([format](saved-format.md)) over HTTP/JSON and gRPC, on
one port. It is a single binary (or a Docker image): no Python at serving
time.

!!! warning "Experimental"
    - **What is promised:** server 1.x reads models saved by similaripy 1.x.
      The `/v1` contract (JSON fields, the gRPC proto, config keys) only grows
      within 1.x: new optional request fields, new answer fields, new keys.
    - **While the label is on**, a change that is not additive can still
      come in a minor version (1.y), always with a migration note in the
      [changelog](https://github.com/bogliosimone/similaripy/blob/master/crates/similaripy-server/CHANGELOG.md).
    - **Upgrade servers before clients:** the JSON endpoint rejects unknown
      fields, so a request with a field from a newer version gets a 400 from
      an older server.
    - **No auth, no TLS:** run it on a private network only
      ([In production](#in-production)).

## Quick start

Save a model in Python:

```python
from similaripy.serving import SparseKNN

SparseKNN(S.T, item_ids=item_ids).save("models/v3")
```

Run the server on it (it listens on `127.0.0.1:8080` by default):

```bash
similaripy-server --model models/v3
```

Ask for recommendations:

```bash
curl -s 127.0.0.1:8080/v1/recommend -d '{"history": [10, 20], "k": 3, "sorted": true}'
```

```json
{"items":[{"id":30,"score":1.0},{"id":40,"score":0.25}],"model":{"name":"v3"},
 "request_id":"8480bc835112df61-0-0","k":3,"sorted":true,"view":null,
 "history":{"sent":2,"truncated":0,"unknown":0}}
```

Same answer as `SparseKNN.load("models/v3").recommend([10, 20], k=3)`: same
ids, same `float32` scores, at most `k` items (only those with a positive
score: here 2).

## Install

**Release archive.** Linux x86_64 and arm64 (glibc ≥ 2.28: Debian 10+,
Ubuntu 18.10+, RHEL 8+) and macOS arm64. Each archive holds the binary, the
license, `recommender.proto` (gRPC) and `openapi.json` (JSON).

```bash
V=1.0.0
T=x86_64-unknown-linux-gnu   # or aarch64-unknown-linux-gnu, aarch64-apple-darwin
URL=https://github.com/bogliosimone/similaripy/releases/download/server-v$V
curl -LO "$URL/similaripy-server-$V-$T.tar.gz" -LO "$URL/SHA256SUMS"
shasum -a 256 --check --ignore-missing SHA256SUMS   # or: sha256sum --check --ignore-missing SHA256SUMS
tar -xzf "similaripy-server-$V-$T.tar.gz"
"./similaripy-server-$V-$T/similaripy-server" --version
```

On macOS the binary is not signed: after downloading it with a browser, run
`xattr -d com.apple.quarantine similaripy-server` once.

**Docker image** (linux/amd64 and linux/arm64), tags `:1.0.0` and `:1.0`
(no `latest`):

```bash
docker run --rm -p 8080:8080 -v "$PWD/models/v3:/model:ro" \
  -e SIMILARIPY_MODEL__NAME=v3 \
  ghcr.io/bogliosimone/similaripy-server:1.0.0 --model /model
```

- The image listens on `0.0.0.0:8080` (outside a container the default is
  `127.0.0.1`), set by the env var `SIMILARIPY_SERVER__ADDR`. An env var
  beats a config file ([Configuration](#configuration)), so change the
  address with `-e SIMILARIPY_SERVER__ADDR=…` or `--server.addr`, not with
  `[server] addr` in a mounted file.
- It runs as a non-root user (uid 65532): the model's files must be readable
  by it.
- Set `model.name`: the default is the directory's name, here `model`.

**From source** (a clone of the repository, Rust from
[rustup](https://rustup.rs)):

```bash
cargo build --release -p similaripy-server
target/release/similaripy-server --version
```

## Configuration

Every setting is a key, `section.key`, settable in three ways. From weakest
to strongest: built-in default < config file < env var < command-line flag.

| Where | Form | Example |
|-------|------|---------|
| File (TOML, `--config FILE`) | `[section]` then `key = value` | `[limits]` then `k_max = 500` |
| Env var | `SIMILARIPY_<SECTION>__<KEY>` (two underscores) | `SIMILARIPY_LIMITS__K_MAX=500` |
| Flag | `--section.key value` (or `--section.key=value`) | `--limits.k_max 500` |

A config file:

```toml
# server.toml
[model]
path = "models/v3"
name = "itemknn-v3"

[limits]
max_history = 500

[views.in_stock]
include_file = "views/in_stock.txt"
```

```bash
similaripy-server --config server.toml --server.threads 4
```

- An unknown key (a typo) stops the server at start, naming the key.
- `--print-config` prints the resolved config (TOML) and exits: a dry run
  for CI or a new deployment.
- The server logs the resolved config at start and serves it at
  `GET /config` (so no key ever holds a secret).
- Only env vars starting with `SIMILARIPY_` and containing `__` are config.

| Key | Default | Meaning |
|-----|---------|---------|
| `model.path` | (required) | A `SparseKNN.save` directory. A symlink is resolved once at start (set `model.name`, else it is the link's name). |
| `model.mmap` | `true` | Map the model's files instead of reading them into memory (one copy shared by every process on the machine). |
| `model.name` | the directory's name | Returned in every answer: tells model versions apart (e.g. `itemknn-2026-11-16`). |
| `server.addr` | `127.0.0.1:8080` | `host:port` to listen on. |
| `server.threads` | P-cores (macOS), available CPUs (Linux, container CPU limit included) | Server threads, one request at a time each. |
| `server.block_size` | `auto` | `auto`, `off` or entries per block, as in [Serving](serving.md#parameters). |
| `limits.k_max` | `2000` | A larger `k` is rejected. |
| `limits.max_history` | `1000` | History ids a request uses (Python has no cap). |
| `limits.history_policy` | `truncate` | A longer history: `truncate` (keep the last `max_history` ids) or `reject` ([History](#history)). |
| `limits.max_body_bytes` | `1048576` (1 MiB) | A larger request body is rejected (JSON 413, gRPC `OUT_OF_RANGE`). Raise it for a large `include` (about 100k ids per MiB in JSON). |
| `limits.max_items` | `10000` | A score request with more items is rejected, never cut ([`POST /v1/score`](#post-v1score-json)). |
| `defaults.k` | `10` | `k` of a request that doesn't send it (at most `limits.k_max`): e.g. `500` for candidate generation. |
| `defaults.sorted` | `false` | `sorted` of a request that doesn't send it ([Sorted or not](#sorted-or-not)); `true` = Python's order. |
| `defaults.view` | `""` (the whole model) | `view` of a request that doesn't send it ([Views](#views)). |
| `timeouts.header_read` | `30` | Seconds a connection has to send a request's headers (and a JSON body: else 408). Also the idle time after which an HTTP/1.1 keep-alive connection is closed. |
| `timeouts.drain_delay` | `0` | After SIGTERM: seconds still accepting connections with `/ready` answering 503 ([Shutdown](#shutdown)). |
| `timeouts.shutdown_grace` | `10` | Then seconds for open requests to finish. |
| `views.reload_interval` | `10` | Seconds between checks of the view files (`0` or at least `0.1`): a changed file is served from the next request on; `0` = read at start only ([Updating a view](#updating-a-view)). |
| `views.NAME.include_file` | | View `NAME` allows only the ids in this file. |
| `views.NAME.exclude_file` | | View `NAME` never returns the ids in this file (either file or both). |

Relative file paths are relative to the server's working directory.

Short flags kept for scripts: `--model DIR` (= `--model.path`), `--addr`,
`--threads`, `--block-size`, `--max-history`, `--k-max`, `--no-mmap`.

## API

The JSON endpoints are also an OpenAPI 3.1 spec,
[`openapi.json`](https://github.com/bogliosimone/similaripy/blob/master/crates/similaripy-server/openapi.json)
(also in the release archive): import it into a client generator or Postman,
or open it in [Swagger Editor](https://editor.swagger.io) to browse it.

### `POST /v1/recommend` (JSON)

Fields and defaults as Python's
[`recommend`](api/serving.md), except `k`, `sorted` and `view` (the
server's defaults):

| Field | Default | Meaning |
|-------|---------|---------|
| `history` | (required) | The user's history, oldest first; ids the model doesn't have are ignored. |
| `weights` | all 1.0 | One per id. |
| `k` | `defaults.k` (`10`) | Items to return (at most `limits.k_max`); `null` = the default. |
| `exclude_seen` | `true` | Never return the history's items. |
| `include` | every item | Only these items can be returned. |
| `exclude` | none | Never these items. |
| `sorted` | `defaults.sorted` | Best first (`true`) or in no particular order. |
| `view` | `defaults.view` | A named view; `""` = the whole model. |

The answer: `items` (best first when sorted), then the request's context:
which `model` answered, the `request_id`, the `k`, `sorted` and `view` used
(defaults applied), and `history` counts ([History](#history)).

Headers:

- `x-request-id`: sent by the client (kept if 1-128 printable ASCII characters without
  spaces, e.g. set by an ingress; else replaced) or generated, echoed in the header and the body:
  the key to join a client's log with the server's.
- `Server-Timing: compute;dur=0.031`: the model's compute time in ms.

### `POST /v1/score` (JSON)

The scores of given items for one history, as Python's
[`score`](api/serving.md): the item-KNN score of each candidate other
retrievers found, as a feature for a ranker. The same bits as
`/v1/recommend`'s score of that item; no top-k, no threshold, history items
included.

| Field | Default | Meaning |
|-------|---------|---------|
| `history` | (required) | As `/v1/recommend`'s (cut past `limits.max_history` the same way). |
| `weights` | all 1.0 | One per id. |
| `items` | (required) | The items to score, at most `limits.max_items`; duplicates are scored again. |
| `view` | `defaults.view` | A named view: its excluded items score 0. |

```bash
curl -s 127.0.0.1:8080/v1/score -d '{"history": [10, 20], "items": [30, 40, 99]}'
```

The answer: `scores`, one per item in the request's order (0 for an item the
model doesn't have, has no column for, e.g. outside a trimmed build's
candidates, or the view excludes), then the request's context:
`model`, `request_id`, `view` and `history` counts, as `/v1/recommend`'s. The
same headers and errors as `/v1/recommend`, plus `items_over_max`.

### Errors

Errors of the JSON endpoints are JSON, with the request id in the body and
the header:

```json
{"error": {"code": "k_over_max", "message": "k = 5000 > k_max = 2000", "request_id": "abc-1"}}
```

Branch on `code` (fixed strings: new ones may be added, none change), show
`message` to people (its wording may change).

| Status | `code` | When |
|--------|--------|------|
| 400 | `bad_json` | Not valid JSON, an unknown or missing field, a wrong type |
| 400 | `k_over_max` | `k` > `limits.k_max` |
| 400 | `items_over_max` | More score `items` than `limits.max_items` |
| 400 | `history_over_max` | A history over `max_history` with `history_policy = reject` |
| 400 | `unknown_view` | A `view` not in the config |
| 400 | `bad_weights` | Weights of the wrong length, or not finite |
| 400 | `bad_body` | The body couldn't be read (e.g. malformed chunked encoding) |
| 408 | `body_timeout` | The body didn't arrive within `timeouts.header_read` |
| 413 | `body_too_large` | Body over `limits.max_body_bytes` (a declared `Content-Length` over it: at once) |
| 500 | `internal` | A server error (also logged to stderr with the request id) |

### gRPC

`similaripy.v1.Recommender/Recommend` and `/Score` on the same port (HTTP/2
without TLS), defined in
[`recommender.proto`](https://github.com/bogliosimone/similaripy/blob/master/crates/similaripy-server/proto/similaripy/v1/recommender.proto)
(also in the release archive). Same fields, defaults, limits and answers as
JSON; ids and scores come back as two packed arrays, the fast path for large
`k`. With [grpcurl](https://github.com/fullstorydev/grpcurl) (the server has
no reflection: pass the proto):

```bash
grpcurl -plaintext -proto recommender.proto -H 'x-request-id: abc-1' \
  -d '{"history": [10, 20], "k": 3}' 127.0.0.1:8080 similaripy.v1.Recommender/Recommend
```

Differences from JSON (protobuf's rules):

- Empty `weights` = unweighted (proto3 can't tell an empty list from none).
- `include` is a message (`IdList`), so an empty list can be sent: absent =
  every item, an empty list = no item (as JSON's absent vs `[]`).
- Unknown fields are ignored, not rejected.
- Errors are status codes: `INVALID_ARGUMENT` (JSON's 400), `OUT_OF_RANGE`
  (a message over `limits.max_body_bytes`), `INTERNAL` (a server error or a
  malformed message). The metadata carries `x-request-id` and, for the
  server's own errors, `x-error-code` with JSON's `code` (e.g.
  `k_over_max`). An oversized or malformed message is rejected by the gRPC
  library before the server sees it: no `x-error-code`, no
  `x-request-id`.

### Other endpoints

| Path | Answer |
|------|--------|
| `GET /health` | `ok` while the process runs (liveness) |
| `GET /ready` | `ok`, or 503 `draining` after SIGTERM (readiness) |
| `GET /metrics` | Prometheus text ([Metrics](#metrics)) |
| `GET /config` | The resolved config (JSON) |
| `GET /v1/views` | The views, their sizes, their files' known ids and modification times, a failed reload's error ([Views](#views)) |

## Answers

**Ids are JSON numbers** (64-bit integers). JavaScript reads numbers as
doubles: ids above 2^53 (9 007 199 254 740 992) lose precision there; use a
parser with big-integer support, or gRPC.

**Ties are fixed for a given model.** Items with exactly the same score are
ordered by the model's internal order, not by id: the same request to the
same saved model always gets the same items and scores in the same order
(and the same bytes with the same `x-request-id`: cache keys, A/B debugging
and tests rely on it), but two models with the same scores may order their ties
differently.

### Sorted or not

The server's default is unsorted (`defaults.sorted = false`), unlike Python's
`recommend`: its main use is candidate generation, where a ranker re-scores
the candidates anyway. `sorted: false` returns the same items and scores in
no particular order and saves the final sort: 19-50% of the compute at `k` ≥
1000 (JSON +8..+13% requests/s at `k` 1000-2000). The order is fixed for a
given model and server config, but can change with `server.block_size` or
`server.threads`. To use the answer's first N items as a top N, send
`"sorted": true` (or set `defaults.sorted = true`); the answer's `sorted`
says which order it has.

### History

The history is used oldest → newest, at most `limits.max_history` ids:

- `truncate` (default): the oldest ids are cut, the last `max_history` kept;
  cut ids are still excluded from the answer when `exclude_seen` is on.
- `reject`: a longer history is a 400 / `INVALID_ARGUMENT`.

An empty history (a new user) is not an error: the answer has no items
(`"items": []`; `/v1/score`: all scores 0). Fall back to something else, e.g. popular items.

Each answer counts its history: `sent` (ids sent), `truncated` (cut) and
`unknown` (ids the model doesn't have, among the rest: ignored). A high
`unknown` rate usually means a client sends ids from another catalogue or
model version.

## Views

A view is a named set of items that can be recommended (e.g. "in stock in
Italy"), built at start from id files: the server-side form of
[`model.view`](serving.md#availability), one bit per item, S shared.

```toml
[views.in_stock]
include_file = "views/in_stock.txt"     # only these items

[views.in_stock_no_adult]
include_file = "views/in_stock.txt"
exclude_file = "views/adult.txt"        # minus these
```

An id file has one integer id per line (`np.savetxt(path, ids, fmt="%d")`).
Ids the model doesn't have are ignored: the start-up log and `GET /v1/views`
show how many ids of each file the model knows (a file in the wrong id
space reads "0 known").

A request picks one view with `"view": "in_stock"`. Without one it gets
`defaults.view`; `"view": ""` is the whole model. An unknown name is a 400.
The request's own `include` / `exclude` apply on top (AND): a request can
narrow a view, never widen it. Example: view `in_stock` = items 1-1000,
request `"include": [5, 7, 2000]` → candidates 5 and 7 (2000 is not in
stock).

### Updating a view

The server checks the view files every `views.reload_interval` seconds
(default 10) and rebuilds the views whose files changed: the next request
uses them, requests in flight finish with the old ones. Only the masks are
rebuilt (one bit per item, milliseconds); the model is not touched. So a
stock list refreshed every few minutes needs no restart and no new model.

- **Write a new file, then rename it over the old one** (`mv` is atomic on
  one file system), or swap a symlink: the reader never sees half a file.
  Don't edit the file in place: a check during the write can read part of
  it, and on file systems with coarse modification times the rest of the
  write may never be noticed.
- **A file that fails to read** (missing, a bad line) keeps its view as it
  was: logged once per change of the file, counted in
  `view_reload_errors_total{view}`, and shown as the view's `error` in
  `GET /v1/views`. The other views still reload. A view in error is read
  again at every check, so a passing failure heals by itself.
- **Is my edit live?** `GET /v1/views` shows each file's `modified` time as
  read and its `ids`; `view_last_reload_timestamp_seconds` the last reload.
- **Adding or removing a view** is a config change: restart (a rolling
  update in Kubernetes).

**Small lists: a ConfigMap.** A ConfigMap holds at most 1 MiB, about 100k
ids. Kubernetes updates a mounted ConfigMap by writing a new directory and
swapping a symlink, which the server follows; the change reaches the pods
within about a minute, plus `reload_interval`. Don't mount it with
`subPath`: those files never update.

```yaml
# kubectl create configmap views --from-file=in_stock.txt \
#   --dry-run=client -o yaml | kubectl apply -f -      (to create or update it)
spec:
  containers:
    - name: server
      env:
        - {name: SIMILARIPY_VIEWS__IN_STOCK__INCLUDE_FILE, value: /views/in_stock.txt}
      volumeMounts:
        - {name: views, mountPath: /views, readOnly: true}
  volumes:
    - name: views
      configMap: {name: views}
```

**Large lists: a sidecar.** A container in the same pod downloads the file
into a shared `emptyDir` and renames it into place; an init container does
the first copy (the server reads the views at start, so the file must be
there):

```yaml
spec:
  initContainers:
    - name: views-init
      image: amazon/aws-cli
      command: [sh, -c, "aws s3 cp s3://my-bucket/views/in_stock.txt /views/in_stock.txt"]
      volumeMounts: [{name: views, mountPath: /views}]
  containers:
    - name: server
      env:
        - {name: SIMILARIPY_VIEWS__IN_STOCK__INCLUDE_FILE, value: /views/in_stock.txt}
      volumeMounts: [{name: views, mountPath: /views, readOnly: true}]
    - name: views-sync
      image: amazon/aws-cli
      command: [sh, -c, "while true; do sleep 300; aws s3 cp s3://my-bucket/views/in_stock.txt /views/.in_stock.tmp && mv /views/.in_stock.tmp /views/in_stock.txt; done"]
      volumeMounts: [{name: views, mountPath: /views}]
  volumes:
    - {name: views, emptyDir: {}}
```

Cost (Yambda 500M, one thread): a request without a view pays nothing; a
broad view (90% of the items) about +3% CPU per request; a narrow one (10%)
+10-14% at `k` 100 and +3% at `k` 2000, as more candidates are checked
against it.

## Metrics

`GET /metrics`, Prometheus text format, names prefixed with
`similaripy_server_`:

| Metric | Meaning |
|--------|---------|
| `info{version, model}` | Always 1: the server's version and the model it serves (which pod serves which model during a rolling update) |
| `threads`, `block_size`, `items`, `active_items` | Gauges: the setup |
| `requests_total{thread}` | Requests computed, per thread |
| `bad_requests_total` | Requests rejected before compute (JSON body: parse, unreadable, 408, 413; `k`, score `items`, history, view, weights of a cut history) |
| `errors_total` | Requests the model rejected (weights) |
| `truncated_total` | Histories cut to `max_history` |
| `history_ids_total`, `known_ids_total` | History ids used, and those the model has |
| `view_requests_total{view}` | Requests per named view (only with views configured) |
| `view_reloads_total` | View reloads that rebuilt a view whose file changed ([Updating a view](#updating-a-view)) |
| `view_reload_errors_total{view}` | Changes to a view's files that failed to read (it kept its previous ids) |
| `view_last_reload_timestamp_seconds` | Unix time of the last reload that rebuilt a view, 0 = none since start: alert when it gets older than your update period |
| `compute_seconds{endpoint, proto}` | Histogram: the model's compute, per endpoint (`recommend`, `score`) and format (`json`, `grpc`) |
| `request_seconds{endpoint, proto}` | Histogram: the handler (JSON: parse + compute + serialize) |

Plus the standard `process_cpu_seconds_total`, `process_start_time_seconds`
and `process_resident_memory_bytes` (Linux and macOS; it includes the pages
of a mapped model read so far).

To see them on a dashboard, the repository has a local stack (server,
Prometheus, Grafana with a ready dashboard, a load generator) that starts
with one command:
[`crates/similaripy-server/observability`](https://github.com/bogliosimone/similaripy/tree/master/crates/similaripy-server/observability).

## In production

- **Private network only.** No auth, no TLS: anyone who reaches the port can
  query it, and `Server-Timing` tells them the compute time. Put an ingress
  or a service mesh with TLS and auth in front of it if it must be reached
  from outside.
- **Threads.** Each thread answers one request at a time, start to finish
  (a request takes tens of µs to about a ms). A connection stays on one
  thread: send concurrent requests over several connections.
- **gRPC clients:** a gRPC channel is one HTTP/2 connection, so one channel
  = one server thread. Open several channels, or put an L7 (HTTP/2-aware)
  load balancer in front, which spreads requests, not connections.
- **Proxies too.** A sidecar or ingress in front keeps a pool of upstream
  connections (e.g. nginx `keepalive 8`): at most that many threads get
  work. Keep the pool at least `server.threads`;
  `requests_total{thread}` shows the balance.
- **Logs.** Start-up, shutdown and one stderr line per server error (500 /
  `INTERNAL`, with the request id); client errors (4xx) are only counted.

### Sizing

- **Memory.** With `mmap` (the default) the model's files are read through
  the page cache, which counts in a container's memory: set the memory
  limit to at least the model directory's size plus a little per thread.
  A lower limit isn't an out-of-memory kill but pages dropped and re-read
  from disk (latency spikes). With `model.mmap = false` the model is
  ordinary memory: a limit under the model's size kills the process.
- **Per thread:** at most 4 bytes x active items (the score accumulator;
  less with column blocks). **Per view:** items / 8 bytes (one bit per
  item).
- **Network volumes.** With `mmap`, a page not in memory is a network read
  inside a request. Copy the model to local disk (bake it into the image,
  or an `emptyDir` filled at start), or set `model.mmap = false`.
- **Threads without a CPU limit.** `server.threads` defaults to the CPU
  *limit*. With only a CPU request (no limit) it sees the whole node (e.g.
  64 threads for a 4-CPU request): set `server.threads` to the request.

### Shutdown

On SIGTERM (or Ctrl-C), in order:

1. `/ready` answers 503, connections are still accepted for
   `timeouts.drain_delay` seconds;
2. the listener closes: new connections are refused;
3. open requests get `timeouts.shutdown_grace` seconds to finish (HTTP/1.1
   connections close after their response, HTTP/2 ones get a GOAWAY);
4. the process exits 0. A second signal exits at once with 1.

### Kubernetes

**Getting the model into the pod.** The simplest way: bake it into an
image on top of the server's, one image tag per model version (a rollout or
a rollback is an image change):

```dockerfile
FROM ghcr.io/bogliosimone/similaripy-server:1.0.0
COPY --chown=65532 models/itemknn-2026-11-16 /model
ENV SIMILARIPY_MODEL__PATH=/model SIMILARIPY_MODEL__NAME=itemknn-2026-11-16
```

The example below mounts a shared volume instead (a
`PersistentVolumeClaim`). Another common way: an init container copies the
model from object storage (S3, GCS) into an `emptyDir` before the server
starts.

A pod is stopped with SIGTERM, then killed after
`terminationGracePeriodSeconds` (default 30). The load balancer takes a few
seconds to notice the pod is going: `drain_delay` keeps it answering until
then.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommender
spec:
  replicas: 3
  selector:
    matchLabels: {app: recommender}
  template:
    metadata:
      labels: {app: recommender}
    spec:
      terminationGracePeriodSeconds: 30
      containers:
        - name: server
          image: ghcr.io/bogliosimone/similaripy-server:1.0.0
          args: ["--model", "/models/itemknn-2026-11-16"]
          env:
            - {name: SIMILARIPY_MODEL__NAME, value: itemknn-2026-11-16}
            - {name: SIMILARIPY_TIMEOUTS__DRAIN_DELAY, value: "5"}
            - {name: SIMILARIPY_TIMEOUTS__SHUTDOWN_GRACE, value: "10"}
          ports:
            - containerPort: 8080
          startupProbe:          # the model loads before the port opens
            httpGet: {path: /health, port: 8080}
            periodSeconds: 2
            failureThreshold: 60   # up to 2 minutes
          livenessProbe:
            httpGet: {path: /health, port: 8080}
          readinessProbe:
            httpGet: {path: /ready, port: 8080}
            periodSeconds: 2
          resources:
            limits: {cpu: "4"}
          volumeMounts:
            - {name: models, mountPath: /models, readOnly: true}
      volumes:
        - name: models
          persistentVolumeClaim: {claimName: models}   # where your saved models live
```

- **Startup** probe: the server loads the model before it opens the port
  (seconds for a multi-GB model on a slow volume). Without it the liveness
  probe can kill the pod before it listens, in a loop.
- **Liveness** `/health` stays ok while draining, so the pod isn't killed
  mid-drain; **readiness** `/ready` turns 503 at SIGTERM, so it stops
  getting traffic.
- `drain_delay` (5 s) + `shutdown_grace` (10 s) must stay under
  `terminationGracePeriodSeconds` (30 s).
- `server.threads` defaults to the CPU limit (here 4). No CPU limit? Set
  `SIMILARIPY_SERVER__THREADS` to the CPU request ([Sizing](#sizing)).
- HTTP/1.1 clients that keep connections open longer than
  `timeouts.header_read` (30 s) without a request see them closed. A pooled
  client usually reconnects, but a POST sent just as the server closes can
  fail (Python `requests` doesn't retry POST): set the client's idle timeout
  under 30 s, or retry.
- A new model = a new directory and a new `--model` path: a rolling update
  replaces the pods one by one.

### Changing the model

- **Never write into a directory a server is serving**, not even to
  overwrite a file with the same contents: with `mmap` (the default) the
  files are mapped, and changing one under the server crashes it (SIGBUS)
  or serves mixed data. Save each model to a new directory
  (`models/itemknn-2026-11-17`); deleting the old one once no server uses it
  is safe.
- **A `current` symlink** to the latest directory works: the server resolves
  it once at start and reads every file from that directory, even if the
  link is flipped during the load. Flipping it later changes nothing until a
  restart. Set `model.name` (e.g. from your deploy script), or every answer
  names the model `current`.
- **Save with item ids** (`SparseKNN(S, item_ids=...)`): the ids in requests
  are then your catalogue's ids, and a client that sends ids from another
  catalogue or model version shows up in `history.unknown` and
  `known_ids_total`. A model saved without ids accepts any id below its
  item count, so such a mismatch goes unnoticed.
