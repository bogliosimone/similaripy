# Local observability: server + Prometheus + Grafana

A local stack to watch `similaripy-server` under steady traffic. For trying
the server and its metrics, not for timing: everything runs in the container
VM, next to the load generator (use the bench runner for numbers).

```bash
cd crates/similaripy-server/observability
podman compose up -d --build      # or: docker compose up -d --build
```

Then open <http://localhost:3000> (Grafana, no login): the
`similaripy-server` dashboard is the home page. Stop with
`podman compose down`. After changing `compose.yaml` or rebuilding, run
`down` before `up`: podman can't recreate the server while the loadgen
containers share its network.

| Service | What it does | Local URL |
|---|---|---|
| `server` | built from this checkout (`Dockerfile`, first build a few minutes), 4 threads | <http://localhost:18080> |
| `loadgen-json`, `loadgen-grpc` | `loadgen --rate 200` each, steps of 300 s, restarted forever | `podman compose logs loadgen-json` |
| `prometheus` | scrapes `server:8080/metrics` every 5 s | <http://localhost:9090> |
| `grafana` | Prometheus data source + the dashboard, provisioned from files | <http://localhost:3000> |

## How the pieces connect

- **Compose** (`compose.yaml`) starts the containers on one private network,
  where each reaches the others by service name (`server:8080`,
  `prometheus:9090`). `ports:` publishes a few of them on your machine,
  127.0.0.1 only.
- **Prometheus** (`prometheus.yml`) polls `/metrics` every 5 s and keeps the
  numbers. Counters only go up; panels show `rate(...)`, the per-second slope.
- **Grafana provisioning** replaces clicking through the UI: at start-up
  Grafana reads `grafana/provisioning/` (the data source: "Prometheus at
  `http://prometheus:9090`"; the dashboards: "load every JSON in
  `/var/lib/grafana/dashboards`"). Edit a panel in the UI, then export it
  (Export -> Export as JSON) over `grafana/dashboards/similaripy-server.json`
  to keep the change.
- **loadgen** takes an IP, not a host name, so its containers share the
  server's network namespace and call `127.0.0.1:8080`.

## The demo data

The default model is the committed `tests/data/sparseknn-format1` (ids 10..70)
with `demo/server.toml` (k_max 50, max_history 8, view `in_stock`) and
`demo/requests.jsonl`, whose 20 lines are mostly valid but include one
`k: 100` (a 400), one 10-id history (truncated), unknown ids 99 and 1000, and
`in_stock` requests, so every panel has data.

A real model, ML-32M (from `benchmarks/server_prep.py`, not in the repo;
the model and request files must match the server's version):

```bash
podman compose --env-file ml-32m.env up -d
```

Knobs (env vars or an env file): `MODEL_DIR`, `REQUESTS`, `SERVER_CONFIG`,
`THREADS` (default 4), `RATE` (per proto, default 200).

## The dashboard

| Panel | Query (simplified) |
|---|---|
| Instances | `similaripy_server_info`: version and model per instance (a rolling update shows both models) |
| Requests/s per proto | `rate(request_seconds_count)` by proto |
| CPU per request | `rate(process_cpu_seconds_total) / rate(request_seconds_count)` |
| Request / compute latency | `histogram_quantile(0.5/0.95/0.99, ...)` by proto; red line = 10 ms budget |
| Rejected and truncated | `bad_requests_total`, `errors_total`, `truncated_total` over requests |
| Unknown history ids | `1 - rate(known_ids_total) / rate(history_ids_total)` |
| Requests/s per view | `view_requests_total` by view, the rest = whole model |
| Requests/s per thread | `requests_total` by thread (round-robin balance) |

Quantiles are interpolated inside the server's buckets (10 us .. 100 ms), so
they are approximate between bucket bounds.
