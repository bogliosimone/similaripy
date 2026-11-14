//! Open-loop load generator for `similaripy-server`: request i is due at
//! t0 + i / rate whatever the server's speed, and its latency counts from
//! then (no coordinated omission: a slow server can't slow the arrivals).
//! Steps the rate up 1.5x until p99 > budget, errors, or the achieved rate
//! falls under 95% of the target; prints the best rate that met the budget.
//! `--check FILE`: the first requests' ids and score bits vs the expected
//! ones (`benchmarks/server_prep.py`: Python `ItemKNN.recommend`).
//! `--grpc`: the same requests as gRPC calls (`proto_from_json`), one
//! HTTP/2 connection per caller unless `--channels` says fewer (then the
//! callers' streams share them: the multiplexed case).

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use bytes::Bytes;
use serde::Deserialize;
use similaripy_server::client::{Conn, connect, metric, send};
use similaripy_server::pb::RecommendRequest;
use similaripy_server::pb::recommender_client::RecommenderClient;
use similaripy_server::{proto_from_json, set_qos};
use tonic::transport::{Channel, Endpoint};

const USAGE: &str = "\
loadgen --requests FILE [options]
  --requests FILE   one /recommend body per line (benchmarks/server_prep.py)
  --addr HOST:PORT  default 127.0.0.1:8080
  --conns N         callers, one keep-alive connection each; default 64 (>= server threads x callers)
  --grpc            send gRPC calls (similaripy.v1.Recommender/Recommend) instead of JSON
  --channels N      --grpc: HTTP/2 connections the callers share; default = --conns (one stream at a time each)
  --threads N       loadgen threads; default 4
  --duration S      seconds per step; default 10
  --rate R          one step at R requests/s (default: ramp from --start)
  --start R         first step of the ramp; default 1000
  --budget-ms MS    the p99 budget; default 10
  --check FILE      compare responses with FILE (ids + score_bits per line), no load";

fn main() {
    if let Err(e) = run() {
        eprintln!("loadgen: {e}");
        std::process::exit(1);
    }
}

struct Opts {
    addr: SocketAddr,
    conns: usize,
    /// gRPC connections (`--channels`), None = JSON.
    grpc: Option<usize>,
    duration: f64,
    budget: Duration,
}

/// The request lines, in the format sent.
enum Bodies {
    Json(Vec<Bytes>),
    Grpc(Vec<RecommendRequest>),
}

impl Bodies {
    fn len(&self) -> usize {
        match self {
            Bodies::Json(b) => b.len(),
            Bodies::Grpc(b) => b.len(),
        }
    }
}

/// One caller's connection (gRPC: a channel, maybe shared).
enum Client {
    Http(Conn),
    Grpc(RecommenderClient<Channel>),
}

enum Sent {
    /// The answer, when asked for.
    Ok(Option<(Vec<i64>, Vec<f32>)>),
    Rejected(String),
    /// The HTTP/1.1 connection broke (a gRPC channel reconnects itself).
    Broken,
}

impl Client {
    async fn call(&mut self, bodies: &Bodies, i: usize, read: bool) -> Sent {
        match (self, bodies) {
            (Client::Http(c), Bodies::Json(b)) => {
                match send(c, "/recommend", Some(b[i].clone())).await {
                    Ok((StatusCode::OK, body)) if read => {
                        match serde_json::from_slice::<Reply>(&body) {
                            Ok(r) => Sent::Ok(Some((r.ids, r.scores))),
                            Err(e) => Sent::Rejected(e.to_string()),
                        }
                    }
                    Ok((StatusCode::OK, _)) => Sent::Ok(None),
                    Ok((status, body)) => {
                        Sent::Rejected(format!("{status} {}", String::from_utf8_lossy(&body)))
                    }
                    Err(_) => Sent::Broken,
                }
            }
            (Client::Grpc(c), Bodies::Grpc(b)) => match c.recommend(b[i].clone()).await {
                Ok(r) => Sent::Ok(read.then(|| {
                    let r = r.into_inner();
                    (r.ids, r.scores)
                })),
                Err(s) => Sent::Rejected(s.to_string()),
            },
            _ => unreachable!("a client of the bodies' format"),
        }
    }
}

/// `n` callers' clients: a connection each (JSON), or `opts.grpc` channels
/// shared round-robin.
async fn clients(opts: &Opts, n: usize) -> Result<Vec<Client>, String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", opts.addr);
    let Some(channels) = opts.grpc else {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(Client::Http(connect(opts.addr).await.map_err(|e| err(&e))?));
        }
        return Ok(v);
    };
    let mut chans = Vec::with_capacity(channels);
    for _ in 0..channels.clamp(1, n) {
        let c = Endpoint::from_shared(format!("http://{}", opts.addr))
            .map_err(|e| err(&e))?
            .tcp_nodelay(true)
            .connect()
            .await
            .map_err(|e| err(&e))?;
        chans.push(c);
    }
    Ok((0..n)
        .map(|i| Client::Grpc(RecommenderClient::new(chans[i % chans.len()].clone())))
        .collect())
}

fn run() -> Result<(), String> {
    let (mut requests, mut check, mut rate) = (None, None, None);
    let (mut grpc, mut channels) = (false, None);
    let mut addr = "127.0.0.1:8080".to_string();
    let (mut conns, mut threads, mut duration, mut start, mut budget_ms) =
        (64usize, 4usize, 10f64, 1000f64, 10f64);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or(format!("{a} needs a value\n{USAGE}"));
        let err = |e: &dyn std::fmt::Display| format!("{a}: {e}");
        match a.as_str() {
            "--requests" => requests = Some(value()?),
            "--check" => check = Some(value()?),
            "--addr" => addr = value()?,
            "--conns" => conns = value()?.parse().map_err(|e| err(&e))?,
            "--grpc" => grpc = true,
            "--channels" => channels = Some(value()?.parse().map_err(|e| err(&e))?),
            "--threads" => threads = value()?.parse().map_err(|e| err(&e))?,
            "--duration" => duration = value()?.parse().map_err(|e| err(&e))?,
            "--rate" => rate = Some(value()?.parse::<f64>().map_err(|e| err(&e))?),
            "--start" => start = value()?.parse().map_err(|e| err(&e))?,
            "--budget-ms" => budget_ms = value()?.parse().map_err(|e| err(&e))?,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {a}\n{USAGE}")),
        }
    }
    let path = requests.ok_or(format!("--requests is required\n{USAGE}"))?;
    let lines: Vec<Bytes> = std::fs::read_to_string(&path)
        .map_err(|e| format!("{path}: {e}"))?
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| Bytes::from(l.to_string()))
        .collect();
    if lines.is_empty() {
        return Err(format!("{path}: no requests"));
    }
    let conns = conns.max(1);
    let bodies = Arc::new(if grpc {
        let q: Result<Vec<_>, _> = lines.iter().map(|l| proto_from_json(l)).collect();
        Bodies::Grpc(q.map_err(|e| format!("{path}: {e}"))?)
    } else {
        Bodies::Json(lines)
    });
    let opts = Opts {
        addr: addr.parse().map_err(|e| format!("--addr {addr}: {e}"))?,
        conns,
        grpc: grpc.then(|| channels.unwrap_or(conns)),
        duration,
        budget: Duration::from_secs_f64(budget_ms / 1e3),
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads.max(1))
        .on_thread_start(|| set_qos(false))
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async {
        if let Some(check) = check {
            return check_responses(&opts, &bodies, &check).await;
        }
        println!(
            "{:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8}",
            "target/s",
            "achieved/s",
            "p50 ms",
            "p95 ms",
            "p99 ms",
            "max ms",
            "errors",
            "late ms",
            "cpu us",
            "handler",
            "compute"
        );
        let mut rate_now = rate.unwrap_or(start);
        let mut best = None;
        loop {
            let before = server_costs(opts.addr).await;
            let s = step(&opts, &bodies, rate_now).await?;
            let costs = server_costs(opts.addr).await.per_request(&before);
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            println!(
                "{rate_now:>10.0} {:>10.0} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>7} {:>8.3} {:>8.1} {:>8.1} {:>8.1}",
                s.achieved,
                ms(s.p50),
                ms(s.p95),
                ms(s.p99),
                ms(s.max),
                s.errors,
                ms(s.late),
                costs[0],
                costs[1],
                costs[2]
            );
            let ok = s.p99 <= opts.budget && s.errors == 0 && s.achieved >= 0.95 * rate_now;
            if ok {
                best = Some(s.achieved);
            }
            if !ok || rate.is_some() {
                break;
            }
            rate_now *= 1.5;
        }
        match best {
            Some(r) => println!("max rps at p99 <= {budget_ms} ms: {r:.0}"),
            None => println!("no step met p99 <= {budget_ms} ms"),
        }
        Ok(())
    })
}

/// The server's cumulative costs from /metrics (NaN if missing).
struct Costs {
    /// Handled requests (400s included), and those the worker computed.
    counts: [f64; 2],
    /// Process CPU, handler time, Worker::recommend time (seconds).
    seconds: [f64; 3],
}

impl Costs {
    /// µs per request since `before`: CPU (everything: HTTP, JSON,
    /// compute, syscalls), handler (JSON + compute), compute.
    fn per_request(&self, before: &Costs) -> [f64; 3] {
        // Compute over computed requests (an edge 400 never reaches the worker)
        std::array::from_fn(|i| {
            let c = (i == 2) as usize;
            (self.seconds[i] - before.seconds[i]) / (self.counts[c] - before.counts[c]) * 1e6
        })
    }
}

async fn server_costs(addr: SocketAddr) -> Costs {
    let text = match connect(addr).await {
        Ok(mut c) => send(&mut c, "/metrics", None)
            .await
            .map(|(_, b)| b)
            .unwrap_or_default(),
        Err(_) => Bytes::new(),
    };
    let text = String::from_utf8_lossy(&text);
    let m = |name| metric(&text, name).unwrap_or(f64::NAN);
    Costs {
        counts: [
            m("similaripy_server_request_seconds_count"),
            m("similaripy_server_compute_seconds_count"),
        ],
        seconds: [
            m("process_cpu_seconds_total"),
            m("similaripy_server_request_seconds_sum"),
            m("similaripy_server_compute_seconds_sum"),
        ],
    }
}

struct Step {
    achieved: f64,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    max: Duration,
    errors: u64,
    /// The latest an idle connection's request left after its due time: the
    /// timer's ~1-2 ms; more = loadgen itself saturated (not in the latency).
    late: Duration,
}

/// `rate` requests/s for `opts.duration` seconds over `opts.conns`
/// connections (opened before t0); latency from each request's due time.
// ponytail: tokio's timer is ~1 ms granular, so requests leave in ~1 ms
// bursts (rate / 1000 at once); µs-level numbers come from the server's own
// histograms (/metrics).
async fn step(opts: &Opts, bodies: &Arc<Bodies>, rate: f64) -> Result<Step, String> {
    let total = (rate * opts.duration).ceil() as u64;
    let conns = clients(opts, opts.conns).await?;
    let next = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let mut tasks = tokio::task::JoinSet::new();
    for mut c in conns {
        let (next, bodies, addr) = (next.clone(), bodies.clone(), opts.addr);
        tasks.spawn(async move {
            let (mut lat, mut errors, mut late) = (Vec::new(), 0u64, Duration::ZERO);
            loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= total {
                    break;
                }
                let due = t0 + Duration::from_secs_f64(i as f64 / rate);
                // Its connection idle at `due`: the clock starts when it
                // leaves (the timer's 1-2 ms lateness is ours, not the
                // server's: it read p50 2 ms at 200/s). Busy (due already
                // passed): from `due`, the wait for the connection included.
                let start = if Instant::now() < due {
                    tokio::time::sleep_until(due.into()).await;
                    let now = Instant::now();
                    late = late.max(now - due);
                    now
                } else {
                    due
                };
                match c.call(&bodies, i as usize % bodies.len(), false).await {
                    Sent::Ok(_) => lat.push(start.elapsed()),
                    Sent::Rejected(_) => errors += 1,
                    Sent::Broken => {
                        errors += 1;
                        match connect(addr).await {
                            Ok(new) => c = Client::Http(new),
                            Err(_) => break,
                        }
                    }
                }
            }
            (lat, errors, late)
        });
    }
    let (mut lat, mut errors, mut late) = (Vec::new(), 0, Duration::ZERO);
    while let Some(r) = tasks.join_next().await {
        let (l, e, d) = r.map_err(|e| e.to_string())?;
        lat.extend(l);
        errors += e;
        late = late.max(d);
    }
    let wall = t0.elapsed().as_secs_f64();
    if lat.is_empty() {
        return Err(format!("no request succeeded ({errors} errors)"));
    }
    lat.sort_unstable();
    let q = |p: f64| lat[((lat.len() - 1) as f64 * p).round() as usize];
    Ok(Step {
        achieved: lat.len() as f64 / wall,
        p50: q(0.50),
        p95: q(0.95),
        p99: q(0.99),
        max: lat[lat.len() - 1],
        errors,
        late,
    })
}

#[derive(Deserialize)]
struct Reply {
    ids: Vec<i64>,
    scores: Vec<f32>,
}

#[derive(Deserialize)]
struct Expected {
    ids: Vec<i64>,
    score_bits: Vec<u32>,
}

/// Request i (one connection, one at a time) against line i of `path`:
/// the same ids, and scores with the same f32 bits.
async fn check_responses(opts: &Opts, bodies: &Bodies, path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut c = clients(opts, 1).await?.pop().expect("one client");
    let (mut n, mut bad) = (0, 0);
    for (i, line) in text.lines().filter(|l| !l.is_empty()).enumerate() {
        let want: Expected =
            serde_json::from_str(line).map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        if i >= bodies.len() {
            return Err(format!("{path} has more lines than --requests"));
        }
        let (ids, scores) = match c.call(bodies, i, true).await {
            Sent::Ok(r) => r.expect("asked for the answer"),
            Sent::Rejected(e) => return Err(format!("request {i}: {e}")),
            Sent::Broken => return Err(format!("request {i}: connection closed")),
        };
        let bits: Vec<u32> = scores.iter().map(|s| s.to_bits()).collect();
        if ids != want.ids || bits != want.score_bits {
            bad += 1;
            if bad <= 5 {
                eprintln!("request {i}: ids or score bits differ");
            }
        }
        n += 1;
    }
    println!("check: {n} requests, {bad} differ");
    if bad > 0 {
        return Err(format!("{bad} of {n} responses differ"));
    }
    Ok(())
}
