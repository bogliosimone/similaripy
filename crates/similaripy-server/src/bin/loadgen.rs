//! Open-loop load generator for `similaripy-server`: request i is due at
//! t0 + i / rate whatever the server's speed, and its latency counts from
//! then (no coordinated omission: a slow server can't slow the arrivals).
//! Steps the rate up 1.5x until p99 > budget, errors, or the achieved rate
//! falls under 95% of the target; prints the best rate that met the budget.
//! `--check FILE`: the first requests' ids and score bits vs the expected
//! ones (`benchmarks/server_prep.py`: Python `ItemKNN.recommend`).

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use bytes::Bytes;
use serde::Deserialize;
use similaripy_server::client::{connect, send};
use similaripy_server::set_qos;

const USAGE: &str = "\
loadgen --requests FILE [options]
  --requests FILE   one /recommend body per line (benchmarks/server_prep.py)
  --addr HOST:PORT  default 127.0.0.1:8080
  --conns N         keep-alive connections; default 64 (>= server threads x callers)
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
    duration: f64,
    budget: Duration,
}

fn run() -> Result<(), String> {
    let (mut requests, mut check, mut rate) = (None, None, None);
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
    let bodies: Arc<Vec<Bytes>> = Arc::new(
        std::fs::read_to_string(&path)
            .map_err(|e| format!("{path}: {e}"))?
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| Bytes::from(l.to_string()))
            .collect(),
    );
    if bodies.is_empty() {
        return Err(format!("{path}: no requests"));
    }
    let opts = Opts {
        addr: addr.parse().map_err(|e| format!("--addr {addr}: {e}"))?,
        conns: conns.max(1),
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
            "{:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>7} {:>8}",
            "target/s", "achieved/s", "p50 ms", "p95 ms", "p99 ms", "max ms", "errors", "late ms"
        );
        let mut rate_now = rate.unwrap_or(start);
        let mut best = None;
        loop {
            let s = step(&opts, &bodies, rate_now).await?;
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            println!(
                "{rate_now:>10.0} {:>10.0} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>7} {:>8.3}",
                s.achieved,
                ms(s.p50),
                ms(s.p95),
                ms(s.p99),
                ms(s.max),
                s.errors,
                ms(s.late)
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
async fn step(opts: &Opts, bodies: &Arc<Vec<Bytes>>, rate: f64) -> Result<Step, String> {
    let total = (rate * opts.duration).ceil() as u64;
    let mut conns = Vec::with_capacity(opts.conns);
    for _ in 0..opts.conns {
        conns.push(
            connect(opts.addr)
                .await
                .map_err(|e| format!("{}: {e}", opts.addr))?,
        );
    }
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
                let body = bodies[i as usize % bodies.len()].clone();
                match send(&mut c, "/recommend", Some(body)).await {
                    Ok((StatusCode::OK, _)) => lat.push(start.elapsed()),
                    Ok(_) => errors += 1,
                    Err(_) => {
                        errors += 1;
                        match connect(addr).await {
                            Ok(new) => c = new,
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
async fn check_responses(opts: &Opts, bodies: &[Bytes], path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut c = connect(opts.addr).await.map_err(|e| e.to_string())?;
    let (mut n, mut bad) = (0, 0);
    for (i, line) in text.lines().filter(|l| !l.is_empty()).enumerate() {
        let want: Expected =
            serde_json::from_str(line).map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        let body = bodies
            .get(i)
            .ok_or(format!("{path} has more lines than --requests"))?;
        let (status, got) = send(&mut c, "/recommend", Some(body.clone()))
            .await
            .map_err(|e| e.to_string())?;
        let got: Reply = serde_json::from_slice(&got).map_err(|e| {
            format!(
                "request {i}: {status} {}: {e}",
                String::from_utf8_lossy(&got)
            )
        })?;
        let bits: Vec<u32> = got.scores.iter().map(|s| s.to_bits()).collect();
        if got.ids != want.ids || bits != want.score_bits {
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
