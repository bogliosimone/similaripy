//! Per-thread counters and latency histograms, each written only by its own
//! server thread (no locks, no read-modify-write atomics on the request path),
//! summed at scrape (Prometheus text).

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

use similaripy_core::serve::Stats;

/// Bucket upper bounds in µs: dense around the 10 ms budget (R1), so the
/// share under budget reads straight off `le="0.01"`.
const BUCKETS_US: [u64; 15] = [
    10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 7_500, 10_000, 15_000, 25_000, 50_000, 100_000,
];

/// One writer: a plain load + store, no atomic read-modify-write; a scrape
/// on another thread sees a recent value.
fn add(a: &AtomicU64, v: u64) {
    a.store(a.load(Relaxed).wrapping_add(v), Relaxed);
}

/// The request's format: the `proto` label of the request histograms.
#[derive(Clone, Copy)]
pub enum Proto {
    Json,
    Grpc,
}

const PROTOS: [&str; 2] = ["json", "grpc"];

#[derive(Default)]
pub struct Histogram {
    /// Per bucket (not cumulative), the last one = over the largest bound.
    counts: [AtomicU64; BUCKETS_US.len() + 1],
    sum_ns: AtomicU64,
}

impl Histogram {
    pub fn record(&self, d: Duration) {
        let ns = d.as_nanos() as u64;
        let b = BUCKETS_US
            .iter()
            .position(|&le| ns <= le * 1_000)
            .unwrap_or(BUCKETS_US.len());
        add(&self.counts[b], 1);
        add(&self.sum_ns, ns);
    }
}

/// One server thread's metrics. 128-byte aligned: two threads never write
/// the same cache line (128 B on Apple silicon, 64 B on x86).
#[repr(align(128))]
#[derive(Default)]
pub struct Thread {
    /// The worker's `Stats` since warm-up (requests, invalid weights,
    /// history ids after the cap, of which the model has).
    requests: AtomicU64,
    errors: AtomicU64,
    ids: AtomicU64,
    known_ids: AtomicU64,
    /// Rejected at the edge (bad JSON, k > k_max): never reached the worker.
    pub bad_requests: AtomicU64,
    /// Histories cut to their last `max_history` ids.
    pub truncated: AtomicU64,
    /// `Worker::recommend` alone, per `Proto`.
    pub compute: [Histogram; 2],
    /// The handler, per `Proto`: JSON body parsed → response serialized;
    /// gRPC message decoded → answer copied (tonic's decode and encode are
    /// outside). Queue wait and HTTP reads are not in it.
    pub total: [Histogram; 2],
}

impl Thread {
    pub fn set_stats(&self, s: Stats) {
        self.requests.store(s.requests, Relaxed);
        self.errors.store(s.errors, Relaxed);
        self.ids.store(s.ids, Relaxed);
        self.known_ids.store(s.known_ids, Relaxed);
    }

    pub fn bad_request(&self) {
        add(&self.bad_requests, 1);
    }

    pub fn truncated(&self) {
        add(&self.truncated, 1);
    }
}

/// Fixed facts about the server, as gauges.
pub struct Info {
    pub threads: usize,
    pub block: usize,
    pub items: usize,
    pub active_items: usize,
}

/// Prometheus text exposition of every thread's metrics, summed (requests
/// also per thread: the round-robin's balance).
pub fn render(threads: &[Thread], info: &Info) -> String {
    let mut o = String::new();
    let gauges = [
        ("threads", "Server threads, one Worker each.", info.threads),
        (
            "block_size",
            "Column block size of every worker, 0 = one block.",
            info.block,
        ),
        ("items", "Items in the model.", info.items),
        (
            "active_items",
            "Items with a column (can be recommended).",
            info.active_items,
        ),
    ];
    for (name, help, v) in gauges {
        head(&mut o, name, "gauge", help);
        let _ = writeln!(o, "similaripy_server_{name} {v}");
    }
    if let Some(cpu) = process_cpu_seconds() {
        let help = "# HELP process_cpu_seconds_total User and system CPU time of the process.";
        let _ = writeln!(o, "{help}\n# TYPE process_cpu_seconds_total counter");
        let _ = writeln!(o, "process_cpu_seconds_total {cpu}");
    }
    head(
        &mut o,
        "requests_total",
        "counter",
        "Requests computed by the worker, per thread.",
    );
    for (i, t) in threads.iter().enumerate() {
        let v = t.requests.load(Relaxed);
        let _ = writeln!(o, "similaripy_server_requests_total{{thread=\"{i}\"}} {v}");
    }
    type Field = fn(&Thread) -> &AtomicU64;
    let totals: [(&str, &str, Field); 5] = [
        (
            "errors_total",
            "Requests the worker rejected (weights).",
            |t| &t.errors,
        ),
        (
            "bad_requests_total",
            "Requests rejected at the edge (JSON, k; not gRPC decode errors).",
            |t| &t.bad_requests,
        ),
        (
            "truncated_total",
            "Histories cut to their last max_history ids.",
            |t| &t.truncated,
        ),
        (
            "history_ids_total",
            "History ids served (after the cap).",
            |t| &t.ids,
        ),
        (
            "known_ids_total",
            "History ids the model has (the rest are ignored).",
            |t| &t.known_ids,
        ),
    ];
    for (name, help, f) in totals {
        head(&mut o, name, "counter", help);
        let v: u64 = threads.iter().map(|t| f(t).load(Relaxed)).sum();
        let _ = writeln!(o, "similaripy_server_{name} {v}");
    }
    let help = "Worker::recommend time, per request format.";
    histogram(&mut o, "compute_seconds", help, threads, |t, p| {
        &t.compute[p]
    });
    let help = "Handler time, per request format: JSON parsed, computed, serialized; \
        gRPC computed, answer copied (decode and encode outside).";
    histogram(&mut o, "request_seconds", help, threads, |t, p| &t.total[p]);
    o
}

/// The process's user + system CPU time (every thread: the acceptor's is
/// near zero with keep-alive): CPU per request = its change / requests.
#[cfg(unix)]
fn process_cpu_seconds() -> Option<f64> {
    // SAFETY: rusage is plain integers: all-zero is a valid value.
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage writes only into the struct we pass.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) } != 0 {
        return None;
    }
    let s = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    Some(s(u.ru_utime) + s(u.ru_stime))
}

#[cfg(not(unix))]
fn process_cpu_seconds() -> Option<f64> {
    None
}

fn head(o: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(o, "# HELP similaripy_server_{name} {help}");
    let _ = writeln!(o, "# TYPE similaripy_server_{name} {kind}");
}

fn histogram(
    o: &mut String,
    name: &str,
    help: &str,
    threads: &[Thread],
    h: fn(&Thread, usize) -> &Histogram,
) {
    head(o, name, "histogram", help);
    let name = format!("similaripy_server_{name}");
    for (p, proto) in PROTOS.iter().enumerate() {
        let mut cum = 0;
        for b in 0..=BUCKETS_US.len() {
            cum += threads
                .iter()
                .map(|t| h(t, p).counts[b].load(Relaxed))
                .sum::<u64>();
            let le = BUCKETS_US
                .get(b)
                .map_or("+Inf".to_string(), |&us| (us as f64 / 1e6).to_string());
            let _ = writeln!(o, "{name}_bucket{{proto=\"{proto}\",le=\"{le}\"}} {cum}");
        }
        let ns: u64 = threads.iter().map(|t| h(t, p).sum_ns.load(Relaxed)).sum();
        let _ = writeln!(o, "{name}_sum{{proto=\"{proto}\"}} {}", ns as f64 / 1e9);
        let _ = writeln!(o, "{name}_count{{proto=\"{proto}\"}} {cum}");
    }
}
