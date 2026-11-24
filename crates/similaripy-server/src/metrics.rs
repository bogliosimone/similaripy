//! Per-thread counters and latency histograms, each written only by its own
//! server thread (no locks, no read-modify-write atomics on the request path),
//! summed at scrape (Prometheus text).

use std::fmt::Write;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, SystemTime};

use similaripy_core::serve::Stats;

/// Bucket upper bounds in µs: dense around a 10 ms budget, so the
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

/// The request's endpoint: the `endpoint` label of the request histograms.
#[derive(Clone, Copy)]
pub enum Endpoint {
    Recommend,
    Score,
}

const ENDPOINTS: [&str; 2] = ["recommend", "score"];

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

/// A counter written by one server thread, on its own cache line.
#[repr(align(128))]
#[derive(Default)]
pub struct Counter(AtomicU64);

impl Counter {
    pub fn add(&self) {
        add(&self.0, 1);
    }

    pub fn get(&self) -> u64 {
        self.0.load(Relaxed)
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
    /// Rejected at the edge, never reached the worker (the causes: the
    /// `bad_requests_total` HELP in `render`).
    pub bad_requests: AtomicU64,
    /// Histories cut to their last `max_history` ids.
    pub truncated: AtomicU64,
    /// The worker's call alone, per `Endpoint` and `Proto`.
    pub compute: [[Histogram; 2]; 2],
    /// The handler, per `Endpoint` and `Proto`: JSON body parsed → response serialized;
    /// gRPC message decoded → answer copied (tonic's decode and encode are
    /// outside). Queue wait and HTTP reads are not in it.
    pub total: [[Histogram; 2]; 2],
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

/// `process_start_time_seconds`: set by `mark_start` (main's first line),
/// else at the first scrape.
// ponytail: main's clock, not the OS's process start (the same to ms).
static START: LazyLock<SystemTime> = LazyLock::new(SystemTime::now);

/// Records the process's start for `/metrics` (call it first in `main`).
pub fn mark_start() {
    LazyLock::force(&START);
}

/// Fixed facts about the server, as gauges, and the requests per view.
pub struct Info<'a> {
    pub version: &'a str,
    /// `model.name`: any string (escaped in the label).
    pub model: &'a str,
    pub threads: usize,
    pub block: usize,
    pub items: usize,
    pub active_items: usize,
    /// (name, requests over every thread, failed reads); names need no
    /// escaping (`view_file::valid_name`).
    pub views: Vec<(&'a str, u64, u64)>,
    /// View scans that added, rebuilt or removed a view and the last one (Unix
    /// seconds, 0 = none).
    pub view_reloads: u64,
    pub last_view_reload: f64,
}

/// Prometheus text exposition of every thread's metrics, summed (requests
/// also per thread: the round-robin's balance).
pub fn render(threads: &[Thread], info: &Info) -> String {
    let mut o = String::new();
    head(
        &mut o,
        "info",
        "gauge",
        "The server's version and the served model's name (always 1).",
    );
    let (version, model) = (escape(info.version), escape(info.model));
    let _ = writeln!(
        o,
        "similaripy_server_info{{version=\"{version}\",model=\"{model}\"}} 1"
    );
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
    let start = START
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let process = [
        (
            "cpu_seconds_total",
            "counter",
            "User and system CPU time of the process.",
            process_cpu_seconds(),
        ),
        (
            "start_time_seconds",
            "gauge",
            "Start time of the process since the Unix epoch, in seconds.",
            Some(start),
        ),
        (
            "resident_memory_bytes",
            "gauge",
            "Resident memory of the process (mapped model pages read so far included).",
            resident_bytes().map(|b| b as f64),
        ),
    ];
    for (name, kind, help, v) in process {
        if let Some(v) = v {
            let _ = writeln!(
                o,
                "# HELP process_{name} {help}\n# TYPE process_{name} {kind}\nprocess_{name} {v}"
            );
        }
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
            "Requests rejected at the edge (JSON body: parse, read, 408, 413; k, history, view(s); not gRPC decode errors).",
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
    head(
        &mut o,
        "view_requests_total",
        "counter",
        "Requests per named view: the view used and each name in `views`.",
    );
    for (name, v, _) in &info.views {
        let _ = writeln!(
            o,
            "similaripy_server_view_requests_total{{view=\"{name}\"}} {v}"
        );
    }
    let help =
        "View scans that added, rebuilt or removed a view (views.dir, views.reload_interval).";
    head(&mut o, "view_reloads_total", "counter", help);
    let _ = writeln!(
        o,
        "similaripy_server_view_reloads_total {}",
        info.view_reloads
    );
    let help = "Changes per view whose files failed to read (the view kept its previous ids, or is not served if it had none).";
    head(&mut o, "view_reload_errors_total", "counter", help);
    for (name, _, e) in &info.views {
        let _ = writeln!(
            o,
            "similaripy_server_view_reload_errors_total{{view=\"{name}\"}} {e}"
        );
    }
    let help = "Unix time of the last view reload that rebuilt a view, 0 = none since start.";
    head(&mut o, "view_last_reload_timestamp_seconds", "gauge", help);
    let _ = writeln!(
        o,
        "similaripy_server_view_last_reload_timestamp_seconds {}",
        info.last_view_reload
    );
    let help = "Worker time (recommend or score), per endpoint and request format.";
    histogram(&mut o, "compute_seconds", help, threads, |t| &t.compute);
    let help = "Handler time, per endpoint and request format: JSON parsed, computed, \
        serialized; gRPC computed, answer copied (decode and encode outside).";
    histogram(&mut o, "request_seconds", help, threads, |t| &t.total);
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

/// The process's resident set size: Linux /proc, macOS proc_pidinfo.
#[cfg(target_os = "linux")]
fn resident_bytes() -> Option<u64> {
    // statm: size resident shared ... (in pages)
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: sysconf reads a constant; no pointers.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    Some(pages * u64::try_from(page).ok()?)
}

#[cfg(target_os = "macos")]
fn resident_bytes() -> Option<u64> {
    // SAFETY: proc_taskinfo is plain integers: all-zero is a valid value.
    let mut t: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // SAFETY: writes at most `size` bytes into `t`, about this process.
    let n = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            (&raw mut t).cast(),
            size,
        )
    };
    (n == size).then_some(t.pti_resident_size)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn resident_bytes() -> Option<u64> {
    None
}

/// A Prometheus label value: backslash, quote and newline escaped.
fn escape(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
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
    h: fn(&Thread) -> &[[Histogram; 2]; 2],
) {
    head(o, name, "histogram", help);
    let name = format!("similaripy_server_{name}");
    for (e, endpoint) in ENDPOINTS.iter().enumerate() {
        for (p, proto) in PROTOS.iter().enumerate() {
            let labels = format!("endpoint=\"{endpoint}\",proto=\"{proto}\"");
            let mut cum = 0;
            for b in 0..=BUCKETS_US.len() {
                cum += threads
                    .iter()
                    .map(|t| h(t)[e][p].counts[b].load(Relaxed))
                    .sum::<u64>();
                let le = BUCKETS_US
                    .get(b)
                    .map_or("+Inf".to_string(), |&us| (us as f64 / 1e6).to_string());
                let _ = writeln!(o, "{name}_bucket{{{labels},le=\"{le}\"}} {cum}");
            }
            let ns: u64 = threads
                .iter()
                .map(|t| h(t)[e][p].sum_ns.load(Relaxed))
                .sum();
            let _ = writeln!(o, "{name}_sum{{{labels}}} {}", ns as f64 / 1e9);
            let _ = writeln!(o, "{name}_count{{{labels}}} {cum}");
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn label_values_escaped() {
        assert_eq!(super::escape("a\\b\"c\nd"), "a\\\\b\\\"c\\nd");
    }
}
