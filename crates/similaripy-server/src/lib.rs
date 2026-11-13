//! The internal server (server plan S2; a design validator, not released):
//! `POST /recommend`, `GET /metrics`, `GET /health` on N threads, each a
//! single-threaded tokio runtime with its own `Worker`. An acceptor hands
//! connections round-robin to the threads (with keep-alive a connection
//! stays on its thread for life); a request runs to completion on the thread
//! that read it (E3), its `Worker` used synchronously from a `thread_local`.

use std::cell::{Cell, RefCell};
use std::io;
use std::net::TcpListener;
use std::sync::{Arc, mpsc};
use std::time::Instant;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use serde::{Deserialize, Serialize};
use similaripy_core::Error;
use similaripy_core::serve::{BlockSize, Model, Request, Stats, Worker};

pub mod client;
mod metrics;

/// Request bodies over this are rejected (413).
pub const BODY_LIMIT: usize = 1 << 20;

pub struct Config {
    /// Server threads (one `Worker` each); `block::perf_cores()` by default.
    pub threads: usize,
    pub block_size: BlockSize,
    /// A longer history keeps its last `max_history` ids (client order =
    /// oldest → newest, Q4); the cut ids stay excluded under `exclude_seen`.
    pub max_history: usize,
    /// A larger k is rejected (400).
    pub k_max: usize,
}

/// What every thread shares: the model, the limits, the metrics.
pub struct App {
    model: Model,
    cfg: Config,
    block: usize,
    metrics: Box<[metrics::Thread]>,
}

impl App {
    pub fn new(model: Model, cfg: Config) -> Result<Self, Error> {
        let block = cfg.block_size.resolve(model.n_active(), cfg.threads)?;
        let metrics = (0..cfg.threads).map(|_| Default::default()).collect();
        Ok(Self {
            model,
            cfg,
            block,
            metrics,
        })
    }

    /// The column block size every worker resolved (`Worker::new` with the
    /// same model, setting and thread count), 0 = one block.
    pub fn block(&self) -> usize {
        self.block
    }

    fn recommend(&self, m: &metrics::Thread, body: &[u8]) -> Response {
        let q: Query = match serde_json::from_slice(body) {
            Ok(q) => q,
            Err(e) => return bad_request(m, e),
        };
        if q.k > self.cfg.k_max {
            return bad_request(m, format!("k = {} > k_max = {}", q.k, self.cfg.k_max));
        }
        let n = q.ids.len();
        let cut = n.saturating_sub(self.cfg.max_history);
        if cut > 0 {
            m.truncated();
        }
        // Python's rounding: numpy.float32(a Python float), not serde's
        // direct f32 parse (it rounds a decimal just above an f32 halfway
        // point up, numpy's f64 → f32 to even; test json_f32_parsed_as_python).
        let weights: Option<Vec<f32>> = q.weights.map(|w| w.iter().map(|&x| x as f32).collect());
        // Weights of another length aren't cut: the worker rejects them.
        let weights = weights
            .as_deref()
            .map(|w| if w.len() == n { &w[cut..] } else { w });
        // Cut ids are still seen: excluded with the request's own (an
        // allocation only for a cut history).
        let exclude = match (cut, q.exclude_seen, q.exclude) {
            (0, _, e) | (_, false, e) => e,
            (_, true, e) => Some([&q.ids[..cut], e.as_deref().unwrap_or_default()].concat()),
        };
        let req = Request {
            ids: &q.ids[cut..],
            weights,
            k: q.k,
            exclude_seen: q.exclude_seen,
            include: q.include.as_deref(),
            exclude: exclude.as_deref(),
            sorted: q.sorted,
        };
        LOCAL.with_borrow_mut(|l| {
            let l = l.as_mut().expect("a request on a server thread");
            l.ids.clear();
            l.scores.clear();
            let t = Instant::now();
            let r = l
                .worker
                .recommend(&self.model, &req, &mut l.ids, &mut l.scores);
            m.compute.record(t.elapsed());
            m.set_stats(since(l.worker.stats(), l.base));
            match r {
                Ok(()) => json(
                    StatusCode::OK,
                    &Reply {
                        ids: &l.ids,
                        scores: &l.scores,
                    },
                ),
                Err(e @ Error::InvalidInput(_)) => error(StatusCode::BAD_REQUEST, e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
            }
        })
    }

    /// One request per path a worker takes, sized to the limits (k_max,
    /// max_history ids, include bits): buffers made before the first accept,
    /// no first-request spike (W2).
    fn warm(&self, w: &mut Worker) -> Result<(), Error> {
        let n = self.cfg.max_history.min(self.model.n_items());
        let ids: Vec<i64> = match self.model.item_ids() {
            Some(e) => e.iter().take(n).copied().collect(),
            None => (0..n as i64).collect(),
        };
        let (mut out_ids, mut out_scores) = (Vec::new(), Vec::new());
        for include in [None, Some(&ids[..])] {
            let req = Request {
                ids: &ids,
                weights: None,
                k: self.cfg.k_max,
                exclude_seen: true,
                include,
                exclude: None,
                sorted: true,
            };
            w.recommend(&self.model, &req, &mut out_ids, &mut out_scores)?;
        }
        Ok(())
    }
}

/// `POST /recommend`'s body; defaults = Python's `ItemKNN.recommend`.
/// Unknown fields are an error (a typo like "weight" would otherwise serve
/// unweighted).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    ids: Vec<i64>,
    #[serde(default)]
    /// f64 as sent, rounded to f32 after (numpy's two steps).
    weights: Option<Vec<f64>>,
    #[serde(default = "ten")]
    k: usize,
    #[serde(default = "yes")]
    exclude_seen: bool,
    #[serde(default)]
    include: Option<Vec<i64>>,
    #[serde(default)]
    exclude: Option<Vec<i64>>,
    #[serde(default = "yes")]
    sorted: bool,
}

fn ten() -> usize {
    10
}

fn yes() -> bool {
    true
}

#[derive(Serialize)]
struct Reply<'a> {
    ids: &'a [i64],
    scores: &'a [f32],
}

fn json(status: StatusCode, body: &impl Serialize) -> Response {
    let bytes = serde_json::to_vec(body).expect("plain numbers and strings serialize");
    (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response()
}

fn error(status: StatusCode, e: impl std::fmt::Display) -> Response {
    json(status, &serde_json::json!({ "error": e.to_string() }))
}

fn bad_request(m: &metrics::Thread, e: impl std::fmt::Display) -> Response {
    m.bad_request();
    error(StatusCode::BAD_REQUEST, e)
}

fn since(s: Stats, base: Stats) -> Stats {
    Stats {
        requests: s.requests - base.requests,
        errors: s.errors - base.errors,
        ids: s.ids - base.ids,
        known_ids: s.known_ids - base.known_ids,
    }
}

/// A server thread's own state.
struct Local {
    worker: Worker,
    /// The worker's stats after warm-up (not served requests).
    base: Stats,
    /// Reused outputs (the worker appends).
    ids: Vec<i64>,
    scores: Vec<f32>,
}

thread_local! {
    static THREAD: Cell<usize> = const { Cell::new(0) };
    static LOCAL: RefCell<Option<Local>> = const { RefCell::new(None) };
}

/// The routes, over `app`.
pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/recommend", post(recommend))
        .route("/metrics", get(metrics))
        .route("/health", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(app)
}

async fn recommend(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let t = Instant::now();
    let m = &app.metrics[THREAD.get()];
    let r = app.recommend(m, &body);
    m.total.record(t.elapsed());
    r
}

async fn metrics(State(app): State<Arc<App>>) -> String {
    let info = metrics::Info {
        threads: app.cfg.threads,
        block: app.block,
        items: app.model.n_items(),
        active_items: app.model.n_active(),
    };
    metrics::render(&app.metrics, &info)
}

/// Serves `router` (`router(app)`, tests add routes) on `listener` with
/// `app`'s threads, each warmed before the first accept. Returns only on a
/// thread's start-up error.
pub fn serve(listener: TcpListener, app: Arc<App>, router: Router) -> io::Result<()> {
    let n = app.cfg.threads;
    let (ready_tx, ready) = mpsc::channel();
    let mut threads = Vec::with_capacity(n);
    for i in 0..n {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<std::net::TcpStream>();
        let (app, router, ready_tx) = (app.clone(), router.clone(), ready_tx.clone());
        std::thread::Builder::new()
            .name(format!("server-{i}"))
            .spawn(move || {
                set_qos(true);
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();
                let start = || -> Result<_, String> {
                    let rt = rt.map_err(|e| e.to_string())?;
                    let mut worker = Worker::new(&app.model, app.cfg.block_size, n)
                        .map_err(|e| e.to_string())?;
                    app.warm(&mut worker).map_err(|e| format!("warm-up: {e}"))?;
                    THREAD.set(i);
                    LOCAL.set(Some(Local {
                        base: worker.stats(),
                        worker,
                        ids: Vec::with_capacity(app.cfg.k_max),
                        scores: Vec::with_capacity(app.cfg.k_max),
                    }));
                    Ok(rt)
                };
                let rt = match start() {
                    Ok(rt) => rt,
                    Err(e) => return ready_tx.send(Err(e)).unwrap_or(()),
                };
                let _ = ready_tx.send(Ok(()));
                rt.block_on(async move {
                    while let Some(s) = rx.recv().await {
                        let Ok(s) = s
                            .set_nonblocking(true)
                            .and_then(|()| tokio::net::TcpStream::from_std(s))
                        else {
                            continue;
                        };
                        let svc = TowerToHyperService::new(router.clone());
                        tokio::spawn(async move {
                            let conn = hyper::server::conn::http1::Builder::new()
                                .serve_connection(TokioIo::new(s), svc);
                            let _ = conn.await;
                        });
                    }
                });
            })?;
        threads.push(tx);
    }
    drop(ready_tx);
    for _ in 0..n {
        ready
            .recv()
            .map_err(|_| io::Error::other("a server thread died at start"))?
            .map_err(io::Error::other)?;
    }
    // ponytail: round-robin per connection; SO_REUSEPORT per thread (Linux)
    // is the S4 probe.
    let mut i = 0;
    for s in listener.incoming() {
        let Ok(s) = s else {
            // E.g. out of file descriptors: fails at once until one closes;
            // retrying at once would spin a core.
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        };
        let _ = s.set_nodelay(true);
        if threads[i % n].send(s).is_err() {
            return Err(io::Error::other(format!("server thread {} died", i % n)));
        }
        i += 1;
    }
    Ok(())
}

/// macOS: the calling thread's QoS class, a scheduling hint (not pinning):
/// `interactive` = the P-cores (the server), else utility = mostly the
/// E-cores (loadgen on the same laptop). Elsewhere a no-op (Linux: pin with
/// taskset or cpusets).
pub fn set_qos(interactive: bool) {
    #[cfg(target_os = "macos")]
    {
        use libc::qos_class_t::{QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_UTILITY};
        let class = if interactive {
            QOS_CLASS_USER_INTERACTIVE
        } else {
            QOS_CLASS_UTILITY
        };
        // SAFETY: changes only the calling thread's own QoS; no pointers.
        unsafe { libc::pthread_set_qos_class_self_np(class, 0) };
    }
    #[cfg(not(target_os = "macos"))]
    let _ = interactive;
}
