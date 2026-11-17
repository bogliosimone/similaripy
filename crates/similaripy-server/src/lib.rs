//! The internal server (server plan S2; a design validator, not released):
//! `POST /v1/recommend` (JSON), gRPC `similaripy.v1.Recommender/Recommend`
//! (proto/, S4 P1), `GET /metrics`, `GET /health`, `GET /config` (the
//! resolved [`Config`]), all on one port and N
//! threads, each a single-threaded tokio runtime with its own `Worker`. An
//! acceptor hands connections round-robin to the threads (with keep-alive a
//! connection stays on its thread for life; HTTP/1.1 or h2c by the client's
//! preface); a request runs to completion on the thread that read it (E3),
//! its `Worker` used synchronously from a `thread_local`. JSON and gRPC run
//! the same `App::run`: same limits, cut and answers.

use std::cell::{Cell, RefCell};
use std::hash::{BuildHasher, RandomState};
use std::io;
use std::net::TcpListener;
use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant, SystemTime};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, FromRequest, State};
use axum::http::{HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use hyper_util::service::TowerToHyperService;
use serde::{Deserialize, Serialize};
use similaripy_core::Error;
use similaripy_core::serve::{Model, Request, Stats, Worker};

pub use config::Config;
use config::HistoryPolicy;

use metrics::Proto;

pub mod client;
pub mod config;
mod metrics;

/// The gRPC messages, client and server (build.rs, from proto/).
pub mod pb {
    tonic::include_proto!("similaripy.v1");
}

/// Request bodies over this are rejected (JSON 413, gRPC OUT_OF_RANGE).
pub const BODY_LIMIT: usize = 1 << 20;

/// Python's `ItemKNN.recommend` defaults, for both formats (exclude_seen
/// defaults to true; sorted to `defaults.sorted`).
const DEFAULT_K: usize = 10;

/// The request id's header (HTTP) and metadata key (gRPC).
const X_REQUEST_ID: &str = "x-request-id";

/// A request with no answer: 400 / INVALID_ARGUMENT or 500 / INTERNAL.
enum Fail {
    Invalid(String),
    Internal(String),
}

/// What every thread shares: the model, the config, the metrics.
pub struct App {
    model: Model,
    cfg: Config,
    block: usize,
    metrics: Box<[metrics::Thread]>,
}

impl App {
    pub fn new(model: Model, mut cfg: Config) -> Result<Self, Error> {
        cfg.model.fill_name();
        let block = cfg
            .server
            .block_size
            .resolve(model.n_active(), cfg.server.threads)?;
        let metrics = (0..cfg.server.threads)
            .map(|_| Default::default())
            .collect();
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

    /// `q` (the whole history) through the edge checks, the history cap and
    /// this thread's worker; `encode` turns the answer into the response
    /// while the worker's buffers are borrowed.
    fn run<T>(
        &self,
        m: &metrics::Thread,
        p: Proto,
        q: &Request,
        encode: impl FnOnce(&Answer) -> T,
    ) -> Result<T, Fail> {
        let limits = &self.cfg.limits;
        if q.k > limits.k_max {
            m.bad_request();
            return Err(Fail::Invalid(format!(
                "k = {} > k_max = {}",
                q.k, limits.k_max
            )));
        }
        let n = q.ids.len();
        let cut = n.saturating_sub(limits.max_history);
        if cut > 0 {
            if limits.history_policy == HistoryPolicy::Reject {
                m.bad_request();
                return Err(Fail::Invalid(format!(
                    "{n} history ids > max_history = {}",
                    limits.max_history
                )));
            }
            m.truncated();
        }
        // Weights of another length aren't cut: the worker rejects them.
        let weights = q.weights.map(|w| if w.len() == n { &w[cut..] } else { w });
        // Cut ids are still seen: excluded with the request's own (an
        // allocation only for a cut history).
        let exclude = match (cut, q.exclude_seen) {
            (0, _) | (_, false) => None,
            (_, true) => Some([&q.ids[..cut], q.exclude.unwrap_or_default()].concat()),
        };
        let req = Request {
            ids: &q.ids[cut..],
            weights,
            exclude: exclude.as_deref().or(q.exclude),
            ..*q
        };
        LOCAL.with_borrow_mut(|l| {
            let l = l.as_mut().expect("a request on a server thread");
            l.ids.clear();
            l.scores.clear();
            let before = l.worker.stats();
            let t = Instant::now();
            let r = l
                .worker
                .recommend(&self.model, &req, &mut l.ids, &mut l.scores);
            let compute = t.elapsed();
            m.compute[p as usize].record(compute);
            let stats = l.worker.stats();
            m.set_stats(since(stats, l.base));
            // This request's counts: the worker's counters, before vs after.
            let d = since(stats, before);
            let history = History {
                sent: n as u64,
                truncated: cut as u64,
                unknown: d.ids - d.known_ids,
            };
            match r {
                Ok(()) => Ok(encode(&Answer {
                    ids: &l.ids,
                    scores: &l.scores,
                    history,
                    compute,
                })),
                Err(e @ Error::InvalidInput(_)) => Err(Fail::Invalid(e.to_string())),
                Err(e) => Err(Fail::Internal(e.to_string())),
            }
        })
    }

    fn recommend_json(&self, m: &metrics::Thread, body: &[u8], request_id: &str) -> Response {
        let q: Query = match serde_json::from_slice(body) {
            Ok(q) => q,
            Err(e) => return bad_request(m, e),
        };
        // Python's rounding: numpy.float32(a Python float), not serde's
        // direct f32 parse (it rounds a decimal just above an f32 halfway
        // point up, numpy's f64 → f32 to even; test json_f32_parsed_as_python).
        let weights: Option<Vec<f32>> = q.weights.map(|w| w.iter().map(|&x| x as f32).collect());
        let req = Request {
            ids: &q.ids,
            weights: weights.as_deref(),
            k: q.k,
            exclude_seen: q.exclude_seen,
            include: q.include.as_deref(),
            exclude: q.exclude.as_deref(),
            sorted: q.sorted.unwrap_or(self.cfg.defaults.sorted),
        };
        let r = self.run(m, Proto::Json, &req, |a| {
            let reply = Reply {
                items: Items(a.ids, a.scores),
                model: ModelInfo {
                    name: &self.cfg.model.name,
                },
                request_id,
                k: req.k,
                sorted: req.sorted,
                view: None,
                history: a.history,
            };
            let mut r = json(StatusCode::OK, &reply);
            let timing = format!("compute;dur={:.3}", a.compute.as_secs_f64() * 1e3);
            r.headers_mut().insert(
                HeaderName::from_static("server-timing"),
                HeaderValue::try_from(timing).expect("ASCII"),
            );
            r
        });
        match r {
            Ok(r) => r,
            Err(Fail::Invalid(e)) => error(StatusCode::BAD_REQUEST, e),
            Err(Fail::Internal(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }

    fn recommend_grpc(
        &self,
        m: &metrics::Thread,
        q: &pb::RecommendRequest,
        request_id: &str,
    ) -> Result<pb::RecommendResponse, tonic::Status> {
        let req = Request {
            ids: &q.ids,
            // proto3: an empty list is not sent; unweighted (T17)
            weights: (!q.weights.is_empty()).then_some(&q.weights[..]),
            k: q.k.map_or(DEFAULT_K, |k| k as usize),
            exclude_seen: q.exclude_seen.unwrap_or(true),
            include: q.include.as_ref().map(|l| &l.ids[..]),
            exclude: (!q.exclude.is_empty()).then_some(&q.exclude[..]),
            sorted: q.sorted.unwrap_or(self.cfg.defaults.sorted),
        };
        // T18: one copy of the answer, prost's default codec encodes it
        let r = self.run(m, Proto::Grpc, &req, |a| pb::RecommendResponse {
            ids: a.ids.to_vec(),
            scores: a.scores.to_vec(),
            model: Some(pb::ModelInfo {
                name: self.cfg.model.name.clone(),
            }),
            request_id: request_id.to_string(),
            k: req.k as u32, // <= k_max: checked in run
            sorted: req.sorted,
            view: None,
            history: Some(pb::History {
                sent: a.history.sent,
                truncated: a.history.truncated,
                unknown: a.history.unknown,
            }),
        });
        r.map_err(|e| match e {
            Fail::Invalid(e) => tonic::Status::invalid_argument(e),
            Fail::Internal(e) => tonic::Status::internal(e),
        })
    }

    /// One request per path a worker takes, sized to the limits (k_max,
    /// max_history ids, include bits): buffers made before the first accept,
    /// no first-request spike (W2).
    fn warm(&self, w: &mut Worker) -> Result<(), Error> {
        let n = self.cfg.limits.max_history.min(self.model.n_items());
        let ids: Vec<i64> = match self.model.item_ids() {
            Some(e) => e.iter().take(n).copied().collect(),
            None => (0..n as i64).collect(),
        };
        let (mut out_ids, mut out_scores) = (Vec::new(), Vec::new());
        for include in [None, Some(&ids[..])] {
            let req = Request {
                ids: &ids,
                weights: None,
                k: self.cfg.limits.k_max,
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

/// `POST /v1/recommend`'s body; defaults = Python's `ItemKNN.recommend`, but
/// `sorted` absent = `defaults.sorted`.
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
    #[serde(default)]
    sorted: Option<bool>,
}

/// A `POST /v1/recommend` body as the gRPC request with the same answer
/// (loadgen, tests): weights rounded to f32 as the server does, `include`
/// kept apart from an empty list. A JSON `"weights": []` (a 400 in JSON for
/// a non-empty history) becomes unweighted.
pub fn proto_from_json(body: &[u8]) -> Result<pb::RecommendRequest, serde_json::Error> {
    let q: Query = serde_json::from_slice(body)?;
    Ok(pb::RecommendRequest {
        weights: q
            .weights
            .unwrap_or_default()
            .iter()
            .map(|&x| x as f32)
            .collect(),
        k: Some(q.k.try_into().unwrap_or(u32::MAX)),
        exclude_seen: Some(q.exclude_seen),
        include: q.include.map(|ids| pb::IdList { ids }),
        exclude: q.exclude.unwrap_or_default(),
        sorted: q.sorted,
        ids: q.ids,
    })
}

fn ten() -> usize {
    DEFAULT_K
}

fn yes() -> bool {
    true
}

/// A served answer, borrowed from the worker's buffers.
struct Answer<'a> {
    ids: &'a [i64],
    scores: &'a [f32],
    history: History,
    compute: Duration,
}

/// The request's history ids (proto `History`): as sent, cut past
/// `max_history`, not in the model among the rest (per occurrence).
#[derive(Serialize, Clone, Copy)]
struct History {
    sent: u64,
    truncated: u64,
    unknown: u64,
}

/// `POST /v1/recommend`'s answer (U4): the items, then the request's context.
#[derive(Serialize)]
struct Reply<'a> {
    items: Items<'a>,
    model: ModelInfo<'a>,
    request_id: &'a str,
    k: usize,
    sorted: bool,
    view: Option<&'a str>,
    history: History,
}

#[derive(Serialize)]
struct ModelInfo<'a> {
    name: &'a str,
}

/// `[{"id": .., "score": ..}, ..]` straight from the two arrays.
struct Items<'a>(&'a [i64], &'a [f32]);

impl Serialize for Items<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Item {
            id: i64,
            score: f32,
        }
        let items = self.0.iter().zip(self.1);
        s.collect_seq(items.map(|(&id, &score)| Item { id, score }))
    }
}

/// The client's id if it is 1-128 printable ASCII bytes (no spaces), else a
/// new one: `<process>-<thread>-<counter>`, the process part a random 64-bit
/// value drawn at the first request (replicas started in the same second
/// differ: the served log joins on it).
fn request_id(sent: Option<&[u8]>) -> String {
    if let Some(v) = sent
        && (1..=128).contains(&v.len())
        && v.iter().all(|b| (0x21..=0x7e).contains(b))
    {
        return String::from_utf8(v.to_vec()).expect("ASCII");
    }
    static PROCESS: OnceLock<u64> = OnceLock::new();
    let process = *PROCESS.get_or_init(|| {
        // std's RandomState keys come from the OS's randomness
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        RandomState::new().hash_one((std::process::id(), nanos))
    });
    let n = COUNTER.get();
    COUNTER.set(n + 1);
    format!("{process:016x}-{}-{n}", THREAD.get())
}

fn with_request_id(mut r: Response, id: &str) -> Response {
    let v = HeaderValue::from_str(id).expect("checked or generated: printable ASCII");
    r.headers_mut().insert(X_REQUEST_ID, v);
    r
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
    /// Request ids this thread generated.
    static COUNTER: Cell<u64> = const { Cell::new(0) };
    static LOCAL: RefCell<Option<Local>> = const { RefCell::new(None) };
}

/// The routes, over `app`: JSON and gRPC (tonic's decode limit is its own:
/// `DefaultBodyLimit` doesn't reach it).
pub fn router(app: Arc<App>) -> Router {
    let grpc = pb::recommender_server::RecommenderServer::new(Grpc(app.clone()))
        .max_decoding_message_size(BODY_LIMIT);
    Router::new()
        .route("/v1/recommend", post(recommend))
        .route("/metrics", get(metrics))
        .route("/health", get(|| async { "ok" }))
        .route("/config", get(config))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(app)
        .merge(tonic::service::Routes::new(grpc).into_axum_router())
}

async fn recommend(State(app): State<Arc<App>>, req: axum::extract::Request) -> Response {
    let t = Instant::now();
    let m = &app.metrics[THREAD.get()];
    let id = request_id(req.headers().get(X_REQUEST_ID).map(|v| v.as_bytes()));
    // Bytes' extractor: the body limit as before (413), here without our id
    let body = match Bytes::from_request(req, &()).await {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };
    let r = with_request_id(app.recommend_json(m, &body, &id), &id);
    m.total[Proto::Json as usize].record(t.elapsed());
    r
}

struct Grpc(Arc<App>);

#[tonic::async_trait]
impl pb::recommender_server::Recommender for Grpc {
    /// Handler time excludes the request's decode and the response's encode
    /// (tonic's, around this): compare formats on CPU per request.
    async fn recommend(
        &self,
        r: tonic::Request<pb::RecommendRequest>,
    ) -> Result<tonic::Response<pb::RecommendResponse>, tonic::Status> {
        let t = Instant::now();
        let m = &self.0.metrics[THREAD.get()];
        let id = request_id(r.metadata().get(X_REQUEST_ID).map(|v| v.as_bytes()));
        let r = self.0.recommend_grpc(m, r.get_ref(), &id);
        m.total[Proto::Grpc as usize].record(t.elapsed());
        let v: tonic::metadata::AsciiMetadataValue =
            id.parse().expect("checked or generated: printable ASCII");
        match r {
            Ok(r) => {
                let mut r = tonic::Response::new(r);
                r.metadata_mut().insert(X_REQUEST_ID, v);
                Ok(r)
            }
            Err(mut e) => {
                e.metadata_mut().insert(X_REQUEST_ID, v);
                Err(e)
            }
        }
    }
}

async fn config(State(app): State<Arc<App>>) -> Response {
    json(StatusCode::OK, &app.cfg)
}

async fn metrics(State(app): State<Arc<App>>) -> String {
    let info = metrics::Info {
        threads: app.cfg.server.threads,
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
    let n = app.cfg.server.threads;
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
                    let mut worker = Worker::new(&app.model, app.cfg.server.block_size, n)
                        .map_err(|e| e.to_string())?;
                    app.warm(&mut worker).map_err(|e| format!("warm-up: {e}"))?;
                    THREAD.set(i);
                    LOCAL.set(Some(Local {
                        base: worker.stats(),
                        worker,
                        ids: Vec::with_capacity(app.cfg.limits.k_max),
                        scores: Vec::with_capacity(app.cfg.limits.k_max),
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
                        // HTTP/1.1 or h2c by the preface; h2 spawns its
                        // streams on this thread's runtime (current-thread)
                        tokio::spawn(async move {
                            let _ = auto::Builder::new(TokioExecutor::new())
                                .serve_connection(TokioIo::new(s), svc)
                                .await;
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
