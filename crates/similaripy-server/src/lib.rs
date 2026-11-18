//! The server (experimental; released as its own artifact, `server-v*` tags):
//! `POST /v1/recommend` and `POST /v1/score` (JSON), gRPC
//! `similaripy.v1.Recommender/Recommend` and `/Score` (proto/, S4 P1; the
//! JSON endpoints' OpenAPI spec: openapi.json), `GET /metrics`, `GET /health` (liveness: always ok),
//! `GET /ready` (readiness: 503 once draining), `GET /config` (the
//! resolved [`Config`]), all on one port and N
//! threads, each a single-threaded tokio runtime with its own `Worker`. An
//! acceptor hands connections round-robin to the threads (with keep-alive a
//! connection stays on its thread for life; HTTP/1.1 or h2c by the client's
//! preface); a request runs to completion on the thread that read it (E3),
//! its `Worker` used synchronously from a `thread_local`. JSON and gRPC run
//! the same `App::run`: same limits, cut and answers (score: V21 B). Named views (S5c):
//! `GET /v1/views`, a request's `view`; the model and its views are one
//! serving state behind one swappable pointer (V15). Hardening (S5d):
//! header-read timeouts, graceful shutdown (`serve`'s `shutdown`).

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::hash::{BuildHasher, RandomState};
use std::io;
use std::net::TcpListener;
use std::sync::atomic::{
    AtomicBool, AtomicU64, Ordering::Acquire, Ordering::Relaxed, Ordering::Release,
};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, mpsc};
use std::time::{Duration, Instant, SystemTime};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use http_body_util::BodyExt;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto;
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use serde::{Deserialize, Serialize};
use similaripy_core::Error;
use similaripy_core::serve::{Model, Request, ScoreRequest, Stats, Worker};

pub use config::Config;
use config::{HistoryPolicy, secs};
pub use metrics::mark_start;

use metrics::{Counter, Endpoint, Proto};
use views::{View, ViewInfo};

pub mod client;
pub mod config;
mod metrics;
mod views;

/// The gRPC messages, client and server (build.rs, from proto/).
pub mod pb {
    tonic::include_proto!("similaripy.v1");
}

/// `limits.max_body_bytes`' default: request bodies over it are rejected
/// (JSON 413, gRPC OUT_OF_RANGE).
pub const BODY_LIMIT: usize = 1 << 20;

/// The request id's header (HTTP) and metadata key (gRPC).
const X_REQUEST_ID: &str = "x-request-id";

/// gRPC's metadata key for an error's `Code` (JSON: `error.code`).
const X_ERROR_CODE: &str = "x-error-code";

/// Why a request has no answer: JSON `error.code`, gRPC `x-error-code`
/// (V21). The strings are a contract: new ones may come, none change.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Code {
    /// Not valid JSON, an unknown or missing field, a wrong type.
    BadJson,
    /// The body couldn't be read (not its size, not its time).
    BadBody,
    KOverMax,
    /// A score request over `limits.max_items`.
    ItemsOverMax,
    /// Over `max_history` under `history_policy = reject`.
    HistoryOverMax,
    UnknownView,
    /// Rejected by the worker: its only request checks are the weights'
    /// (length, finite; test `edge_rejections`; recommend and score); a cut
    /// history's weights of the wrong length are rejected at the edge.
    BadWeights,
    BodyTimeout,
    BodyTooLarge,
    Internal,
}

impl Code {
    fn as_str(self) -> &'static str {
        match self {
            Code::BadJson => "bad_json",
            Code::BadBody => "bad_body",
            Code::KOverMax => "k_over_max",
            Code::ItemsOverMax => "items_over_max",
            Code::HistoryOverMax => "history_over_max",
            Code::UnknownView => "unknown_view",
            Code::BadWeights => "bad_weights",
            Code::BodyTimeout => "body_timeout",
            Code::BodyTooLarge => "body_too_large",
            Code::Internal => "internal",
        }
    }

    fn http(self) -> StatusCode {
        match self {
            Code::BodyTimeout => StatusCode::REQUEST_TIMEOUT,
            Code::BodyTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Code::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::BAD_REQUEST,
        }
    }

    fn grpc(self) -> tonic::Code {
        match self {
            Code::BodyTooLarge => tonic::Code::OutOfRange,
            Code::Internal => tonic::Code::Internal,
            _ => tonic::Code::InvalidArgument,
        }
    }
}

/// A request with no answer: its `Code` and a message for people.
struct Fail(Code, String);

impl Fail {
    /// `{"error": {"code", "message", "request_id"}}` with the code's status.
    fn json(self, request_id: &str) -> Response {
        self.log(request_id);
        let Fail(code, message) = self;
        let body = serde_json::json!({
            "error": {"code": code.as_str(), "message": message, "request_id": request_id}
        });
        json(code.http(), &body)
    }

    /// The code's status, the message, `x-error-code` in the metadata.
    fn grpc(self, request_id: &str) -> tonic::Status {
        self.log(request_id);
        let Fail(code, message) = self;
        let mut s = tonic::Status::new(code.grpc(), message);
        s.metadata_mut().insert(
            X_ERROR_CODE,
            tonic::metadata::AsciiMetadataValue::from_static(code.as_str()),
        );
        s
    }

    /// One stderr line per server error (F13); client errors only count.
    fn log(&self, request_id: &str) {
        if self.0 == Code::Internal {
            eprintln!("similaripy-server: error: request {request_id}: {}", self.1);
        }
    }
}

/// What is served: the model and its named views (each a `Model::view` of
/// it). Never changed: replaced whole by `App::install`.
struct Served {
    model: Model,
    views: BTreeMap<String, View>,
}

/// What every thread shares: the config, the metrics, the serving state.
pub struct App {
    cfg: Config,
    block: usize,
    /// The model's `n_active` at start: `block` was resolved from it.
    n_active: usize,
    metrics: Box<[metrics::Thread]>,
    /// The serving state (V15). Each thread keeps an `Arc` of it and loads
    /// `version` once per request (one atomic load, no lock); after a
    /// `swap` its next request takes the new one, in-flight requests keep
    /// theirs.
    // ponytail: an idle thread holds the old state until its next request;
    // 1.1's hot swap nudges idle threads if an old model must be freed.
    state: Mutex<Arc<Served>>,
    version: AtomicU64,
    /// Requests per view and thread: kept across swaps (the views' names
    /// come from the config, which a swap doesn't change).
    served: BTreeMap<String, Arc<[Counter]>>,
    /// Held through a whole swap or view reload, never by a serving thread:
    /// one writer at a time, so a reload rebuilds over the model it then
    /// installs, never an older one (V22); the state lock is only taken to
    /// read and to install (a scrape never waits on file reads). Holds the
    /// stamps of the files the served views were read from.
    writer: Mutex<views::Stamps>,
    /// View reloads (V21 C, written under `writer`) that rebuilt a view,
    /// the last one (Unix seconds as f64 bits, 0 = none), failed reads per
    /// view.
    reloads: AtomicU64,
    last_reload: AtomicU64,
    reload_errors: BTreeMap<String, AtomicU64>,
    /// Set by `serve` on shutdown: `/ready` answers 503.
    draining: AtomicBool,
}

impl App {
    /// The app serving `model` with `cfg`'s views (id files read here: errors
    /// name the view, file and line).
    pub fn new(model: Model, mut cfg: Config) -> Result<Self, String> {
        cfg.model.fill_name();
        cfg.check_timeouts()?;
        cfg.check_limits()?;
        let block = cfg
            .server
            .block_size
            .resolve(model.n_active(), cfg.server.threads)
            .map_err(|e| e.to_string())?;
        let metrics = (0..cfg.server.threads)
            .map(|_| Default::default())
            .collect();
        let served: BTreeMap<_, Arc<[Counter]>> = cfg
            .views
            .named
            .keys()
            .map(|k| {
                (
                    k.clone(),
                    (0..cfg.server.threads)
                        .map(|_| Counter::default())
                        .collect(),
                )
            })
            .collect();
        let stamps = views::stamps(&cfg);
        let views = views::build(&model, &cfg, &served)?;
        Ok(Self {
            reload_errors: served
                .keys()
                .map(|k| (k.clone(), AtomicU64::new(0)))
                .collect(),
            served,
            writer: Mutex::new(stamps),
            reloads: AtomicU64::new(0),
            last_reload: AtomicU64::new(0),
            n_active: model.n_active(),
            cfg,
            block,
            metrics,
            state: Mutex::new(Arc::new(Served { model, views })),
            version: AtomicU64::new(0),
            draining: AtomicBool::new(false),
        })
    }

    /// Serves `model` with the config's views, rebuilt from their files, from
    /// each thread's next request on (1.1's model hot swap calls this). An
    /// error leaves the served state as it was.
    // ponytail: the workers' block size was resolved from the first model's
    // n_active: another n_active is refused, not re-resolved (1.1's hot swap
    // re-resolves it per worker if models of other sizes must swap).
    pub fn swap(&self, model: Model) -> Result<(), String> {
        if model.n_active() != self.n_active {
            return Err(format!(
                "n_active {} != {} at start: restart to serve it",
                model.n_active(),
                self.n_active
            ));
        }
        let mut seen = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        let stamps = views::stamps(&self.cfg);
        let views = views::build(&model, &self.cfg, &self.served)?;
        *seen = stamps;
        self.install(Served { model, views });
        Ok(())
    }

    /// Rebuilds the views whose files changed since they were read (V21 C)
    /// over the served model, from each thread's next request on; one whose
    /// files fail keeps what it served (logged and counted once per change,
    /// its `error` in `GET /v1/views`). True if a file changed.
    pub fn reload_views(&self) -> bool {
        let mut seen = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        // Under `writer`: no swap can serve another model before ours lands.
        let s = self.state();
        let Some(r) = views::rebuild(&s.model, &self.cfg, &s.views, &mut seen) else {
            return false;
        };
        for name in &r.failed {
            self.reload_errors[name].fetch_add(1, Relaxed);
        }
        let items = s.model.n_items();
        let lines: Vec<String> = (r.rebuilt.iter().chain(&r.failed))
            .map(|name| views::describe_one(name, &r.views[name], items))
            .collect();
        self.install(Served {
            model: s.model.view(None, None),
            views: r.views,
        });
        if !r.rebuilt.is_empty() {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default();
            self.last_reload.store(now.as_secs_f64().to_bits(), Relaxed);
            self.reloads.fetch_add(1, Relaxed);
        }
        for line in lines {
            eprintln!("similaripy-server: reload: {line}");
        }
        true
    }

    /// `s` served from each thread's next request on.
    fn install(&self, s: Served) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        *state = Arc::new(s);
        // Under the lock: a thread that sees this version gets this state
        // (or a newer one).
        self.version.fetch_add(1, Release);
    }

    /// The current state (a lock: start-up, scrapes, a thread after a swap).
    fn state(&self) -> Arc<Served> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The start-up log's view lines (none without views).
    pub fn describe_views(&self) -> Vec<String> {
        let s = self.state();
        views::describe(&s.views, s.model.n_items())
    }

    /// The column block size every worker resolved (`Worker::new` with the
    /// same model, setting and thread count), 0 = one block.
    pub fn block(&self) -> usize {
        self.block
    }

    /// A history (whole, with its weights) through the edge checks, the
    /// history cap and this thread's worker (`call`, given the kept ids and
    /// weights and the reused outputs), over the model of `view`
    /// (None: `defaults.view`; "": the whole model); `encode` turns the
    /// answer into the response while the worker's buffers are borrowed.
    #[allow(clippy::too_many_arguments)]
    fn run<T>(
        &self,
        m: &metrics::Thread,
        (e, p): (Endpoint, Proto),
        history: &[i64],
        weights: Option<&[f32]>,
        view: Option<&str>,
        call: impl FnOnce(&mut Worker, &Model, Cut, &mut Vec<i64>, &mut Vec<f32>) -> Result<(), Error>,
        encode: impl FnOnce(&Answer) -> T,
    ) -> Result<T, Fail> {
        let limits = &self.cfg.limits;
        LOCAL.with_borrow_mut(|l| {
            let l = l.as_mut().expect("a request on a server thread");
            // V15: the version first, then the state (as new or newer)
            let v = self.version.load(Acquire);
            if v != l.version {
                l.state = self.state();
                l.version = v;
            }
            let Local {
                worker,
                base,
                ids,
                scores,
                state,
                ..
            } = l;
            let name = view.unwrap_or(&self.cfg.defaults.view);
            let (model, view, served) = if name.is_empty() {
                (&state.model, None, None)
            } else {
                match state.views.get_key_value(name) {
                    Some((name, v)) => (&v.model, Some(name.as_str()), Some(&v.served)),
                    None => {
                        m.bad_request();
                        return Err(Fail(
                            Code::UnknownView,
                            format!("unknown view {name:?} (GET /v1/views lists them)"),
                        ));
                    }
                }
            };
            let n = history.len();
            let cut = n.saturating_sub(limits.max_history);
            if cut > 0 {
                if limits.history_policy == HistoryPolicy::Reject {
                    m.bad_request();
                    return Err(Fail(
                        Code::HistoryOverMax,
                        format!("{n} history ids > max_history = {}", limits.max_history),
                    ));
                }
                m.truncated();
            }
            // Weights of another length aren't cut: the worker rejects them,
            // unless they match the kept ids (it can't see the cut): here.
            if let Some(w) = weights
                && cut > 0
                && w.len() != n
            {
                m.bad_request();
                let e = format!("len(weights) = {} != len(history) = {n}", w.len());
                return Err(Fail(Code::BadWeights, e));
            }
            let weights = weights.map(|w| &w[cut..]);
            let kept = Cut {
                ids: &history[cut..],
                weights,
            };
            ids.clear();
            scores.clear();
            let before = worker.stats();
            let t = Instant::now();
            let r = call(worker, model, kept, ids, scores);
            let compute = t.elapsed();
            m.compute[e as usize][p as usize].record(compute);
            let stats = worker.stats();
            m.set_stats(since(stats, *base));
            // This request's counts: the worker's counters, before vs after.
            let d = since(stats, before);
            let history = History {
                sent: n as u64,
                truncated: cut as u64,
                unknown: d.ids - d.known_ids,
            };
            match r {
                Ok(()) => {
                    if let Some(c) = served {
                        c[THREAD.get()].add();
                    }
                    Ok(encode(&Answer {
                        ids,
                        scores,
                        view,
                        history,
                        compute,
                    }))
                }
                Err(e @ Error::InvalidInput(_)) => Err(Fail(Code::BadWeights, e.to_string())),
                Err(e) => Err(Fail(Code::Internal, e.to_string())),
            }
        })
    }

    /// `q` (the whole history): `k` checked, then `run` with
    /// `Worker::recommend`; cut ids are still seen (excluded with the
    /// request's own: an allocation only for a cut history).
    fn recommend<T>(
        &self,
        m: &metrics::Thread,
        p: Proto,
        q: &Request,
        view: Option<&str>,
        encode: impl FnOnce(&Answer) -> T,
    ) -> Result<T, Fail> {
        let k_max = self.cfg.limits.k_max;
        if q.k > k_max {
            m.bad_request();
            return Err(Fail(
                Code::KOverMax,
                format!("k = {} > k_max = {k_max}", q.k),
            ));
        }
        // Built before `run` (outside the compute timer, as before score):
        // `run` cuts the same `cut` oldest ids.
        let limits = &self.cfg.limits;
        let cut = q.ids.len().saturating_sub(limits.max_history);
        let truncate = limits.history_policy == HistoryPolicy::Truncate;
        let exclude = match (cut, q.exclude_seen && truncate) {
            (0, _) | (_, false) => None,
            (_, true) => Some([&q.ids[..cut], q.exclude.unwrap_or_default()].concat()),
        };
        let call = |w: &mut Worker, model: &Model, c: Cut, ids: &mut _, scores: &mut _| {
            let req = Request {
                ids: c.ids,
                weights: c.weights,
                exclude: exclude.as_deref().or(q.exclude),
                ..*q
            };
            w.recommend(model, &req, ids, scores)
        };
        let at = (Endpoint::Recommend, p);
        self.run(m, at, q.ids, q.weights, view, call, encode)
    }

    /// The scores of `items` (at most `limits.max_items`) for a history:
    /// `run` with `Worker::score`.
    #[allow(clippy::too_many_arguments)]
    fn score<T>(
        &self,
        m: &metrics::Thread,
        p: Proto,
        history: &[i64],
        weights: Option<&[f32]>,
        items: &[i64],
        view: Option<&str>,
        encode: impl FnOnce(&Answer) -> T,
    ) -> Result<T, Fail> {
        let max = self.cfg.limits.max_items;
        if items.len() > max {
            m.bad_request();
            return Err(Fail(
                Code::ItemsOverMax,
                format!("{} items > max_items = {max}", items.len()),
            ));
        }
        let call = |w: &mut Worker, model: &Model, c: Cut, _: &mut _, scores: &mut _| {
            let req = ScoreRequest {
                ids: c.ids,
                weights: c.weights,
                items,
                ..Default::default()
            };
            w.score(model, &req, scores)
        };
        let at = (Endpoint::Score, p);
        self.run(m, at, history, weights, view, call, encode)
    }

    fn recommend_json(
        &self,
        m: &metrics::Thread,
        body: &[u8],
        request_id: &str,
    ) -> Result<Response, Fail> {
        let q: Query = parse(m, body)?;
        let weights = f32s(q.weights);
        let req = Request {
            ids: &q.history,
            weights: weights.as_deref(),
            k: q.k.unwrap_or(self.cfg.defaults.k),
            exclude_seen: q.exclude_seen,
            include: q.include.as_deref(),
            exclude: q.exclude.as_deref(),
            sorted: q.sorted.unwrap_or(self.cfg.defaults.sorted),
            ..Default::default()
        };
        self.recommend(m, Proto::Json, &req, q.view.as_deref(), |a| {
            let reply = Reply {
                items: Items(a.ids, a.scores),
                model: ModelInfo {
                    name: &self.cfg.model.name,
                },
                request_id,
                k: req.k,
                sorted: req.sorted,
                view: a.view,
                history: a.history,
            };
            timed(json(StatusCode::OK, &reply), a.compute)
        })
    }

    fn score_json(
        &self,
        m: &metrics::Thread,
        body: &[u8],
        request_id: &str,
    ) -> Result<Response, Fail> {
        let q: ScoreQuery = parse(m, body)?;
        let weights = f32s(q.weights);
        let w = weights.as_deref();
        self.score(
            m,
            Proto::Json,
            &q.history,
            w,
            &q.items,
            q.view.as_deref(),
            |a| {
                let reply = ScoreReply {
                    scores: a.scores,
                    model: ModelInfo {
                        name: &self.cfg.model.name,
                    },
                    request_id,
                    view: a.view,
                    history: a.history,
                };
                timed(json(StatusCode::OK, &reply), a.compute)
            },
        )
    }

    fn recommend_grpc(
        &self,
        m: &metrics::Thread,
        q: &pb::RecommendRequest,
        request_id: &str,
    ) -> Result<pb::RecommendResponse, Fail> {
        let req = Request {
            ids: &q.history,
            // proto3: an empty list is not sent; unweighted (T17)
            weights: (!q.weights.is_empty()).then_some(&q.weights[..]),
            k: q.k.map_or(self.cfg.defaults.k, |k| k as usize),
            exclude_seen: q.exclude_seen.unwrap_or(default_exclude_seen()),
            include: q.include.as_ref().map(|l| &l.ids[..]),
            exclude: (!q.exclude.is_empty()).then_some(&q.exclude[..]),
            sorted: q.sorted.unwrap_or(self.cfg.defaults.sorted),
            ..Default::default()
        };
        // T18: one copy of the answer, prost's default codec encodes it
        let view = q.view.as_deref();
        self.recommend(m, Proto::Grpc, &req, view, |a| pb::RecommendResponse {
            ids: a.ids.to_vec(),
            scores: a.scores.to_vec(),
            model: Some(self.model_info()),
            request_id: request_id.to_string(),
            k: req.k as u32, // <= k_max: checked in recommend
            sorted: req.sorted,
            view: a.view.map(String::from),
            history: Some(a.history.into()),
        })
    }

    fn score_grpc(
        &self,
        m: &metrics::Thread,
        q: &pb::ScoreRequest,
        request_id: &str,
    ) -> Result<pb::ScoreResponse, Fail> {
        let weights = (!q.weights.is_empty()).then_some(&q.weights[..]);
        let view = q.view.as_deref();
        self.score(m, Proto::Grpc, &q.history, weights, &q.items, view, |a| {
            pb::ScoreResponse {
                scores: a.scores.to_vec(),
                model: Some(self.model_info()),
                request_id: request_id.to_string(),
                view: a.view.map(String::from),
                history: Some(a.history.into()),
            }
        })
    }

    fn model_info(&self) -> pb::ModelInfo {
        pb::ModelInfo {
            name: self.cfg.model.name.clone(),
        }
    }

    /// One request per path a worker takes, sized to the limits (k_max,
    /// max_history ids, include bits, max_items): buffers made before the first accept,
    /// no first-request spike (W2).
    fn warm(&self, model: &Model, w: &mut Worker) -> Result<(), Error> {
        let n = self.cfg.limits.max_history.min(model.n_items());
        let ids: Vec<i64> = match model.item_ids() {
            Some(e) => e.iter().take(n).copied().collect(),
            None => (0..n as i64).collect(),
        };
        let (mut out_ids, mut out_scores) = (Vec::new(), Vec::new());
        for include in [None, Some(&ids[..])] {
            let req = Request {
                ids: &ids,
                k: self.cfg.limits.k_max,
                include,
                ..Default::default()
            };
            w.recommend(model, &req, &mut out_ids, &mut out_scores)?;
        }
        let items: Vec<i64> = match model.item_ids() {
            // Repeated up to max_items: the candidate buffer at full size.
            Some(e) => e
                .iter()
                .cycle()
                .take(self.cfg.limits.max_items)
                .copied()
                .collect(),
            None => (0..model.n_items() as i64)
                .cycle()
                .take(self.cfg.limits.max_items)
                .collect(),
        };
        let req = ScoreRequest {
            ids: &ids,
            items: &items,
            ..Default::default()
        };
        w.score(model, &req, &mut out_scores)?;
        Ok(())
    }
}

/// `POST /v1/recommend`'s body; defaults = Python's `SparseKNN.recommend`, but
/// `k` and `sorted` absent = `defaults.k` and `defaults.sorted`.
/// Unknown fields are an error (a typo like "weight" would otherwise serve
/// unweighted).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    /// The history ids, oldest → newest (Python's `history`).
    history: Vec<i64>,
    #[serde(default)]
    /// f64 as sent, rounded to f32 after (numpy's two steps).
    weights: Option<Vec<f64>>,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default = "default_exclude_seen")]
    exclude_seen: bool,
    #[serde(default)]
    include: Option<Vec<i64>>,
    #[serde(default)]
    exclude: Option<Vec<i64>>,
    #[serde(default)]
    sorted: Option<bool>,
    /// Absent or null = `defaults.view`; "" = the whole model.
    #[serde(default)]
    view: Option<String>,
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
        k: q.k.map(|k| k.try_into().unwrap_or(u32::MAX)),
        exclude_seen: Some(q.exclude_seen),
        include: q.include.map(|ids| pb::IdList { ids }),
        exclude: q.exclude.unwrap_or_default(),
        sorted: q.sorted,
        view: q.view,
        history: q.history,
    })
}

/// `POST /v1/score`'s body (unknown fields are an error, as `Query`'s).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScoreQuery {
    /// As `Query`'s.
    history: Vec<i64>,
    #[serde(default)]
    weights: Option<Vec<f64>>,
    /// The items to score, in the answer's order.
    items: Vec<i64>,
    /// As `Query`'s.
    #[serde(default)]
    view: Option<String>,
}

/// A JSON body as `T`, else a counted `bad_json`.
fn parse<'a, T: Deserialize<'a>>(m: &metrics::Thread, body: &'a [u8]) -> Result<T, Fail> {
    serde_json::from_slice(body).map_err(|e| {
        m.bad_request();
        Fail(Code::BadJson, e.to_string())
    })
}

/// Python's rounding: numpy.float32(a Python float), not serde's direct f32
/// parse (it rounds a decimal just above an f32 halfway point up, numpy's
/// f64 → f32 to even; test json_f32_parsed_as_python).
fn f32s(w: Option<Vec<f64>>) -> Option<Vec<f32>> {
    w.map(|w| w.iter().map(|&x| x as f32).collect())
}

/// `r` with the worker's time as `Server-Timing`.
fn timed(mut r: Response, compute: Duration) -> Response {
    let timing = format!("compute;dur={:.3}", compute.as_secs_f64() * 1e3);
    r.headers_mut().insert(
        HeaderName::from_static("server-timing"),
        HeaderValue::try_from(timing).expect("ASCII"),
    );
    r
}

/// The history a worker call gets: its oldest ids past `max_history` cut.
struct Cut<'a> {
    ids: &'a [i64],
    weights: Option<&'a [f32]>,
}

/// `exclude_seen` when a JSON request leaves it out: the core's default.
fn default_exclude_seen() -> bool {
    Request::default().exclude_seen
}

/// A served answer, borrowed from the worker's buffers.
struct Answer<'a> {
    ids: &'a [i64],
    scores: &'a [f32],
    /// The view's name; None = the whole model.
    view: Option<&'a str>,
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

impl From<History> for pb::History {
    fn from(h: History) -> Self {
        pb::History {
            sent: h.sent,
            truncated: h.truncated,
            unknown: h.unknown,
        }
    }
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

/// `POST /v1/score`'s answer: the scores in the items' order, then the
/// request's context (as `Reply`'s).
#[derive(Serialize)]
struct ScoreReply<'a> {
    scores: &'a [f32],
    model: ModelInfo<'a>,
    request_id: &'a str,
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
    /// The serving state as of `version` (`App::state`).
    state: Arc<Served>,
    version: u64,
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

/// The routes, over `app`: JSON (bodies limited by `read_body`) and gRPC
/// (tonic's decode limit).
pub fn router(app: Arc<App>) -> Router {
    let limit = app.cfg.limits.max_body_bytes;
    let grpc = pb::recommender_server::RecommenderServer::new(Grpc(app.clone()))
        .max_decoding_message_size(limit);
    Router::new()
        .route("/v1/recommend", post(recommend))
        .route("/v1/score", post(score))
        .route("/metrics", get(metrics))
        .route("/health", get(|| async { "ok" }))
        .route("/ready", get(ready))
        .route("/config", get(config))
        .route("/v1/views", get(views))
        .with_state(app)
        .merge(tonic::service::Routes::new(grpc).into_axum_router())
}

async fn recommend(State(app): State<Arc<App>>, req: axum::extract::Request) -> Response {
    post_json(&app, req, Endpoint::Recommend).await
}

async fn score(State(app): State<Arc<App>>, req: axum::extract::Request) -> Response {
    post_json(&app, req, Endpoint::Score).await
}

/// A JSON endpoint: the body read, answered (or the error), the request id
/// on both.
async fn post_json(app: &App, req: axum::extract::Request, e: Endpoint) -> Response {
    let t = Instant::now();
    let m = &app.metrics[THREAD.get()];
    let id = request_id(req.headers().get(X_REQUEST_ID).map(|v| v.as_bytes()));
    let r = match read_body(app, req).await {
        Ok(body) => {
            let r = match e {
                Endpoint::Recommend => app.recommend_json(m, &body, &id),
                Endpoint::Score => app.score_json(m, &body, &id),
            };
            m.total[e as usize][Proto::Json as usize].record(t.elapsed());
            r
        }
        Err(f) => {
            m.bad_request();
            Err(f)
        }
    };
    with_request_id(r.unwrap_or_else(|f| f.json(&id)), &id)
}

/// A JSON endpoint's body: over `limits.max_body_bytes` 413 at once (a
/// declared Content-Length before reading, else as it streams), not within
/// header_read of its headers 408 (a dribbled body would hold the
/// connection; hyper's timer covers headers only), unreadable 400. The rest
/// of an oversized body is read and dropped after the 413 (`drain`).
// ponytail: JSON only; a gRPC body is read inside tonic (no timeout).
async fn read_body(app: &App, req: axum::extract::Request) -> Result<Bytes, Fail> {
    let limit = app.cfg.limits.max_body_bytes;
    let deadline = secs(app.cfg.timeouts.header_read);
    let declared = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok()?.parse::<u64>().ok());
    // A client waiting for `100 Continue` never sends the body after a 413:
    // no drain (it would hold the connection until header_read).
    let expects = req.headers().contains_key(header::EXPECT);
    let mut body = req.into_body();
    if let Some(n) = declared.filter(|&n| n > limit as u64) {
        if !expects {
            drain(body, limit, deadline);
        }
        let e = format!("body of {n} bytes > limits.max_body_bytes = {limit}");
        return Err(Fail(Code::BodyTooLarge, e));
    }
    // One chunk (the usual small JSON body) is kept as is (no copy, no
    // allocation), more are joined.
    let (mut first, mut joined) = (None::<Bytes>, Vec::new());
    let mut size = 0;
    let read = async {
        while let Some(frame) = body.frame().await {
            let Ok(data) = frame.map_err(|e| e.to_string())?.into_data() else {
                continue; // trailers
            };
            size += data.len();
            if size > limit {
                return Ok(false);
            }
            match first.take() {
                None if joined.is_empty() => first = Some(data),
                f => {
                    joined.extend_from_slice(&f.unwrap_or_default());
                    joined.extend_from_slice(&data);
                }
            }
        }
        Ok(true)
    };
    match tokio::time::timeout(deadline, read).await {
        Ok(Ok(true)) => Ok(first.unwrap_or_else(|| Bytes::from(joined))),
        Ok(Ok(false)) => {
            drain(body, limit, deadline);
            let e = format!("body over limits.max_body_bytes = {limit}");
            Err(Fail(Code::BodyTooLarge, e))
        }
        Ok(Err(e)) => Err(Fail(Code::BadBody, e)),
        Err(_) => {
            let e = "request body not received within timeouts.header_read";
            Err(Fail(Code::BodyTimeout, e.into()))
        }
    }
}

/// Reads and drops the rest of a rejected body in the background (up to
/// 8 x `limit` bytes, within `deadline`): closing a connection with unread
/// bytes makes the OS reset it, and a client still sending would lose the
/// 413 (nginx's lingering close). Past either bound the connection closes.
fn drain(mut body: axum::body::Body, limit: usize, deadline: Duration) {
    let cap = limit.saturating_mul(8);
    tokio::spawn(tokio::time::timeout(deadline, async move {
        let mut n = 0;
        while let Some(Ok(frame)) = body.frame().await {
            n += frame.data_ref().map_or(0, |d| d.len());
            if n > cap {
                break;
            }
        }
    }));
}

struct Grpc(Arc<App>);

#[tonic::async_trait]
impl pb::recommender_server::Recommender for Grpc {
    async fn recommend(
        &self,
        r: tonic::Request<pb::RecommendRequest>,
    ) -> Result<tonic::Response<pb::RecommendResponse>, tonic::Status> {
        self.call(r, Endpoint::Recommend, App::recommend_grpc)
    }

    async fn score(
        &self,
        r: tonic::Request<pb::ScoreRequest>,
    ) -> Result<tonic::Response<pb::ScoreResponse>, tonic::Status> {
        self.call(r, Endpoint::Score, App::score_grpc)
    }
}

impl Grpc {
    /// `f` timed, the request id in its answer's or error's metadata. Handler
    /// time excludes the request's decode and the response's encode (tonic's,
    /// around this): compare formats on CPU per request.
    fn call<Q, R>(
        &self,
        r: tonic::Request<Q>,
        e: Endpoint,
        f: fn(&App, &metrics::Thread, &Q, &str) -> Result<R, Fail>,
    ) -> Result<tonic::Response<R>, tonic::Status> {
        let t = Instant::now();
        let m = &self.0.metrics[THREAD.get()];
        let id = request_id(r.metadata().get(X_REQUEST_ID).map(|v| v.as_bytes()));
        let r = f(&self.0, m, r.get_ref(), &id);
        m.total[e as usize][Proto::Grpc as usize].record(t.elapsed());
        let v: tonic::metadata::AsciiMetadataValue =
            id.parse().expect("checked or generated: printable ASCII");
        match r {
            Ok(r) => {
                let mut r = tonic::Response::new(r);
                r.metadata_mut().insert(X_REQUEST_ID, v);
                Ok(r)
            }
            Err(f) => {
                let mut e = f.grpc(&id);
                e.metadata_mut().insert(X_REQUEST_ID, v);
                Err(e)
            }
        }
    }
}

/// Readiness: 503 once `serve` is shutting down (the load balancer stops
/// sending; liveness, `/health`, stays ok so the pod isn't killed mid-drain).
async fn ready(State(app): State<Arc<App>>) -> (StatusCode, &'static str) {
    match app.draining.load(Relaxed) {
        false => (StatusCode::OK, "ok"),
        true => (StatusCode::SERVICE_UNAVAILABLE, "draining"),
    }
}

async fn config(State(app): State<Arc<App>>) -> Response {
    json(StatusCode::OK, &app.cfg)
}

/// `GET /v1/views`: the default view (null = the whole model), the model's
/// items, and per view what it allows and its files' known ids.
async fn views(State(app): State<Arc<App>>) -> Response {
    #[derive(Serialize)]
    struct Views<'a> {
        default: Option<&'a str>,
        items: usize,
        views: BTreeMap<&'a str, &'a ViewInfo>,
    }
    let s = app.state();
    let d = &app.cfg.defaults.view;
    let body = Views {
        default: (!d.is_empty()).then_some(d.as_str()),
        items: s.model.n_items(),
        views: s.views.iter().map(|(k, v)| (k.as_str(), &v.info)).collect(),
    };
    json(StatusCode::OK, &body)
}

async fn metrics(State(app): State<Arc<App>>) -> String {
    let s = app.state();
    let info = metrics::Info {
        version: env!("CARGO_PKG_VERSION"),
        model: &app.cfg.model.name,
        threads: app.cfg.server.threads,
        block: app.block,
        items: s.model.n_items(),
        active_items: s.model.n_active(),
        views: app
            .served
            .iter()
            .map(|(k, c)| {
                let errors = app.reload_errors[k].load(Relaxed);
                (k.as_str(), c.iter().map(Counter::get).sum(), errors)
            })
            .collect(),
        view_reloads: app.reloads.load(Relaxed),
        last_view_reload: f64::from_bits(app.last_reload.load(Relaxed)),
    };
    metrics::render(&app.metrics, &info)
}

/// Serves `router` (`router(app)`, tests add routes) on `listener` with
/// `app`'s threads, each warmed before the first accept, until `shutdown`
/// completes (main: SIGTERM / SIGINT). Then: `/ready` answers 503 and
/// connections are still accepted for `timeouts.drain_delay`; the listener
/// closes (new connections are refused); each thread lets its connections
/// finish (HTTP/1.1: after the response in flight, idle ones at once; h2:
/// GOAWAY, open streams finish) for up to `timeouts.shutdown_grace`, then
/// closes what is left. Returns Ok then, an error only if a thread fails.
pub fn serve(
    listener: TcpListener,
    app: Arc<App>,
    router: Router,
    shutdown: impl Future<Output = ()>,
) -> io::Result<()> {
    let n = app.cfg.server.threads;
    let t = &app.cfg.timeouts;
    let (header_read, grace) = (secs(t.header_read), secs(t.shutdown_grace));
    let (ready_tx, ready) = mpsc::channel();
    let mut threads = Vec::with_capacity(n);
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<std::net::TcpStream>();
        let (app, router, ready_tx) = (app.clone(), router.clone(), ready_tx.clone());
        let handle = std::thread::Builder::new()
            .name(format!("server-{i}"))
            .spawn(move || {
                set_qos(true);
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();
                let start = || -> Result<_, String> {
                    let rt = rt.map_err(|e| e.to_string())?;
                    // The version first, then the state (as `run`)
                    let version = app.version.load(Acquire);
                    let state = app.state();
                    let mut worker = Worker::new(&state.model, app.cfg.server.block_size, n)
                        .map_err(|e| e.to_string())?;
                    app.warm(&state.model, &mut worker)
                        .map_err(|e| format!("warm-up: {e}"))?;
                    THREAD.set(i);
                    LOCAL.set(Some(Local {
                        base: worker.stats(),
                        worker,
                        state,
                        version,
                        ids: Vec::with_capacity(app.cfg.limits.k_max),
                        scores: Vec::with_capacity(
                            app.cfg.limits.k_max.max(app.cfg.limits.max_items),
                        ),
                    }));
                    Ok(rt)
                };
                let rt = match start() {
                    Ok(rt) => rt,
                    Err(e) => return ready_tx.send(Err(e)).unwrap_or(()),
                };
                let _ = ready_tx.send(Ok(()));
                // HTTP/1.1 or h2c by the preface; h2 spawns its streams on
                // this thread's runtime (current-thread)
                let mut http = auto::Builder::new(TokioExecutor::new());
                // The timer also restarts while an HTTP/1.1 connection waits
                // for its next request: idle keep-alive closes after it.
                http.http1()
                    .timer(TokioTimer::new())
                    .header_read_timeout(header_read);
                // Pings: a dead peer's h2 connection (no ack) is closed.
                // ponytail: fixed, a key if a network needs other values
                http.http2()
                    .timer(TokioTimer::new())
                    .keep_alive_interval(Duration::from_secs(60))
                    .keep_alive_timeout(Duration::from_secs(20));
                let graceful = GracefulShutdown::new();
                rt.block_on(async {
                    // Until the acceptor closes the channel (shutdown)
                    while let Some(s) = rx.recv().await {
                        let Ok(s) = s
                            .set_nonblocking(true)
                            .and_then(|()| tokio::net::TcpStream::from_std(s))
                        else {
                            continue;
                        };
                        let svc = TowerToHyperService::new(router.clone());
                        let decided = Arc::new(AtomicBool::new(false));
                        let io = Preface {
                            s,
                            seen: 0,
                            decided: decided.clone(),
                        };
                        let conn = http.serve_connection(TokioIo::new(io), svc).into_owned();
                        let conn = graceful.watch(conn);
                        tokio::spawn(async move {
                            let mut conn = std::pin::pin!(conn);
                            // The protocol not decided within header_read of
                            // the accept (silent, a dribbled preface): closed.
                            // After it, HTTP/1.1 has hyper's header timer; an
                            // h2 connection may idle like any gRPC channel
                            // (pings find dead peers; shutdown sends GOAWAY).
                            // ponytail: so h2 headers dribbled after the
                            // preface have no deadline (private network,
                            // R-S3); a per-stream header timer if exposed.
                            let first = tokio::time::timeout(header_read, conn.as_mut()).await;
                            if first.is_err() && decided.load(Relaxed) {
                                let _ = conn.await;
                            }
                        });
                    }
                    let open = graceful.count();
                    if tokio::time::timeout(grace, graceful.shutdown())
                        .await
                        .is_err()
                    {
                        eprintln!(
                            "similaripy-server: thread {i}: shutdown_grace over, closing the \
                             connections still open (of {open})"
                        );
                    }
                });
                // The runtime's drop closes what is left.
            })?;
        threads.push(tx);
        handles.push(handle);
    }
    drop(ready_tx);
    for _ in 0..n {
        ready
            .recv()
            .map_err(|_| io::Error::other("a server thread died at start"))?
            .map_err(io::Error::other)?;
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = rt.block_on(async {
        let every = app.cfg.views.reload_interval;
        if every > 0.0 && !app.served.is_empty() {
            tokio::spawn(reload_views(app.clone(), secs(every)));
        }
        listener.set_nonblocking(true)?;
        let listener = tokio::net::TcpListener::from_std(listener)?;
        let mut next = 0;
        accept(&listener, &threads, &mut next, shutdown).await?;
        app.draining.store(true, Relaxed);
        let drain = secs(app.cfg.timeouts.drain_delay);
        eprintln!(
            "similaripy-server: draining: /ready 503, accepting for {drain:?} more, then \
             up to {grace:?} for open connections"
        );
        accept(&listener, &threads, &mut next, tokio::time::sleep(drain)).await?;
        drop(listener);
        // Each thread sees its channel closed: graceful shutdown
        drop(threads);
        // Joined off this runtime: its other tasks (main's second-signal
        // watch) keep running meanwhile.
        let joined =
            tokio::task::spawn_blocking(move || handles.into_iter().all(|h| h.join().is_ok()));
        match joined.await {
            Ok(true) => {
                eprintln!("similaripy-server: stopped");
                Ok(())
            }
            _ => Err(io::Error::other("a server thread panicked at shutdown")),
        }
    });
    // Not waiting for a view reload stuck on a slow file system.
    rt.shutdown_background();
    result
}

/// Checks the view files every `every` and reloads them on a change (V21
/// C), on the acceptor's runtime (never a serving thread); the stat and
/// reads in a blocking task. Ends with the runtime.
async fn reload_views(app: Arc<App>, every: Duration) {
    let mut tick = tokio::time::interval(every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let app = app.clone();
        let _ = tokio::task::spawn_blocking(move || app.reload_views()).await;
    }
}

/// Hands `listener`'s connections round-robin to `threads` until `until`
/// completes.
// ponytail: round-robin per connection; SO_REUSEPORT per thread (Linux)
// is the S4 probe.
async fn accept(
    listener: &tokio::net::TcpListener,
    threads: &[tokio::sync::mpsc::UnboundedSender<std::net::TcpStream>],
    next: &mut usize,
    until: impl Future<Output = ()>,
) -> io::Result<()> {
    let mut until = std::pin::pin!(until);
    loop {
        let s = tokio::select! {
            s = listener.accept() => s,
            () = &mut until => return Ok(()),
        };
        let s = match s.and_then(|(s, _)| s.into_std()) {
            Ok(s) => s,
            Err(_) => {
                // E.g. out of file descriptors: fails at once until one
                // closes; retrying at once would spin a core.
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }
        };
        let _ = s.set_nodelay(true);
        let i = *next % threads.len();
        if threads[i].send(s).is_err() {
            return Err(io::Error::other(format!("server thread {i} died")));
        }
        *next += 1;
    }
}

/// h2c's connection preface: the auto builder picks h2 when a connection
/// starts with all of it, HTTP/1.1 at the first byte that differs.
const H2_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// A connection's socket that sets `decided` once its bytes so far settle
/// the protocol as the auto builder does (`serve`'s first deadline). After
/// that, a read costs one compare.
struct Preface {
    s: tokio::net::TcpStream,
    /// Leading bytes equal to `H2_PREFACE`; its length once decided.
    seen: usize,
    decided: Arc<AtomicBool>,
}

impl tokio::io::AsyncRead for Preface {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        let before = buf.filled().len();
        let r = std::pin::Pin::new(&mut self.s).poll_read(cx, buf);
        let seen = self.seen;
        if seen < H2_PREFACE.len() {
            let new = &buf.filled()[before..];
            let n = new.len().min(H2_PREFACE.len() - seen);
            self.seen = match new[..n] == H2_PREFACE[seen..seen + n] {
                true => seen + n,
                false => H2_PREFACE.len(),
            };
            if self.seen == H2_PREFACE.len() {
                self.decided.store(true, Relaxed);
            }
        }
        r
    }
}

impl tokio::io::AsyncWrite for Preface {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.s).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> std::task::Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.s).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.s.is_write_vectored()
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.s).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.s).poll_shutdown(cx)
    }
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
