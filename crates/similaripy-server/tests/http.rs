//! The server over real HTTP on a loopback port: responses equal core
//! `serve::recommend` as the Python binding calls it (block resolved for
//! every core busy; the server's block differs on purpose: same bits), JSON
//! f32 parsed as Python + numpy do, edge rejections, a panicking connection,
//! metrics; gRPC on the same port: the same answers (ids and f32 bits) as
//! JSON, proto3 presence rules, errors as status codes; the answer's context
//! (model, request id, k, sorted, history counts) and same request -> same
//! bytes (U5); hardening (S5d): graceful shutdown (in-flight requests
//! finish, new connections refused, grace), header-read / first-request /
//! keep-alive timeouts, and the binary's exit 0 on SIGTERM.

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

use axum::http::StatusCode;
use axum::routing::get;
use bytes::Bytes;
use serde::Deserialize;
use similaripy_core::CsrView;
use similaripy_core::serve::{BlockSize, ItemOrder, Model, Request, recommend};
use similaripy_server::client::{connect, metric, send, send_with};
use similaripy_server::config::{self, HistoryPolicy};
use similaripy_server::pb::recommender_client::RecommenderClient;
use similaripy_server::pb::{IdList, RecommendRequest};
use similaripy_server::{App, BODY_LIMIT, Config, proto_from_json, router, serve};
use tonic::Code;
use tonic::transport::Channel;

const N: usize = 60;

/// xorshift64: deterministic data without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A random S over N items (float values: the bits matter), external ids
/// 1000 + 7i. Deterministic: two calls build equal models.
fn model() -> Model {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut indptr, mut indices, mut data) = (vec![0i64], vec![], vec![]);
    for _ in 0..N {
        let mut cols: Vec<i32> = (0..8).map(|_| rng.below(N) as i32).collect();
        cols.sort_unstable();
        cols.dedup();
        for c in cols {
            indices.push(c);
            data.push(rng.unit() + 0.01);
        }
        indptr.push(indices.len() as i64);
    }
    let s = CsrView::new("S", &indptr, &indices, &data, N).unwrap();
    let ext = (0..N as i64).map(|i| 1000 + 7 * i).collect();
    Model::new(s, Some(ext), None, ItemOrder::Auto).unwrap()
}

fn config(threads: usize) -> Config {
    let mut c = Config::default();
    c.server.threads = threads;
    // Several blocks here, one block in the Python reference: same bits.
    c.server.block_size = BlockSize::Size(16);
    c.limits.max_history = 20;
    c.limits.k_max = 50;
    c.model.name = "knn-test".into();
    c
}

/// A server on a free loopback port (its threads live until the test
/// process ends); `with_panic` adds a `/panic` route.
fn start(cfg: Config, with_panic: bool) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Arc::new(App::new(model(), cfg).unwrap());
    let mut r = router(app.clone());
    if with_panic {
        r = r.route("/panic", get(panics));
    }
    std::thread::spawn(move || serve(listener, app, r, std::future::pending()));
    addr
}

fn block_on<F: Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(f)
}

/// `POST /v1/recommend`'s answer, every field (unknown ones are an error:
/// the shape is the contract).
#[derive(Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
struct Reply {
    items: Vec<Item>,
    model: ModelInfo,
    request_id: String,
    k: usize,
    sorted: bool,
    view: Option<String>,
    history: History,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
struct Item {
    id: i64,
    score: f32,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
struct ModelInfo {
    name: String,
}

#[derive(Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(deny_unknown_fields)]
struct History {
    sent: u64,
    truncated: u64,
    unknown: u64,
}

/// An error's body, every field (V21: the shape is the contract).
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ErrorBody {
    error: ErrorInfo,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ErrorInfo {
    code: String,
    message: String,
    request_id: String,
}

/// `text`'s error, its request id the header's.
fn error_of(h: &axum::http::HeaderMap, text: &str) -> ErrorInfo {
    let e = serde_json::from_str::<ErrorBody>(text)
        .unwrap_or_else(|e| panic!("{e}: {text}"))
        .error;
    assert_eq!(h["x-request-id"], e.request_id.as_str(), "{text}");
    e
}

/// A gRPC error's `x-error-code`.
fn grpc_code(e: &tonic::Status) -> &str {
    e.metadata()
        .get("x-error-code")
        .map_or("", |v| v.to_str().unwrap())
}

impl Reply {
    /// (ids, score bits)
    fn answer(&self) -> (Vec<i64>, Vec<u32>) {
        self.items.iter().map(|x| (x.id, x.score.to_bits())).unzip()
    }
}

/// Python's call: `SparseKNN.recommend` → `serve::recommend` with the block
/// resolved for every core busy (`ServeModel`).
fn python(m: &Model, req: &Request) -> (Vec<i64>, Vec<u32>) {
    let block = BlockSize::Auto.resolve(m.n_active(), usize::MAX).unwrap();
    let (mut ids, mut scores) = (vec![], vec![]);
    recommend(m, block, req, &mut ids, &mut scores).unwrap();
    (ids, scores.iter().map(|s| s.to_bits()).collect())
}

async fn post(addr: SocketAddr, body: &str) -> (StatusCode, String) {
    let mut c = connect(addr).await.unwrap();
    let (status, b) = send(&mut c, "/v1/recommend", Some(Bytes::from(body.to_string())))
        .await
        .unwrap();
    (status, String::from_utf8(b.to_vec()).unwrap())
}

async fn get_text(addr: SocketAddr, path: &str) -> String {
    let mut c = connect(addr).await.unwrap();
    let (status, b) = send(&mut c, path, None).await.unwrap();
    assert_eq!(status, StatusCode::OK);
    String::from_utf8(b.to_vec()).unwrap()
}

#[test]
fn same_ids_and_bits_as_python_recommend() {
    let addr = start(config(2), false);
    let reference = model();
    let mut rng = Rng(42);
    block_on(async {
        for case in 0..200 {
            let len = 1 + rng.below(30); // past max_history = 20 sometimes
            // Some unknown ids (ignored), some duplicates.
            let ids: Vec<i64> = (0..len)
                .map(|_| 1000 + rng.below(N + 5) as i64 * 7)
                .collect();
            let weights: Option<Vec<f32>> =
                (case % 2 == 0).then(|| (0..len).map(|_| rng.unit() * 3.0).collect());
            let include: Option<Vec<i64>> =
                (case % 5 == 1).then(|| (0..30).map(|_| 1000 + rng.below(N) as i64 * 7).collect());
            let exclude: Option<Vec<i64>> =
                (case % 7 == 2).then(|| (0..5).map(|_| 1000 + rng.below(N) as i64 * 7).collect());
            let (k, exclude_seen) = (1 + rng.below(50), case % 3 != 0);
            let body = serde_json::json!({
                "history": ids, "weights": weights, "k": k, "exclude_seen": exclude_seen,
                "include": include, "exclude": exclude,
            });
            let (status, text) = post(addr, &body.to_string()).await;
            assert_eq!(status, StatusCode::OK, "{text}");
            let got: Reply = serde_json::from_str(&text).unwrap();
            // The server keeps the last max_history ids (and their weights).
            // Cut ids are still seen: excluded under exclude_seen.
            let cut = len.saturating_sub(20);
            let mut excluded = exclude.clone();
            if exclude_seen && cut > 0 {
                excluded.get_or_insert_default().extend(&ids[..cut]);
            }
            let want = python(
                &reference,
                &Request {
                    ids: &ids[cut..],
                    weights: weights.as_deref().map(|w| &w[cut..]),
                    k,
                    exclude_seen,
                    include: include.as_deref(),
                    exclude: excluded.as_deref(),
                    sorted: true,
                },
            );
            assert_eq!(got.answer(), want, "case {case}: {body}");
        }
    });
}

#[test]
fn defaults_are_pythons() {
    // k = 10, exclude_seen = true, sorted = true when not sent.
    let addr = start(config(1), false);
    let reference = model();
    let ids = [1000, 1007, 1014, 1070];
    block_on(async {
        let (status, text) = post(addr, r#"{"history": [1000, 1007, 1014, 1070]}"#).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let got: Reply = serde_json::from_str(&text).unwrap();
        let req = Request {
            ids: &ids,
            weights: None,
            k: 10,
            exclude_seen: true,
            include: None,
            exclude: None,
            sorted: true,
        };
        assert_eq!(got.answer(), python(&reference, &req));
    });
}

/// Requests: a weight's JSON decimal → f32 as `numpy.float32(json.loads(..))`
/// does (correctly rounded f64, then rounded to f32), through the server:
/// decimals just above an f32 halfway point are where serde's direct f32
/// parse rounds otherwise. Responses: an f32 the server writes, read back
/// that way, keeps its bits: every finite f32 but one (±7.038531e-26: f64 then
/// f32 rounds twice; checked exhaustively in review, accepted).
#[test]
fn json_f32_parsed_as_python() {
    let addr = start(config(1), false);
    let reference = model();
    let mut rng = Rng(7);
    block_on(async {
        for _ in 0..300 {
            // Halfway between two f32 in [1, 2): exact in f64 (24 fractional
            // binary digits = 24 decimals); a trailing 1 puts it just above.
            let a = 1.0 + rng.unit();
            let mid = (f64::from(a) + f64::from(f32::from_bits(a.to_bits() + 1))) / 2.0;
            let random = f64::from(rng.unit()) * 10.0 + 1e-9;
            for s in [
                format!("{mid:.40}"),
                format!("{mid:.40}1"),
                format!("{random:e}"),
            ] {
                let body = format!(r#"{{"history": [1000], "weights": [{s}], "k": 50}}"#);
                let (status, text) = post(addr, &body).await;
                assert_eq!(status, StatusCode::OK, "{text}");
                let got: Reply = serde_json::from_str(&text).unwrap();
                let w = [s.parse::<f64>().unwrap() as f32];
                let req = Request {
                    ids: &[1000],
                    weights: Some(&w),
                    k: 50,
                    exclude_seen: true,
                    include: None,
                    exclude: None,
                    sorted: true,
                };
                assert_eq!(got.answer(), python(&reference, &req), "{s}");
            }
        }
    });
    for _ in 0..100_000 {
        let f = f32::from_bits(rng.next() as u32 & 0x7f7f_ffff); // finite
        let text = serde_json::to_string(&f).unwrap();
        let back = text.parse::<f64>().unwrap() as f32;
        assert_eq!(back.to_bits(), f.to_bits(), "{text}");
    }
}

#[test]
fn edge_rejections() {
    let addr = start(config(1), false);
    block_on(async {
        let big = format!(r#"{{"history": [{}1]}}"#, "1,".repeat(BODY_LIMIT / 2));
        for (body, status, code, msg) in [
            (
                r#"{"history": [1000], "weight": [2.0]}"#,
                400,
                "bad_json",
                "unknown field",
            ),
            (
                r#"{"history": [1000], "k": 51}"#,
                400,
                "k_over_max",
                "k_max",
            ),
            // The pre-1.0 name: a clear error, never an empty history.
            (r#"{"ids": [1000]}"#, 400, "bad_json", "unknown field `ids`"),
            // The worker's two checks, one code (Code::BadWeights).
            (
                r#"{"history": [1000, 1007], "weights": [1.0]}"#,
                400,
                "bad_weights",
                "len(weights)",
            ),
            (r#"{"history": "x"}"#, 400, "bad_json", "invalid type"),
            (r#"{"k": 3}"#, 400, "bad_json", "missing field"),
            (r#"{"history": [1"#, 400, "bad_json", "EOF"),
            // 413 as JSON with the id too (its Content-Length, before the body).
            (&big, 413, "body_too_large", "max_body_bytes"),
        ] {
            let (got, h, text) = post_with(addr, body, &[]).await;
            assert_eq!(got.as_u16(), status, "{body:.80}: {text}");
            let e = error_of(&h, &text);
            assert_eq!(e.code, code, "{body:.80}: {text}");
            assert!(e.message.contains(msg), "{body:.80}: {text}");
        }
        // The worker's other weights check: the same code.
        let body = r#"{"history": [1000], "weights": [1e39]}"#;
        let (_, h, text) = post_with(addr, body, &[]).await;
        let e = error_of(&h, &text);
        assert_eq!(
            (&e.code[..], e.message.contains("finite")),
            ("bad_weights", true),
            "{text}"
        );
        let m = get_text(addr, "/metrics").await;
        // JSON x 5 + k + 413: the edge; weights x 2: the worker.
        assert!(
            m.contains("similaripy_server_bad_requests_total 7\n"),
            "{m}"
        );
        assert!(m.contains("similaripy_server_errors_total 2\n"), "{m}");
    });
}

#[test]
fn a_panicking_request_leaves_its_thread_serving() {
    let addr = start(config(1), true);
    block_on(async {
        let mut c = connect(addr).await.unwrap();
        assert!(send(&mut c, "/panic", None).await.is_err());
        // The one thread serves the next connection, its worker clean.
        let (status, text) = post(addr, r#"{"history": [1000, 1007]}"#).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        assert_eq!(get_text(addr, "/health").await, "ok");
    });
}

#[test]
fn metrics_count_served_requests_per_thread() {
    let addr = start(config(2), false);
    block_on(async {
        // Two connections: round-robin puts one on each thread. Warm-up
        // requests are not counted.
        let long: Vec<i64> = (0..25).map(|i| 1000 + 7 * i).collect();
        post(addr, r#"{"history": [1000, 1007, 99]}"#).await;
        post(addr, &serde_json::json!({ "history": long }).to_string()).await;
        let m = get_text(addr, "/metrics").await;
        for line in [
            "similaripy_server_threads 2\n",
            "similaripy_server_block_size 16\n",
            "similaripy_server_requests_total{thread=\"0\"} 1\n",
            "similaripy_server_requests_total{thread=\"1\"} 1\n",
            "similaripy_server_truncated_total 1\n",
            "similaripy_server_history_ids_total 23\n", // 3 + the last 20
            "similaripy_server_known_ids_total 22\n",   // 99 is unknown
            "similaripy_server_compute_seconds_count{proto=\"json\"} 2\n",
            "similaripy_server_compute_seconds_count{proto=\"grpc\"} 0\n",
            "similaripy_server_request_seconds_bucket{proto=\"json\",le=\"+Inf\"} 2\n",
        ] {
            assert!(m.contains(line), "{line:?} not in\n{m}");
        }
        let info = format!(
            "similaripy_server_info{{version=\"{}\",model=\"knn-test\"}} 1\n",
            env!("CARGO_PKG_VERSION")
        );
        assert!(m.contains(&info), "{info:?} not in\n{m}");
        #[cfg(unix)]
        assert!(
            metric(&m, "process_cpu_seconds_total").is_some_and(|s| s > 0.0),
            "{m}"
        );
        // After 2026-01-01, not in the future
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        assert!(
            metric(&m, "process_start_time_seconds").is_some_and(|s| (1.767e9..=now).contains(&s)),
            "{m}"
        );
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(
            metric(&m, "process_resident_memory_bytes").is_some_and(|b| b > 1e6),
            "{m}"
        );
        assert_eq!(
            metric(&m, "similaripy_server_compute_seconds_count"),
            Some(2.0)
        );
    });
}

async fn panics() -> &'static str {
    panic!("a handler panic (test)")
}

async fn grpc(addr: SocketAddr) -> RecommenderClient<Channel> {
    RecommenderClient::connect(format!("http://{addr}"))
        .await
        .unwrap()
}

/// (ids, score bits) of a gRPC answer.
async fn ask(c: &mut RecommenderClient<Channel>, q: RecommendRequest) -> (Vec<i64>, Vec<u32>) {
    let r = c.recommend(q).await.unwrap().into_inner();
    (r.ids, r.scores.iter().map(|s| s.to_bits()).collect())
}

/// (ids, score bits) of a JSON answer.
async fn ask_json(addr: SocketAddr, body: &str) -> (Vec<i64>, Vec<u32>) {
    let (status, text) = post(addr, body).await;
    assert_eq!(status, StatusCode::OK, "{body}: {text}");
    serde_json::from_str::<Reply>(&text).unwrap().answer()
}

#[test]
fn grpc_answers_equal_json() {
    // Random requests both ways on one port: sorted / unsorted, weighted,
    // empty histories, k past the candidates, histories past max_history
    // (cut ids still excluded), include / exclude.
    let addr = start(config(2), false);
    let mut rng = Rng(11);
    block_on(async {
        let mut c = grpc(addr).await;
        let mut sizes = [0usize; 3]; // empty answers, short answers, cut histories
        for case in 0..300 {
            let len = rng.below(31);
            let ids: Vec<i64> = (0..len)
                .map(|_| 1000 + rng.below(N + 5) as i64 * 7)
                .collect();
            let weights: Option<Vec<f32>> =
                (case % 2 == 0).then(|| (0..len).map(|_| rng.unit() * 3.0).collect());
            let include: Option<Vec<i64>> =
                (case % 5 == 1).then(|| (0..30).map(|_| 1000 + rng.below(N) as i64 * 7).collect());
            let exclude: Option<Vec<i64>> =
                (case % 7 == 2).then(|| (0..5).map(|_| 1000 + rng.below(N) as i64 * 7).collect());
            let k = 1 + rng.below(50);
            let body = serde_json::json!({
                "history": ids, "weights": weights, "k": k,
                "exclude_seen": case % 3 != 0, "include": include, "exclude": exclude,
                "sorted": case % 4 != 3,
            })
            .to_string();
            let want = ask_json(addr, &body).await;
            let got = ask(&mut c, proto_from_json(body.as_bytes()).unwrap()).await;
            assert_eq!(got, want, "case {case}: {body}");
            sizes[0] += want.0.is_empty() as usize;
            sizes[1] += (!want.0.is_empty() && want.0.len() < k) as usize;
            sizes[2] += (len > 20) as usize;
        }
        assert!(sizes.iter().all(|&n| n > 5), "{sizes:?}");
    });
}

#[test]
fn grpc_presence_rules() {
    let addr = start(config(1), false);
    let ids = vec![1000, 1007, 1014, 1070];
    block_on(async {
        let mut c = grpc(addr).await;
        // Nothing but ids: JSON's defaults (k 10, exclude_seen, sorted).
        let only_ids = RecommendRequest {
            history: ids.clone(),
            ..Default::default()
        };
        let defaults = ask(&mut c, only_ids.clone()).await;
        assert_eq!(defaults.0.len(), 10);
        assert_eq!(
            defaults,
            ask_json(addr, r#"{"history": [1000, 1007, 1014, 1070]}"#).await
        );
        // Empty weights = not sent = unweighted.
        let unweighted = RecommendRequest {
            weights: vec![],
            ..only_ids.clone()
        };
        assert_eq!(ask(&mut c, unweighted).await, defaults);
        // include absent = every item; an empty IdList = no item.
        let none = RecommendRequest {
            include: Some(IdList { ids: vec![] }),
            ..only_ids.clone()
        };
        assert_eq!(ask(&mut c, none).await, (vec![], vec![]));
        let json = ask_json(
            addr,
            r#"{"history": [1000, 1007, 1014, 1070], "include": []}"#,
        )
        .await;
        assert_eq!(json, (vec![], vec![]));
        // An empty history: an empty answer, not an error.
        let empty = ask(&mut c, RecommendRequest::default()).await;
        assert_eq!(empty, (vec![], vec![]));
    });
}

#[test]
fn grpc_errors_are_status_codes() {
    let addr = start(config(1), false);
    block_on(async {
        let mut c = grpc(addr).await;
        for (q, code, msg) in [
            (
                RecommendRequest {
                    history: vec![1000],
                    k: Some(51),
                    ..Default::default()
                },
                "k_over_max",
                "k_max",
            ),
            (
                RecommendRequest {
                    history: vec![1000, 1007],
                    weights: vec![1.0],
                    ..Default::default()
                },
                "bad_weights",
                "len(weights)",
            ),
        ] {
            let e = c.recommend(q).await.unwrap_err();
            assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
            assert_eq!(grpc_code(&e), code, "{e:?}");
            assert!(e.message().contains(msg), "{e:?}");
        }
        // Over the 1 MB limit (tonic's own, not DefaultBodyLimit's).
        let big = RecommendRequest {
            history: vec![i64::MAX; BODY_LIMIT / 8], // 9 bytes each as a varint
            ..Default::default()
        };
        let e = c.recommend(big).await.unwrap_err();
        assert_eq!(e.code(), Code::OutOfRange, "{e:?}");
        // tonic's, before our handler: no code (documented)
        assert_eq!(grpc_code(&e), "", "{e:?}");
        // The connection still serves; JSON on the same port too.
        let q = RecommendRequest {
            history: vec![1000, 1007],
            ..Default::default()
        };
        assert!(!ask(&mut c, q).await.0.is_empty());
        assert!(
            !ask_json(addr, r#"{"history": [1000, 1007]}"#)
                .await
                .0
                .is_empty()
        );
        let m = get_text(addr, "/metrics").await;
        for line in [
            "similaripy_server_bad_requests_total 1\n", // k
            "similaripy_server_errors_total 1\n",       // weights
            "similaripy_server_compute_seconds_count{proto=\"grpc\"} 2\n",
            "similaripy_server_compute_seconds_count{proto=\"json\"} 1\n",
            "similaripy_server_request_seconds_count{proto=\"grpc\"} 3\n",
        ] {
            assert!(m.contains(line), "{line:?} not in\n{m}");
        }
        // Label-aware reads sum the formats.
        assert_eq!(
            metric(&m, "similaripy_server_compute_seconds_count"),
            Some(3.0)
        );
    });
}

#[test]
fn config_sets_sorted_default_and_history_policy() {
    let mut cfg = config(1);
    cfg.defaults.sorted = false;
    cfg.limits.history_policy = HistoryPolicy::Reject;
    let addr = start(cfg.clone(), false);
    let ids: Vec<i64> = (0..20).map(|i| 1000 + 7 * i).collect(); // = max_history
    let body = |extra: &str| format!(r#"{{"history": {ids:?}, "k": 50{extra}}}"#);
    block_on(async {
        // `sorted` not sent = the server's default (false), both formats.
        let absent = ask_json(addr, &body("")).await;
        assert_eq!(absent, ask_json(addr, &body(r#", "sorted": false"#)).await);
        let sorted = ask_json(addr, &body(r#", "sorted": true"#)).await;
        assert_ne!(absent, sorted);
        let mut c = grpc(addr).await;
        let q = RecommendRequest {
            history: ids.clone(),
            k: Some(50),
            ..Default::default()
        };
        assert_eq!(ask(&mut c, q.clone()).await, absent);
        let q_sorted = RecommendRequest {
            sorted: Some(true),
            ..q.clone()
        };
        assert_eq!(ask(&mut c, q_sorted).await, sorted);
        // reject: one id past max_history is a 400 / INVALID_ARGUMENT.
        let long = [&ids[..], &[1140]].concat();
        let body = serde_json::json!({ "history": long }).to_string();
        let (status, h, text) = post_with(addr, &body, &[]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
        let e = error_of(&h, &text);
        assert_eq!(e.code, "history_over_max", "{text}");
        assert!(
            e.message.contains("21 history ids > max_history = 20"),
            "{text}"
        );
        let e = c
            .recommend(RecommendRequest { history: long, ..q })
            .await
            .unwrap_err();
        assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
        assert_eq!(grpc_code(&e), "history_over_max", "{e:?}");
        let m = get_text(addr, "/metrics").await;
        assert!(
            m.contains("similaripy_server_bad_requests_total 2\n"),
            "{m}"
        );
        assert!(m.contains("similaripy_server_truncated_total 0\n"), "{m}");
        // GET /config: the resolved config.
        let got: Config = serde_json::from_str(&get_text(addr, "/config").await).unwrap();
        assert_eq!(got, cfg);
    });
}

#[test]
fn defaults_k_and_max_body_bytes() {
    let mut cfg = config(1);
    cfg.defaults.k = 3;
    cfg.limits.max_body_bytes = 200;
    let addr = start(cfg, false);
    let history = r#""history": [1000, 1007, 1014, 1070]"#;
    block_on(async {
        // k absent or null = defaults.k, both formats; sent = as sent.
        for body in [
            format!("{{{history}}}"),
            format!(r#"{{{history}, "k": null}}"#),
        ] {
            assert_eq!(ask_json(addr, &body).await.0.len(), 3, "{body}");
        }
        let five = format!(r#"{{{history}, "k": 5}}"#);
        assert_eq!(ask_json(addr, &five).await.0.len(), 5);
        let mut c = grpc(addr).await;
        let q = RecommendRequest {
            history: vec![1000, 1007, 1014, 1070],
            ..Default::default()
        };
        assert_eq!(ask(&mut c, q).await.0.len(), 3);
        // A body over max_body_bytes: 413 / OUT_OF_RANGE.
        let big = format!(r#"{{"history": [{}1]}}"#, "1,".repeat(100));
        let (status, h, text) = post_with(addr, &big, &[]).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{text}");
        assert_eq!(error_of(&h, &text).code, "body_too_large");
        let big = RecommendRequest {
            history: vec![i64::MAX; 30], // 270 bytes
            ..Default::default()
        };
        let e = c.recommend(big).await.unwrap_err();
        assert_eq!(e.code(), Code::OutOfRange, "{e:?}");
    });
}

/// A JSON answer with the response headers.
async fn post_with(
    addr: SocketAddr,
    body: &str,
    headers: &[(&str, &str)],
) -> (StatusCode, axum::http::HeaderMap, String) {
    let mut c = connect(addr).await.unwrap();
    let body = Some(Bytes::from(body.to_string()));
    let (status, h, b) = send_with(&mut c, "/v1/recommend", body, headers)
        .await
        .unwrap();
    (status, h, String::from_utf8(b.to_vec()).unwrap())
}

#[test]
fn context_and_history_counts() {
    // max_history 20: 25 ids sent, the oldest 5 cut (an unknown id among
    // them not counted); among the 20 used, 1420 (twice) and 5 are unknown.
    let addr = start(config(1), false);
    let mut ids: Vec<i64> = (0..17).map(|i| 1000 + 7 * i).collect();
    ids.splice(0..0, [1427, 1000, 1007, 1014, 1021]);
    ids.extend([1420, 5, 1420]);
    assert_eq!(ids.len(), 25);
    let body = serde_json::json!({ "history": ids, "k": 7, "sorted": false }).to_string();
    block_on(async {
        let (status, h, text) = post_with(addr, &body, &[("x-request-id", "abc-1")]).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let got: Reply = serde_json::from_str(&text).unwrap();
        let history = History {
            sent: 25,
            truncated: 5,
            unknown: 3,
        };
        assert_eq!(got.model.name, "knn-test");
        assert_eq!(got.request_id, "abc-1");
        assert_eq!((got.k, got.sorted, &got.view), (7, false, &None));
        assert_eq!(got.history, history);
        let timing = h["server-timing"].to_str().unwrap();
        assert!(timing.starts_with("compute;dur="), "{timing}");
        timing["compute;dur=".len()..].parse::<f64>().unwrap();
        // gRPC: the same context, the id from the metadata.
        let mut c = grpc(addr).await;
        let mut q = tonic::Request::new(proto_from_json(body.as_bytes()).unwrap());
        q.metadata_mut()
            .insert("x-request-id", "abc-2".parse().unwrap());
        let r = c.recommend(q).await.unwrap();
        assert_eq!(r.metadata().get("x-request-id").unwrap(), "abc-2");
        let r = r.into_inner();
        assert_eq!(r.model.unwrap().name, "knn-test");
        assert_eq!(
            (r.request_id.as_str(), r.k, r.sorted, r.view),
            ("abc-2", 7, false, None)
        );
        let h = r.history.unwrap();
        assert_eq!((h.sent, h.truncated, h.unknown), (25, 5, 3));
        // Exactly max_history ids: nothing cut.
        let at_max = serde_json::json!({ "history": &ids[5..] }).to_string();
        let (_, _, text) = post_with(addr, &at_max, &[]).await;
        let got: Reply = serde_json::from_str(&text).unwrap();
        assert_eq!(
            (got.history.sent, got.history.truncated, got.history.unknown),
            (20, 0, 3)
        );
        // Defaults applied: k 10, sorted true; nothing cut, nothing unknown.
        let (_, _, text) = post_with(addr, r#"{"history": [1000, 1007]}"#, &[]).await;
        let got: Reply = serde_json::from_str(&text).unwrap();
        assert_eq!((got.k, got.sorted), (10, true));
        assert_eq!(
            (got.history.sent, got.history.truncated, got.history.unknown),
            (2, 0, 0)
        );
    });
}

#[test]
fn request_id_kept_if_printable_else_generated() {
    let addr = start(config(2), false);
    let body = r#"{"history": [1000, 1007]}"#;
    let long = "a".repeat(129);
    block_on(async {
        let ask = |id: Option<&'static str>| async move {
            let h: Vec<(&str, &str)> = id.map(|v| ("x-request-id", v)).into_iter().collect();
            let (status, h, text) = post_with(addr, body, &h).await;
            assert_eq!(status, StatusCode::OK, "{text}");
            let got: Reply = serde_json::from_str(&text).unwrap();
            assert_eq!(h["x-request-id"], got.request_id[..]);
            got.request_id
        };
        let max = "b".repeat(128).leak();
        assert_eq!(ask(Some("req-42/x")).await, "req-42/x");
        assert_eq!(ask(Some(max)).await, *max);
        // Generated: <16 hex>-<thread>-<counter>, one process part, all distinct.
        let mut seen = vec![];
        for id in [
            None,
            Some("has space"),
            Some(long.clone().leak()),
            Some(""),
            Some("tab\tx"),
        ] {
            seen.push(ask(id).await);
        }
        for id in &seen {
            let p: Vec<&str> = id.split('-').collect();
            assert_eq!(p.len(), 3, "{id}");
            assert!(
                p[0].len() == 16 && p[0].bytes().all(|b| b.is_ascii_hexdigit()),
                "{id}"
            );
            assert!(p[0] == seen[0].split('-').next().unwrap(), "{id}");
            p[1].parse::<usize>().unwrap();
            p[2].parse::<u64>().unwrap();
        }
        let mut distinct = seen.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), seen.len(), "{seen:?}");
        // Errors carry it too: JSON header, gRPC status metadata.
        let (status, h, text) = post_with(
            addr,
            r#"{"history": [], "k": 51}"#,
            &[("x-request-id", "e1")],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error_of(&h, &text).request_id, "e1");
        let mut c = grpc(addr).await;
        let mut q = tonic::Request::new(RecommendRequest {
            k: Some(51),
            ..Default::default()
        });
        q.metadata_mut()
            .insert("x-request-id", "e2".parse().unwrap());
        let e = c.recommend(q).await.unwrap_err();
        assert_eq!(e.metadata().get("x-request-id").unwrap(), "e2");
    });
}

#[test]
fn same_request_same_bytes() {
    // U5: ties go to the lower internal id, fixed per model: the same request
    // (and request id) gets the same bytes, on either thread, sorted or not.
    let addr = start(config(2), false);
    block_on(async {
        for sorted in [true, false] {
            let body = serde_json::json!({
                "history": [1000, 1007, 1014, 1070, 1077, 1300], "k": 40, "sorted": sorted,
            })
            .to_string();
            let h = [("x-request-id", "same")];
            let mut answers = vec![];
            for _ in 0..6 {
                // a new connection each: the acceptor deals them round-robin
                let (status, _, text) = post_with(addr, &body, &h).await;
                assert_eq!(status, StatusCode::OK, "{text}");
                answers.push(text);
            }
            assert!(answers.iter().all(|a| *a == answers[0]), "{answers:?}");
            let mut grpc_answers = vec![];
            for _ in 0..4 {
                // a channel (connection) each: both threads
                let mut c = grpc(addr).await;
                let mut q = tonic::Request::new(proto_from_json(body.as_bytes()).unwrap());
                q.metadata_mut()
                    .insert("x-request-id", "same".parse().unwrap());
                grpc_answers.push(c.recommend(q).await.unwrap().into_inner());
            }
            assert!(grpc_answers.iter().all(|a| *a == grpc_answers[0]));
        }
    });
}

/// An id file in the temp dir (one id per line), removed on drop.
struct IdFile(std::path::PathBuf);

impl Drop for IdFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn id_file(name: &str, text: &str) -> IdFile {
    // Tests run in parallel: a name per call.
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let p = std::env::temp_dir().join(format!("similaripy-http-{pid}-{n}-{name}.txt"));
    std::fs::write(&p, text).unwrap();
    IdFile(p)
}

fn lines(ids: impl IntoIterator<Item = i64>) -> String {
    ids.into_iter().map(|i| format!("{i}\n")).collect()
}

/// Views over `model()`: `even` = even items (+ two ids it doesn't have),
/// `head` = all but the first 10, `both` = even minus the first 10.
fn views_config(threads: usize) -> (Config, [IdFile; 2]) {
    let even = id_file(
        "even",
        &format!(
            "{}5\n\n 6 \n",
            lines((0..N as i64).step_by(2).map(|i| 1000 + 7 * i))
        ),
    );
    let head = id_file("head", &lines((0..10).map(|i| 1000 + 7 * i)));
    let mut c = config(threads);
    let view = |inc: Option<&IdFile>, exc: Option<&IdFile>| config::ViewConfig {
        include_file: inc.map(|f| f.0.clone()),
        exclude_file: exc.map(|f| f.0.clone()),
    };
    c.views = [
        ("even".to_string(), view(Some(&even), None)),
        ("head".to_string(), view(None, Some(&head))),
        ("both".to_string(), view(Some(&even), Some(&head))),
    ]
    .into();
    (c, [even, head])
}

/// The core view each name stands for.
fn core_view(m: &Model, name: &str, files: &[IdFile; 2]) -> Model {
    let ids = |f: &IdFile| read_ids(&f.0);
    match name {
        "" => m.view(None, None),
        "even" => m.view(Some(&ids(&files[0])), None),
        "head" => m.view(None, Some(&ids(&files[1]))),
        "both" => m.view(Some(&ids(&files[0])), Some(&ids(&files[1]))),
        _ => unreachable!(),
    }
}

fn read_ids(p: &std::path::Path) -> Vec<i64> {
    std::fs::read_to_string(p)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect()
}

#[test]
fn view_answers_equal_core_view_and_and_with_include_exclude() {
    let (cfg, files) = views_config(2);
    let addr = start(cfg, false);
    let reference = model();
    let mut rng = Rng(7);
    block_on(async {
        let mut c = grpc(addr).await;
        for case in 0..300 {
            let len = 1 + rng.below(20);
            let ids: Vec<i64> = (0..len).map(|_| 1000 + rng.below(N) as i64 * 7).collect();
            let include: Option<Vec<i64>> =
                (case % 3 == 1).then(|| (0..30).map(|_| 1000 + rng.below(N) as i64 * 7).collect());
            let exclude: Option<Vec<i64>> =
                (case % 4 == 2).then(|| (0..8).map(|_| 1000 + rng.below(N) as i64 * 7).collect());
            let name = ["", "even", "head", "both"][case % 4];
            let k = 1 + rng.below(50);
            let body = serde_json::json!({
                "history": ids, "k": k, "include": include, "exclude": exclude, "view": name,
            })
            .to_string();
            let want = python(
                &core_view(&reference, name, &files),
                &Request {
                    ids: &ids,
                    weights: None,
                    k,
                    exclude_seen: true,
                    include: include.as_deref(),
                    exclude: exclude.as_deref(),
                    sorted: true,
                },
            );
            let (status, text) = post(addr, &body).await;
            assert_eq!(status, StatusCode::OK, "{text}");
            let got: Reply = serde_json::from_str(&text).unwrap();
            assert_eq!(got.answer(), want, "JSON case {case}: {body}");
            let used = (!name.is_empty()).then(|| name.to_string());
            assert_eq!(got.view, used, "{body}");
            let q = proto_from_json(body.as_bytes()).unwrap();
            let r = c.recommend(q).await.unwrap().into_inner();
            let bits = r.scores.iter().map(|s| s.to_bits()).collect();
            assert_eq!((r.ids, bits), want, "gRPC case {case}: {body}");
            assert_eq!(r.view, used);
        }
    });
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Views {
    default: Option<String>,
    items: usize,
    views: std::collections::BTreeMap<String, ViewInfo>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ViewInfo {
    allowed: usize,
    include: Option<FileInfo>,
    exclude: Option<FileInfo>,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
struct FileInfo {
    file: String,
    ids: usize,
    known: usize,
}

#[test]
fn default_view_empty_name_unknown_name_and_views_listing() {
    let (mut cfg, files) = views_config(1);
    cfg.defaults.view = "even".into();
    let addr = start(cfg, false);
    let reference = model();
    let ids = [1000i64, 1014, 1021];
    let req = |m: &Model| {
        python(
            m,
            &Request {
                ids: &ids,
                weights: None,
                k: 10,
                exclude_seen: true,
                include: None,
                exclude: None,
                sorted: true,
            },
        )
    };
    block_on(async {
        // Absent and null = defaults.view; "" = the whole model.
        for body in [
            r#"{"history": [1000, 1014, 1021]}"#,
            r#"{"history": [1000, 1014, 1021], "view": null}"#,
        ] {
            let (_, text) = post(addr, body).await;
            let got: Reply = serde_json::from_str(&text).unwrap();
            assert_eq!(got.view.as_deref(), Some("even"));
            assert_eq!(got.answer(), req(&core_view(&reference, "even", &files)));
        }
        let (_, text) = post(addr, r#"{"history": [1000, 1014, 1021], "view": ""}"#).await;
        let got: Reply = serde_json::from_str(&text).unwrap();
        assert_eq!((got.answer(), got.view), (req(&reference), None));
        let mut c = grpc(addr).await;
        let q = RecommendRequest {
            history: ids.to_vec(),
            ..Default::default()
        };
        let r = c.recommend(q.clone()).await.unwrap().into_inner();
        assert_eq!(r.view.as_deref(), Some("even"));
        let whole = RecommendRequest {
            view: Some(String::new()),
            ..q.clone()
        };
        let r = c.recommend(whole).await.unwrap().into_inner();
        let bits = r.scores.iter().map(|s| s.to_bits()).collect();
        assert_eq!((r.view, (r.ids, bits)), (None, req(&reference)));
        // Unknown: 400 / INVALID_ARGUMENT, counted as a bad request; a
        // history past max_history is rejected before the cut (not counted
        // as truncated).
        let long: Vec<i64> = (0..30).map(|i| 1000 + 7 * i).collect();
        let body = serde_json::json!({"history": long, "view": "nope"}).to_string();
        let (status, h, text) = post_with(addr, &body, &[]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let e = error_of(&h, &text);
        assert_eq!(e.code, "unknown_view", "{text}");
        assert!(e.message.contains("unknown view \"nope\""), "{text}");
        let e = c
            .recommend(RecommendRequest {
                view: Some("nope".into()),
                ..q
            })
            .await
            .unwrap_err();
        assert_eq!(e.code(), Code::InvalidArgument);
        assert_eq!(grpc_code(&e), "unknown_view", "{e:?}");
        let m = get_text(addr, "/metrics").await;
        assert_eq!(
            metric(&m, "similaripy_server_bad_requests_total"),
            Some(2.0)
        );
        assert_eq!(metric(&m, "similaripy_server_truncated_total"), Some(0.0));
        // even: 2 JSON + 1 gRPC; "" and unknown names aren't counted.
        let even = "similaripy_server_view_requests_total{view=\"even\"}";
        assert_eq!(metric(&m, even), Some(3.0), "{m}");
        let head = "similaripy_server_view_requests_total{view=\"head\"}";
        assert_eq!(metric(&m, head), Some(0.0));
        // The listing.
        let v: Views = serde_json::from_str(&get_text(addr, "/v1/views").await).unwrap();
        assert_eq!((v.default.as_deref(), v.items), (Some("even"), N));
        assert_eq!(v.views.keys().collect::<Vec<_>>(), ["both", "even", "head"]);
        let file = |f: &IdFile, ids, known| FileInfo {
            file: f.0.display().to_string(),
            ids,
            known,
        };
        let even = &v.views["even"];
        assert_eq!(even.allowed, N / 2);
        assert_eq!(even.include, Some(file(&files[0], N / 2 + 2, N / 2)));
        assert_eq!(even.exclude, None);
        assert_eq!(v.views["head"].allowed, N - 10);
        assert_eq!(v.views["head"].exclude, Some(file(&files[1], 10, 10)));
        assert_eq!(v.views["both"].allowed, N / 2 - 5);
    });
}

#[test]
fn swap_serves_new_views_from_the_next_request() {
    let (cfg, files) = views_config(2);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Arc::new(App::new(model(), cfg).unwrap());
    let r = router(app.clone());
    let served = app.clone();
    std::thread::spawn(move || serve(listener, served, r, std::future::pending()));
    let body = r#"{"history": [1000, 1014, 1021], "k": 50, "view": "even"}"#;
    block_on(async {
        let before = ask_json(addr, body).await;
        assert!(before.0.iter().all(|id| (id - 1000) / 7 % 2 == 0));
        // The file now holds the odd items: served only after the swap.
        std::fs::write(
            &files[0].0,
            lines((1..N as i64).step_by(2).map(|i| 1000 + 7 * i)),
        )
        .unwrap();
        assert_eq!(ask_json(addr, body).await, before);
        let even = "similaripy_server_view_requests_total{view=\"even\"}";
        assert_eq!(metric(&get_text(addr, "/metrics").await, even), Some(2.0));
        app.swap(model()).unwrap();
        // Counters survive the swap.
        assert_eq!(metric(&get_text(addr, "/metrics").await, even), Some(2.0));
        // Both threads (round-robin per connection) take the new state.
        for _ in 0..4 {
            let after = ask_json(addr, body).await;
            assert!(!after.0.is_empty());
            assert!(
                after.0.iter().all(|id| (id - 1000) / 7 % 2 == 1),
                "{after:?}"
            );
        }
        // A broken file: the swap fails, the served state stays.
        std::fs::write(&files[0].0, "1007\nx\n").unwrap();
        let e = app.swap(model()).unwrap_err();
        assert!(e.contains(":2:") && e.contains("\"x\""), "{e}");
        assert!(
            ask_json(addr, body)
                .await
                .0
                .iter()
                .all(|id| (id - 1000) / 7 % 2 == 1)
        );
    });
}

#[test]
fn view_errors_at_start_name_view_file_and_line() {
    let (mut cfg, files) = views_config(1);
    std::fs::write(&files[1].0, "1000\n1007\n\nten\n").unwrap();
    let e = App::new(model(), cfg.clone()).err().unwrap();
    let at = format!("{}:4:", files[1].0.display());
    assert!(
        e.contains("\"both\"") && e.contains(&at) && e.contains("ten"),
        "{e}"
    );
    cfg.views.get_mut("head").unwrap().exclude_file = Some("/nonexistent/ids.txt".into());
    cfg.views.remove("both");
    let e = App::new(model(), cfg.clone()).err().unwrap();
    assert!(
        e.contains("\"head\"") && e.contains("/nonexistent/ids.txt"),
        "{e}"
    );
    cfg.views.remove("head");
    cfg.defaults.view = "head".into();
    let e = App::new(model(), cfg.clone()).err().unwrap();
    assert!(e.contains("defaults.view"), "{e}");
    cfg.defaults.view = String::new();
    // Not settable as a flag / env var (a dot splits the key, env is
    // lowercase), or not a plain label.
    for bad in ["bad name", "in.stock", "InStock", ""] {
        let mut c = cfg.clone();
        c.views.insert(bad.into(), cfg.views["even"].clone());
        let e = App::new(model(), c).err().unwrap();
        assert!(e.contains(&format!("{bad:?}")), "{e}");
    }
    cfg.views.insert("none".into(), Default::default());
    assert!(
        App::new(model(), cfg)
            .err()
            .unwrap()
            .contains("needs include_file")
    );
}

// ---- S5d hardening ----

/// A server with test routes `/slow` (400 ms) and `/hang` (30 s), stopped
/// by sending on the returned channel; the handle gives `serve`'s result.
fn start_stoppable(
    cfg: Config,
) -> (
    SocketAddr,
    tokio::sync::oneshot::Sender<()>,
    std::thread::JoinHandle<std::io::Result<()>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Arc::new(App::new(model(), cfg).unwrap());
    let sleep = |ms| {
        move || async move {
            tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
            "slept"
        }
    };
    let r = router(app.clone())
        .route("/slow", get(sleep(400)))
        .route("/hang", get(sleep(30_000)));
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let h = std::thread::spawn(move || {
        serve(listener, app, r, async {
            let _ = stopped.await;
        })
    });
    (addr, stop, h)
}

fn timeouts(threads: usize, header_read: f64, drain: f64, grace: f64) -> Config {
    let mut c = config(threads);
    c.timeouts.header_read = header_read;
    c.timeouts.drain_delay = drain;
    c.timeouts.shutdown_grace = grace;
    c
}

/// A raw connection (HTTP/1.1 by hand: the closing is what's tested).
fn raw(addr: SocketAddr) -> std::net::TcpStream {
    let s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    s
}

/// Everything until the server closes (a reset counts as closed; a read
/// timeout fails the test).
fn read_to_close(s: &mut std::net::TcpStream) -> String {
    use std::io::{ErrorKind, Read};
    let mut buf = vec![];
    match s.read_to_end(&mut buf) {
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::ConnectionReset => {}
        Err(e) => panic!("not closed: {e} after {:?}", String::from_utf8_lossy(&buf)),
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn get_raw(addr: SocketAddr, path: &str) -> std::net::TcpStream {
    use std::io::Write;
    let mut s = raw(addr);
    write!(s, "GET {path} HTTP/1.1\r\nHost: t\r\n\r\n").unwrap();
    s
}

/// Waits (up to 3 s) until a connect is refused: the listener is closed.
fn wait_refused(addr: SocketAddr) {
    let t = std::time::Instant::now();
    while std::net::TcpStream::connect(addr).is_ok() {
        assert!(t.elapsed().as_secs() < 3, "still accepting");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn shutdown_finishes_in_flight_requests_and_refuses_new_connections() {
    let (addr, stop, h) = start_stoppable(timeouts(2, 30.0, 0.0, 5.0));
    let mut slow = get_raw(addr, "/slow");
    // An idle keep-alive connection: closed at once by the shutdown
    let mut idle = get_raw(addr, "/health");
    std::thread::sleep(std::time::Duration::from_millis(100));
    let t = std::time::Instant::now();
    stop.send(()).unwrap();
    wait_refused(addr);
    // The response in flight completes, then the connection closes.
    let text = read_to_close(&mut slow);
    assert!(
        text.starts_with("HTTP/1.1 200") && text.ends_with("slept"),
        "{text}"
    );
    assert!(read_to_close(&mut idle).ends_with("ok"));
    h.join().unwrap().unwrap();
    // Within the grace: done when the slow request was (~0.3 s left)
    assert!(t.elapsed().as_secs_f64() < 2.0, "{:?}", t.elapsed());
}

#[test]
fn drain_delay_keeps_accepting_with_ready_503() {
    let (addr, stop, h) = start_stoppable(timeouts(1, 30.0, 1.5, 5.0));
    assert!(read_head(&mut get_raw(addr, "/ready")).ends_with("ok"));
    let t = std::time::Instant::now();
    stop.send(()).unwrap();
    // Draining: new connections still served; /ready 503, /health ok
    let ready = loop {
        let text = read_head(&mut get_raw(addr, "/ready"));
        if text.starts_with("HTTP/1.1 503") {
            break text;
        }
        assert!(t.elapsed().as_secs_f64() < 1.0, "{text}");
    };
    assert!(ready.ends_with("draining"), "{ready}");
    assert!(read_head(&mut get_raw(addr, "/health")).ends_with("ok"));
    wait_refused(addr);
    assert!(t.elapsed().as_secs_f64() >= 1.4, "{:?}", t.elapsed());
    h.join().unwrap().unwrap();
}

/// The response's head and body as far as `Content-Length` says (the
/// connection stays open).
fn read_head(s: &mut std::net::TcpStream) -> String {
    use std::io::Read;
    let mut buf = vec![];
    let mut b = [0u8; 1024];
    loop {
        let n = s.read(&mut b).unwrap();
        assert!(n > 0, "closed: {:?}", String::from_utf8_lossy(&buf));
        buf.extend_from_slice(&b[..n]);
        let text = String::from_utf8_lossy(&buf).into_owned();
        if let Some((head, body)) = text.split_once("\r\n\r\n") {
            let len: usize = head
                .lines()
                .find_map(|l| {
                    l.to_lowercase()
                        .strip_prefix("content-length: ")
                        .map(|v| v.parse().unwrap())
                })
                .unwrap();
            if body.len() >= len {
                return text;
            }
        }
    }
}

#[test]
fn shutdown_grace_closes_what_is_left() {
    let (addr, stop, h) = start_stoppable(timeouts(1, 30.0, 0.0, 0.3));
    let mut hang = get_raw(addr, "/hang");
    std::thread::sleep(std::time::Duration::from_millis(100));
    let t = std::time::Instant::now();
    stop.send(()).unwrap();
    h.join().unwrap().unwrap();
    let took = t.elapsed().as_secs_f64();
    assert!((0.25..2.0).contains(&took), "{took}");
    // Closed with no response
    assert_eq!(read_to_close(&mut hang), "");
}

#[test]
fn slow_silent_and_idle_connections_closed_after_header_read() {
    use std::io::Write;
    let (addr, _stop, _h) = start_stoppable(timeouts(2, 0.3, 0.0, 5.0));
    let cases: [(&str, &[u8]); 4] = [
        ("silent", b""),
        ("one preface byte", b"P"),
        ("most of h2's preface", b"PRI * HTTP/2.0\r\n"),
        (
            "half the headers",
            b"POST /v1/recommend HTTP/1.1\r\nHost: t\r\n",
        ),
    ];
    for (name, sent) in cases {
        let mut s = raw(addr);
        s.write_all(sent).unwrap();
        let t = std::time::Instant::now();
        read_to_close(&mut s);
        let took = t.elapsed().as_secs_f64();
        assert!((0.25..2.5).contains(&took), "{name}: closed after {took}");
    }
    // A keep-alive connection: served, then closed when idle past header_read
    let mut s = get_raw(addr, "/health");
    assert!(read_head(&mut s).ends_with("ok"));
    let t = std::time::Instant::now();
    assert_eq!(read_to_close(&mut s), "");
    let took = t.elapsed().as_secs_f64();
    assert!((0.25..2.5).contains(&took), "idle: closed after {took}");
    // Requests in time keep a connection open past header_read
    let mut s = get_raw(addr, "/health");
    for _ in 0..3 {
        assert!(read_head(&mut s).ends_with("ok"));
        std::thread::sleep(std::time::Duration::from_millis(200));
        write!(s, "GET /health HTTP/1.1\r\nHost: t\r\n\r\n").unwrap();
    }
    assert!(read_head(&mut s).ends_with("ok"));
    // h2 (preface + an empty SETTINGS frame) without a request yet: stays
    // open like an idle gRPC channel
    let mut h2 = raw(addr);
    h2.write_all(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n").unwrap();
    h2.write_all(&[0, 0, 0, 4, 0, 0, 0, 0, 0]).unwrap();
    assert!(open_for(&mut h2, std::time::Duration::from_millis(900)));
}

/// True if `s` is still open after `d` (bytes the server sends are read).
fn open_for(s: &mut std::net::TcpStream, d: std::time::Duration) -> bool {
    use std::io::{ErrorKind, Read};
    let end = std::time::Instant::now() + d;
    let mut b = [0u8; 1024];
    loop {
        let left = end.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return true;
        }
        s.set_read_timeout(Some(left)).unwrap();
        match s.read(&mut b) {
            Ok(0) => return false,
            Ok(_) => {}
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return true;
            }
            Err(_) => return false,
        }
    }
}

#[test]
fn json_body_not_sent_in_time_is_408() {
    use std::io::Write;
    let (addr, _stop, _h) = start_stoppable(timeouts(1, 0.3, 0.0, 5.0));
    let mut s = raw(addr);
    let head = "POST /v1/recommend HTTP/1.1\r\nHost: t\r\nContent-Length: 100\r\n\r\n";
    write!(s, "{head}{{\"ids\": [").unwrap();
    let t = std::time::Instant::now();
    let text = read_head(&mut s);
    let took = t.elapsed().as_secs_f64();
    assert!(text.starts_with("HTTP/1.1 408"), "{text}");
    assert!(text.to_lowercase().contains("x-request-id: "), "{text}");
    assert!(text.contains(r#""code":"body_timeout""#), "{text}");
    assert!((0.25..2.5).contains(&took), "{took}");
    let m = read_head(&mut get_raw(addr, "/metrics"));
    assert_eq!(
        metric(&m, "similaripy_server_bad_requests_total"),
        Some(1.0),
        "{m}"
    );
}

#[test]
fn declared_body_over_the_limit_is_413_at_once() {
    use std::io::Write;
    // A 408 would take header_read (5 s): a 413 comes before the body.
    let mut cfg = timeouts(1, 5.0, 0.0, 5.0);
    cfg.limits.max_body_bytes = 100;
    let (addr, _stop, _h) = start_stoppable(cfg);
    let mut s = raw(addr);
    let n = 101;
    let head = format!("POST /v1/recommend HTTP/1.1\r\nHost: t\r\nContent-Length: {n}\r\n\r\n");
    write!(s, "{head}{{\"history\": [").unwrap();
    let t = std::time::Instant::now();
    let text = read_head(&mut s);
    assert!(text.starts_with("HTTP/1.1 413"), "{text}");
    assert!(text.contains(r#""code":"body_too_large""#), "{text}");
    assert!(text.to_lowercase().contains("x-request-id: "), "{text}");
    assert!(t.elapsed().as_secs_f64() < 2.0, "{:?}", t.elapsed());
    // No Content-Length (chunked): axum's limit as it streams, same answer.
    let mut s = raw(addr);
    let chunk = "1".repeat(200);
    let head = "POST /v1/recommend HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n";
    write!(s, "{head}c8\r\n{chunk}\r\n").unwrap();
    let text = read_head(&mut s);
    assert!(text.starts_with("HTTP/1.1 413"), "{text}");
    assert!(text.contains(r#""code":"body_too_large""#), "{text}");
    // Expect: 100-continue (curl, bodies over 1 MiB): the 413, and the
    // connection closed at once (no drain waiting for a body never sent).
    let mut s = raw(addr);
    let head = "POST /v1/recommend HTTP/1.1\r\nHost: t\r\nContent-Length: 5000\r\n\
        Expect: 100-continue\r\n\r\n";
    write!(s, "{head}").unwrap();
    let t = std::time::Instant::now();
    let text = read_to_close(&mut s);
    assert!(text.starts_with("HTTP/1.1 413"), "{text}");
    assert!(t.elapsed().as_secs_f64() < 2.0, "{:?}", t.elapsed());
}

/// The binary on a saved model: SIGTERM → logs, exit 0 within the grace.
#[cfg(unix)]
#[test]
fn binary_exits_0_on_sigterm() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let dir = std::env::temp_dir().join(format!("similaripy-server-bin-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    model().save(&dir.join("v1"), "test").unwrap();
    // Served through a symlink: loaded from the directory it names (V22).
    std::os::unix::fs::symlink("v1", dir.join("current")).unwrap();
    /// Killed and its model removed however the test ends.
    struct Child(std::process::Child, std::path::PathBuf);
    impl Drop for Child {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
            let _ = std::fs::remove_dir_all(&self.1);
        }
    }
    let link = dir.join("current");
    let resolved = dir.join("v1").canonicalize().unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_similaripy-server"))
        .args(["--model", link.to_str().unwrap(), "--addr", "127.0.0.1:0"])
        .args(["--threads", "1", "--timeouts.shutdown_grace", "3"])
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut child = Child(child, dir);
    let child = &mut child.0;
    let mut log = BufReader::new(child.stderr.take().unwrap()).lines();
    let addr: SocketAddr = loop {
        let line = log.next().expect("a listening line").unwrap();
        if let Some((_, a)) = line.split_once("listening on ") {
            let shown = format!("= {} (", resolved.display());
            assert!(line.contains(&shown), "{line}");
            break a.trim().parse().unwrap();
        }
    };
    assert!(read_head(&mut get_raw(addr, "/ready")).ends_with("ok"));
    let mut idle = get_raw(addr, "/health");
    assert!(read_head(&mut idle).ends_with("ok"));
    let t = std::time::Instant::now();
    // SAFETY: a signal to our own child
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(t.elapsed().as_secs() < 3, "no exit within the grace");
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let rest: Vec<String> = log.map_while(Result::ok).collect();
    assert_eq!(status.code(), Some(0), "{rest:?}");
    assert!(rest.iter().any(|l| l.contains("SIGTERM")), "{rest:?}");
    assert!(rest.iter().any(|l| l.contains("stopped")), "{rest:?}");
}

#[test]
fn oversized_bodies_get_their_413_and_the_connection_keeps_serving() {
    // The server answers 413 at once and reads the rest of the body after
    // it: a client still sending gets the 413, not a reset (before: about
    // 1 run in 5 of edge_rejections), and the connection serves on.
    let mut cfg = config(1);
    cfg.limits.max_body_bytes = 4096;
    let addr = start(cfg, false);
    block_on(async {
        let mut c = connect(addr).await.unwrap();
        // About 4x the limit: well within the drain's cap (8x).
        let big = Bytes::from(format!(r#"{{"history": [{}1]}}"#, "1,".repeat(2 * 4096)));
        for i in 0..50 {
            let (status, h, b) = send_with(&mut c, "/v1/recommend", Some(big.clone()), &[])
                .await
                .unwrap_or_else(|e| panic!("round {i}: {e}"));
            assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "round {i}");
            let e = error_of(&h, std::str::from_utf8(&b).unwrap());
            assert_eq!(e.code, "body_too_large");
            let ok = Some(Bytes::from_static(br#"{"history": [1000]}"#));
            let (status, _, _) = send_with(&mut c, "/v1/recommend", ok, &[]).await.unwrap();
            assert_eq!(
                status,
                StatusCode::OK,
                "round {i}: same connection after a 413"
            );
        }
    });
}
