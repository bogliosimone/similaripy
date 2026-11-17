//! The server over real HTTP on a loopback port: responses equal core
//! `serve::recommend` as the Python binding calls it (block resolved for
//! every core busy; the server's block differs on purpose: same bits), JSON
//! f32 parsed as Python + numpy do, edge rejections, a panicking connection,
//! metrics; gRPC on the same port: the same answers (ids and f32 bits) as
//! JSON, proto3 presence rules, errors as status codes; the answer's context
//! (model, request id, k, sorted, history counts) and same request -> same
//! bytes (U5).

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
    std::thread::spawn(move || serve(listener, app, r));
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

impl Reply {
    /// (ids, score bits)
    fn answer(&self) -> (Vec<i64>, Vec<u32>) {
        self.items.iter().map(|x| (x.id, x.score.to_bits())).unzip()
    }
}

/// Python's call: `ItemKNN.recommend` → `serve::recommend` with the block
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
                "ids": ids, "weights": weights, "k": k, "exclude_seen": exclude_seen,
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
        let (status, text) = post(addr, r#"{"ids": [1000, 1007, 1014, 1070]}"#).await;
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
                let body = format!(r#"{{"ids": [1000], "weights": [{s}], "k": 50}}"#);
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
        for (body, status, msg) in [
            (r#"{"ids": [1000], "weight": [2.0]}"#, 400, "unknown field"),
            (r#"{"ids": [1000], "k": 51}"#, 400, "k_max"),
            (
                r#"{"ids": [1000, 1007], "weights": [1.0]}"#,
                400,
                "len(weights)",
            ),
            (r#"{"ids": "x"}"#, 400, "invalid type"),
            (r#"{"k": 3}"#, 400, "missing field"),
        ] {
            let (got, text) = post(addr, body).await;
            assert_eq!(got.as_u16(), status, "{body}: {text}");
            assert!(text.contains(msg), "{body}: {text}");
        }
        let big = format!(r#"{{"ids": [{}1]}}"#, "1,".repeat(BODY_LIMIT / 2));
        let (got, _) = post(addr, &big).await;
        assert_eq!(got, StatusCode::PAYLOAD_TOO_LARGE);
        let m = get_text(addr, "/metrics").await;
        // JSON x 3 + k: the edge; weights: the worker.
        assert!(
            m.contains("similaripy_server_bad_requests_total 4\n"),
            "{m}"
        );
        assert!(m.contains("similaripy_server_errors_total 1\n"), "{m}");
    });
}

#[test]
fn a_panicking_request_leaves_its_thread_serving() {
    let addr = start(config(1), true);
    block_on(async {
        let mut c = connect(addr).await.unwrap();
        assert!(send(&mut c, "/panic", None).await.is_err());
        // The one thread serves the next connection, its worker clean.
        let (status, text) = post(addr, r#"{"ids": [1000, 1007]}"#).await;
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
        post(addr, r#"{"ids": [1000, 1007, 99]}"#).await;
        post(addr, &serde_json::json!({ "ids": long }).to_string()).await;
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
        #[cfg(unix)]
        assert!(
            metric(&m, "process_cpu_seconds_total").is_some_and(|s| s > 0.0),
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
                "ids": ids, "weights": weights, "k": k,
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
            ids: ids.clone(),
            ..Default::default()
        };
        let defaults = ask(&mut c, only_ids.clone()).await;
        assert_eq!(defaults.0.len(), 10);
        assert_eq!(
            defaults,
            ask_json(addr, r#"{"ids": [1000, 1007, 1014, 1070]}"#).await
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
        let json = ask_json(addr, r#"{"ids": [1000, 1007, 1014, 1070], "include": []}"#).await;
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
        for (q, msg) in [
            (
                RecommendRequest {
                    ids: vec![1000],
                    k: Some(51),
                    ..Default::default()
                },
                "k_max",
            ),
            (
                RecommendRequest {
                    ids: vec![1000, 1007],
                    weights: vec![1.0],
                    ..Default::default()
                },
                "len(weights)",
            ),
        ] {
            let e = c.recommend(q).await.unwrap_err();
            assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
            assert!(e.message().contains(msg), "{e:?}");
        }
        // Over the 1 MB limit (tonic's own, not DefaultBodyLimit's).
        let big = RecommendRequest {
            ids: vec![i64::MAX; BODY_LIMIT / 8], // 9 bytes each as a varint
            ..Default::default()
        };
        let e = c.recommend(big).await.unwrap_err();
        assert_eq!(e.code(), Code::OutOfRange, "{e:?}");
        // The connection still serves; JSON on the same port too.
        let q = RecommendRequest {
            ids: vec![1000, 1007],
            ..Default::default()
        };
        assert!(!ask(&mut c, q).await.0.is_empty());
        assert!(
            !ask_json(addr, r#"{"ids": [1000, 1007]}"#)
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
    let body = |extra: &str| format!(r#"{{"ids": {ids:?}, "k": 50{extra}}}"#);
    block_on(async {
        // `sorted` not sent = the server's default (false), both formats.
        let absent = ask_json(addr, &body("")).await;
        assert_eq!(absent, ask_json(addr, &body(r#", "sorted": false"#)).await);
        let sorted = ask_json(addr, &body(r#", "sorted": true"#)).await;
        assert_ne!(absent, sorted);
        let mut c = grpc(addr).await;
        let q = RecommendRequest {
            ids: ids.clone(),
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
        let (status, text) = post(addr, &serde_json::json!({ "ids": long }).to_string()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
        assert!(text.contains("21 history ids > max_history = 20"), "{text}");
        let e = c
            .recommend(RecommendRequest { ids: long, ..q })
            .await
            .unwrap_err();
        assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
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
    let body = serde_json::json!({ "ids": ids, "k": 7, "sorted": false }).to_string();
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
        let at_max = serde_json::json!({ "ids": &ids[5..] }).to_string();
        let (_, _, text) = post_with(addr, &at_max, &[]).await;
        let got: Reply = serde_json::from_str(&text).unwrap();
        assert_eq!(
            (got.history.sent, got.history.truncated, got.history.unknown),
            (20, 0, 3)
        );
        // Defaults applied: k 10, sorted true; nothing cut, nothing unknown.
        let (_, _, text) = post_with(addr, r#"{"ids": [1000, 1007]}"#, &[]).await;
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
    let body = r#"{"ids": [1000, 1007]}"#;
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
        let (status, h, _) =
            post_with(addr, r#"{"ids": [], "k": 51}"#, &[("x-request-id", "e1")]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(h["x-request-id"], "e1");
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
                "ids": [1000, 1007, 1014, 1070, 1077, 1300], "k": 40, "sorted": sorted,
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
                "ids": ids, "k": k, "include": include, "exclude": exclude, "view": name,
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
            r#"{"ids": [1000, 1014, 1021]}"#,
            r#"{"ids": [1000, 1014, 1021], "view": null}"#,
        ] {
            let (_, text) = post(addr, body).await;
            let got: Reply = serde_json::from_str(&text).unwrap();
            assert_eq!(got.view.as_deref(), Some("even"));
            assert_eq!(got.answer(), req(&core_view(&reference, "even", &files)));
        }
        let (_, text) = post(addr, r#"{"ids": [1000, 1014, 1021], "view": ""}"#).await;
        let got: Reply = serde_json::from_str(&text).unwrap();
        assert_eq!((got.answer(), got.view), (req(&reference), None));
        let mut c = grpc(addr).await;
        let q = RecommendRequest {
            ids: ids.to_vec(),
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
        let body = serde_json::json!({"ids": long, "view": "nope"}).to_string();
        let (status, text) = post(addr, &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(text.contains("unknown view \\\"nope\\\""), "{text}");
        let e = c
            .recommend(RecommendRequest {
                view: Some("nope".into()),
                ..q
            })
            .await
            .unwrap_err();
        assert_eq!(e.code(), Code::InvalidArgument);
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
    std::thread::spawn(move || serve(listener, served, r));
    let body = r#"{"ids": [1000, 1014, 1021], "k": 50, "view": "even"}"#;
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
