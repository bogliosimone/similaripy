//! The server over real HTTP on a loopback port: responses equal core
//! `serve::recommend` as the Python binding calls it (block resolved for
//! every core busy; the server's block differs on purpose: same bits), JSON
//! f32 parsed as Python + numpy do, edge rejections, a panicking connection,
//! metrics.

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

use axum::http::StatusCode;
use axum::routing::get;
use bytes::Bytes;
use serde::Deserialize;
use similaripy_core::CsrView;
use similaripy_core::serve::{BlockSize, ItemOrder, Model, Request, recommend};
use similaripy_server::client::{connect, send};
use similaripy_server::{App, BODY_LIMIT, Config, router, serve};

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
    Config {
        threads,
        // Several blocks here, one block in the Python reference: same bits.
        block_size: BlockSize::Size(16),
        max_history: 20,
        k_max: 50,
    }
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

#[derive(Deserialize, Debug)]
struct Reply {
    ids: Vec<i64>,
    scores: Vec<f32>,
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
    let (status, b) = send(&mut c, "/recommend", Some(Bytes::from(body.to_string())))
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
            let bits: Vec<u32> = got.scores.iter().map(|s| s.to_bits()).collect();
            assert_eq!((got.ids, bits), want, "case {case}: {body}");
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
        let bits: Vec<u32> = got.scores.iter().map(|s| s.to_bits()).collect();
        let req = Request {
            ids: &ids,
            weights: None,
            k: 10,
            exclude_seen: true,
            include: None,
            exclude: None,
            sorted: true,
        };
        assert_eq!((got.ids, bits), python(&reference, &req));
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
                let bits: Vec<u32> = got.scores.iter().map(|s| s.to_bits()).collect();
                assert_eq!((got.ids, bits), python(&reference, &req), "{s}");
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
            "similaripy_server_compute_seconds_count 2\n",
            "similaripy_server_request_seconds_bucket{le=\"+Inf\"} 2\n",
        ] {
            assert!(m.contains(line), "{line:?} not in\n{m}");
        }
    });
}

async fn panics() -> &'static str {
    panic!("a handler panic (test)")
}
