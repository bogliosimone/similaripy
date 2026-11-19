//! `similaripy-server --model DIR`: serves a `SparseKNN.save` directory.
//! Logs go to stderr. Exit 0 after a SIGTERM / SIGINT shutdown, 1 on any
//! error (config, model, bind, a second signal).

use std::net::TcpListener;
use std::sync::Arc;

use similaripy_core::serve::Model;
use similaripy_server::config::{Cli, parse, version};
use similaripy_server::{App, router, serve};

const USAGE: &str = "\
similaripy-server [--config FILE] [--section.key VALUE ...]
  Experimental. No auth, no TLS: run it on a private network only (it binds
  127.0.0.1 by default; put an ingress / mesh with TLS and auth in front).

Config: built-in defaults < FILE (TOML) < env SIMILARIPY_<SECTION>__<KEY> < flags.
  --model.path DIR            a SparseKNN.save directory (required); a symlink is
                              resolved once at start (then set model.name)
  --model.mmap BOOL           map the model's files; default true
  --model.name NAME           the model name in every answer; default the directory's name
  --server.addr HOST:PORT     default 127.0.0.1:8080
  --server.threads N          one Worker each; default: P-cores (macOS), available CPUs (Linux)
  --server.block_size B       auto | off | N entries; default auto (from threads)
  --limits.k_max N            reject a larger k; default 2000
  --limits.max_history N      default 1000
  --limits.history_policy P   truncate (keep the last max_history ids) | reject; default truncate
  --limits.max_body_bytes N   reject a larger request body (JSON 413, gRPC OUT_OF_RANGE); default 1048576
  --limits.max_items N        reject a score request with more items; default 10000
  --defaults.k N              a request's k when it doesn't send one (<= k_max); default 10
  --defaults.sorted BOOL      a request's sorted when it doesn't send one; default false
  --defaults.view NAME        a request's view when it doesn't send one; default empty (the whole model)
  --views.reload_interval S   seconds between checks of the view files: a changed
                              file is served from the next request on; 0 = read at
                              start only; default 10
  --views.NAME.include_file F  view NAME allows only the ids in F (one per line)
  --views.NAME.exclude_file F  view NAME never returns the ids in F; either or both
                               (NAME: 1-64 of a-z 0-9 _ -; F relative to the working directory)
  --timeouts.header_read S    seconds to send a request's headers (and a JSON body: else 408),
                              the first bytes after connecting, HTTP/1.1 keep-alive idle; default 30
  --timeouts.drain_delay S    after SIGTERM / SIGINT: seconds still accepting with /ready 503
                              (Kubernetes: e.g. 5); default 0
  --timeouts.shutdown_grace S then seconds for open requests to finish; default 10
                              (a second signal exits at once, code 1)
  e.g. SIMILARIPY_LIMITS__K_MAX=500 = --limits.k_max 500 = [limits] k_max = 500
Short forms: --model, --addr, --threads, --block-size, --max-history, --k-max, --no-mmap.
  --print-config   print the resolved config (TOML) and exit
  --version, --help";

fn main() {
    similaripy_server::mark_start();
    if let Err(e) = run() {
        eprintln!("similaripy-server: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    // Non-UTF-8 env vars can't be config (names are ASCII): skipped.
    let env = std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)));
    let cli =
        parse(std::env::args().skip(1), env).map_err(|e| format!("{e} (--help for usage)"))?;
    let cfg = match cli {
        Cli::Help => {
            println!("{USAGE}");
            return Ok(());
        }
        Cli::Version => {
            println!("{}", version());
            return Ok(());
        }
        Cli::PrintConfig(cfg) => {
            print!("{}", cfg.to_toml());
            return Ok(());
        }
        Cli::Run(cfg) => cfg,
    };
    eprint!("{}\nconfig:\n{}", version(), cfg.to_toml());
    let addr = cfg.server.addr.clone();
    // Resolved once: each file is opened through `dir`, so a symlink flipped
    // to another version mid-load would mix two models' files (V22).
    let path = &cfg.model.path;
    let dir = path
        .canonicalize()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let t = std::time::Instant::now();
    let model = Model::load(&dir, cfg.model.mmap).map_err(|e| e.to_string())?;
    let loaded = t.elapsed();
    // Which similaripy saved it: the start-up log only (V6 revised)
    let written_by = std::fs::read_to_string(dir.join("meta.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|m| m["written_by"].as_str().map(String::from))
        .unwrap_or_else(|| "?".into());
    if cfg.model.name == "model" {
        eprintln!(
            "similaripy-server: warning: model.name is \"model\" (the directory's name): \
             set model.name to tell model versions apart in answers"
        );
    }
    let name = cfg.model.name.clone();
    let (items, active) = (model.n_items(), model.n_active());
    let threads = cfg.server.threads;
    let app = Arc::new(App::new(model, cfg).map_err(|e| e.to_string())?);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("{addr}: {e}"))?;
    // The bound address (port 0 = a free one)
    let addr = listener.local_addr().map_or(addr, |a| a.to_string());
    eprintln!(
        "similaripy-server: model {name:?} = {} ({items} items, {active} active, written by \
         {written_by}, loaded in {loaded:.1?}), {threads} threads, block {}, listening on {addr}",
        dir.display(),
        app.block(),
    );
    for line in app.describe_views() {
        eprintln!("similaripy-server: {line}");
    }
    serve(listener, app.clone(), router(app), shutdown()).map_err(|e| e.to_string())
}

/// The first SIGTERM / SIGINT (Ctrl-C) starts the graceful shutdown; a
/// second one exits at once with 1. The handlers are installed when `serve`
/// starts accepting: a signal before that (model load, warm-up) has the
/// default action (the process ends, nothing to drain).
async fn shutdown() {
    // One pair of handlers for both signals: none lost in between.
    let mut signals = Signals::new();
    let first = signals.next().await;
    eprintln!("similaripy-server: {first}: shutting down (again to exit now)");
    tokio::spawn(async move {
        let second = signals.next().await;
        eprintln!("similaripy-server: {second} again: exiting now");
        std::process::exit(1);
    });
}

#[cfg(unix)]
struct Signals(tokio::signal::unix::Signal, tokio::signal::unix::Signal);

#[cfg(unix)]
impl Signals {
    fn new() -> Self {
        use tokio::signal::unix::{SignalKind, signal};
        Self(
            signal(SignalKind::terminate()).expect("a SIGTERM handler"),
            signal(SignalKind::interrupt()).expect("a SIGINT handler"),
        )
    }

    async fn next(&mut self) -> &'static str {
        tokio::select! {
            _ = self.0.recv() => "SIGTERM",
            _ = self.1.recv() => "SIGINT",
        }
    }
}

#[cfg(not(unix))]
struct Signals;

#[cfg(not(unix))]
impl Signals {
    fn new() -> Self {
        Self
    }

    async fn next(&mut self) -> &'static str {
        let _ = tokio::signal::ctrl_c().await;
        "Ctrl-C"
    }
}
