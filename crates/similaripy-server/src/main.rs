//! `similaripy-server --model DIR`: serves an `ItemKNN.save` directory.

use std::net::TcpListener;
use std::sync::Arc;

use similaripy_core::serve::Model;
use similaripy_server::config::{Cli, parse, version};
use similaripy_server::{App, router, serve};

const USAGE: &str = "\
similaripy-server [--config FILE] [--section.key VALUE ...]
  (experimental: no auth, no TLS; private networks only)

Config: built-in defaults < FILE (TOML) < env SIMILARIPY_<SECTION>__<KEY> < flags.
  --model.path DIR            an ItemKNN.save directory (required)
  --model.mmap BOOL           map the model's files; default true
  --model.name NAME           the model name in every answer; default the directory's name
  --server.addr HOST:PORT     default 127.0.0.1:8080
  --server.threads N          one Worker each; default: P-cores (macOS), available CPUs (Linux)
  --server.block_size B       auto | off | N entries; default auto (from threads)
  --limits.k_max N            reject a larger k; default 2000
  --limits.max_history N      default 200
  --limits.history_policy P   truncate (keep the last max_history ids) | reject; default truncate
  --defaults.sorted BOOL      a request's sorted when it doesn't send one; default true
  e.g. SIMILARIPY_LIMITS__K_MAX=500 = --limits.k_max 500 = [limits] k_max = 500
Short forms: --model, --addr, --threads, --block-size, --max-history, --k-max, --no-mmap.
  --print-config   print the resolved config (TOML) and exit
  --version, --help";

fn main() {
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
    let (dir, addr) = (cfg.model.path.clone(), cfg.server.addr.clone());
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
    eprintln!(
        "similaripy-server: model {name:?} = {} ({items} items, {active} active, written by \
         {written_by}, loaded in {loaded:.1?}), {threads} threads, block {}, listening on {addr}",
        dir.display(),
        app.block(),
    );
    serve(listener, app.clone(), router(app)).map_err(|e| e.to_string())
}
