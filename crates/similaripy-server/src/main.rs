//! `similaripy-server --model DIR`: serves an `ItemKNN.save` directory.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use similaripy_core::block::perf_cores;
use similaripy_core::serve::{BlockSize, Model};
use similaripy_server::{App, Config, router, serve};

const USAGE: &str = "\
similaripy-server --model DIR [options]   (internal: no auth, no TLS; private networks only)
  --model DIR          an ItemKNN.save directory
  --addr HOST:PORT     default 127.0.0.1:8080
  --threads N          server threads, one Worker each; default: P-cores (macOS), available CPUs (Linux)
  --block-size B       auto | off | N entries; default auto (from --threads)
  --max-history N      keep the last N history ids; default 200
  --k-max N            reject a larger k; default 1000
  --no-mmap            read the model into memory instead of mapping it";

fn main() {
    if let Err(e) = run() {
        eprintln!("similaripy-server: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut model = None;
    let mut addr = "127.0.0.1:8080".to_string();
    let mut cfg = Config {
        threads: perf_cores(),
        block_size: BlockSize::Auto,
        max_history: 200,
        k_max: 1000,
    };
    let mut mmap = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or(format!("{a} needs a value\n{USAGE}"));
        let num = |v: String| v.parse::<usize>().map_err(|e| format!("{a} {v}: {e}"));
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--addr" => addr = value()?,
            "--threads" => cfg.threads = num(value()?)?.max(1),
            "--max-history" => cfg.max_history = num(value()?)?,
            "--k-max" => cfg.k_max = num(value()?)?,
            "--block-size" => {
                cfg.block_size = match value()?.as_str() {
                    "auto" => BlockSize::Auto,
                    "off" => BlockSize::Off,
                    v => BlockSize::Size(num(v.to_string())?),
                }
            }
            "--no-mmap" => mmap = false,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {a}\n{USAGE}")),
        }
    }
    let dir = model.ok_or(format!("--model is required\n{USAGE}"))?;
    let t = std::time::Instant::now();
    let model = Model::load(&dir, mmap).map_err(|e| e.to_string())?;
    let loaded = t.elapsed();
    let (items, active) = (model.n_items(), model.n_active());
    let threads = cfg.threads;
    let app = Arc::new(App::new(model, cfg).map_err(|e| e.to_string())?);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("{addr}: {e}"))?;
    eprintln!(
        "similaripy-server: {} ({items} items, {active} active, loaded in {loaded:.1?}), \
         {threads} threads, block {}, listening on {addr}",
        dir.display(),
        app.block(),
    );
    serve(listener, app.clone(), router(app)).map_err(|e| e.to_string())
}
