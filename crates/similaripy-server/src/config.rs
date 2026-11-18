//! The server's config (V1): built-in defaults < the `--config` TOML file <
//! `SIMILARIPY_<SECTION>__<KEY>` env vars < `--section.key value` flags.
//! One mechanism: the defaults as a TOML table, the file merged in, each
//! override set as a dotted key, then one deserialize into [`Config`] with
//! `deny_unknown_fields` at every level (a typo is a start-up error naming
//! the key). Keys are a contract (R-S2: additive within 1.x); `GET /config`
//! serves the resolved config, so no key may hold a secret.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use similaripy_core::block::perf_cores;
use similaripy_core::serve::{BlockSize, FORMAT};
use toml::{Table, Value};

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub model: ModelConfig,
    pub server: ServerConfig,
    pub limits: Limits,
    pub defaults: Defaults,
    pub timeouts: Timeouts,
    pub views: Views,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
    /// A `SparseKNN.save` directory (required; "" = not set).
    pub path: PathBuf,
    /// Map the model's files instead of reading them into memory.
    pub mmap: bool,
    /// The `model.name` of every response (V6); "" = the directory's
    /// basename (a saved model is never replaced: the directory is a version).
    pub name: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    pub addr: String,
    /// Server threads (one `Worker` each); `block::perf_cores()` by default.
    pub threads: usize,
    /// "auto" | "off" | entries per block.
    #[serde(with = "block_size")]
    pub block_size: BlockSize,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Limits {
    /// A larger k is rejected (400 / INVALID_ARGUMENT).
    pub k_max: usize,
    /// History ids a request uses; a longer history follows `history_policy`.
    /// Default 1000: Python has no cap, so offline evaluation and the server
    /// agree on all but the longest histories.
    pub max_history: usize,
    pub history_policy: HistoryPolicy,
    /// A larger request body is rejected (JSON 413, gRPC OUT_OF_RANGE).
    pub max_body_bytes: usize,
    /// A score request with more items is rejected (400 / INVALID_ARGUMENT),
    /// never cut: a ranker would get features for some candidates only.
    pub max_items: usize,
}

/// A history longer than `limits.max_history` (V7).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HistoryPolicy {
    /// Keep the last `max_history` ids (client order = oldest → newest, Q4);
    /// the cut ids stay excluded under `exclude_seen`.
    Truncate,
    /// 400 / INVALID_ARGUMENT.
    Reject,
}

/// What a request gets for a field it doesn't send.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Defaults {
    /// Python's `recommend` default; at most `limits.k_max`.
    pub k: usize,
    /// Python's `recommend` sorts; candidate generation re-scored by a ranker
    /// can set false (U1).
    pub sorted: bool,
    /// The view of a request that doesn't name one; "" = the whole model.
    pub view: String,
}

/// Seconds (a TOML integer or float).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Timeouts {
    /// A connection must send a request's headers within this (slow-loris):
    /// its first bytes after the accept (until the protocol is decided),
    /// each HTTP/1.1 request head (so also idle HTTP/1.1 keep-alive) and a
    /// JSON request's body after its headers (408).
    pub header_read: f64,
    /// After SIGTERM / SIGINT: `/ready` answers 503 and new connections are
    /// still accepted for this long (Kubernetes: time to drop the pod from
    /// its endpoints; e.g. 5), then the listener closes.
    pub drain_delay: f64,
    /// Then open connections get this long to finish their requests;
    /// what is left is closed. The process exits 0 either way.
    pub shutdown_grace: f64,
}

/// `[views]`: the named views (V2) and how often their files are checked.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Views {
    /// Seconds between checks of the view files (V21 C): a changed file
    /// (contents, or a renamed / symlinked new file in its place) rebuilds
    /// the views, served from the next request on; 0 = read at start only.
    pub reload_interval: f64,
    /// `[views.<name>]` with an include and / or exclude id file
    /// (`reload_interval` is not a view name; flatten allows no
    /// deny_unknown_fields: `parse` names a stray key).
    #[serde(flatten)]
    pub named: BTreeMap<String, ViewConfig>,
}

/// A named view: the items in `include_file` (absent: all) minus those in
/// `exclude_file` (absent: none); files hold one id per line.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViewConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_file: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_file: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            mmap: true,
            name: String::new(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080".into(),
            threads: perf_cores(),
            block_size: BlockSize::Auto,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            k_max: 2000,
            max_history: 1000,
            history_policy: HistoryPolicy::Truncate,
            max_body_bytes: crate::BODY_LIMIT,
            max_items: 10_000,
        }
    }
}

/// The core's request defaults (D-A0-4: one source).
impl Default for Defaults {
    fn default() -> Self {
        let r = similaripy_core::serve::Request::default();
        Self {
            k: r.k,
            sorted: r.sorted,
            view: String::new(),
        }
    }
}

impl Default for Views {
    fn default() -> Self {
        Self {
            reload_interval: 10.0,
            named: BTreeMap::new(),
        }
    }
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            header_read: 30.0,
            drain_delay: 0.0,
            shutdown_grace: 10.0,
        }
    }
}

/// `secs` as a Duration (checked by `Config::check_timeouts`).
pub fn secs(secs: f64) -> std::time::Duration {
    std::time::Duration::try_from_secs_f64(secs).expect("checked by Config::check_timeouts")
}

impl ModelConfig {
    /// An empty `name` = the directory's basename (`/models/x/` and `x` too;
    /// `.` / `..` by the directory they name; no last part at all (`/`, or
    /// `..` not found) as written). `parse` and `App::new` both call it.
    pub fn fill_name(&mut self) {
        if !self.name.is_empty() {
            return;
        }
        let path = &self.path;
        let full = path.file_name().is_none().then(|| path.canonicalize().ok());
        let named = full.flatten().unwrap_or_else(|| path.clone());
        self.name = match named.file_name() {
            Some(n) => n.to_string_lossy().into_owned(),
            None => path.display().to_string(),
        };
    }
}

impl Config {
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("plain keys and values")
    }

    /// The view rules a config alone can break: names of 1-64 of
    /// `a-z 0-9 _ -` (settable as a flag or env var: no `.`, lowercase;
    /// Prometheus labels and URLs without escaping), at least one file each,
    /// `defaults.view` "" or a view.
    /// The timeouts and `views.reload_interval`: seconds from 0 to a year
    /// (more is a typo in ms), header_read > 0;
    /// `parse` and `App::new` both call it (`secs` relies on it).
    pub fn check_timeouts(&self) -> Result<(), String> {
        let t = &self.timeouts;
        for (key, v) in [
            ("timeouts.header_read", t.header_read),
            ("timeouts.drain_delay", t.drain_delay),
            ("timeouts.shutdown_grace", t.shutdown_grace),
            ("views.reload_interval", self.views.reload_interval),
        ] {
            let positive = key == "timeouts.header_read";
            if !(0.0..=31_536_000.0).contains(&v) || (positive && v == 0.0) {
                let from = if positive { "more than 0" } else { "0" };
                return Err(format!("{key} = {v}: seconds, {from} to a year"));
            }
            // A stat loop, or a zero Duration (tokio's interval panics)
            if key == "views.reload_interval" && v > 0.0 && v < 0.1 {
                return Err(format!("{key} = {v}: 0 (off) or at least 0.1 seconds"));
            }
        }
        Ok(())
    }

    /// `defaults.k` <= `limits.k_max`, `max_body_bytes` and `max_items` > 0; `parse` and
    /// `App::new` both call it.
    pub fn check_limits(&self) -> Result<(), String> {
        let (k, k_max) = (self.defaults.k, self.limits.k_max);
        if k > k_max {
            return Err(format!("defaults.k = {k} > limits.k_max = {k_max}"));
        }
        if self.limits.max_body_bytes == 0 {
            return Err("limits.max_body_bytes must be >= 1".into());
        }
        if self.limits.max_items == 0 {
            return Err("limits.max_items must be >= 1".into());
        }
        Ok(())
    }

    pub fn check_views(&self) -> Result<(), String> {
        for (name, v) in &self.views.named {
            let ok = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-".contains(&b);
            if !(1..=64).contains(&name.len()) || !name.bytes().all(ok) {
                return Err(format!("view {name:?}: a name is 1-64 of a-z 0-9 _ -"));
            }
            if v.include_file.is_none() && v.exclude_file.is_none() {
                return Err(format!(
                    "view {name:?}: needs include_file and / or exclude_file"
                ));
            }
        }
        let d = &self.defaults.view;
        if !d.is_empty() && !self.views.named.contains_key(d) {
            return Err(format!("defaults.view = {d:?}: no such view"));
        }
        Ok(())
    }
}

/// What the command line asks for.
#[derive(Debug)]
pub enum Cli {
    Run(Config),
    /// `--print-config`: the resolved config, then exit (a dry run).
    PrintConfig(Config),
    Help,
    Version,
}

pub fn version() -> String {
    format!(
        "similaripy-server {} (reads similaripy saved models up to format {FORMAT})",
        env!("CARGO_PKG_VERSION")
    )
}

/// The flags before V1, kept for scripts (bench-runner gates run both CLIs).
const ALIASES: [(&str, &str); 6] = [
    ("--model", "model.path"),
    ("--addr", "server.addr"),
    ("--threads", "server.threads"),
    ("--block-size", "server.block_size"),
    ("--max-history", "limits.max_history"),
    ("--k-max", "limits.k_max"),
];

/// The config from `args` (without the program name) and `env` (only
/// `SIMILARIPY_*` names containing `__` are config: `SIMILARIPY_BENCH_DATA`,
/// Kubernetes' `SIMILARIPY_PORT` and the like are not).
pub fn parse(
    args: impl IntoIterator<Item = String>,
    env: impl IntoIterator<Item = (String, String)>,
) -> Result<Cli, String> {
    let (mut file, mut flags, mut print) = (None, vec![], false);
    let mut args = args.into_iter();
    while let Some(a) = args.next() {
        // `--key=value` = `--key value`
        let (a, mut inline) = match a.split_once('=') {
            Some((k, v)) if a.starts_with("--") => (k.to_string(), Some(v.to_string())),
            _ => (a, None),
        };
        let mut value = || {
            inline
                .take()
                .or_else(|| args.next())
                .ok_or(format!("{a} needs a value"))
        };
        match a.as_str() {
            "-h" | "--help" => return Ok(Cli::Help),
            "-V" | "--version" => return Ok(Cli::Version),
            "--print-config" => print = true,
            "--config" => file = Some(value()?),
            "--no-mmap" => flags.push(("model.mmap".to_string(), "false".to_string())),
            _ => {
                let key = match ALIASES.iter().find(|(f, _)| *f == a) {
                    Some((_, key)) => key.to_string(),
                    None => match a.strip_prefix("--") {
                        Some(key) if key.contains('.') => key.to_string(),
                        _ => return Err(format!("unknown argument {a}")),
                    },
                };
                flags.push((key, value()?));
            }
        }
    }
    let Value::Table(defaults) = Value::try_from(Config::default()).expect("plain keys") else {
        unreachable!("a struct serializes to a table")
    };
    let mut table = defaults.clone();
    if let Some(path) = file {
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
        let t: Table = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        merge(&mut table, t);
    }
    let mut env: Vec<_> = env
        .into_iter()
        .filter_map(|(name, v)| {
            let rest = name.strip_prefix("SIMILARIPY_")?;
            let key = rest
                .contains("__")
                .then(|| rest.to_lowercase().replace("__", "."))?;
            Some((key, v, name))
        })
        .collect();
    // Sorted: two spellings of one key (case) resolve the same every run.
    env.sort();
    let flags = flags
        .into_iter()
        .map(|(key, v)| (key.clone(), v, format!("--{key}")));
    for (key, v, source) in env.into_iter().chain(flags) {
        set(&mut table, &defaults, &key, v).map_err(|e| format!("{source}: {e}"))?;
    }
    // [views] holds reload_interval and view sections: any other value is a typo
    if let Some(Value::Table(views)) = table.get("views")
        && let Some((k, _)) = views
            .iter()
            .find(|(k, v)| *k != "reload_interval" && !v.is_table())
    {
        return Err(format!(
            "views.{k}: unknown key (a view is a [views.NAME] section)"
        ));
    }
    let mut cfg: Config = Value::Table(table)
        .try_into()
        .map_err(|e| format!("config: {}", e.to_string().trim_end().replace('\n', " ")))?;
    if cfg.model.path.as_os_str().is_empty() {
        return Err(
            "model.path is required (--model DIR, SIMILARIPY_MODEL__PATH or [model] path)".into(),
        );
    }
    cfg.model.fill_name();
    if cfg.server.threads == 0 {
        return Err("server.threads must be >= 1".into());
    }
    cfg.check_timeouts()?;
    cfg.check_limits()?;
    cfg.check_views()?;
    Ok(if print {
        Cli::PrintConfig(cfg)
    } else {
        Cli::Run(cfg)
    })
}

/// `src`'s keys into `dst`, tables merged key by key.
fn merge(dst: &mut Table, src: Table) {
    for (k, v) in src {
        match (dst.get_mut(&k), v) {
            (Some(Value::Table(d)), Value::Table(s)) => merge(d, s),
            (_, v) => {
                dst.insert(k, v);
            }
        }
    }
}

/// `key` (dotted) = `raw`, typed as the key's default (so "2026" stays a
/// string for a string key, and `block_size`, default "auto", takes "off" or
/// "64" whatever an earlier source set); a `*_file` key (views: no default)
/// is a string; another key without a default takes `raw` as a TOML literal,
/// else a string. An unknown key is set too: the deserialize names it.
// ponytail: no list keys yet; S5e's `compression.algorithms` adds "a,b".
fn set(table: &mut Table, defaults: &Table, key: &str, raw: String) -> Result<(), String> {
    if key.split('.').any(str::is_empty) {
        return Err(format!("{key:?}: an empty name part"));
    }
    let (path, last) = key.rsplit_once('.').unwrap_or(("", key));
    let default = path
        .split('.')
        .filter(|p| !p.is_empty())
        .try_fold(defaults, |t, part| t.get(part)?.as_table())
        .and_then(|t| t.get(last));
    let mut t = table;
    for part in path.split('.').filter(|p| !p.is_empty()) {
        t = match t.entry(part).or_insert_with(|| Value::Table(Table::new())) {
            Value::Table(t) => t,
            _ => return Err(format!("{key}: {part} is a value, not a section")),
        };
    }
    let bad = |e: &dyn std::fmt::Display| format!("{key} = {raw:?}: {e}");
    let v = match default {
        Some(Value::String(_)) => Value::String(raw),
        Some(Value::Integer(_)) => Value::Integer(raw.parse().map_err(|e| bad(&e))?),
        Some(Value::Boolean(_)) => Value::Boolean(raw.parse().map_err(|e| bad(&e))?),
        Some(Value::Float(_)) => Value::Float(raw.parse().map_err(|e| bad(&e))?),
        Some(_) => return Err(bad(&"not settable from a flag or env var")),
        None if last.ends_with("_file") => Value::String(raw),
        None => toml::from_str::<Table>(&format!("v = {raw}"))
            .ok()
            .and_then(|mut t| t.remove("v"))
            .unwrap_or(Value::String(raw)),
    };
    t.insert(last.to_string(), v);
    Ok(())
}

/// `BlockSize` as "auto", "off" or entries, from a TOML integer or a string
/// ("64" from an env var or flag).
mod block_size {
    use super::*;

    pub fn serialize<S: Serializer>(b: &BlockSize, s: S) -> Result<S::Ok, S::Error> {
        match b {
            BlockSize::Auto => s.serialize_str("auto"),
            BlockSize::Off => s.serialize_str("off"),
            BlockSize::Size(n) => s.serialize_u64(*n as u64),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BlockSize, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            N(i64),
            S(String),
        }
        let bad = || serde::de::Error::custom("expected \"auto\", \"off\" or entries > 0");
        match Raw::deserialize(d).map_err(|_| bad())? {
            Raw::N(n) if n > 0 => Ok(BlockSize::Size(n as usize)),
            Raw::N(_) => Err(bad()),
            Raw::S(s) => match s.as_str() {
                "auto" => Ok(BlockSize::Auto),
                "off" => Ok(BlockSize::Off),
                s => match s.parse() {
                    Ok(0) | Err(_) => Err(bad()),
                    Ok(n) => Ok(BlockSize::Size(n)),
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(args: &[&str], env: &[(&str, &str)]) -> Result<Config, String> {
        let args = args.iter().map(|s| s.to_string());
        let env = env.iter().map(|(k, v)| (k.to_string(), v.to_string()));
        match parse(args, env)? {
            Cli::Run(c) => Ok(c),
            other => panic!("{other:?}"),
        }
    }

    /// A config file, removed on drop (no tempfile dependency).
    struct File(String);

    impl Drop for File {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn file(text: &str) -> File {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("similaripy-server-{}-{n}.toml", std::process::id()));
        std::fs::write(&p, text).unwrap();
        File(p.to_str().unwrap().to_string())
    }

    #[test]
    fn defaults_then_file_then_env_then_flags() {
        let f = file(
            "[model]\npath = \"/m\"\n[limits]\nk_max = 1\nmax_history = 1\n[defaults]\nk = 1\nsorted = false\n",
        );
        let env = [
            ("SIMILARIPY_LIMITS__MAX_HISTORY", "2"),
            ("SIMILARIPY_DEFAULTS__SORTED", "true"),
        ];
        let flags = ["--config", &f.0, "--defaults.sorted", "false"];
        let c = cfg(&flags, &env).unwrap();
        assert_eq!(c.limits.k_max, 1); // file over default
        assert_eq!(c.limits.max_history, 2); // env over file
        assert!(!c.defaults.sorted); // flag over env over file
        assert_eq!(c.limits.history_policy, HistoryPolicy::Truncate); // default
        assert_eq!(c.server, ServerConfig::default());
        // Without the file: the default under env.
        assert_eq!(cfg(&["--model", "/m"], &env).unwrap().limits.k_max, 2000);
        assert_eq!(
            cfg(&["--model", "/m"], &[]).unwrap().limits.max_history,
            1000
        );
    }

    #[test]
    fn every_key_three_ways() {
        let want = Config {
            model: ModelConfig {
                path: "/models/x".into(),
                mmap: false,
                name: "knn-7".into(),
            },
            server: ServerConfig {
                addr: "0.0.0.0:9".into(),
                threads: 3,
                block_size: BlockSize::Size(64),
            },
            limits: Limits {
                k_max: 7,
                max_history: 8,
                history_policy: HistoryPolicy::Reject,
                max_body_bytes: 4096,
                max_items: 9,
            },
            defaults: Defaults {
                k: 5,
                sorted: false,
                view: "instock".into(),
            },
            timeouts: Timeouts {
                header_read: 2.5,
                drain_delay: 5.0,
                shutdown_grace: 0.25,
            },
            views: Views {
                reload_interval: 2.0,
                named: [(
                    "instock".to_string(),
                    ViewConfig {
                        include_file: Some("2026.txt".into()),
                        exclude_file: Some("/ids/adult.txt".into()),
                    },
                )]
                .into(),
            },
        };
        let keys = [
            ("model.path", "/models/x"),
            ("model.mmap", "false"),
            ("model.name", "knn-7"),
            ("server.addr", "0.0.0.0:9"),
            ("server.threads", "3"),
            ("server.block_size", "64"),
            ("limits.k_max", "7"),
            ("limits.max_history", "8"),
            ("limits.history_policy", "reject"),
            ("limits.max_body_bytes", "4096"),
            ("limits.max_items", "9"),
            ("defaults.k", "5"),
            ("defaults.sorted", "false"),
            ("defaults.view", "instock"),
            ("timeouts.header_read", "2.5"),
            // an integer for a float key
            ("timeouts.drain_delay", "5"),
            ("timeouts.shutdown_grace", "0.25"),
            ("views.reload_interval", "2"),
            // a number-like path stays a string (`*_file`)
            ("views.instock.include_file", "2026.txt"),
            ("views.instock.exclude_file", "/ids/adult.txt"),
        ];
        // Every field of Config is in `keys`: the TOML of `want` has as many.
        assert_eq!(want.to_toml().matches(" = ").count(), keys.len());
        let flags: Vec<String> = keys
            .iter()
            .flat_map(|(k, v)| [format!("--{k}"), v.to_string()])
            .collect();
        let flags: Vec<&str> = flags.iter().map(|s| s.as_str()).collect();
        assert_eq!(cfg(&flags, &[]).unwrap(), want);
        let env: Vec<(String, String)> = keys
            .iter()
            .map(|(k, v)| {
                (
                    format!("SIMILARIPY_{}", k.to_uppercase().replace('.', "__")),
                    v.to_string(),
                )
            })
            .collect();
        let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(cfg(&[], &env).unwrap(), want);
        // The file: the printed config reads back as itself.
        let f = file(&want.to_toml());
        assert_eq!(cfg(&["--config", &f.0], &[]).unwrap(), want);
    }

    #[test]
    fn unknown_keys_are_errors_naming_them() {
        let f = file("[model]\npath = \"/m\"\n[limits]\nk_maxx = 3\n");
        let e = cfg(&["--config", &f.0], &[]).unwrap_err();
        assert!(e.contains("k_maxx") && e.contains("limits"), "{e}");
        let e = cfg(&["--model", "/m", "--limits.k_maxx", "3"], &[]).unwrap_err();
        assert!(e.contains("k_maxx"), "{e}");
        let e = cfg(&["--model", "/m"], &[("SIMILARIPY_SERVR__ADDR", "x")]).unwrap_err();
        assert!(e.contains("servr"), "{e}");
        let e = cfg(&["--model", "/m", "--treads", "3"], &[]).unwrap_err();
        assert!(e.contains("--treads"), "{e}");
    }

    #[test]
    fn values_typed_by_their_key() {
        // A string key keeps a number-like value; a bad number names the key.
        let c = cfg(&["--model", "2026", "--addr", "8080"], &[]).unwrap();
        assert_eq!(
            (c.model.path.to_str(), &c.server.addr[..]),
            (Some("2026"), "8080")
        );
        let e = cfg(&["--model", "/m", "--k-max", "lots"], &[]).unwrap_err();
        assert!(e.contains("limits.k_max"), "{e}");
        for (v, want) in [
            ("auto", BlockSize::Auto),
            ("off", BlockSize::Off),
            ("16", BlockSize::Size(16)),
        ] {
            let c = cfg(&["--model", "/m", "--block-size", v], &[]).unwrap();
            assert_eq!(c.server.block_size, want);
        }
        for v in ["0", "big"] {
            assert!(
                cfg(&["--model", "/m", "--block-size", v], &[]).is_err(),
                "{v}"
            );
        }
        let e = cfg(&["--model", "/m", "--limits.history_policy", "drop"], &[]).unwrap_err();
        assert!(e.contains("drop"), "{e}");
        // defaults.k within k_max (whichever source set them); a body > 0
        let e = cfg(&["--model", "/m", "--k-max", "5", "--defaults.k", "6"], &[]).unwrap_err();
        assert!(e.contains("defaults.k = 6 > limits.k_max = 5"), "{e}");
        let e = cfg(&["--model", "/m", "--limits.max_body_bytes", "0"], &[]).unwrap_err();
        assert!(e.contains("max_body_bytes"), "{e}");
        let e = cfg(&["--model", "/m", "--limits.max_items", "0"], &[]).unwrap_err();
        assert!(e.contains("max_items"), "{e}");
        // views.reload_interval: seconds; a typo in [views] reads as a view
        let e = cfg(&["--model", "/m", "--views.reload_interval", "-1"], &[]).unwrap_err();
        assert!(e.contains("views.reload_interval = -1"), "{e}");
        let e = cfg(&["--model", "/m", "--views.reload_interval", "1e-10"], &[]).unwrap_err();
        assert!(e.contains("at least 0.1"), "{e}");
        let e = cfg(&["--model", "/m", "--views.reload_intervall", "5"], &[]).unwrap_err();
        assert!(e.contains("views.reload_intervall: unknown key"), "{e}");
        let f = file("[model]\npath = \"/m\"\n[views]\nreload_interval = 0\n");
        assert_eq!(
            cfg(&["--config", &f.0], &[]).unwrap().views.reload_interval,
            0.0
        );
        // Timeouts: an integer in the file is seconds too; out of range = error
        let f = file("[model]\npath = \"/m\"\n[timeouts]\nheader_read = 3\n");
        let c = cfg(&["--config", &f.0], &[]).unwrap();
        assert_eq!(
            secs(c.timeouts.header_read),
            std::time::Duration::from_secs(3)
        );
        for (key, v) in [
            ("header_read", "0"),
            ("drain_delay", "-1"),
            ("shutdown_grace", "nan"),
            ("shutdown_grace", "1e300"),
        ] {
            let e = cfg(&["--model", "/m", &format!("--timeouts.{key}"), v], &[]).unwrap_err();
            assert!(e.contains(key), "{e}");
        }
    }

    #[test]
    fn block_size_overrides_an_integer_from_the_file() {
        // `--print-config` writes a set block as an integer.
        let f = file("[model]\npath = \"/m\"\n[server]\nblock_size = 64\n");
        for (v, want) in [
            ("off", BlockSize::Off),
            ("auto", BlockSize::Auto),
            ("8", BlockSize::Size(8)),
        ] {
            let c = cfg(&["--config", &f.0, "--block-size", v], &[]).unwrap();
            assert_eq!(c.server.block_size, want, "{v}");
        }
        let f = file("[model]\npath = \"/m\"\n[server]\nblock_size = -3\n");
        let e = cfg(&["--config", &f.0], &[]).unwrap_err();
        assert!(e.contains("entries > 0"), "{e}");
    }

    #[test]
    fn errors_name_the_env_var_or_flag() {
        let e = cfg(&["--model", "/m"], &[("SIMILARIPY_MODEL__", "1")]).unwrap_err();
        assert!(
            e.contains("SIMILARIPY_MODEL__") && e.contains("empty"),
            "{e}"
        );
        let e = cfg(&["--model", "/m", "--limits.k_max", "x"], &[]).unwrap_err();
        assert!(e.starts_with("--limits.k_max:"), "{e}");
    }

    #[test]
    fn flags_take_key_equals_value() {
        let c = cfg(
            &[
                "--model=/m",
                "--limits.k_max=5",
                "--defaults.k=5",
                "--addr=h:1",
            ],
            &[],
        )
        .unwrap();
        assert_eq!((c.limits.k_max, &c.server.addr[..]), (5, "h:1"));
    }

    #[test]
    fn env_names_outside_config_are_ignored() {
        let env = [
            ("SIMILARIPY_BENCH_DATA", "/x"),
            ("SIMILARIPY_PORT", "tcp://10.0.0.1:8080"),
            ("RAYON_NUM_THREADS", "2"),
        ];
        assert_eq!(
            cfg(&["--model", "/m"], &env).unwrap().model.path.to_str(),
            Some("/m")
        );
    }

    #[test]
    fn model_name_defaults_to_the_directory() {
        let name = |args: &[&str]| cfg(args, &[]).unwrap().model.name;
        assert_eq!(
            name(&["--model", "/models/knn-2026-11-16"]),
            "knn-2026-11-16"
        );
        assert_eq!(
            name(&["--model", "/models/knn-2026-11-16/"]),
            "knn-2026-11-16"
        );
        assert_eq!(name(&["--model", "/"]), "/");
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            name(&["--model", "."]),
            cwd.file_name().unwrap().to_str().unwrap()
        );
        assert_eq!(name(&["--model", "/m", "--model.name", "a"]), "a");
    }

    #[test]
    fn aliases_and_cli_actions() {
        let c = cfg(
            &[
                "--model",
                "/m",
                "--addr",
                "a:1",
                "--threads",
                "2",
                "--block-size",
                "off",
                "--max-history",
                "5",
                "--k-max",
                "6",
                "--defaults.k",
                "6",
                "--no-mmap",
            ],
            &[],
        )
        .unwrap();
        assert_eq!(
            (
                c.server.threads,
                c.limits.max_history,
                c.limits.k_max,
                c.model.mmap
            ),
            (2, 5, 6, false)
        );
        let none = std::iter::empty();
        assert!(matches!(
            parse(["--version".into()], none),
            Ok(Cli::Version)
        ));
        // `--print-config` prints the core's request defaults (D-A2-3; the
        // Python signatures are checked against the same in test_serve.py).
        let d = similaripy_core::serve::Request::default();
        let toml = cfg(&["--model", "/m"], &[]).unwrap().to_toml();
        assert!(toml.contains(&format!("\nk = {}\n", d.k)), "{toml}");
        assert!(
            toml.contains(&format!("\nsorted = {}\n", d.sorted)),
            "{toml}"
        );
        let args = ["--print-config", "--model", "/m"].map(String::from);
        assert!(matches!(
            parse(args, std::iter::empty()),
            Ok(Cli::PrintConfig(_))
        ));
        assert!(
            cfg(&[], &[])
                .unwrap_err()
                .contains("model.path is required")
        );
        assert!(cfg(&["--model", "/m", "--threads", "0"], &[]).is_err());
        assert!(
            cfg(&["--model"], &[])
                .unwrap_err()
                .contains("needs a value")
        );
    }
}
