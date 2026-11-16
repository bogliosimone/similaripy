//! The server's config (V1): built-in defaults < the `--config` TOML file <
//! `SIMILARIPY_<SECTION>__<KEY>` env vars < `--section.key value` flags.
//! One mechanism: the defaults as a TOML table, the file merged in, each
//! override set as a dotted key, then one deserialize into [`Config`] with
//! `deny_unknown_fields` at every level (a typo is a start-up error naming
//! the key). Keys are a contract (R-S2: additive within 1.x); `GET /config`
//! serves the resolved config, so no key may hold a secret.

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
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
    /// An `ItemKNN.save` directory (required; "" = not set).
    pub path: PathBuf,
    /// Map the model's files instead of reading them into memory.
    pub mmap: bool,
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
    pub max_history: usize,
    pub history_policy: HistoryPolicy,
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
    /// Python's `recommend` sorts; candidate generation re-scored by a ranker
    /// can set false (U1).
    pub sorted: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            mmap: true,
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
            max_history: 200,
            history_policy: HistoryPolicy::Truncate,
        }
    }
}

impl Default for Defaults {
    fn default() -> Self {
        Self { sorted: true }
    }
}

impl Config {
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("plain keys and values")
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
    let cfg: Config = Value::Table(table)
        .try_into()
        .map_err(|e| format!("config: {}", e.to_string().trim_end().replace('\n', " ")))?;
    if cfg.model.path.as_os_str().is_empty() {
        return Err(
            "model.path is required (--model DIR, SIMILARIPY_MODEL__PATH or [model] path)".into(),
        );
    }
    if cfg.server.threads == 0 {
        return Err("server.threads must be >= 1".into());
    }
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
/// "64" whatever an earlier source set); a key without a default takes `raw`
/// as a TOML literal, else a string. An unknown key is set too: the
/// deserialize names it.
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
            "[model]\npath = \"/m\"\n[limits]\nk_max = 1\nmax_history = 1\n[defaults]\nsorted = false\n",
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
    }

    #[test]
    fn every_key_three_ways() {
        let want = Config {
            model: ModelConfig {
                path: "/models/x".into(),
                mmap: false,
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
            },
            defaults: Defaults { sorted: false },
        };
        let keys = [
            ("model.path", "/models/x"),
            ("model.mmap", "false"),
            ("server.addr", "0.0.0.0:9"),
            ("server.threads", "3"),
            ("server.block_size", "64"),
            ("limits.k_max", "7"),
            ("limits.max_history", "8"),
            ("limits.history_policy", "reject"),
            ("defaults.sorted", "false"),
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
        let c = cfg(&["--model=/m", "--limits.k_max=5", "--addr=h:1"], &[]).unwrap();
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
