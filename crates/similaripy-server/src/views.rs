//! Named views: each a core `Model::view` of the served model (S shared, one
//! bit per item), read from the `views.dir` folder: `NAME.include.txt` (the
//! items it allows; absent: all) and / or `NAME.exclude.txt` (the items it
//! removes), one id per line. Each scan adds new names, rebuilds the views
//! whose files changed, shares the others (`Arc`, same model only) and
//! removes a view whose files are gone two scans in a row.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use serde::Serialize;
use similaripy_core::serve::Model;
use similaripy_core::serve::view_file::{self, Rule};

use crate::metrics::Counter;

/// One named view: the model behind it, what it was built from, its
/// requests per server thread and its failed reads.
pub struct View {
    /// None: the view failed to load and isn't served (`info.error`).
    pub model: Option<Model>,
    pub info: ViewInfo,
    /// Kept across the view's rebuilds; a removed view's go with it (a
    /// re-added name starts at 0).
    pub served: Arc<[Counter]>,
    pub errors: Arc<AtomicU64>,
}

/// `GET /v1/views`' entry for a view.
#[derive(Serialize, Clone)]
pub struct ViewInfo {
    /// Items the view allows (of the model's `items`); null: not loaded.
    pub allowed: Option<usize>,
    pub include: Option<FileInfo>,
    pub exclude: Option<FileInfo>,
    /// The last read's failure (null: none): with `allowed` set the view
    /// still serves the files as described here, read before it.
    pub error: Option<String>,
}

/// An id file: distinct ids in it, of which the model has `known` (the rest
/// are ignored: a file in another id space reads "0 known").
#[derive(Serialize, Clone)]
pub struct FileInfo {
    pub file: String,
    pub ids: usize,
    pub known: usize,
    /// The file's modification time when read (Unix seconds; null if the
    /// file system has none): an edit is served once this changes.
    pub modified: Option<f64>,
}

/// What tells a file's change: the file it resolves to (a swapped
/// symlink, as Kubernetes' ConfigMap updates), its mtime, size and inode
/// (an atomic rename). None: absent (or a dangling symlink, a directory).
#[derive(PartialEq, Debug, Clone)]
pub struct Stamp {
    path: PathBuf,
    modified: Option<std::time::SystemTime>,
    len: u64,
    #[cfg(unix)]
    inode: (u64, u64),
}

pub fn stamp(path: &Path) -> Option<Stamp> {
    let m = std::fs::metadata(path).ok().filter(|m| m.is_file())?;
    Some(Stamp {
        path: path.canonicalize().ok()?,
        modified: m.modified().ok(),
        len: m.len(),
        #[cfg(unix)]
        inode: {
            use std::os::unix::fs::MetadataExt;
            (m.dev(), m.ino())
        },
    })
}

/// A view's files in the folder: [include, exclude] stamps, and whether
/// `NAME.meta.json` is there (reserved for 1.1's view settings).
#[derive(PartialEq, Debug, Clone, Default)]
struct Files {
    stamps: [Option<Stamp>; 2],
    meta: bool,
}

/// What the writer knows of the folder between scans.
#[derive(Default)]
pub struct Seen {
    /// Per view, the files it was last read from.
    views: BTreeMap<String, Files>,
    /// Files (view, rule) absent on the last scan, still served: absent on
    /// the next one too = gone (an `rm` + `cp` sync never drops a view).
    missing: BTreeSet<(String, usize)>,
    /// Skipped files (bad name) and views (past `max_views`), logged once
    /// while they stay skipped.
    bad: BTreeSet<String>,
    skipped: BTreeSet<String>,
    /// The folder failed to read on the last scan (logged once).
    pub dir_failed: bool,
}

/// The folder's view files now (dot-files, other suffixes, sub-dirs
/// ignored; symlinks followed); a file with a view suffix and a bad name
/// is skipped, logged once.
fn list(
    dir: &Path,
    logged: &mut BTreeSet<String>,
    lines: &mut Vec<String>,
) -> std::io::Result<BTreeMap<String, Files>> {
    let mut out = BTreeMap::<String, Files>::new();
    let mut bad = BTreeSet::new();
    for e in std::fs::read_dir(dir)? {
        let path = e?.path();
        let Some(file) = path.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        if file.starts_with('.') {
            continue;
        }
        let Some((name, rule)) = view_file::parse_file_name(file) else {
            continue;
        };
        if !view_file::valid_name(name) {
            if !logged.contains(file) {
                lines.push(format!(
                    "{}: not a view name (1-64 of a-z 0-9 _ -): skipped",
                    path.display()
                ));
            }
            bad.insert(file.to_string());
            continue;
        }
        let f = out.entry(name.to_string()).or_default();
        match rule {
            None => f.meta = path.is_file(),
            Some(r) => f.stamps[r as usize] = stamp(&path),
        }
    }
    out.retain(|_, f| f.meta || f.stamps.iter().any(Option::is_some));
    *logged = bad;
    Ok(out)
}

/// A scan's outcome: every view, the log lines of those added, rebuilt,
/// failed (once per change) or removed, and of the files and views skipped
/// (bad name, past `max_views`; once while they stay skipped).
pub struct Rebuilt {
    pub views: BTreeMap<String, Arc<View>>,
    pub lines: Vec<String>,
    pub skipped: Vec<String>,
    /// A view was added, rebuilt or removed (`view_reloads_total`; a
    /// failed read alone is not a reload).
    pub changed: bool,
}

/// The folder's views over `base`, from `old` (served over `base` if
/// `same_model`, else every view is rebuilt: a mask is per model) and
/// `seen` (updated). A view whose files fail to read keeps its old mask
/// (same model) with the error, else is listed unloaded; retried every
/// scan, counted and logged once per change. None: nothing changed. Err:
/// the folder is unreadable (nothing changed either).
pub fn scan(
    base: &Model,
    cfg: &crate::config::Views,
    old: &BTreeMap<String, Arc<View>>,
    seen: &mut Seen,
    same_model: bool,
    threads: usize,
) -> Result<Option<Rebuilt>, String> {
    let dir = &cfg.dir;
    let mut skipped_lines = vec![];
    // Stamped before the reads: a write during them is a change next time.
    let mut now = list(dir, &mut seen.bad, &mut skipped_lines)
        .map_err(|e| format!("views.dir {}: {e}", dir.display()))?;
    // A file gone since the last scan stays one more scan (A15).
    let mut missing = BTreeSet::new();
    for (name, was) in &seen.views {
        let f = now.entry(name.clone()).or_default();
        for i in 0..2 {
            if f.stamps[i].is_none() && was.stamps[i].is_some() {
                let key = (name.clone(), i);
                if !seen.missing.contains(&key) {
                    f.stamps[i] = was.stamps[i].clone();
                    missing.insert(key);
                }
            }
        }
    }
    seen.missing = missing;
    now.retain(|_, f| f.meta || f.stamps.iter().any(Option::is_some));
    // Served names first, then new ones by name, up to max_views.
    let (served, new): (Vec<_>, Vec<_>) = now.keys().partition(|n| old.contains_key(*n));
    let kept: BTreeSet<&String> = served.into_iter().chain(new).take(cfg.max_views).collect();
    let skipped: BTreeSet<String> = now.keys().filter(|n| !kept.contains(n)).cloned().collect();
    for name in skipped.difference(&seen.skipped) {
        let max = cfg.max_views;
        skipped_lines.push(format!(
            "view {name:?}: past views.max_views = {max}: skipped"
        ));
    }
    seen.skipped = skipped;

    let mut r = Rebuilt {
        views: BTreeMap::new(),
        lines: vec![],
        skipped: skipped_lines,
        changed: false,
    };
    let items = base.n_items();
    for name in kept {
        let files = &now[name];
        let old = old.get(name);
        let changed = !same_model || seen.views.get(name) != Some(files);
        // A file absent for one scan (A15) is not read: its view stays as
        // it is until the file is back or gone for good.
        // ponytail: a swap during that scan has no mask over the new model
        // to keep: the view is unloaded (error) until the next scan.
        let pending = (0..2).any(|i| seen.missing.contains(&(name.clone(), i)));
        if let Some(v) = old
            && same_model
            && (pending || !changed && v.info.error.is_none())
        {
            r.views.insert(name.clone(), v.clone());
            continue;
        }
        seen.views.insert(name.clone(), files.clone());
        let (served, errors) = match old {
            Some(v) => (v.served.clone(), v.errors.clone()),
            None => (
                (0..threads).map(|_| Counter::default()).collect(),
                Arc::default(),
            ),
        };
        let paths =
            [Rule::Include, Rule::Exclude].map(|r| dir.join(format!("{name}{}", r.suffix())));
        let built = if files.meta {
            Err(format!(
                "view {name:?}: {name}.meta.json (view settings) needs a newer server: not served"
            ))
        } else {
            let p = |i: usize| files.stamps[i].as_ref().map(|_| paths[i].as_path());
            build_one(base, name, p(0), p(1))
        };
        let view = match built {
            Ok((model, info)) => {
                r.changed = true;
                View {
                    model: Some(model),
                    info,
                    served,
                    errors,
                }
            }
            Err(e) => {
                // A view in error is retried every scan (a passing EMFILE,
                // EIO), but logged and counted once per change.
                // A kept mask: over this model, and never a meta.json view.
                let keep = old.filter(|v| same_model && !files.meta && v.model.is_some());
                let mut view = match keep {
                    Some(v) => View {
                        model: v.model.as_ref().map(|m| m.view(None, None)),
                        info: v.info.clone(),
                        served,
                        errors,
                    },
                    None => View {
                        model: None,
                        info: ViewInfo {
                            allowed: None,
                            include: None,
                            exclude: None,
                            error: None,
                        },
                        served,
                        errors,
                    },
                };
                view.info.error = Some(e);
                if changed || old.is_none() {
                    view.errors.fetch_add(1, Relaxed);
                    r.lines.push(describe_one(name, &view, items));
                }
                r.views.insert(name.clone(), Arc::new(view));
                continue;
            }
        };
        r.lines.push(describe_one(name, &view, items));
        r.views.insert(name.clone(), Arc::new(view));
    }
    for name in old.keys().filter(|n| !r.views.contains_key(*n)) {
        r.changed = true;
        seen.views.remove(name);
        r.lines
            .push(format!("view {name:?}: its files are gone: removed"));
    }
    seen.views.retain(|n, _| r.views.contains_key(n));
    // A failure alone changes the listing (`error`): installed, not a reload.
    Ok((r.changed || !r.lines.is_empty() || !r.skipped.is_empty()).then_some(r))
}

fn build_one(
    base: &Model,
    name: &str,
    include: Option<&Path>,
    exclude: Option<&Path>,
) -> Result<(Model, ViewInfo), String> {
    // The mtime before the read: a write during it shows as a change next time.
    /// A file, its ids and mtime.
    type Read<'a> = (&'a Path, Vec<i64>, Option<f64>);
    fn read(p: Option<&Path>) -> Result<Option<Read<'_>>, String> {
        p.map(|p| {
            let modified = std::fs::metadata(p)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64());
            let ids = view_file::read_ids(p).map_err(|e| e.to_string())?;
            Ok((p, ids, modified))
        })
        .transpose()
    }
    let named = |e: String| format!("view {name:?}: {e}");
    let include = read(include).map_err(named)?;
    let exclude = read(exclude).map_err(named)?;
    let model = base.view(
        include.as_ref().map(|(_, ids, _)| &ids[..]),
        exclude.as_ref().map(|(_, ids, _)| &ids[..]),
    );
    let file = |f: Option<Read>| {
        f.map(|(p, mut ids, modified)| {
            let known = base.view(Some(&ids), None).n_allowed();
            ids.sort_unstable();
            ids.dedup();
            FileInfo {
                file: p.display().to_string(),
                ids: ids.len(),
                known,
                modified,
            }
        })
    };
    let info = ViewInfo {
        allowed: Some(model.n_allowed()),
        include: file(include),
        exclude: file(exclude),
        error: None,
    };
    Ok((model, info))
}

/// The start-up log's lines: one per view, then the memory they hold.
pub fn describe(views: &BTreeMap<String, Arc<View>>, items: usize) -> Vec<String> {
    let mut lines: Vec<String> = views
        .iter()
        .map(|(name, v)| describe_one(name, v, items))
        .collect();
    let loaded = views.values().filter(|v| v.model.is_some()).count();
    if loaded > 0 {
        let each = items.div_ceil(64) * 8;
        lines.push(format!(
            "views: {loaded} x {each} B = {} KiB",
            (loaded * each).div_ceil(1024)
        ));
    }
    lines
}

/// A view's log line: its files, what it allows (a warning at 0), or the
/// failure it kept its previous ids through (or isn't served for).
pub fn describe_one(name: &str, v: &View, items: usize) -> String {
    let i = &v.info;
    if let Some(e) = &i.error {
        let then = match i.allowed {
            Some(_) => "serving the previous ids",
            None => "not served",
        };
        return format!("{e} ({then})");
    }
    let allowed = i.allowed.unwrap_or_default();
    let file = |kind: &str, f: &Option<FileInfo>| match f {
        Some(f) => format!(" {kind} {} ({} of {} ids known)", f.file, f.known, f.ids),
        None => String::new(),
    };
    let warn = if allowed == 0 {
        " (warning: every answer will be empty)"
    } else {
        ""
    };
    format!(
        "view {name:?}:{}{}, allows {} of {items} items{warn}",
        file("include", &i.include),
        file("exclude", &i.exclude),
        allowed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A14: a scan shares the unchanged views (pointer-equal) and rebuilds
    /// the changed one; over another model it rebuilds every view.
    #[test]
    fn unchanged_views_shared_only_over_the_same_model() {
        let model = || {
            let (indptr, indices, data) = ([0i64, 1, 2, 3], [1i32, 2, 0], [1.0f32; 3]);
            let s = similaripy_core::CsrView::new("S", &indptr, &indices, &data, 3).unwrap();
            let order = similaripy_core::serve::ItemOrder::Auto;
            Model::new(s, Some(vec![10, 20, 30]), None, order).unwrap()
        };
        let dir =
            std::env::temp_dir().join(format!("similaripy-views-share-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.include.txt"), "10\n").unwrap();
        std::fs::write(dir.join("b.exclude.txt"), "20\n").unwrap();
        let cfg = crate::config::Views {
            dir: dir.clone(),
            ..Default::default()
        };
        let (base, mut seen) = (model(), Seen::default());
        let none = BTreeMap::new();
        let v1 = scan(&base, &cfg, &none, &mut seen, true, 1)
            .unwrap()
            .unwrap()
            .views;
        assert!(
            scan(&base, &cfg, &v1, &mut seen, true, 1)
                .unwrap()
                .is_none()
        );
        std::fs::write(dir.join("b.exclude.txt"), "20\n30\n").unwrap();
        let v2 = scan(&base, &cfg, &v1, &mut seen, true, 1)
            .unwrap()
            .unwrap()
            .views;
        assert!(Arc::ptr_eq(&v1["a"], &v2["a"]));
        assert!(!Arc::ptr_eq(&v1["b"], &v2["b"]));
        assert_eq!(v2["b"].info.allowed, Some(1));
        // A swap: nothing shared, the counters kept.
        let v3 = scan(&model(), &cfg, &v2, &mut seen, false, 1)
            .unwrap()
            .unwrap()
            .views;
        assert!(v3.iter().all(|(k, v)| !Arc::ptr_eq(v, &v2[k])));
        assert!(Arc::ptr_eq(&v3["a"].served, &v2["a"].served));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
