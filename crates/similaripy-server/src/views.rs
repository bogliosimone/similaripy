//! Named views: each a core `Model::view` of the served model
//! (S shared, one bit per item), built at start from the config's id files.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;
use similaripy_core::serve::Model;

use crate::config::{Config, ViewConfig};
use crate::metrics::Counter;

/// One named view: the model behind it, what it was built from, and its
/// requests per server thread.
pub struct View {
    pub model: Model,
    pub info: ViewInfo,
    pub served: Arc<[Counter]>,
}

impl View {
    /// This view again (the mask copied, S shared).
    fn copy(&self) -> View {
        View {
            model: self.model.view(None, None),
            info: self.info.clone(),
            served: self.served.clone(),
        }
    }
}

/// `GET /v1/views`' entry for a view.
#[derive(Serialize, Clone)]
pub struct ViewInfo {
    /// Items the view allows (of the model's `items`).
    pub allowed: usize,
    pub include: Option<FileInfo>,
    pub exclude: Option<FileInfo>,
    /// The last reload's failure (null: none): the view still serves the
    /// files as described here, read before it.
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
/// (an atomic rename). None: unreadable (a change too, when it comes back).
#[derive(PartialEq, Debug)]
pub struct Stamp {
    path: std::path::PathBuf,
    modified: Option<std::time::SystemTime>,
    len: u64,
    #[cfg(unix)]
    inode: (u64, u64),
}

pub fn stamp(path: &Path) -> Option<Stamp> {
    let m = std::fs::metadata(path).ok()?;
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

/// Per view of `cfg`, the stamps of its files.
pub type Stamps = BTreeMap<String, Vec<Option<Stamp>>>;

pub fn stamps(cfg: &Config) -> Stamps {
    cfg.views
        .named
        .iter()
        .map(|(name, v)| (name.clone(), view_stamps(v)))
        .collect()
}

fn view_stamps(v: &ViewConfig) -> Vec<Option<Stamp>> {
    [&v.include_file, &v.exclude_file]
        .into_iter()
        .flatten()
        .map(|p| stamp(p))
        .collect()
}

/// The ids of `path`: one integer per line, blank lines skipped, spaces
/// around a number allowed. Errors name the file (and the line).
pub fn read_ids(path: &Path) -> Result<Vec<i64>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut ids = Vec::new();
    for (i, line) in text.lines().enumerate() {
        // A BOM (Windows / Excel exports) before the first id
        let t = line.trim_start_matches('\u{feff}').trim();
        if t.is_empty() {
            continue;
        }
        let id = t.parse().map_err(|_| {
            format!(
                "{}:{}: {t:?} is not an integer id (numpy: np.savetxt(path, ids, fmt=\"%d\"))",
                path.display(),
                i + 1
            )
        })?;
        ids.push(id);
    }
    Ok(ids)
}

/// Every view of `cfg` over `base`, each with its counters from `served`
/// (one entry per view name, kept across swaps); the first error fails.
pub fn build(
    base: &Model,
    cfg: &Config,
    served: &BTreeMap<String, Arc<[Counter]>>,
) -> Result<BTreeMap<String, View>, String> {
    cfg.check_views()?;
    cfg.views
        .named
        .iter()
        .map(|(name, v)| Ok((name.clone(), build_one(base, name, v, &served[name])?)))
        .collect()
}

/// A reload's outcome: every view, and the names of those whose files
/// changed, rebuilt or failed.
pub struct Rebuilt {
    pub views: BTreeMap<String, View>,
    pub rebuilt: Vec<String>,
    pub failed: Vec<String>,
}

/// The views of `old` whose files changed since `seen` (then updated) or
/// whose last read failed, rebuilt over `base`; one whose files fail to
/// read keeps its old mask with the error; the others stay as they are.
/// None if nothing changed.
pub fn rebuild(
    base: &Model,
    cfg: &Config,
    old: &BTreeMap<String, View>,
    seen: &mut Stamps,
) -> Option<Rebuilt> {
    let mut r = Rebuilt {
        views: BTreeMap::new(),
        rebuilt: vec![],
        failed: vec![],
    };
    for (name, v) in &cfg.views.named {
        let old = &old[name];
        // Stamped before the reads: a write during them is a change next time.
        let now = view_stamps(v);
        let changed = seen[name] != now;
        let view = if !changed && old.info.error.is_none() {
            old.copy()
        } else {
            seen.insert(name.clone(), now);
            match build_one(base, name, v, &old.served) {
                Ok(view) => {
                    r.rebuilt.push(name.clone());
                    view
                }
                // A view in error is retried every check (a passing EMFILE,
                // EIO), but logged and counted once per change.
                Err(e) => {
                    if changed {
                        r.failed.push(name.clone());
                    }
                    let mut kept = old.copy();
                    kept.info.error = Some(e);
                    kept
                }
            }
        };
        r.views.insert(name.clone(), view);
    }
    (!r.rebuilt.is_empty() || !r.failed.is_empty()).then_some(r)
}

fn build_one(
    base: &Model,
    name: &str,
    v: &ViewConfig,
    served: &Arc<[Counter]>,
) -> Result<View, String> {
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
            Ok((p, read_ids(p)?, modified))
        })
        .transpose()
    }
    let named = |e: String| format!("view {name:?}: {e}");
    let include = read(v.include_file.as_deref()).map_err(named)?;
    let exclude = read(v.exclude_file.as_deref()).map_err(named)?;
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
        allowed: model.n_allowed(),
        include: file(include),
        exclude: file(exclude),
        error: None,
    };
    Ok(View {
        model,
        info,
        served: served.clone(),
    })
}

/// The start-up log's lines: one per view, then the memory they hold.
pub fn describe(views: &BTreeMap<String, View>, items: usize) -> Vec<String> {
    let mut lines: Vec<String> = views
        .iter()
        .map(|(name, v)| describe_one(name, v, items))
        .collect();
    if !views.is_empty() {
        let each = items.div_ceil(64) * 8;
        lines.push(format!(
            "views: {} x {each} B = {} KiB",
            views.len(),
            (views.len() * each).div_ceil(1024)
        ));
    }
    lines
}

/// A view's log line: its files, what it allows (a warning at 0), or the
/// failed reload it kept its previous ids through.
pub fn describe_one(name: &str, v: &View, items: usize) -> String {
    let i = &v.info;
    if let Some(e) = &i.error {
        return format!("{e} (reload failed: serving the previous ids)");
    }
    let file = |kind: &str, f: &Option<FileInfo>| match f {
        Some(f) => format!(" {kind} {} ({} of {} ids known)", f.file, f.known, f.ids),
        None => String::new(),
    };
    let warn = if i.allowed == 0 {
        " (warning: every answer will be empty)"
    } else {
        ""
    };
    format!(
        "view {name:?}:{}{}, allows {} of {items} items{warn}",
        file("include", &i.include),
        file("exclude", &i.exclude),
        i.allowed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file removed on drop.
    struct File(std::path::PathBuf);

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
            std::env::temp_dir().join(format!("similaripy-views-{}-{n}.txt", std::process::id()));
        std::fs::write(&p, text).unwrap();
        File(p)
    }

    #[test]
    fn ids_one_per_line_errors_name_file_and_line() {
        let f = file("\u{feff}3\r\n\n  -7 \n10\n");
        assert_eq!(read_ids(&f.0).unwrap(), [3, -7, 10]);
        let f = file("3\n4\nabc\n");
        let e = read_ids(&f.0).unwrap_err();
        assert!(
            e.contains(&format!("{}:3:", f.0.display())) && e.contains("abc"),
            "{e}"
        );
        let f = file("1 2\n");
        assert!(read_ids(&f.0).unwrap_err().contains(":1:"));
        let missing = Path::new("/nonexistent/similaripy-views.txt");
        assert!(
            read_ids(missing)
                .unwrap_err()
                .contains("/nonexistent/similaripy-views.txt")
        );
    }
}
