//! Named views (V2, S5c): each a core `Model::view` of the served model
//! (S shared, one bit per item), built at start from the config's id files.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;
use similaripy_core::serve::Model;

use crate::config::Config;
use crate::metrics::Counter;

/// One named view: the model behind it, what it was built from, and its
/// requests per server thread.
pub struct View {
    pub model: Model,
    pub info: ViewInfo,
    pub served: Arc<[Counter]>,
}

/// `GET /v1/views`' entry for a view.
#[derive(Serialize)]
pub struct ViewInfo {
    /// Items the view allows (of the model's `items`).
    pub allowed: usize,
    pub include: Option<FileInfo>,
    pub exclude: Option<FileInfo>,
}

/// An id file: distinct ids in it, of which the model has `known` (the rest
/// are ignored: a file in another id space reads "0 known").
#[derive(Serialize)]
pub struct FileInfo {
    pub file: String,
    pub ids: usize,
    pub known: usize,
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
/// (one entry per view name, kept across swaps).
pub fn build(
    base: &Model,
    cfg: &Config,
    served: &BTreeMap<String, Arc<[Counter]>>,
) -> Result<BTreeMap<String, View>, String> {
    cfg.check_views()?;
    fn read(p: Option<&Path>) -> Result<Option<(&Path, Vec<i64>)>, String> {
        p.map(|p| Ok((p, read_ids(p)?))).transpose()
    }
    let mut views = BTreeMap::new();
    for (name, v) in &cfg.views {
        let named = |e: String| format!("view {name:?}: {e}");
        let include = read(v.include_file.as_deref()).map_err(named)?;
        let exclude = read(v.exclude_file.as_deref()).map_err(named)?;
        let model = base.view(
            include.as_ref().map(|(_, ids)| &ids[..]),
            exclude.as_ref().map(|(_, ids)| &ids[..]),
        );
        let file = |f: Option<(&Path, Vec<i64>)>| {
            f.map(|(p, mut ids)| {
                let known = allowed(&base.view(Some(&ids), None));
                ids.sort_unstable();
                ids.dedup();
                FileInfo {
                    file: p.display().to_string(),
                    ids: ids.len(),
                    known,
                }
            })
        };
        let info = ViewInfo {
            allowed: allowed(&model),
            include: file(include),
            exclude: file(exclude),
        };
        let served = served[name].clone();
        views.insert(
            name.clone(),
            View {
                model,
                info,
                served,
            },
        );
    }
    Ok(views)
}

/// Items `m`'s mask allows (all without a mask). The last word's bits
/// past `n_items` can be set (an exclude-only view starts from all ones):
/// not items, not counted.
fn allowed(m: &Model) -> usize {
    let n = m.n_items();
    let Some(bits) = m.parts().allowed else {
        return n;
    };
    let full: usize = bits[..n / 64].iter().map(|w| w.count_ones() as usize).sum();
    let tail = bits
        .get(n / 64)
        .map_or(0, |w| (w & ((1 << (n % 64)) - 1)).count_ones());
    full + tail as usize
}

/// The start-up log's lines: one per view, then the memory they hold.
pub fn describe(views: &BTreeMap<String, View>, items: usize) -> Vec<String> {
    let file = |kind: &str, f: &Option<FileInfo>| match f {
        Some(f) => format!(" {kind} {} ({} of {} ids known)", f.file, f.known, f.ids),
        None => String::new(),
    };
    let mut lines: Vec<String> = views
        .iter()
        .map(|(name, v)| {
            let i = &v.info;
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
        })
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
