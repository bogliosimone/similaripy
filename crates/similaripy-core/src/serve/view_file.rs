//! The view file format (stable in 1.0; `docs/server.md`, "View folder
//! format"): a view `NAME` is `NAME.include.txt` (the items it allows) and /
//! or `NAME.exclude.txt` (the items it removes) in one folder, one integer id
//! per line, full-line `#` comments; `NAME.meta.json` is reserved for later
//! view settings. Read by the server's folder scan, written by pipelines (and
//! a later Python `write_view`). Cold code: files, never the request path.

use std::fs::File;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::Error;
use crate::store::io;

/// A view's id file: the rule it holds; `as usize` = its slot in an
/// `[include, exclude]` pair (fixed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Only these items (absent: every item; empty: none).
    Include = 0,
    /// Never these items, after `Include` (absent or empty: none).
    Exclude = 1,
}

impl Rule {
    pub fn suffix(self) -> &'static str {
        match self {
            Rule::Include => ".include.txt",
            Rule::Exclude => ".exclude.txt",
        }
    }
}

/// `NAME.meta.json`: reserved for a later server's view settings (a 1.0
/// server lists the view with an error and doesn't serve it).
pub const META_SUFFIX: &str = ".meta.json";

/// The next temporary file number of `write_ids` in this process.
static TMP_N: AtomicU64 = AtomicU64::new(0);

/// A view name: 1-64 of `a-z 0-9 _ -` (Prometheus labels and URLs without
/// escaping, settable as a flag value, no path separator).
pub fn valid_name(name: &str) -> bool {
    let ok = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-".contains(&b);
    (1..=64).contains(&name.len()) && name.bytes().all(ok)
}

/// What a file name in a views folder is: `Some((name, Some(rule)))` an id
/// file, `Some((name, None))` a `NAME.meta.json`, None not a view file (a
/// name starting with `.`, another suffix). The name is not checked
/// (`valid_name`: the caller skips and reports a bad one).
pub fn parse_file_name(file: &str) -> Option<(&str, Option<Rule>)> {
    if file.starts_with('.') {
        return None;
    }
    [Rule::Include, Rule::Exclude]
        .into_iter()
        .find_map(|r| Some((file.strip_suffix(r.suffix())?, Some(r))))
        .or_else(|| Some((file.strip_suffix(META_SUFFIX)?, None)))
}

/// The ids of one id file: one integer (i64) per line with spaces or tabs
/// around it, a BOM before the first, blank lines and full-line `#`
/// comments skipped. Any other line fails the file: the error names the file
/// and line (`ErrorKind::InvalidData`; a read failure: its own kind).
pub fn read_ids(path: &Path) -> Result<Vec<i64>, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Io(e.kind(), format!("{}: {e}", path.display())))?;
    let mut ids = Vec::new();
    for (i, line) in text.lines().enumerate() {
        // A BOM (Windows / Excel exports) before the first id
        let t = line.trim_start_matches('\u{feff}').trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let id = t.parse().map_err(|_| {
            let e = format!(
                "{}:{}: {t:?} is not an integer id (numpy: np.savetxt(path, ids, fmt=\"%d\"))",
                path.display(),
                i + 1
            );
            Error::Io(ErrorKind::InvalidData, e)
        })?;
        ids.push(id);
    }
    Ok(ids)
}

/// Writes `ids` as `dir/NAME<rule suffix>`, one per line, replacing the file
/// atomically: a temporary `.NAME...tmp` in `dir` (a reader scanning the
/// folder skips dot names), synced, renamed over the target, then `dir`
/// synced (the rename survives a crash). A reader sees the old file or the
/// new one, never part of one. A bad name is `InvalidInput` before any file
/// operation; on a failure the temporary file is removed. An error from the
/// last step (syncing `dir`) comes after the rename: the file is replaced,
/// only its survival of a crash is unknown. Returns the path.
pub fn write_ids(dir: &Path, name: &str, rule: Rule, ids: &[i64]) -> Result<PathBuf, Error> {
    if !valid_name(name) {
        return Err(Error::InvalidInput(format!(
            "view name {name:?}: 1-64 of a-z 0-9 _ -"
        )));
    }
    let file = format!("{name}{}", rule.suffix());
    let path = dir.join(&file);
    let mut text = String::with_capacity(ids.len() * 12);
    for id in ids {
        use std::fmt::Write;
        let _ = writeln!(text, "{id}");
    }
    // A new temp file, never another writer's: the name repeats across
    // processes (PID 1 in every container) and restarts (a killed writer's
    // stale temp), so a taken one moves on to the next.
    let (mut f, tmp) = (0..64)
        .map(|_| {
            let n = TMP_N.fetch_add(1, Relaxed);
            dir.join(format!(".{file}.{}.{n}.tmp", std::process::id()))
        })
        .find_map(|tmp| match File::create_new(&tmp) {
            Err(e) if e.kind() == ErrorKind::AlreadyExists => None,
            r => Some(r.map(|f| (f, tmp.clone())).map_err(io(&tmp))),
        })
        .unwrap_or_else(|| {
            let e = format!("{}: 64 temporary file names taken", dir.display());
            Err(Error::Io(ErrorKind::AlreadyExists, e))
        })?;
    let written = (|| {
        f.write_all(text.as_bytes()).map_err(io(&tmp))?;
        f.sync_all().map_err(io(&tmp))?;
        drop(f);
        std::fs::rename(&tmp, &path).map_err(io(&path))
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    // The rename itself on disk (a directory opens read-only on unix; Miri
    // supports fsync on files only).
    #[cfg(all(unix, not(miri)))]
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(io(dir))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder in the temp dir, removed on drop.
    struct Dir(PathBuf);

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn dir() -> Dir {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Relaxed);
        let d =
            std::env::temp_dir().join(format!("similaripy-view-file-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }

    fn read(d: &Dir, text: &str) -> Result<Vec<i64>, Error> {
        let p = d.0.join("v.include.txt");
        std::fs::write(&p, text).unwrap();
        read_ids(&p)
    }

    #[test]
    fn syntax_ids_comments_blank_lines_errors_name_file_and_line() {
        let d = dir();
        let ok = "\u{feff}# in_stock, job export 2026-11-24\n3\r\n\n  -7 \n\t+5\t\n  # x\n#\n9223372036854775807";
        assert_eq!(read(&d, ok).unwrap(), [3, -7, 5, i64::MAX]);
        assert_eq!(read(&d, "").unwrap(), Vec::<i64>::new());
        assert_eq!(read(&d, "# only a comment\n").unwrap(), Vec::<i64>::new());
        let p = d.0.join("v.include.txt").display().to_string();
        for (bad, line) in [
            ("3\n4\nabc\n", 3),
            ("1 2\n", 1),
            ("1 # inline\n", 1),
            ("1,2\n", 1),
            ("1.0\n", 1),
            ("9223372036854775808\n", 1),
        ] {
            match read(&d, bad) {
                Err(Error::Io(ErrorKind::InvalidData, e)) => {
                    assert!(e.contains(&format!("{p}:{line}:")), "{bad:?}: {e}")
                }
                other => panic!("{bad:?}: {other:?}"),
            }
        }
        let missing = Path::new("/nonexistent/similaripy-views.txt");
        match read_ids(missing) {
            Err(Error::Io(ErrorKind::NotFound, e)) => assert!(e.contains("/nonexistent/"), "{e}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn names() {
        for ok in ["a", "in_stock", "eu-2026", &"x".repeat(64)] {
            assert!(valid_name(ok), "{ok}");
        }
        for bad in [
            "",
            "InStock",
            "in.stock",
            "a b",
            "../x",
            "a/b",
            &"x".repeat(65),
        ] {
            assert!(!valid_name(bad), "{bad}");
        }
        assert_eq!(
            parse_file_name("in_stock.include.txt"),
            Some(("in_stock", Some(Rule::Include)))
        );
        assert_eq!(
            parse_file_name("eu.exclude.txt"),
            Some(("eu", Some(Rule::Exclude)))
        );
        assert_eq!(parse_file_name("eu.meta.json"), Some(("eu", None)));
        assert_eq!(
            parse_file_name("Bad Name.include.txt"),
            Some(("Bad Name", Some(Rule::Include)))
        );
        for ignored in [
            ".eu.include.txt",
            "..data",
            "eu.txt",
            "notes.md",
            "eu.include.txt.gz",
        ] {
            assert_eq!(parse_file_name(ignored), None, "{ignored}");
        }
    }

    #[test]
    fn write_then_read_round_trip_atomic_and_no_temp_left() {
        let d = dir();
        let ids = [7, -1, i64::MIN, i64::MAX, 7];
        let p = write_ids(&d.0, "in_stock", Rule::Include, &ids).unwrap();
        assert_eq!(p, d.0.join("in_stock.include.txt"));
        assert_eq!(read_ids(&p).unwrap(), ids);
        // Replaced: a new file (a new inode on unix: what a reader sees change).
        let p2 = write_ids(&d.0, "in_stock", Rule::Include, &[]).unwrap();
        assert_eq!(read_ids(&p2).unwrap(), Vec::<i64>::new());
        write_ids(&d.0, "in_stock", Rule::Exclude, &[1]).unwrap();
        let mut files: Vec<String> = std::fs::read_dir(&d.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        files.sort();
        assert_eq!(files, ["in_stock.exclude.txt", "in_stock.include.txt"]);
    }

    #[test]
    fn write_refuses_a_bad_name_before_any_file_and_cleans_up_on_failure() {
        let d = dir();
        for bad in ["../escape", "a/b", "", "InStock"] {
            match write_ids(&d.0, bad, Rule::Include, &[1]) {
                Err(Error::InvalidInput(e)) => assert!(e.contains("view name"), "{e}"),
                other => panic!("{bad:?}: {other:?}"),
            }
        }
        assert_eq!(std::fs::read_dir(&d.0).unwrap().count(), 0);
        assert!(!d.0.parent().unwrap().join("escape.include.txt").exists());
        // The target is a directory: the rename fails, the temp is removed.
        std::fs::create_dir(d.0.join("v.include.txt")).unwrap();
        assert!(write_ids(&d.0, "v", Rule::Include, &[1]).is_err());
        let left: Vec<_> = std::fs::read_dir(&d.0).unwrap().collect();
        assert_eq!(left.len(), 1, "only the directory: {left:?}");
        // Stale temps (a killed writer, another pod with the same PID): the
        // next free name is used, the stale ones are left alone.
        std::fs::remove_dir(d.0.join("v.include.txt")).unwrap();
        let next = TMP_N.load(Relaxed);
        for n in next..next + 3 {
            let stale = format!(".v.include.txt.{}.{n}.tmp", std::process::id());
            std::fs::write(d.0.join(stale), "1\n").unwrap();
        }
        let p = write_ids(&d.0, "v", Rule::Include, &[2]).unwrap();
        assert_eq!(read_ids(&p).unwrap(), [2]);
        assert_eq!(std::fs::read_dir(&d.0).unwrap().count(), 4);
        // A missing folder: an error naming it.
        let gone = d.0.join("missing");
        match write_ids(&gone, "v", Rule::Include, &[1]) {
            Err(Error::Io(ErrorKind::NotFound, e)) => assert!(e.contains("missing"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
}
