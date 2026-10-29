//! Saved models: one `.npy` file per array (header written and checked here,
//! readable by `numpy.load`) plus `meta.json`. Arrays are read into a `Vec`
//! or mapped read-only (`memmap2`). A mapped file must not change while
//! mapped (SIGBUS), so saving never rewrites a file in place: it writes new
//! files (`File::create_new`) and replaces a saved model by renaming
//! directories.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Deref, Range};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::Error;

const MAGIC: &[u8] = b"\x93NUMPY";

/// Element types stored as `.npy`.
///
/// # Safety
/// Plain data: no padding bytes, every bit pattern a valid value.
pub(crate) unsafe trait Pod: Copy + Default + Send + Sync + 'static {
    /// numpy's little-endian `descr`.
    const DESCR: &'static str;
}

// SAFETY: plain numbers.
unsafe impl Pod for i32 {
    const DESCR: &'static str = "<i4";
}
// SAFETY: plain numbers.
unsafe impl Pod for i64 {
    const DESCR: &'static str = "<i8";
}
// SAFETY: plain numbers.
unsafe impl Pod for u64 {
    const DESCR: &'static str = "<u8";
}
// SAFETY: plain numbers (every bit pattern is some f32, NaN included).
unsafe impl Pod for f32 {
    const DESCR: &'static str = "<f4";
}

fn bytes<T: Pod>(s: &[T]) -> &[u8] {
    // SAFETY: Pod has no padding; u8 needs no alignment; same length in bytes.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), size_of_val(s)) }
}

fn bytes_mut<T: Pod>(s: &mut [T]) -> &mut [u8] {
    // SAFETY: as `bytes`, and any bytes written make a valid Pod value.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast(), size_of_val(s)) }
}

/// `b` as `&[T]`; None unless aligned for `T` and whole elements.
fn cast<T: Pod>(b: &[u8]) -> Option<&[T]> {
    if b.as_ptr().align_offset(align_of::<T>()) != 0 || !b.len().is_multiple_of(size_of::<T>()) {
        return None;
    }
    // SAFETY: aligned, in bounds, whole elements; Pod accepts any bytes.
    Some(unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / size_of::<T>()) })
}

/// An array owned, or mapped read-only from a saved model's file (shared
/// by the models made from it: views, trimmed).
pub(crate) enum Buf<T> {
    Owned(Vec<T>),
    /// The byte range of the elements, checked by `cast` at load.
    Mapped(Arc<Mmap>, Range<usize>, PhantomData<T>),
}

impl<T: Pod> Deref for Buf<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Buf::Owned(v) => v,
            Buf::Mapped(map, range, _) => cast(&map[range.clone()]).expect("checked at load"),
        }
    }
}

impl<T: Pod> Clone for Buf<T> {
    fn clone(&self) -> Self {
        match self {
            Buf::Owned(v) => Buf::Owned(v.clone()),
            Buf::Mapped(map, range, _) => Buf::Mapped(Arc::clone(map), range.clone(), PhantomData),
        }
    }
}

impl<T> From<Vec<T>> for Buf<T> {
    fn from(v: Vec<T>) -> Self {
        Buf::Owned(v)
    }
}

pub(crate) fn io(path: &Path) -> impl Fn(std::io::Error) -> Error + '_ {
    move |e| Error::Io(e.kind(), format!("{}: {e}", path.display()))
}

fn bad(path: &Path, msg: impl std::fmt::Display) -> Error {
    Error::InvalidInput(format!("{}: {msg}", path.display()))
}

fn little_endian() -> Result<(), Error> {
    if cfg!(target_endian = "big") {
        return Err(Error::InvalidInput(
            "saved models need a little-endian machine".into(),
        ));
    }
    Ok(())
}

/// numpy's format 1.0 header for a 1-d C-order array of `n` elements: data
/// starts at a multiple of 64 bytes (aligned for any `T` when mapped).
fn header<T: Pod>(n: usize) -> Vec<u8> {
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({n},), }}",
        T::DESCR
    );
    let len = (10 + dict.len() + 1).next_multiple_of(64) - 10;
    let mut h = Vec::with_capacity(10 + len);
    h.extend_from_slice(MAGIC);
    h.extend_from_slice(&[1, 0]);
    h.extend_from_slice(&(len as u16).to_le_bytes());
    h.extend_from_slice(dict.as_bytes());
    h.resize(10 + len - 1, b' ');
    h.push(b'\n');
    h
}

/// (elements, data offset) of a `.npy` header for a 1-d C-order array of
/// `T`: anything else (another dtype, big-endian, Fortran order, objects,
/// more dimensions) is an error. Versions 1.0-3.0, as numpy writes them.
fn read_header<T: Pod>(r: &mut impl Read) -> Result<(usize, usize), String> {
    let short = |_| "truncated header".to_string();
    let mut pre = [0u8; 8];
    r.read_exact(&mut pre).map_err(short)?;
    if &pre[..6] != MAGIC {
        return Err("not a .npy file".into());
    }
    let (len, start) = match pre[6] {
        1 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).map_err(short)?;
            (u16::from_le_bytes(b) as usize, 10)
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).map_err(short)?;
            (u32::from_le_bytes(b) as usize, 12)
        }
        v => return Err(format!("unsupported .npy version {v}")),
    };
    if len > 1 << 16 {
        return Err("header too long".into());
    }
    let mut h = vec![0u8; len];
    r.read_exact(&mut h).map_err(short)?;
    let want = || format!("expected a 1-d C-order array of {}", T::DESCR);
    let h = std::str::from_utf8(&h).map_err(|_| want())?;
    // numpy pads with spaces and ends with '\n'
    let n = h
        .trim_end()
        .strip_prefix(&format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': (",
            T::DESCR
        ))
        .and_then(|r| r.strip_suffix(",), }"))
        .and_then(|d| d.parse::<usize>().ok())
        .ok_or_else(want)?;
    Ok((n, start + len))
}

/// Writes `a` as `dir/name.npy`; never replaces an existing file.
pub(crate) fn write<T: Pod>(dir: &Path, name: &str, a: &[T]) -> Result<(), Error> {
    little_endian()?;
    let path = dir.join(format!("{name}.npy"));
    let mut f = File::create_new(&path).map_err(io(&path))?;
    f.write_all(&header::<T>(a.len())).map_err(io(&path))?;
    f.write_all(bytes(a)).map_err(io(&path))?;
    f.sync_all().map_err(io(&path))
}

/// Reads `dir/name.npy`, mapped or into a `Vec`; the header and the file
/// size are checked before anything is allocated.
pub(crate) fn read<T: Pod>(dir: &Path, name: &str, mmap: bool) -> Result<Buf<T>, Error> {
    little_endian()?;
    let path = dir.join(format!("{name}.npy"));
    let mut f = File::open(&path).map_err(io(&path))?;
    let file_len = f.metadata().map_err(io(&path))?.len();
    let (n, start) = read_header::<T>(&mut f).map_err(|m| bad(&path, m))?;
    let end = n
        .checked_mul(size_of::<T>())
        .and_then(|b| b.checked_add(start))
        .filter(|&end| end as u64 == file_len)
        .ok_or_else(|| bad(&path, "file size does not match its header"))?;
    if mmap {
        // SAFETY: a read-only map. The file must not change while mapped:
        // saves never rewrite a file in place (documented for other writers).
        let map = unsafe { Mmap::map(&f) }.map_err(io(&path))?;
        if map.len() < end {
            return Err(bad(&path, "file changed while loading"));
        }
        if cast::<T>(&map[start..end]).is_none() {
            return Err(bad(&path, "data not aligned"));
        }
        Ok(Buf::Mapped(Arc::new(map), start..end, PhantomData))
    } else {
        let mut v = vec![T::default(); n];
        f.read_exact(bytes_mut(&mut v)).map_err(io(&path))?;
        Ok(Buf::Owned(v))
    }
}

/// Writes `text` as `dir/name`; never replaces an existing file.
pub(crate) fn write_text(dir: &Path, name: &str, text: &str) -> Result<(), Error> {
    let path = dir.join(name);
    let mut f = File::create_new(&path).map_err(io(&path))?;
    f.write_all(text.as_bytes()).map_err(io(&path))?;
    f.sync_all().map_err(io(&path))
}

/// Runs `write` into a new sibling directory, then renames it to `dir`, so
/// `dir` only ever appears complete (a failed or concurrent save never
/// leaves a partial model there). `dir` may be missing or empty; a
/// non-empty one is an error unless `overwrite` and `is_model` says it holds
/// a saved model and nothing else: it is then renamed away first and
/// removed, so readers that mapped the old files keep them. Between the two
/// renames `dir` doesn't exist (a concurrent load fails): one saver at a
/// time. Never through a symlink.
pub(crate) fn save_dir(
    dir: &Path,
    overwrite: bool,
    is_model: impl Fn(&Path) -> bool,
    write: impl Fn(&Path) -> Result<(), Error>,
) -> Result<(), Error> {
    let exists = |msg: &str| {
        Err(Error::Io(
            std::io::ErrorKind::AlreadyExists,
            format!("{}: {msg}", dir.display()),
        ))
    };
    if fs::symlink_metadata(dir).is_ok_and(|m| m.file_type().is_symlink()) {
        return exists(
            "a symlink: not replaced (save to a new directory and point the link at it)",
        );
    }
    let occupied = dir.exists() && fs::read_dir(dir).map_err(io(dir))?.next().is_some();
    if occupied && !overwrite {
        return exists("not empty (overwrite=True replaces a saved model)");
    }
    if occupied && !is_model(dir) {
        return exists("not only a saved model: not replaced");
    }
    if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(io(parent))?;
    }
    let new = sibling(dir, "new")?;
    fs::create_dir(&new).map_err(io(&new))?;
    if let Err(e) = write(&new) {
        let _ = fs::remove_dir_all(&new);
        return Err(e);
    }
    if !occupied {
        if dir.exists() {
            let _ = fs::remove_dir(dir); // empty: only an empty directory goes
        }
        if let Err(e) = fs::rename(&new, dir) {
            let _ = fs::remove_dir_all(&new);
            return Err(io(dir)(e));
        }
        return Ok(());
    }
    let old = sibling(dir, "old")?;
    if let Err(e) = fs::rename(dir, &old) {
        let _ = fs::remove_dir_all(&new);
        return Err(io(dir)(e));
    }
    if let Err(e) = fs::rename(&new, dir) {
        // Another saver got in between: leave its model, drop both copies.
        // Nothing at `dir` and the old model can't go back: keep it.
        let _ = fs::remove_dir_all(&new);
        if fs::rename(&old, dir).is_err() {
            if !dir.exists() {
                return Err(Error::Io(
                    e.kind(),
                    format!(
                        "not saved to {}; the replaced model is left at {}: {e}",
                        dir.display(),
                        old.display()
                    ),
                ));
            }
            let _ = fs::remove_dir_all(&old);
        }
        return Err(io(dir)(e));
    }
    fs::remove_dir_all(&old).map_err(|e| {
        Error::Io(
            e.kind(),
            format!(
                "saved to {}, but the replaced model is left at {}: {e}",
                dir.display(),
                old.display()
            ),
        )
    })
}

/// `.name.tag-pid-nanos` next to `dir`.
fn sibling(dir: &Path, tag: &str) -> Result<PathBuf, Error> {
    let name = dir
        .file_name()
        .ok_or_else(|| Error::InvalidInput(format!("{}: not a directory name", dir.display())))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    Ok(dir.with_file_name(format!(
        ".{}.{tag}-{}-{nanos}",
        name.to_string_lossy(),
        std::process::id()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_numpys_and_round_trips() {
        let h = header::<f32>(5);
        assert_eq!(h.len() % 64, 0);
        assert_eq!(
            &h[10..],
            format!(
                "{:<117}\n",
                "{'descr': '<f4', 'fortran_order': False, 'shape': (5,), }"
            )
            .as_bytes()
        );
        assert_eq!(read_header::<f32>(&mut &h[..]), Ok((5, 128)));
        assert!(read_header::<i32>(&mut &h[..]).is_err()); // other dtype
        assert!(read_header::<f32>(&mut &h[..20]).is_err()); // truncated
        let mut v2 = h.clone();
        v2[6] = 9;
        assert!(read_header::<f32>(&mut &v2[..]).is_err());
        // as numpy writes some arrays: spaces for growth after the dict, version 2.0
        let dict =
            "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }                    \n";
        let mut np2 = b"\x93NUMPY\x02\x00".to_vec();
        np2.extend((dict.len() as u32).to_le_bytes());
        np2.extend(dict.as_bytes());
        assert_eq!(read_header::<i64>(&mut &np2[..]), Ok((3, 12 + dict.len())));
        for bad in [
            "{'descr': '>i8', 'fortran_order': False, 'shape': (3,), }",
            "{'descr': '<i8', 'fortran_order': True, 'shape': (3,), }",
            "{'descr': '<i8', 'fortran_order': False, 'shape': (3, 2), }",
            "{'descr': '|O', 'fortran_order': False, 'shape': (3,), }",
            "{'descr': '<i8', 'fortran_order': False, 'shape': (-3,), }",
        ] {
            let mut b = b"\x93NUMPY\x01\x00".to_vec();
            b.extend((bad.len() as u16).to_le_bytes());
            b.extend(bad.as_bytes());
            assert!(read_header::<i64>(&mut &b[..]).is_err(), "{bad}");
        }
    }

    #[test]
    fn cast_checks_alignment_and_length() {
        let words = [1u64, 2, 3];
        let b = bytes(&words);
        assert_eq!(cast::<u64>(b), Some(&words[..]));
        assert_eq!(cast::<i32>(b).map(<[i32]>::len), Some(6));
        assert!(cast::<u64>(&b[4..]).is_none()); // misaligned
        assert!(cast::<u64>(&b[..12]).is_none()); // not whole elements
        let mut v = [0i32; 2];
        bytes_mut(&mut v).copy_from_slice(&[1, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(v, [1, -1]);
    }
}
