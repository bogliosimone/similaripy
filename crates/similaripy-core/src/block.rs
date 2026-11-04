//! `block_size='auto'` (4c step 1b): today's `DEFAULT_BLOCK_SIZE`, or, behind
//! the private switch `SIMILARIPY_AUTO_BLOCK=estimate`, a size estimated from
//! the accumulator width, the L2 cache and the threads sharing it
//! (`4c/estimator-design.md`). Block size never changes results (same
//! addition order), only speed.
// ponytail: env switch until the estimate is validated on x86 + M5 (plan
// step 6); then it becomes 'auto' and the switch goes.

use std::sync::OnceLock;

use crate::DEFAULT_BLOCK_SIZE;

/// Block size for `block_size='auto'` over `n_cols` accumulator slots on
/// `threads` threads; 0 = one block.
pub fn auto_block_size(n_cols: usize, threads: usize) -> usize {
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var("SIMILARIPY_AUTO_BLOCK").is_ok_and(|v| v == "estimate")) {
        estimate(n_cols, threads, l2_cache())
    } else {
        DEFAULT_BLOCK_SIZE
    }
}

/// The estimate: one block while the f32 accumulator fits the thread's L2
/// share (L2 / threads on it); else the largest power of two that fits, at
/// least `DEFAULT_BLOCK_SIZE` (smaller blocks lost 11-41% in every M2 sweep).
/// Unknown cache: `DEFAULT_BLOCK_SIZE`.
// ponytail: width only; the hot-footprint input (work spread over columns)
// only if the flat-column sweep (tonight3) needs it.
pub fn estimate(n_cols: usize, threads: usize, l2: Option<(usize, usize)>) -> usize {
    let Some((bytes, cpus)) = l2 else {
        return DEFAULT_BLOCK_SIZE;
    };
    let share = bytes / threads.clamp(1, cpus.max(1));
    if n_cols.saturating_mul(4) <= share {
        return 0;
    }
    let slots = (share / 4).max(1);
    (1usize << (usize::BITS - 1 - slots.leading_zeros())).max(DEFAULT_BLOCK_SIZE)
}

/// (L2 bytes, logical CPUs sharing one L2) of the performance cores, read
/// once; None where unknown.
pub fn l2_cache() -> Option<(usize, usize)> {
    static L2: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    *L2.get_or_init(detect)
}

#[cfg(target_os = "macos")]
fn detect() -> Option<(usize, usize)> {
    fn sysctl(name: &str) -> Option<Vec<u64>> {
        let c = std::ffi::CString::new(name).ok()?;
        let mut buf = [0u64; 16];
        let mut len = std::mem::size_of_val(&buf);
        // SAFETY: a valid C string, a writable buffer of `len` bytes.
        let rc = unsafe {
            libc::sysctlbyname(
                c.as_ptr(),
                buf.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && len > 0).then(|| {
            // 4-byte values (cpusperl2 is an int) or 8-byte arrays (cacheconfig)
            if len == 4 {
                vec![buf[0] & 0xffff_ffff]
            } else {
                buf[..len / 8].to_vec()
            }
        })
    }
    // Apple silicon: perflevel0 = the performance cores; Intel Macs: hw.*
    let size = sysctl("hw.perflevel0.l2cachesize").or_else(|| sysctl("hw.l2cachesize"))?[0];
    let cpus = sysctl("hw.perflevel0.cpusperl2")
        .map(|v| v[0])
        .or_else(|| sysctl("hw.cacheconfig").and_then(|v| v.get(2).copied()))?;
    (size > 0 && cpus > 0).then_some((size as usize, cpus as usize))
}

#[cfg(target_os = "linux")]
fn detect() -> Option<(usize, usize)> {
    let base = std::path::Path::new("/sys/devices/system/cpu/cpu0/cache");
    for e in std::fs::read_dir(base).ok()?.flatten() {
        let read = |f: &str| {
            std::fs::read_to_string(e.path().join(f))
                .ok()
                .map(|s| s.trim().to_string())
        };
        if read("level").as_deref() != Some("2") || read("type").as_deref() == Some("Instruction") {
            continue;
        }
        return Some((
            parse_size(&read("size")?)?,
            count_cpus(&read("shared_cpu_list")?)?,
        ));
    }
    None
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect() -> Option<(usize, usize)> {
    None
}

/// "1024K", "2M", "512" (bytes) → bytes.
#[cfg_attr(not(any(test, target_os = "linux")), allow(dead_code))]
fn parse_size(s: &str) -> Option<usize> {
    let (num, mul) = match s.as_bytes().last()? {
        b'K' => (&s[..s.len() - 1], 1 << 10),
        b'M' => (&s[..s.len() - 1], 1 << 20),
        b'G' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    num.parse::<usize>().ok().map(|n| n * mul)
}

/// "0-1,8-9" / "0,16" → number of CPUs.
#[cfg_attr(not(any(test, target_os = "linux")), allow(dead_code))]
fn count_cpus(s: &str) -> Option<usize> {
    s.split(',')
        .map(|r| match r.split_once('-') {
            Some((a, b)) => {
                Some(b.trim().parse::<usize>().ok()? + 1 - a.trim().parse::<usize>().ok()?)
            }
            None => r.trim().parse::<usize>().ok().map(|_| 1),
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const M2_PRO: Option<(usize, usize)> = Some((16 << 20, 3));

    #[test]
    fn estimate_matches_the_m2_sweep() {
        // tonight1 sweep winners (4c Deviations 2026-11-04), cosine kernel + serving
        assert_eq!(estimate(84_432, 6, M2_PRO), 0); // ML-32M
        assert_eq!(estimate(934_057, 1, M2_PRO), 0); // Y50M 1t: none
        assert_eq!(estimate(934_057, 6, M2_PRO), 0); // Y50M 6t: none
        assert_eq!(estimate(3_004_578, 1, M2_PRO), 0); // Y500M 1t: none
        assert_eq!(estimate(3_004_578, 6, M2_PRO), 1 << 20); // Y500M 6t: 1M
    }

    #[test]
    fn estimate_floor_and_unknown_cache() {
        let zen4 = Some((1 << 20, 2)); // 1 MB L2 per core, 2 SMT threads
        assert_eq!(estimate(3_004_578, 16, zen4), DEFAULT_BLOCK_SIZE); // 512 KB share → floor
        assert_eq!(estimate(100_000, 16, zen4), 0); // 400 KB fits
        assert_eq!(estimate(3_004_578, 6, None), DEFAULT_BLOCK_SIZE);
        assert_eq!(estimate(3_004_578, 0, M2_PRO), 0); // 0 threads read as 1
        assert_eq!(estimate(usize::MAX, 1, M2_PRO), 4 << 20); // no overflow
    }

    #[test]
    fn sysfs_parsers() {
        assert_eq!(parse_size("1024K"), Some(1 << 20));
        assert_eq!(parse_size("2M"), Some(2 << 20));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size("xK"), None);
        assert_eq!(count_cpus("0-1"), Some(2));
        assert_eq!(count_cpus("0,16"), Some(2));
        assert_eq!(count_cpus("0-3,8-11"), Some(8));
        assert_eq!(count_cpus("a"), None);
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn detects_this_machine() {
        let (bytes, cpus) = l2_cache().expect("L2 readable on macOS / Linux");
        assert!(
            bytes >= 128 << 10 && (1..=256).contains(&cpus),
            "{bytes} {cpus}"
        );
    }
}
