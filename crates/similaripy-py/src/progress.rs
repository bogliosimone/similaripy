//! Progress bar with the same output as the C++ `progress::ProgressBar`
//! (`progress_bar.h`): stderr, `\r` redraw at most `refresh_hz` times a second,
//! timer started by the first finished row. No ticker thread: the worker that
//! finishes a row after the refresh interval redraws (D25).

use std::cell::Cell;
use std::io::Write;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use pyo3::prelude::*;

/// Rows a worker counts locally before adding them to the shared counter: one
/// shared atomic add and clock read per batch instead of per row (~+8% total
/// time on 2M tiny rows when per row). `close` sets the final count anyway.
const BATCH: usize = 64;

thread_local! {
    /// (bar this worker last ticked, rows not yet added to it). Switching bars
    /// (global pool shared by concurrent calls) drops the pending rows: display only.
    static PENDING: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

struct Line {
    desc: String,
    /// Length of the last drawn line, to blank it when the next is shorter.
    last_len: usize,
}

#[pyclass(frozen)]
pub struct ProgressBar {
    total: usize,
    width: usize,
    interval_ms: u64,
    count: AtomicUsize,
    created: Instant,
    /// Set by the first finished row (prep time is not counted, as in C++).
    start: OnceLock<Instant>,
    /// Last redraw, in ms since `created`.
    last_draw_ms: AtomicU64,
    line: Mutex<Line>,
}

#[pymethods]
impl ProgressBar {
    #[new]
    fn new(total: usize, refresh_hz: u64, width: usize) -> Self {
        Self {
            total,
            width,
            interval_ms: (1000 / refresh_hz.max(1)).max(1),
            count: AtomicUsize::new(0),
            created: Instant::now(),
            start: OnceLock::new(),
            last_draw_ms: AtomicU64::new(0),
            line: Mutex::new(Line {
                desc: "Progress".into(),
                last_len: 0,
            }),
        }
    }

    fn set_description(&self, desc: &str) {
        let mut line = self.lock();
        line.desc = desc.into();
        self.render(&mut line);
        self.last_draw_ms.store(self.now_ms(), Ordering::Relaxed);
    }

    #[pyo3(signature = (final_desc=""))]
    fn close(&self, final_desc: &str) {
        let mut line = self.lock();
        if !final_desc.is_empty() {
            line.desc = final_desc.into();
        }
        self.count.store(self.total, Ordering::Relaxed);
        let elapsed = self.elapsed();
        let rate = if elapsed > 0.0 {
            self.total as f64 / elapsed
        } else {
            0.0
        };
        let out = format!(
            "{}: 100%|{}| {}/{} [{}, {rate:.2}it/s]",
            line.desc,
            "#".repeat(self.width),
            self.total,
            self.total,
            fmt_time(elapsed),
        );
        draw(&mut line, &out, true);
    }
}

impl ProgressBar {
    /// Called by the workers after each finished row.
    pub fn tick(&self) {
        if self.start.get().is_none() {
            self.count.fetch_add(1, Ordering::Relaxed);
            let _ = self.start.set(Instant::now());
            self.last_draw_ms.store(self.now_ms(), Ordering::Relaxed);
            return;
        }
        let id = self as *const Self as usize;
        let ready = PENDING.with(|p| {
            let (bar, n) = p.get();
            let n = if bar == id { n + 1 } else { 1 };
            let ready = if n == BATCH { n } else { 0 };
            p.set((id, n - ready));
            ready
        });
        if ready == 0 {
            return;
        }
        self.count.fetch_add(ready, Ordering::Relaxed);
        if !self.due() {
            return;
        }
        // A busy lock means another worker is drawing: skip, never wait.
        if let Ok(mut line) = self.line.try_lock()
            && self.due()
        {
            self.render(&mut line);
            self.last_draw_ms.store(self.now_ms(), Ordering::Relaxed);
        }
    }

    fn due(&self) -> bool {
        self.now_ms()
            .saturating_sub(self.last_draw_ms.load(Ordering::Relaxed))
            >= self.interval_ms
    }

    fn now_ms(&self) -> u64 {
        self.created.elapsed().as_millis() as u64
    }

    fn elapsed(&self) -> f64 {
        self.start.get().map_or(0.0, |s| s.elapsed().as_secs_f64())
    }

    fn lock(&self) -> MutexGuard<'_, Line> {
        self.line.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn render(&self, line: &mut Line) {
        // min: rows pending for a dropped bar at the same address can overshoot.
        let total = self.total;
        let cur = self.count.load(Ordering::Relaxed).min(total);
        let elapsed = self.elapsed();
        let rate = if elapsed > 0.0 {
            cur as f64 / elapsed
        } else {
            0.0
        };
        let (pct, filled) = if total > 0 {
            (
                (cur as f64 * 100.0 / total as f64) as usize,
                (cur * self.width / total).min(self.width),
            )
        } else {
            (0, 0)
        };
        let eta = if rate > 0.01 && cur < total && cur > 0 {
            let remaining = (total - cur) as f64 / rate;
            if remaining < 86400.0 {
                fmt_time(remaining)
            } else {
                "--:--".into()
            }
        } else {
            "--:--".into()
        };
        let out = format!(
            "{}: {pct:>3}%|{}{}| {cur}/{total} [{}<{eta}, {rate:.2}it/s]",
            line.desc,
            "#".repeat(filled),
            " ".repeat(self.width - filled),
            fmt_time(elapsed),
        );
        draw(line, &out, false);
    }
}

/// `\r`-redraws `out`, blanking a longer previous line first.
fn draw(line: &mut Line, out: &str, newline: bool) {
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r");
    if out.len() < line.last_len {
        let _ = write!(err, "{}\r", " ".repeat(line.last_len));
    }
    let _ = write!(err, "{out}{}", if newline { "\n" } else { "" });
    let _ = err.flush();
    line.last_len = out.len();
}

/// `MM:SS`, as the C++ bar.
fn fmt_time(secs: f64) -> String {
    let s = secs as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}
