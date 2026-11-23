//! Progress bar for `verbose=True`, written to Python's `sys.stderr`.
//!
//! The workers only count rows: each adds to its own cache-line slot
//! (`tick`), no clock, no lock, no I/O. One drawing thread per bar wakes every
//! `INTERVAL` (or at once on a phase change), sums the slots and writes the
//! line, taking the GIL only for the write. Two modes, picked by the caller:
//! live (`\r` redraws: a terminal, Jupyter) and log (one line per phase and
//! per 10% of the rows, for files and CI logs). One live bar per process: a
//! second concurrent bar falls back to log lines instead of sharing the line.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::PyType;

const INTERVAL: Duration = Duration::from_millis(100);
/// Bar cells at most; fewer when a terminal is too narrow for the line.
const WIDTH: usize = 25;
/// Eighths of a bar cell, for a smooth fill.
const PARTIAL: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];

/// A live bar owns the terminal line.
static LIVE: AtomicBool = AtomicBool::new(false);
/// Every write takes this lock: true while the cursor sits at the end of a
/// live line, so a log line from another bar first moves to a new line.
static MID_LINE: Mutex<bool> = Mutex::new(false);

/// One worker's row count, on its own cache line (128 B: Apple M-series).
#[repr(align(128))]
struct Slot(AtomicU64);

#[derive(Clone, Copy, PartialEq)]
enum End {
    Done,
    Failed,
    /// Dropped without `__exit__`: stop drawing.
    Dropped,
}

struct State {
    /// Current phase: label, short name for the timings, start.
    name: String,
    short: String,
    started: Instant,
    /// Finished phases: (short name, seconds).
    timings: Vec<(String, f64)>,
    /// The current phase is the one counting rows.
    computing: bool,
    end: Option<End>,
}

struct Shared {
    live: bool,
    /// Terminal width: live lines are fitted to it (no wrap, so `\r` redraws).
    columns: Option<usize>,
    created: Instant,
    state: Mutex<State>,
    wake: Condvar,
    /// (total rows, one slot per worker of the pool), set by the row phase.
    rows: OnceLock<(u64, Box<[Slot]>)>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn done_rows(&self) -> Option<(u64, u64)> {
        let (total, slots) = self.rows.get()?;
        let done: u64 = slots.iter().map(|s| s.0.load(Ordering::Relaxed)).sum();
        Some((done.min(*total), *total))
    }

    /// Ends the current phase and starts `name`. In log mode the new phase's
    /// line is written here, so no phase is skipped however short.
    fn start(&self, name: &str, short: &str, computing: bool) {
        let line = {
            let mut st = self.lock();
            if st.end.is_some() {
                return;
            }
            self.next(&mut st, Some((name, short)), computing);
            (!self.live).then(|| render(self, &st, WIDTH))
        };
        if let Some(line) = line {
            emit(Out::Log(&line));
        }
    }

    /// Ends the current phase and starts `name` (`None`: the run ended).
    fn next(&self, st: &mut State, name: Option<(&str, &str)>, computing: bool) {
        let now = Instant::now();
        if !st.short.is_empty() {
            let secs = now.duration_since(st.started).as_secs_f64();
            st.timings.push((std::mem::take(&mut st.short), secs));
        }
        st.computing = computing;
        if let Some((name, short)) = name {
            st.name = name.into();
            st.short = short.into();
            st.started = now;
        }
        self.wake.notify_one();
    }
}

#[pyclass(frozen)]
pub struct ProgressBar {
    shared: Arc<Shared>,
    drawer: Mutex<Option<JoinHandle<()>>>,
}

#[pymethods]
impl ProgressBar {
    /// `live`: redraw one line (a terminal, Jupyter); else log lines.
    /// `columns`: the terminal's width, if known.
    #[new]
    #[pyo3(signature = (live, columns=None))]
    fn new(live: bool, columns: Option<usize>) -> PyResult<Self> {
        let live = live
            && LIVE
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok();
        let now = Instant::now();
        let shared = Arc::new(Shared {
            live,
            columns: columns.filter(|_| live),
            created: now,
            state: Mutex::new(State {
                name: String::new(),
                short: String::new(),
                started: now,
                timings: Vec::new(),
                computing: false,
                end: None,
            }),
            wake: Condvar::new(),
            rows: OnceLock::new(),
        });
        let s = shared.clone();
        let drawer = std::thread::Builder::new()
            .name("similaripy-progress".into())
            .spawn(move || draw_loop(&s))
            .inspect_err(|_| {
                if live {
                    LIVE.store(false, Ordering::Release);
                }
            })?;
        Ok(Self {
            shared,
            drawer: Mutex::new(Some(drawer)),
        })
    }

    /// Starts a phase without rows (`short`: its name in the final timings).
    pub(crate) fn phase(&self, name: &str, short: &str) {
        self.shared.start(name, short, false);
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    /// Draws the final line ("Done" or "Failed") and waits for the drawer,
    /// with the GIL released (the drawer takes it to write).
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyType>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.finish(if exc_type.is_none() {
            End::Done
        } else {
            End::Failed
        });
        let drawer = self.drawer.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(d) = drawer {
            let _ = py.detach(|| d.join());
        }
        false
    }

    // Formatting hooks for the tests (the crate has no Rust test target).
    #[staticmethod]
    fn _fmt_count(n: f64) -> String {
        fmt_count(n)
    }

    #[staticmethod]
    fn _fmt_clock(secs: f64) -> String {
        fmt_clock(secs)
    }

    #[staticmethod]
    #[pyo3(signature = (done, total, width=WIDTH))]
    fn _bar(done: u64, total: u64, width: usize) -> String {
        bar(done, total, width)
    }
}

impl ProgressBar {
    /// Starts the phase that counts `total` rows on the current rayon pool.
    pub(crate) fn rows_phase(&self, name: &str, short: &str, total: usize) {
        let slots = (0..rayon::current_num_threads())
            .map(|_| Slot(AtomicU64::new(0)))
            .collect();
        let _ = self.shared.rows.set((total as u64, slots));
        self.shared.start(name, short, true);
    }

    /// One finished row, on a worker of the pool `rows_phase` ran on.
    #[inline]
    pub(crate) fn tick(&self) {
        if let (Some((_, slots)), Some(i)) = (self.shared.rows.get(), rayon::current_thread_index())
            && let Some(s) = slots.get(i)
        {
            // Single writer per slot: a plain load + store, no atomic add.
            s.0.store(s.0.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
    }

    fn finish(&self, end: End) {
        let mut st = self.shared.lock();
        if st.end.is_none() {
            self.shared.next(&mut st, None, false);
            st.end = Some(end);
        }
    }
}

impl Drop for ProgressBar {
    fn drop(&mut self) {
        // Not joined: the drawer may wait for the GIL this thread holds.
        self.finish(End::Dropped);
    }
}

/// The drawing thread: wakes every `INTERVAL` or on a phase change.
fn draw_loop(sh: &Shared) {
    /// Frees the terminal line however the loop ends (a panic included).
    struct Release;
    impl Drop for Release {
        fn drop(&mut self) {
            LIVE.store(false, Ordering::Release);
        }
    }
    let live = sh.live;
    let _release = live.then_some(Release);
    let (mut last, mut last_len, mut decile) = (String::new(), 0usize, 0u64);
    loop {
        let (line, end, started, computing) = {
            let st = sh.lock();
            let st = if st.end.is_none() {
                sh.wake
                    .wait_timeout(st, INTERVAL)
                    .unwrap_or_else(|e| e.into_inner())
                    .0
            } else {
                st
            };
            if st.end == Some(End::Dropped) {
                break;
            }
            (fitted(sh, &st), st.end, !st.name.is_empty(), st.computing)
        };
        // The GIL only outside the lock: a Python thread may hold the GIL
        // while it waits for the lock (`phase`).
        if end.is_some() {
            if live {
                emit(Out::LiveEnd(&redraw(&line, last_len)));
            } else {
                emit(Out::Log(&line));
            }
            break;
        }
        if live {
            if line != last && started {
                emit(Out::Live(&redraw(&line, last_len)));
                last_len = line.chars().count();
                last = line;
            }
        } else if let Some((done, total)) = sh.done_rows().filter(|_| computing)
            && total > 0
            && done * 10 / total > decile
            && done < total
        {
            decile = done * 10 / total;
            emit(Out::Log(&line));
        }
    }
}

/// `\r` + `line`, padded to blank a longer previous line: one write, no flicker.
fn redraw(line: &str, last_len: usize) -> String {
    let pad = last_len.saturating_sub(line.chars().count());
    format!("\r{line}{}", " ".repeat(pad))
}

enum Out<'a> {
    /// A `\r` redraw: the cursor stays at the line's end.
    Live(&'a str),
    /// The live bar's final redraw, newline added.
    LiveEnd(&'a str),
    /// A whole line, newline added; on a new line if a live line is open.
    Log(&'a str),
}

/// `sys.stderr.write` + `flush` under `MID_LINE`; errors ignored (a closed or
/// missing stderr). The lock is taken detached: `write` may release the GIL
/// while another writer holds it, and that writer may be waiting for it.
fn emit(out: Out<'_>) {
    Python::attach(|py| {
        let mut mid = MID_LINE
            .lock_py_attached(py)
            .unwrap_or_else(|e| e.into_inner());
        let s = match out {
            Out::Live(s) => {
                *mid = true;
                s.to_owned()
            }
            Out::LiveEnd(s) => {
                *mid = false;
                format!("{s}\n")
            }
            Out::Log(s) => {
                let nl = if *mid { "\n" } else { "" };
                *mid = false;
                format!("{nl}{s}\n")
            }
        };
        let _ = (|| -> PyResult<()> {
            let err = py.import("sys")?.getattr("stderr")?;
            if !err.is_none() {
                err.call_method1("write", (s,))?;
                err.call_method0("flush")?;
            }
            Ok(())
        })();
    });
}

/// The line, its bar narrowed to fit the terminal (one column spare: some
/// terminals wrap at the last one). A live line still too long is cut; the
/// final line ends with a newline, so it may wrap.
fn fitted(sh: &Shared, st: &State) -> String {
    let line = render(sh, st, WIDTH);
    let Some(max) = sh.columns.map(|c| c.saturating_sub(1)) else {
        return line;
    };
    let n = line.chars().count();
    if n <= max {
        return line;
    }
    let line = render(sh, st, WIDTH.saturating_sub(n - max));
    if st.end.is_some() {
        line
    } else {
        line.chars().take(max).collect()
    }
}

fn render(sh: &Shared, st: &State, width: usize) -> String {
    let mut rows = sh.done_rows();
    if let Some(end) = st.end {
        if end == End::Done {
            // every row is done, counted or not (the kernel skips rows when k = 0)
            rows = rows.map(|(_, t)| (t, t));
        }
        let total = sh.created.elapsed().as_secs_f64();
        let head = match (end, rows) {
            (End::Done, _) => "Done",
            _ => "Failed",
        };
        let counts = rows.map_or(String::new(), |(d, t)| {
            format!(" {}", bar_counts(d, t, width))
        });
        let times: Vec<String> = st
            .timings
            .iter()
            .map(|(name, s)| format!("{name} {}", fmt_secs(*s)))
            .collect();
        let times = if times.is_empty() {
            String::new()
        } else {
            format!(" = {}", times.join(" + "))
        };
        return format!("{head}{counts} [{}{times}]", fmt_clock(total));
    }
    let secs = st.started.elapsed().as_secs_f64();
    match rows.filter(|_| st.computing) {
        Some((done, total)) => {
            let rate = done as f64 / secs.max(1e-9);
            let eta = if done > 0 && done < total {
                fmt_clock((total - done) as f64 / rate)
            } else {
                "--:--".into()
            };
            format!(
                "{} {} [{}<{eta}, {} rows/s]",
                st.name,
                bar_counts(done, total, width),
                fmt_clock(secs),
                fmt_count(rate),
            )
        }
        None => format!("{} [{}]", st.name, fmt_clock(secs)),
    }
}

/// ` 43%|██████████▊              | 25.8k/60.0k rows`
/// No room for a bar (`width` 0): ` 43% 25.8k/60.0k rows`.
fn bar_counts(done: u64, total: u64, width: usize) -> String {
    let pct = (done * 100).checked_div(total).unwrap_or(100);
    let bar = if width > 0 {
        format!("|{}|", bar(done, total, width))
    } else {
        String::new()
    };
    let (done, total) = (fmt_count(done as f64), fmt_count(total as f64));
    format!("{pct:>3}%{bar} {done}/{total} rows")
}

/// `width` cells, filled in eighths.
fn bar(done: u64, total: u64, width: usize) -> String {
    let eighths = if total > 0 {
        (u128::from(done.min(total)) * (width as u128 * 8) / u128::from(total)) as usize
    } else {
        width * 8
    };
    let (full, part) = (eighths / 8, eighths % 8);
    let mut s = "█".repeat(full);
    if full < width {
        s.push(PARTIAL[part]);
        s.extend(std::iter::repeat_n(' ', width - full - 1));
    }
    s
}

/// `999`, `25.8k`, `1.23M`, `4.56G`: at most 4 significant characters.
fn fmt_count(n: f64) -> String {
    if n < 999.5 {
        return format!("{n:.0}");
    }
    for (unit, scale) in [("k", 1e3), ("M", 1e6), ("G", 1e9)] {
        let v = n / scale;
        if v < 9.995 {
            return format!("{v:.2}{unit}");
        }
        if v < 999.5 {
            return format!("{v:.1}{unit}");
        }
    }
    format!("{:.1}T", n / 1e12)
}

/// `MM:SS`, `H:MM:SS` from one hour.
fn fmt_clock(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s < 3600 {
        format!("{:02}:{:02}", s / 60, s % 60)
    } else {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    }
}

/// `0.4s` under a minute, else the clock.
fn fmt_secs(secs: f64) -> String {
    if secs < 59.95 {
        format!("{secs:.1}s")
    } else {
        fmt_clock(secs)
    }
}
