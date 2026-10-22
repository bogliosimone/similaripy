# 📦 Installation

## Installation

SimilariPy can be installed from PyPI with:

```cmd
pip install similaripy
```

Pre-built wheels are available for Python 3.11–3.15 on:

- Linux x86_64 and aarch64 (glibc 2.28+, or musl 1.2+ e.g. Alpine)
- macOS 12+ (Apple Silicon) and 10.15+ (Intel)
- Windows x64

No compiler is needed on these platforms. Elsewhere, pip builds the package from source (see below).

Free-threaded Python (3.14t, 3.15t) is supported: the GIL stays disabled. When calling SimilariPy from several Python threads at once, set `num_threads` on each call to avoid oversubscribing the CPU cores.

## Build from source

Building from source is needed only when no wheel matches your platform (pip then builds the source distribution), or to build for your exact CPU. It needs a Rust toolchain; no C compiler or OpenMP.

Install Rust with [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

On Windows, run `rustup-init.exe` from [rustup.rs](https://rustup.rs) instead; it also offers to install the Visual C++ build tools that Rust needs for linking.

Then `pip install similaripy` builds the package. The source distribution pins the Rust version it was tested with (`rust-toolchain.toml`), and rustup installs that version on the first build. Without rustup (e.g. a distribution's `cargo` package), Rust 1.88 or newer is needed.

### Optional: native CPU optimizations

Wheels target a generic CPU baseline. To build for your exact CPU:

```bash
export RUSTFLAGS="-C target-cpu=native"
pip install similaripy --no-binary similaripy --no-cache-dir
```

(PowerShell: `$env:RUSTFLAGS="-C target-cpu=native"`)

The speedup depends on the CPU and the workload: benchmark it on yours.

## Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|
| python                          |   >= 3.11      |
| numpy                           |   >= 1.23.5    |
| scipy                           |   >= 1.15.0    |
