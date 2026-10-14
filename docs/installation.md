# 📦 Installation

## Installation

SimilariPy can be installed from PyPI with:

```cmd
pip install similaripy
```

Pre-built wheels are available for Python 3.11–3.15 on:

- Linux x86_64 and aarch64 (glibc 2.28+)
- macOS 15+ (Apple Silicon and Intel)
- Windows x64

No compiler is needed on these platforms. Elsewhere, pip builds the package from source (see below).

Free-threaded Python (3.14t, 3.15t) is supported: the GIL stays disabled. When calling SimilariPy from several Python threads at once, set `num_threads` on each call to avoid oversubscribing the CPU cores.

## Build from source

Building from source is needed only when no wheel matches your platform, or to compile with CPU-specific optimizations. It requires a GCC-compatible compiler with OpenMP.

### Ubuntu / Debian

Install the official dev-tools:

```bash
sudo apt update && sudo apt install build-essential
```

### MacOS (Intel & Apple Silicon)

Install GCC with homebrew:

```bash
brew install gcc
```

### Windows

Install the official **[Visual C++ Build Tools](https://visualstudio.microsoft.com/en/visual-cpp-build-tools/)**.

### Optional: native CPU optimizations

Wheels target a generic CPU baseline. To build for your exact CPU (use `-mcpu=native` on ARM):

```bash
pip install similaripy --no-binary similaripy -C cmake.define.CMAKE_CXX_FLAGS="-march=native"
```

The speedup depends on the CPU: on Apple Silicon we measured none, so benchmark it on your workload.

### Optional Optimization: Intel MKL for Intel CPUs

For Intel CPUs, using SciPy/Numpy with MKL (Math Kernel Library) is highly recommended for best performance.
The easiest way to achieve this is to install them via Anaconda.

## Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|
| python                          |   >= 3.11      |
| numpy                           |   >= 1.22.4    |
| scipy                           |   >= 1.12.0    |
