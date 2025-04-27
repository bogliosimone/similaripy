# 📦 Installation

## Installation

SimilariPy can be installed from PyPI with:

```cmd
pip install similaripy
```

## GCC Compiler - Required

To install the package and compile the Cython code, a GCC-compatible compiler with OpenMP is required.

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

⚠️ On Windows, use the default *format_output='coo'* in all similarity functions, as *'csr'* is currently not supported.


### Optional Optimization: Intel MKL for Intel CPUs

For Intel CPUs, using SciPy/Numpy with MKL (Math Kernel Library) is highly recommended for best performance.
The easiest way to achieve this is to install them via Anaconda.

## Requirements

| Package                         | Version        |
| --------------------------------|:--------------:|
| numpy                           |   >= 1.21      |
| scipy                           |   >= 1.10.1    |
| tqdm                            |   >= 4.65.2    |
