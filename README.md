# Needle
**Needle** (**Ne**cessary **E**lements of **D**eep **Le**arning) is a minimalist deep learning library built from scratch in Python and C++. It is inspired by the Carnegie Mellon University course [CMU 10-414/714: Deep Learning Systems](https://dlsyscourse.org/).

While deep learning systems like PyTorch and TensorFlow are widely available, understanding their internals is crucial for effective use and extension. This project is a "full-stack" journey to build, own, and understand every component of a modern framework, from the high-level Python API down to the low-level hardware kernels.

### Follow My Journey

This repository is the cornerstone of my learning project. I'll be documenting my progress, design decisions, and implementation challenges on [**my personal blog**](https://minhdang26403.github.io/).

---

## 🎯 Core Features

Needle currently supports:

- **Autograd `Tensor`:** Eager `Tensor` API with a tape-based automatic differentiation engine (topological traversal).
- **Operator Library:** Elementwise ops (add, mul, div, pow), scalar ops, comparisons, unary ops (`log`, `exp`, `relu`), reshaping (`reshape`, `transpose`, `broadcast_to`), reductions (`sum`), and `matmul`.
- **Multi-Backend `NDArray`:**
  - `numpy` backend for a simple, readable baseline.
  - `cpu` backend (C++/pybind11) with vectorizable kernels and tiled `matmul`.
- **C++/Python Interop:** A C++ core exposed to Python using `pybind11` and built with `scikit-build-core`.
- **Initialization utilities:** `needle.init` provides basic initializers and helpers (e.g., `ones`) used by autograd.

Planned/future work:
- Additional backends (Metal, CUDA).
- `nn` library (`Module`, layers, optimizers) and a data pipeline.
- Broader operator coverage and performance improvements.

## 💡 Project Goals

This is primarily a learning project and a portfolio piece. **It is not intended for production use.**

The main goal is to build a complete DL system to gain a fundamental understanding of its internals. It also serves as a public demonstration of my skills in:

* Systems Programming (C++)
* Python/C++ Interoperability (`pybind11`)
* ML Systems Architecture (Autodiff, Operators, Schedulers)
* Modern Software Engineering (CI/CD, Testing, Packaging)

---

## 🛠️ Technical Stack

* **Core:** Python 3.12+
* **Bindings:** `pybind11`
* **Build System:** `scikit-build-core` & `CMake`
* **Backends:** NumPy, C++ (CPU), Metal, CUDA
* **Testing & Linting:** `pytest`, `mypy`, `ruff`
* **Env:** Conda (`environment.yml`) + pip for Python deps

---

## 📦 Repository Layout

```bash
repo-root/
  python/needle/           # Python package
  src/cpu/                 # Native CPU backend (CMake targets)
  src/metal/               # Native Metal backend (CMake targets)
  src/cuda/                # Native CUDA backend (CMake targets)
  tests/                   # Unit + integration tests
  examples/
  benchmarks/              # Microbenchmarks (e.g., matmul)
  docs/
  environment.yml          # Conda env for Python/CMake/Ninja
  pyproject.toml           # Packaging + tool configs (ruff/mypy/pytest)
  CMakeLists.txt           # Root CMake, adds src/{cpu,metal,cuda}
  pytest.ini               # Pytest defaults
```

---

## ⚙️ Tooling and Configuration

- `pyproject.toml`: Single-source packaging (name/version/metadata), scikit-build-core for CMake integration, and centralized configs for ruff/mypy/pytest. It also points to the modern package location `python/needle`.
- `environment.yml`: Conda env for Python + build tools (CMake, Ninja). Python dependencies are installed via pip using `pyproject.toml` for consistent versions.
- `pytest.ini`: Keeps tests quiet (`-q`) and scoped to the `tests` directory for consistent discovery locally and in CI.
- Optional: Pre-commit hooks. If you add a `.pre-commit-config.yaml`, you can enable it with `pre-commit install` to auto-run ruff/mypy on commit.

---

## 🗺️ Project Roadmap

This project is broken down into phases:

* [x] **Phase 0 — Decisions, scaffold, tooling**
    * Set up `pyproject.toml` (`scikit-build-core`, `pybind11`), CMake, `pytest`.
    * Set up CI stub on GitHub Actions.

* [x] **Phase 1 — Containers and backends**
    * Implement NDArray with NumPy backend.
    * Implement shape/stride invariants, elementwise ops, broadcasting, reductions, and `matmul`.

* [x] **Phase 2 — Compute graph and autograd**
    * Implement `Tensor` with autograd.
    * Autograd engine: topological sort, backward traversal, accumulation, default grad for scalars.
    * `TensorOp` base with `compute()`/`gradient()` contracts.
    * Core ops with gradients: add, mul, neg, `matmul`, `sum`, `reshape`, `transpose`, `broadcast_to`, `log`, `exp`, `relu`, elementwise power/divide.
    * Gradient checks using finite differences on small random tensors.

* [ ] **Phase 3 — Optimizers, NN library, data pipeline**
    * Implement `nn.Module`, `nn.Parameter`, `nn.Sequential`.
    * Optimizers: SGD/Momentum/Adam, param groups, weight decay, step/zero_grad.
    * Layers/losses: Linear, activations, MSE/CrossEntropy, Dropout, BatchNorm; Conv2d via im2col+GEMM; minimal RNN/LSTM/GRU.
    * Data: CIFAR10/PTB datasets, transforms, simple DataLoader.
    * Integration tests: small training loops showing decreasing loss.

* [x] **Phase 4 — C++ CPU backend**
    * C++ `NDArray`, kernels for ewise unary/binary, broadcast, reductions, `matmul` (tiled).
    * pybind11 bindings exposed as `needle.backend.ndarray_backend_cpu`.
    * `needle.backend.device.cpu()` to select the native CPU backend.
    * Parity tests and micro-benchmarks vs. NumPy.

* [ ] **Phase 5 — Metal backend**
    * Objective-C++ bridge, buffers/queues/pipelines; kernels for copy/ewise/reduce/matmul/conv subset.
    * Regression tests with with Metal backend.

* [ ] **Phase 6 — CUDA backends CUDA**
    * CUDA kernels, cuBLAS (matmul), optional cuDNN (conv).
    * Regression tests with CUDA backend.

* [ ] **Phase 7 — Efficiency experiments (R&D)**
    * Kernel fusion (Python-level → C++ passes), profiling harness.
    * PTQ quantization (Linear/Conv), calibration, quantized kernels.
    * KV cache utilities and attention improvements; explore FlashAttention-like approach.
    * Benchmark performance improvements.

* [ ] **Phase 8 — Benchmarks, docs, blog, release**
    * Ops microbenchmarks; model throughput/latency across backends.
    * Publish blog posts on design decisions and performance.
    * Release v1.0 to PyPI.

---

## 🚀 Quick Start

### 1. Conda environment (recommended)

This project uses `scikit-build-core` and `CMake` to manage the C++ build process. We recommend a minimal Conda env for Python/CMake/Ninja, then install Python deps via `pip`.

```bash
# Clone the repository
git clone [https://github.com/minhdang26403/needle.git](https://github.com/minhdang26403/needle.git)
cd needle

# Create and activate the conda environment
conda env create -f environment.yml    # creates env named "needle"
conda activate needle

# Install the package in editable mode with dev tools
python -m pip install -U pip
pip install -e ".[dev]"

# Run tests
pytest -q
```

### 2. Plain virtualenv (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
pytest -q
```
