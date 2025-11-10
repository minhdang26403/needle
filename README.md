# Needle
**Needle** (**Ne**cessary **E**lements of **D**eep **Le**arning) is a minimalist deep learning library built from scratch in Python, C++, and Metal. It is inspired by the Carnegie Mellon University course [CMU 10-414/714: Deep Learning Systems](https://dlsyscourse.org/).

While deep learning systems like PyTorch and TensorFlow are widely available, understanding their internals is crucial for effective use and extension. This project is a "full-stack" journey to build, own, and understand every component of a modern framework, from the high-level Python API down to the low-level hardware kernels.

### Follow My Journey

This repository is the cornerstone of my learning project. I'll be documenting my progress, design decisions, and the challenges of C++/Python interop on [**my personal blog**](https://minhdang26403.github.io/).

---

## 🎯 Core Features

Needle is being built to support the full "full stack" of a deep learning system:

* **Dynamic Autograd Engine:** A complete, tape-based automatic differentiation engine (`Tensor.backward()`).
* **Multi-Backend `NDArray`:** A clean backend API to dispatch compute to:
    * `numpy` (for simple, readable baseline)
    * `cpu` (high-performance C++ kernels)
    * `metal` (macOS GPU acceleration)
    * `cuda` (NVIDIA GPU acceleration)
* **C++/Python Interop:** A high-performance C++ core exposed to Python using `pybind11` and `scikit-build-core`.
* **A Full `nn` Library:** A `Module`-based API with common layers (`Linear`, `Conv2d`, `RNN/LSTM`), optimizers (`SGD`, `Adam`), and loss functions.
* **Data Pipeline:** A `DataLoader` and `Dataset` implementation for handling common datasets (CIFAR10, PTB).

## 💡 Project Goals

This is primarily a learning project and a portfolio piece. **It is not intended for production use.**

The main goal is to build, own, and trace a complete DL system to gain a fundamental understanding of its internals. It also serves as a public demonstration of my skills in:

* Systems Programming (C++, Metal, CUDA)
* Python/C++ Interoperability (`pybind11`)
* ML Systems Architecture (Autodiff, Operators, Schedulers)
* Modern Software Engineering (CI/CD, Testing, Packaging)

---

## 🛠️ Technical Stack

* **Core:** Python 3.12+
* **Bindings:** `pybind11`
* **Build System:** `scikit-build-core` & `CMake`
* **Backends:** C++, NumPy, Objective-C++ (Metal), CUDA
* **Testing & Linting:** `pytest`, `mypy`, `ruff` (optional: `pre-commit`)
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
  benchmarks/
  docs/
  environment.yml          # Conda env for Python/CMake/Ninja
  pyproject.toml           # Packaging + tool configs (ruff/mypy/pytest)
  CMakeLists.txt           # Root CMake, adds src/{cpu,metal,cuda}
  pytest.ini               # Pytest defaults
  .github/workflows/ci.yml # CI: lint/type-check/test on macOS/Linux
```

---

## ⚙️ Tooling and Configuration

- `pyproject.toml`: Single-source packaging (name/version/metadata), scikit-build-core for CMake integration, and centralized configs for ruff/mypy/pytest. It also points to the modern package location `python/needle`.
- `environment.yml`: Conda env for Python + build tools (CMake, Ninja). Python dependencies are installed via pip using `pyproject.toml` for consistent versions.
- `pytest.ini`: Keeps tests quiet (`-q`) and scoped to the `tests` directory for consistent discovery locally and in CI.
- `.github/workflows/ci.yml`: CI matrix (macOS, Ubuntu; Python 3.10–3.12) that runs lint, type-check, and tests on PRs and pushes.
- Optional: Pre-commit hooks. If you add a `.pre-commit-config.yaml`, you can enable it with `pre-commit install` to auto-run ruff/mypy on commit.

---

## 🗺️ Project Roadmap

This project is broken down into eight distinct phases.

* [x] **Phase 0 — Decisions, scaffold, tooling**
    * Set up `pyproject.toml` (`scikit-build-core`, `pybind11`), CMake, `pytest`.
    * Set up CI stub on GitHub Actions.

* [x] **Phase 1 — Containers only (no autograd)**
    * Define backend protocol (NDArray): shape, strides, dtype, device, allocate/copy, reshape, transpose, elementwise add/mul, matmul, reduce sum/mean, broadcasting rules.
    * Implement NumPy backend to this protocol.
    * Tests: shape/stride invariants, broadcasting semantics, dtype/device handling, to/from numpy parity.

* [ ] **Phase 2 — Compute graph and autograd**
    * Implement `Tensor` (no grad) that wraps backend arrays, dtype/device conversions, to/from NumPy.
    * Autograd engine: tape data structures, grad mode/no_grad, topological sort, backward traversal, accumulation, default grad for scalars.
    * `Op` base API with `compute()` and `gradient()` contracts.
    * Core ops with gradients: add, mul, neg, matmul, sum/mean, reshape, transpose, broadcast_to; plus log/exp/relu/sigmoid/tanh/softmax; argmax (non-diff).
    * Gradient checks: finite differences on small random tensors; edge cases for broadcasting and reductions.

* [ ] **Phase 3 — Optimizers, NN library, data pipeline**
    * Implement `nn.Module`, `nn.Parameter`, `nn.Sequential`.
    * Optimizers: SGD/Momentum/Adam, param groups, weight decay, step/zero_grad.
    * Layers/losses: Linear, activations, MSE/CrossEntropy, Dropout, BatchNorm; Conv2d via im2col+GEMM; minimal RNN/LSTM/GRU.
    * Data: CIFAR10/PTB datasets, transforms, simple DataLoader.
    * Integration tests: small training loops showing decreasing loss.

* [ ] **Phase 4 — C++ CPU backend**
    * C++ `NDArray` (shape/stride, allocation), kernels for ewise unary/binary, broadcast, reductions, matmul (optional BLAS), conv2d.
    * pybind11 bindings, module `needle_cpu`, device selection.
    * Ensure all tests pass on the `device("cpu")`.
    * Cross-backend parity tests; micro-benchmarks vs. NumPy.

* [ ] **Phase 5 — Metal backend (macOS)**
    * Objective-C++ bridge, buffers/queues/pipelines; kernels for copy/ewise/reduce/matmul/conv subset.
    * Ensure all tests pass on `device("metal")`.

* [ ] **Phase 6 — CUDA backend (optional)**
    * CUDA kernels, cuBLAS (matmul), optional cuDNN (conv).
    * Gate build via `NEEDLE_ENABLE_CUDA=ON`.
    * Ensure all tests pass on `device("cuda")`.

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
