# Needle
Needle (Necessary Elements of Deep Learning) is a minimalist deep learning library built from scratch in Python, C++, and Metal. It is inspired by the Carnegie Mellon University course [11-785: Deep Learning Systems](https://dlsyscourse.org/).

While deep learning systems like PyTorch and TensorFlow are widely available, understanding their internals is crucial for effective use and extension. This project is my "full-stack" journey to build, own, and understand every component of a modern framework, from the high-level Python API down to the low-level hardware kernels.

As a systems engineer pivoting into MLSys, my goal is to trace the execution from `loss.backward()` all the way to the C++ and GPU operations that make it possible.

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
* **Testing & Linting:** `pytest`, `mypy`, `ruff`, `pre-commit`

---

## 🗺️ Project Roadmap

This project is broken down into eight distinct phases.

* [ ] **Phase 0 — Decisions, scaffold, tooling**
    * Set up `pyproject.toml` (`scikit-build-core`, `pybind11`), CMake, `pytest`, `pre-commit`.
    * Set up CI stub on GitHub Actions.

* [ ] **Phase 1 — Containers only (no autograd)**
    * Implement `NDArray` concept with a NumPy backend.
    * Implement `Tensor` container (no grad) holding the backend array.
    * Unit tests for ops, shapes, strides, and broadcasting.

* [ ] **Phase 2 — Compute graph and autograd**
    * Implement the autograd "tape" and `backward()` traversal.
    * Define `Op` base class with `forward()` and `backward()` contracts.
    * Implement core ops (math, reduce, broadcast) and their gradients.
    * Write gradient checks using finite-differences.

* [ ] **Phase 3 — Optimizers, NN library, data pipeline**
    * Implement Optimizers: `SGD`, `Adam`.
    * Implement `nn.Module`, `nn.Parameter`, `nn.Sequential`.
    * Implement NN layers: `Linear`, `ReLU`, `Losses`, `Conv2d`, `RNN`.
    * Implement `Dataset` and `DataLoader` for CIFAR10/PTB.
    * Integration tests: small training loops showing decreasing loss.

* [ ] **Phase 4 — C++ CPU backend**
    * Implement C++ `NDArray` with optimized kernels for all ops.
    * Write `pybind11` bindings to expose the backend.
    * Ensure all tests pass on the `device("cpu")`.
    * Micro-benchmarks vs. NumPy.

* [ ] **Phase 5 — Metal backend (macOS)**
    * Write Objective-C++ bridge to Metal.
    * Implement Metal kernels for key ops (ewise, reduce, matmul, conv).
    * Enable universal2 wheels.
    * Ensure all tests pass on `device("metal")`.

* [ ] **Phase 6 — CUDA backend (optional)**
    * Implement CUDA kernels; integrate `cuBLAS` and `cuDNN`.
    * Gate build via `NEEDLE_ENABLE_CUDA=ON`.
    * Ensure all tests pass on `device("cuda")`.

* [ ] **Phase 7 — Efficiency experiments (R&D)**
    * Explore kernel fusion, quantization (PTQ), KV Caching, and FlashAttention-like concepts.
    * Benchmark performance improvements.

* [ ] **Phase 8 — Benchmarks, docs, blog, release**
    * Finalize documentation and tutorials.
    * Publish blog posts on design decisions and performance.
    * Release v1.0 to PyPI.

---

## 🚀 Quick Start

### 1. Build from Source (Development)

This project uses `scikit-build-core` and `CMake` to manage the C++ build process.

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/needle.git](https://github.com/YOUR_USERNAME/needle.git)
cd needle

# 2. Install in editable mode
# This will compile the C++ backends
pip install -e .

# 3. Run the tests
pytest
```