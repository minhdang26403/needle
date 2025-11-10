import importlib
from typing import Any, Optional

from . import ndarray_backend_numpy

# Try to import the native CPU backend dynamically (may not be built in dev)
try:
    ndarray_backend_cpu: Optional[Any] = importlib.import_module(
        "needle.backend.ndarray_backend_cpu"
    )
except Exception as _e:  # pragma: no cover
    ndarray_backend_cpu = None


class BackendDevice:
    def __init__(self, name: str, module: Any) -> None:
        self.name = name
        self.module = module

    def __repr__(self) -> str:
        return f"{self.name}()"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)


def cpu_numpy() -> BackendDevice:
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu() -> BackendDevice:
    """
    Return the native C++ CPU backend device.
    Requires the `ndarray_backend_cpu` extension to be built.
    """
    if ndarray_backend_cpu is None:
        raise RuntimeError(
            "CPU backend not available. "
            "Build the C++ extension first (ndarray_backend_cpu)."
        )
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device() -> BackendDevice:
    return cpu_numpy()
