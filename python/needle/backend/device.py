from typing import Any

from . import ndarray_backend_numpy


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


def default_device() -> BackendDevice:
    return cpu_numpy()
