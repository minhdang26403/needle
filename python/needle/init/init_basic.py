from typing import Optional

from needle.autograd import Tensor
from needle.backend.device import Device, default_device


def rand(
    *shape: int,
    low: float = 0.0,
    high: float = 1.0,
    device: Optional[Device] = None,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate random numbers uniform between low and high"""
    device = default_device() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(
    *shape: int,
    mean: float = 0.0,
    std: float = 1.0,
    device: Optional[Device] = None,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate random normal with specified mean and std deviation"""
    device = default_device() if device is None else device
    array = device.randn(*shape) * std + mean
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(
    *shape: int,
    c: float = 1.0,
    device: Optional[Device] = None,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate constant Tensor"""
    device = default_device() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(
    *shape: int,
    device: Optional[Device] = None,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(
    *shape: int,
    device: Optional[Device] = None,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate all-zeros Tensor"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(
    *shape: int,
    p: float = 0.5,
    device: Optional[Device] = None,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate binary random Tensor"""
    device = default_device() if device is None else device
    array = device.rand(*shape) <= p
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(
    n: int,
    i: Tensor,
    device: Optional[Device] = None,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Generate one-hot encoding Tensor"""
    device = default_device() if device is None else device
    return Tensor(
        device.one_hot(n, i.numpy().astype("int64"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(
    array: Tensor, *, device: Optional[Device] = None, requires_grad: bool = False
) -> Tensor:
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(
    array: Tensor, *, device: Optional[Device] = None, requires_grad: bool = False
) -> Tensor:
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
