"""Elementary tensor operators (ewise, scalars, linear algebra, transforms).

Each operator subclasses `TensorOp` and implements:
  - compute(*ndarrays) -> ndarray     forward pass on backend arrays
  - gradient(out_grad, node) -> Tensor | tuple[Tensor, ...]
"""

import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

import needle.backend.ndarray as nd
from needle.ops import TensorOp

if TYPE_CHECKING:
    from needle.autograd import Tensor


class EWiseAdd(TensorOp):
    """Elementwise addition of two tensors."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a, b) = args
        return a + b

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor", "Tensor"]:
        return out_grad, out_grad


class AddScalar(TensorOp):
    """Add a scalar to a tensor."""

    def __init__(self, scalar: float) -> None:
        self.scalar = scalar

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a + self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        return out_grad


class EWiseMul(TensorOp):
    """Elementwise multiplication of two tensors."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a, b) = args
        return a * b

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor", "Tensor"]:
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


class MulScalar(TensorOp):
    """Multiply a tensor by a scalar."""

    def __init__(self, scalar: float) -> None:
        self.scalar = scalar

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a * self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        return out_grad * self.scalar


class EWisePow(TensorOp):
    """Elementwise power: a ** b."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a, b) = args
        return a**b

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor", "Tensor"]:
        a, b = node.inputs
        a_grad = out_grad * b * a ** (b - 1)
        b_grad = out_grad * a**b * a.log()
        return a_grad, b_grad


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: float):
        self.scalar = scalar

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a**self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        a = node.inputs[0]
        return out_grad * self.scalar * a ** (self.scalar - 1)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a, b) = args
        return a / b

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor", "Tensor"]:
        a, b = node.inputs
        return out_grad / b, out_grad * -a / b**2


class DivScalar(TensorOp):
    """Divide a tensor by a scalar."""

    def __init__(self, scalar: float) -> None:
        self.scalar = scalar

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a / self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        return out_grad / self.scalar


class Transpose(TensorOp):
    """Swap two axes (defaults to swapping the last two)."""

    def __init__(self, axes: Optional[tuple[int, ...]] = None) -> None:
        self.axes = axes

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        axes = self.axes
        ndim = a.ndim
        if axes is None:
            axes = (ndim - 2, ndim - 1)

        new_dims = list(range(ndim))
        new_dims[axes[0]] = axes[1]
        new_dims[axes[1]] = axes[0]

        return a.permute(new_dims)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        return out_grad.transpose(self.axes)


class Reshape(TensorOp):
    """Reshape a tensor to a new shape."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        return out_grad.reshape(node.inputs[0].shape)


class BroadcastTo(TensorOp):
    """Broadcast a tensor to a target shape."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        ndiff = len(self.shape) - len(a.shape)
        if ndiff > 0:
            added_axes = (1,) * ndiff
            # need to reassign since compact returns a new copy
            a = a.compact().reshape(added_axes + a.shape)
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        input_shape = node.inputs[0].shape

        # 1. Identify new axes that were added by the broadcast
        # e.g., broadcasting from (3,) to (2, 3) adds a new axis 0.
        ndim_diff = len(self.shape) - len(input_shape)
        axes = list(range(ndim_diff))

        # 2. Identify existing axes that were "stretched" from 1
        # e.g., broadcasting from (3, 1) to (3, 5) stretches axis 1.
        for i in range(len(input_shape)):
            if input_shape[i] != self.shape[i + ndim_diff]:
                axes.append(i + ndim_diff)

        # 3. Sum the gradient along these axes
        sum_grad = out_grad.sum(tuple(axes))

        # 4. Reshape the result to match the original input's shape
        return sum_grad.reshape(input_shape)


class Summation(TensorOp):
    """Sum reduction over specified axes (or all axes if None)."""

    def __init__(
        self,
        axes: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        keepdims: bool = False,
    ) -> None:
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a.sum(self.axes, keepdims=self.keepdims)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        input_shape = node.inputs[0].shape

        # 1. Determine the shape to reshape `out_grad` into.
        # This shape will have `1`s for the axes that were summed.
        new_shape = list(input_shape)

        # Handle the case where self.axes is None (all axes summed)
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        for axis in axes:
            new_shape[axis] = 1

        # 2. Reshape the gradient
        reshaped_grad = out_grad.reshape(tuple(new_shape))

        # 3. Broadcast it to the original input_shape
        return reshaped_grad.broadcast_to(input_shape)


class Max(TensorOp):
    """Maximum reduction over specified axes (or all axes if None)."""

    def __init__(
        self,
        axes: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        keepdims: bool = False,
    ) -> None:
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a.max(self.axes, keepdims=self.keepdims)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        a = node.inputs[0]
        max_a = a.max(self.axes, keepdims=self.keepdims)
        # Only the max value will have a gradient of 1, all other values will have a
        # gradient of 0.
        return out_grad.broadcast_to(a.shape) * (a == max_a)


class MatMul(TensorOp):
    """Matrix multiplication for tensors of any dimension."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a, b) = args

        # This is the 2D-only path
        if a.ndim == 2 and b.ndim == 2:
            m, k_a = a.shape
            k_b, p = b.shape

            assert k_a == k_b, "Matrix inner dimensions must agree"

            # Call the 2D matmul from NDArray class
            return a @ b

        # This is the N-D @ 2D path
        elif a.ndim > 2 and b.ndim == 2:
            m, k_a = a.shape[-2:]
            k_b, p = b.shape

            assert k_a == k_b, "Matrix inner dimensions must agree"

            # Get batch dimensions
            batch_shape = a.shape[:-2]
            # Flatten A by removing the last two dimensions and reshaping the remaining
            # dimensions into a single dimension
            remaining_shape = math.prod(a.shape) // k_a
            a_flat = a.reshape((remaining_shape, k_a))

            # Multiply the flattened A by B
            c_flat = a_flat @ b

            # Reshape the result to the output shape
            output_shape = batch_shape + (m, p)

            return c_flat.reshape(output_shape)

        # Full broadcasted matmul is complex.
        # We can "cheat" by just using numpy.
        return nd.array(np.matmul(a.numpy(), b.numpy()))

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor", "Tensor"]:
        a, b = node.inputs
        a_grad = out_grad @ b.transpose()
        b_grad = a.transpose() @ out_grad

        a_ndim = a.ndim
        b_ndim = b.ndim

        if a_ndim > b_ndim:
            b_grad = b_grad.sum(tuple(range(a_ndim - b_ndim)))

        if b_ndim > a_ndim:
            a_grad = a_grad.sum(tuple(range(b_ndim - a_ndim)))

        return a_grad, b_grad


class Negate(TensorOp):
    """Elementwise negation."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return -a

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        return -out_grad


class Log(TensorOp):
    """Elementwise natural logarithm."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a.log()

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        return out_grad / node.inputs[0]


class Exp(TensorOp):
    """Elementwise exponential."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a.exp()

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        a = node.inputs[0]
        return out_grad * a.exp()


class ReLU(TensorOp):
    """Rectified Linear Unit: max(x, 0)."""

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (a,) = args
        return a.maximum(0)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        a = node.inputs[0]
        # lazy import to avoid circular import
        from needle.autograd import Tensor

        return out_grad * Tensor(a.realize_cached_data() > 0)
