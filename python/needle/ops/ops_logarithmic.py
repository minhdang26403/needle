from typing import TYPE_CHECKING

import needle.backend.ndarray as nd
from needle.ops import TensorOp

if TYPE_CHECKING:
    from needle.autograd import Tensor


class LogSoftmax(TensorOp):
    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (Z,) = args

        reduced_shape = list(Z.shape)
        reduced_shape[1] = 1

        # We can compute softmax via logsumexp and reuse the code.
        # Since the logsumexp is computed along the axis 1, we need to reshape the
        # input to shape that allows broadcasting.
        log_sum_exp = LogSumExp(axes=(1,)).compute(Z).reshape(tuple(reduced_shape))

        # Explicit broadcast to make the operation work with generic ndarray backend.
        return Z - log_sum_exp.broadcast_to(Z.shape)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        Z = node.inputs[0]

        # Get the softmax of the input.
        softmax_Z = Z.logsoftmax().exp()
        assert Z.shape == softmax_Z.shape

        sum_out_grad = out_grad.sum(axes=(1,), keepdims=True)

        return out_grad - softmax_Z * sum_out_grad.broadcast_to(Z.shape)


class LogSumExp(TensorOp):
    def __init__(self, axes: int | tuple[int, ...] | list[int] | None = None) -> None:
        self.axes = axes

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        (Z,) = args

        # Preserve the shape of the input so that we can broadcast the max_Z to the
        # original shape.
        self.max_Z = Z.max(axis=self.axes, keepdims=True)

        # Our NDArray does not support implicit broadcasting, so we need to explicitly
        # broadcast the max_Z to the original shape.
        log_sum_exp = (
            (Z - self.max_Z.broadcast_to(Z.shape)).exp().sum(axis=self.axes).log()
        )

        # We perform the max on max_Z to get the correct shape as log_sum_exp.
        return log_sum_exp + self.max_Z.max(axis=self.axes)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        Z = node.inputs[0]

        max_Z = Z.max(axes=self.axes, keepdims=True)
        exp_Z = (Z - max_Z.broadcast_to(Z.shape)).exp()

        # During the summation, the axes are summed, thus the summed axes are removed.
        # Therefore, we need to construct the shape that allows broadcasting to the
        # original shape.
        reduced_shape = list(Z.shape)
        if self.axes:
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
            for axis in self.axes:
                reduced_shape[axis] = 1
        else:
            reduced_shape = [1] * len(Z.shape)

        # Retain the shape after the summation. This has the same effect as the
        # keepdims=True parameter in ndarray.sum() of numpy.
        exp_sum = exp_Z.sum(axes=self.axes, keepdims=True)

        # out_grad's axes are removed, thus we need to reshape it to the original shape.
        return (
            out_grad.reshape(tuple(reduced_shape)).broadcast_to(Z.shape)
            * exp_Z
            / exp_sum.broadcast_to(Z.shape)
        )
