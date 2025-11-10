"""Operator base class for Tensor-producing ops.

An op defines two contracts:
  - compute(*ndarrays) -> ndarray    : forward pass on realized backend arrays
  - gradient(out_grad, node) -> Tensor | tuple[Tensor, ...]  : adjoints

`TensorOp` implements `__call__` to create Tensor graph nodes and delegates
execution and gradients to subclasses.
"""

from typing import TYPE_CHECKING, Tuple

import needle.backend.ndarray as nd

if TYPE_CHECKING:
    from needle.autograd import Tensor


class TensorOp:
    """Operator that produces a Tensor output and defines compute/gradient.

    There may be alternate subclasses for other structures.
    """

    def __call__(self, *args: "Tensor") -> "Tensor":
        """Create a Tensor graph node by applying this op to input tensors."""
        from needle.autograd import Tensor  # lazy to avoid circular import

        return Tensor.make_from_op(self, args)

    def compute(self, *args: nd.NDArray) -> nd.NDArray:
        """Calculate the forward pass on realized backend arrays.

        Parameters
        ----------
        input: tuple of NDArray
            The realized input arrays to the function.

        Returns
        -------
        output: NDArray
            Array output of the operation.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def gradient(
        self, out_grad: "Tensor", node: "Tensor"
    ) -> "Tensor" | Tuple["Tensor", ...]:
        """Return adjoint(s) for each input, given output adjoint `out_grad`."""
        raise NotImplementedError("Subclasses must implement this method.")

    def gradient_as_tuple(
        self, out_grad: "Tensor", node: "Tensor"
    ) -> Tuple["Tensor", ...]:
        """Convenience method to always return a tuple from gradient call."""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        else:
            return (output,)
