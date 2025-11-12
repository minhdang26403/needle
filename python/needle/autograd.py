"""Autograd engine for Tensor computations.

This module defines the public `Tensor` container and its minimal graph
machinery (producer op, inputs, cached data). The design keeps the compute
surface simple and backend-agnostic while allowing eager execution by default.

This module is a thin wrapper around the `Tensor` class that provides
autograd functionality. It is used to compute the gradient of a Tensor with
respect to a given input.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

import needle.ops as ops
from needle.backend.device import Device, default_device
from needle.backend.ndarray import NDArray, array

LAZY_MODE = False
TENSOR_COUNTER = 0


class Tensor:
    """User-facing multi-dimensional array with minimal autograd scaffolding.

    Notes
    -----
    - Owns a backend array (`nd.NDArray`) as `cached_data` when realized.
    - When constructed from another `Tensor`, reuses device/dtype by default.
    - Eager execution is enabled when `LAZY_MODE=False`.
    """

    # Trace of computational graph
    op: Optional[ops.TensorOp]
    inputs: Tuple["Tensor", ...]
    # Cached fields for dynamic computation
    cached_data: Optional[NDArray]
    requires_grad: bool

    grad: Optional["Tensor"]

    def __init__(
        self,
        array: Any,
        *,
        device: Optional[Device] = None,
        dtype: Optional[str] = None,
        requires_grad: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a tensor from array-like or an existing Tensor.

        Parameters
        ----------
        array
            Array-like data or another `Tensor`.
        device
            Target device; defaults to the global default device.
        dtype
            Optional dtype string; if omitted, backend default is used.
        requires_grad
            Whether to track gradients for this value (not implemented yet).
        """
        if isinstance(array, Tensor):
            if device is None:
                device = array.device

            if dtype is None:
                dtype = array.dtype

            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            (),
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(
        numpy_array: np.ndarray, *, device: Device, dtype: Optional[str] = None
    ) -> NDArray:
        """Create a backend array from a NumPy array on a device."""
        return array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: ops.TensorOp, inputs: Tuple["Tensor", ...]) -> "Tensor":
        """Construct a Tensor node produced by an op over given input tensors."""
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data: Any, requires_grad: bool = False) -> "Tensor":
        """Construct a leaf Tensor with already-available data."""
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            (),
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self) -> "Tensor":
        """Return a new tensor that shares the data but detaches from the graph."""
        return self.detach()

    @data.setter
    def data(self, value: "Tensor") -> None:
        """Set the data of the tensor."""
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self) -> "Tensor":
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    def realize_cached_data(self) -> NDArray:
        """Return realized backend data, computing if necessary."""
        # avoid recomputing the data if it has already been computed
        if self.cached_data is not None:
            return self.cached_data

        # compute the data
        assert self.op is not None
        self.cached_data = self.op.compute(
            *tuple(input.realize_cached_data() for input in self.inputs)
        )
        return self.cached_data

    def is_leaf(self) -> bool:
        """Return True if this Tensor has no producing op."""
        return self.op is None

    def __del__(self) -> None:
        """Best-effort counter bookkeeping for debugging."""
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[ops.TensorOp],
        inputs: Tuple["Tensor", ...],
        *,
        num_outputs: int = 1,
        cached_data: Optional[NDArray] = None,
        requires_grad: Optional[bool] = None,
    ) -> None:
        """Internal initializer used by constructors and `make_from_op`."""
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1

        # if requires_grad is not specified, the value requires grad if any of the
        # inputs requires grad
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)

        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of dimension sizes."""
        return self.realize_cached_data().shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def dtype(self) -> str:
        """String dtype descriptor (backend-defined)."""
        return self.realize_cached_data().dtype

    @property
    def device(self) -> Device:
        """Backend device where this Tensor's storage resides."""
        return self.realize_cached_data().device

    def __repr__(self) -> str:
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self) -> str:
        return self.realize_cached_data().__str__()

    def numpy(self) -> np.ndarray:
        """Return a NumPy view or copy of the underlying data."""
        return self.realize_cached_data().numpy()

    def backward(self, out_grad: Optional["Tensor"] = None) -> None:
        """Compute the gradient of the tensor with respect to the inputs."""
        if out_grad is None:
            # Lazy import to avoid circular import at module import time
            from needle.init import ones

            out_grad = ones(*self.shape, dtype=self.dtype, device=self.device)
        compute_gradient_of_variables(self, out_grad)

    def __add__(self, other: Union["Tensor", float]) -> "Tensor":
        """Elementwise addition, optionally with a scalar."""
        if isinstance(other, Tensor):
            return ops.EWiseAdd()(self, other)
        else:
            return ops.AddScalar(other)(self)

    def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
        """Elementwise multiplication, optionally with a scalar."""
        if isinstance(other, Tensor):
            return ops.EWiseMul()(self, other)
        else:
            return ops.MulScalar(other)(self)

    def __pow__(self, other: Union["Tensor", float]) -> "Tensor":
        """Elementwise power, optionally with a scalar exponent."""
        if isinstance(other, Tensor):
            return ops.EWisePow()(self, other)
        else:
            return ops.PowerScalar(other)(self)

    def __sub__(self, other: Union["Tensor", float]) -> "Tensor":
        """Elementwise subtraction, optionally by a scalar."""
        if isinstance(other, Tensor):
            return ops.EWiseAdd()(self, ops.Negate()(other))
        else:
            return ops.AddScalar(-other)(self)

    def __truediv__(self, other: Union["Tensor", float]) -> "Tensor":
        """Elementwise division, optionally by a scalar."""
        if isinstance(other, Tensor):
            return ops.EWiseDiv()(self, other)
        else:
            return ops.DivScalar(other)(self)

    def log(self) -> "Tensor":
        """Elementwise natural logarithm."""
        return ops.Log()(self)

    def exp(self) -> "Tensor":
        """Elementwise exponential."""
        return ops.Exp()(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication (2D by 2D)."""
        return ops.MatMul()(self, other)

    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication (alias for `__matmul__`)."""
        return ops.MatMul()(self, other)

    def sum(
        self,
        axes: Optional[Union[int, tuple[int, ...], list[int]]] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        """Sum reduction over specified axes (or all axes if None)."""
        return ops.Summation(axes, keepdims)(self)

    def broadcast_to(self, shape: tuple[int, ...]) -> "Tensor":
        """Return a broadcasted view to `shape` (no data copy)."""
        return ops.BroadcastTo(shape)(self)

    def reshape(self, shape: tuple[int, ...]) -> "Tensor":
        """Return a reshaped view to `shape` (no data copy)."""
        return ops.Reshape(shape)(self)

    def __neg__(self) -> "Tensor":
        """Elementwise negation."""
        return ops.Negate()(self)

    def transpose(self, axes: Optional[tuple[int, ...]] = None) -> "Tensor":
        """Return a transposed view (swap last two dims by default)."""
        return ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__


def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor) -> None:
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # A map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}

    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are
    # taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        output_grads = node_to_output_grads_list[node]
        node.grad = sum_node_list(output_grads)

        # Skip leaf node
        if node.op is None:
            continue

        assert isinstance(node.op, ops.TensorOp)

        # Compute the gradients with respect to the input nodes.
        input_grads = node.op.gradient_as_tuple(node.grad, node)
        for input_node, input_grad in zip(node.inputs, input_grads):
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = [input_grad]
            else:
                node_to_output_grads_list[input_node].append(input_grad)


def find_topo_sort(node_list: List[Tensor]) -> List[Tensor]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    visited: Set[Tensor] = set()
    topo_order: List[Tensor] = []
    for node in node_list:
        if node not in visited:
            topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node: Tensor, visited: Set[Tensor], topo_order: List[Tensor]) -> None:
    """Post-order DFS to find topological order of the graph."""
    visited.add(node)
    for pred in node.inputs:
        if pred not in visited:
            topo_sort_dfs(pred, visited, topo_order)
    topo_order.append(node)


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list: List[Tensor]) -> Tensor:
    """Custom sum function in order to avoid creating redundant nodes in Python sum
    implementation."""
    from functools import reduce
    from operator import add

    return reduce(add, node_list)
