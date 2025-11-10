import math
from typing import Any, Callable, Union, cast

import numpy as np

from .device import Device, default_device


class NDArray:
    """Backend-agnostic N-dimensional array.

    This class abstracts the underlying array implementation (NumPy, CPU/C++,
    Metal, CUDA, etc.) and provides a uniform, minimal interface for array
    creation, views, indexing, broadcasting, elementwise ops, reductions, and
    matrix multiplication.
    """

    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _offset: int
    _device: Device
    _handle: Any

    def __init__(self, other: Any, device: Device | None = None) -> None:
        """Construct an NDArray from another NDArray, NumPy array, or array-like.

        Parameters
        ----------
        other : NDArray | numpy.ndarray | array_like
            Source to create from. If an NDArray, a device-aware copy is made.
            If a NumPy array or array-like, data are copied onto the chosen device.
        device : Device | None, optional
            Target device. If omitted and ``other`` is an NDArray, the other's
            device is used; otherwise the global default device is used.
        """
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(other, array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: "NDArray") -> None:
        """Initialize this instance by taking metadata and handle from ``other``."""
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute compact (row-major) strides for a shape.

        Parameters
        ----------
        shape : tuple of int
            Target shape.

        Returns
        -------
        tuple of int
            Strides in elements for a compact layout.
        """
        stride = 1
        strides = []
        for i in range(len(shape) - 1, -1, -1):
            strides.append(stride)
            stride *= shape[i]
        return tuple(reversed(strides))

    @staticmethod
    def make(
        shape: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        device: Device | None = None,
        handle: Any = None,
        offset: int = 0,
    ) -> "NDArray":
        """Create a new NDArray with explicit metadata and optional existing storage.

        Parameters
        ----------
        shape : tuple of int
            Desired logical shape.
        strides : tuple of int | None, optional
            Strides in elements. If None, compact strides are computed.
        device : Device | None, optional
            Target device. Defaults to the global default device.
        handle : Any, optional
            Existing backend handle representing the storage. If None, new
            storage is allocated.
        offset : int, optional
            Element offset into ``handle`` storage. Defaults to 0.

        Returns
        -------
        NDArray
            A new NDArray with the specified layout and storage.
        """
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = (
            NDArray.compact_strides(shape) if strides is None else tuple(strides)
        )
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(math.prod(shape))
        else:
            array._handle = handle
        return array

    ### Properties and string representations
    @property
    def shape(self) -> tuple[int, ...]:
        """tuple[int, ...]: Logical shape of the array."""
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        """tuple[int, ...]: Strides in elements for each dimension."""
        return self._strides

    @property
    def device(self) -> Device:
        """Device: The device on which this array's storage resides."""
        return self._device

    @property
    def dtype(self) -> str:
        """str: Data type name (currently always ``'float32'``)."""
        return "float32"

    @property
    def ndim(self) -> int:
        """int: Number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """int: Total number of elements as the product of ``shape``."""
        return math.prod(self._shape)

    def __repr__(self) -> str:
        """Return an unambiguous string representation."""
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self) -> str:
        """Return a readable string representation of the array contents."""
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value: float) -> None:
        """Fill the entire array with a scalar value.

        Parameters
        ----------
        value : float
            Scalar to write to all elements.
        """
        self.device.fill(self._handle, value)

    def to(self, device: Device) -> "NDArray":
        """Move or copy the array to another device.

        Parameters
        ----------
        device : Device
            Target device.

        Returns
        -------
        NDArray
            ``self`` if already on ``device``; otherwise a new NDArray on ``device``.
        """
        if self.device == device:
            return self
        else:
            return NDArray(self.numpy(), device)

    def numpy(self) -> np.ndarray:
        """Convert to a NumPy ndarray (host representation).

        Returns
        -------
        numpy.ndarray
            A NumPy view or copy of this array on the host.
        """
        return cast(
            np.ndarray,
            self.device.to_numpy(self._handle, self.shape, self.strides, self._offset),
        )

    def is_compact(self) -> bool:
        """Return whether the array is compact in memory.

        The array is compact if its strides match compact row-major strides and
        the underlying storage size equals ``prod(shape)``.

        Returns
        -------
        bool
            True if compact, False otherwise.
        """
        return (
            self._strides == self.compact_strides(self._shape)
            and math.prod(self.shape) == self._handle.size
        )

    def compact(self) -> "NDArray":
        """Return a compact copy if needed, otherwise return ``self``.

        Returns
        -------
        NDArray
            A compact array with identical contents.
        """
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape: tuple[int, ...], strides: tuple[int, ...]) -> "NDArray":
        """Create a new view with given shape and strides (no data copy).

        Parameters
        ----------
        shape : tuple of int
            New logical shape.
        strides : tuple of int
            Strides in elements for the new view.

        Returns
        -------
        NDArray
            A view sharing storage with this array.
        """
        assert len(shape) == len(strides)
        return NDArray.make(
            shape,
            strides=strides,
            device=self.device,
            handle=self._handle,
            offset=self._offset,
        )

    @property
    def flat(self) -> "NDArray":
        """NDArray: A view of the array flattened to 1-D."""
        return self.reshape((self.size,))

    def reshape(self, new_shape: tuple[int, ...]) -> "NDArray":
        """Reshape to ``new_shape`` without copying memory.

        Parameters
        ----------
        new_shape : tuple of int
            Target shape. Product must equal current size.

        Returns
        -------
        NDArray
            A view with new shape sharing the same storage.

        Raises
        ------
        ValueError
            If the size changes or if the array is not compact.
        """

        if math.prod(self.shape) != math.prod(new_shape):
            raise ValueError("Product of current shape must equal product of new shape")
        if not self.is_compact():
            raise ValueError("Array must be compact to reshape")

        return NDArray.make(
            new_shape,
            NDArray.compact_strides(new_shape),
            self.device,
            self._handle,
            self._offset,
        )

    def permute(self, new_axes: tuple[int, ...]) -> "NDArray":
        """Permute dimensions according to ``new_axes`` without copying memory.

        Parameters
        ----------
        new_axes : tuple of int
            A permutation of ``range(ndim)`` describing the new axis order.

        Returns
        -------
        NDArray
            A view with permuted shape/strides sharing the same storage.
        """

        new_shape = tuple(self.shape[i] for i in new_axes)
        new_strides = tuple(self.strides[i] for i in new_axes)

        return NDArray.make(
            new_shape,
            new_strides,
            self.device,
            self._handle,
            self._offset,
        )

    def broadcast_to(self, new_shape: tuple[int, ...]) -> "NDArray":
        """Broadcast to ``new_shape`` by adjusting strides (no copy).

        Broadcasting is allowed only along dimensions with size 1.

        Parameters
        ----------
        new_shape : tuple of int
            Target shape.

        Returns
        -------
        NDArray
            A broadcasted view sharing storage with this array.

        Raises
        ------
        AssertionError
            If a non-singleton dimension is changed.
        """

        assert len(new_shape) == len(self.shape), (
            "New shape must have the same number of dimensions as the original shape"
        )

        for x, y in zip(self.shape, new_shape):
            assert x == y or x == 1, (
                "New shape must have the same dimensions as the original shape, "
                "or the dimension of the original shape must be 1"
            )

        new_strides = tuple(
            0 if x != y else self.strides[i]
            for i, (x, y) in enumerate(zip(self.shape, new_shape))
        )

        return NDArray.make(
            new_shape,
            new_strides,
            self.device,
            self._handle,
            self._offset,
        )

    ### Get and set elements

    def process_slice(self, sl: slice, dim: int) -> slice:
        """Normalize a slice to explicit ``start``, ``stop``, and positive ``step``.

        Parameters
        ----------
        sl : slice
            Original slice.
        dim : int
            Dimension index the slice applies to (used for defaults).

        Returns
        -------
        slice
            Normalized slice with all fields set and positive step.
        """
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step is None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs: int | slice | tuple[int | slice, ...]) -> "NDArray":
        """Return a strided view per the given index/slice specification.

        Parameters
        ----------
        idxs : int | slice | tuple[int | slice, ...]
            Indexing specification; will be normalized to per-dimension slices.

        Returns
        -------
        NDArray
            A view into the same storage with updated shape/strides/offset.

        Raises
        ------
        AssertionError
            If invalid slices are provided or dimensions mismatch.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        slices = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(slices) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = tuple((s.stop - s.start + s.step - 1) // s.step for s in slices)
        new_strides = tuple(stride * s.step for stride, s in zip(self.strides, slices))
        new_offset = self._offset
        for stride, s in zip(self.strides, slices):
            new_offset += stride * s.start

        return NDArray.make(
            new_shape,
            new_strides,
            self.device,
            self._handle,
            new_offset,
        )

    def __setitem__(
        self,
        idxs: int | slice | tuple[int | slice, ...],
        other: Union["NDArray", float],
    ) -> None:
        """Assign to a strided view specified by ``idxs``.

        Parameters
        ----------
        idxs : int | slice | tuple[int | slice, ...]
            Indexing specification producing the output view (same semantics
            as ``__getitem__``).
        other : NDArray | float
            Source data. If NDArray, shapes must match; otherwise broadcast
            a scalar.
        """
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert math.prod(view.shape) == math.prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                math.prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Element-wise and scalar operations

    def ewise_or_scalar(
        self,
        other: Union["NDArray", float],
        ewise_func: Callable[[Any, Any, Any], None],
        scalar_func: Callable[[Any, Any, Any], None],
    ) -> "NDArray":
        """Apply an elementwise or scalar backend function depending on ``other``.

        Parameters
        ----------
        other : NDArray | float
            Second operand.
        ewise_func : Callable
            Backend function for NDArray vs NDArray operation.
        scalar_func : Callable
            Backend function for NDArray vs scalar operation.

        Returns
        -------
        NDArray
            Resulting array on the same device.
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise addition."""
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise subtraction."""
        return self + (-other)

    def __rsub__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise reverse subtraction."""
        return other + (-self)

    def __mul__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise multiplication."""
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise true division."""
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self) -> "NDArray":
        """Elementwise negation."""
        return self * (-1)

    def __pow__(self, other: float) -> "NDArray":
        """Elementwise exponentiation by a scalar."""
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise maximum."""
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    def log(self) -> "NDArray":
        """Elementwise natural logarithm."""
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self) -> "NDArray":
        """Elementwise exponential."""
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self) -> "NDArray":
        """Elementwise hyperbolic tangent."""
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Binary operations
    def __eq__(self, other: Union["NDArray", float]) -> "NDArray":  # type: ignore[override]
        """Elementwise equality returning a float32 mask (1.0 or 0.0)."""
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise greater-or-equal returning a float32 mask (1.0 or 0.0)."""
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other: Union["NDArray", float]) -> "NDArray":  # type: ignore[override]
        """Elementwise not-equal derived from equality."""
        return 1 - (self == other)

    def __gt__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise greater-than derived from >= and !=."""
        return (self >= other) * (self != other)

    def __lt__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise less-than derived from >=."""
        return 1 - (self >= other)

    def __le__(self, other: Union["NDArray", float]) -> "NDArray":
        """Elementwise less-or-equal derived from >."""
        return 1 - (self > other)

    ### Matrix multiplication
    def __matmul__(self, other: "NDArray") -> "NDArray":
        """Matrix multiplication of two 2D arrays.

        Both operands must be 2D with matching inner dimensions. On certain
        CPU backends, a tiled implementation may be used when shapes are
        multiples of a tile size; otherwise a generic implementation is used.
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a: "NDArray", tile: int) -> "NDArray":
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return out.permute((0, 2, 1, 3)).compact().reshape((m, p))
        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over a given axis
    def reduce_view_out(
        self, axis: int | tuple[int, ...] | list[int] | None, keepdims: bool = False
    ) -> tuple["NDArray", "NDArray"]:
        """Prepare a compact reduction view and corresponding output array.

        Parameters
        ----------
        axis : int | tuple[int, ...] | list[int] | None
            Axis to reduce over. Only a single axis is supported; ``None`` reduces
            over all elements.
        keepdims : bool, optional
            If True, keep the reduced dimension with size 1.

        Returns
        -------
        tuple[NDArray, NDArray]
            A tuple of (view, out), where ``view`` is a permuted/reshaped input
            such that the last dimension is reduced, and ``out`` is the output
            array with appropriate shape.
        """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape(
                (1,) * (self.ndim - 1) + (math.prod(self.shape),)
            )
            out = NDArray.make((1,), device=self.device)
        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(
                tuple(a for a in range(self.ndim) if a != axis) + (axis,)
            )
            out = NDArray.make(
                tuple(1 if i == axis else s for i, s in enumerate(self.shape))
                if keepdims
                else tuple(s for i, s in enumerate(self.shape) if i != axis),
                device=self.device,
            )

        return view, out

    def sum(
        self,
        axis: int | tuple[int, ...] | list[int] | None = None,
        keepdims: bool = False,
    ) -> "NDArray":
        """Sum of array elements over a given axis.

        Parameters
        ----------
        axis : int | tuple[int, ...] | list[int] | None, optional
            Axis to reduce over. Only one axis is supported. If None, sum over all
            elements and return a scalar-shaped array.
        keepdims : bool, optional
            If True, keep the reduced dimension with size 1.

        Returns
        -------
        NDArray
            The reduced array.
        """
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(
        self,
        axis: int | tuple[int, ...] | list[int] | None = None,
        keepdims: bool = False,
    ) -> "NDArray":
        """Maximum of array elements over a given axis.

        Parameters
        ----------
        axis : int | tuple[int, ...] | list[int] | None, optional
            Axis to reduce over. Only one axis is supported. If None, max over all
            elements and return a scalar-shaped array.
        keepdims : bool, optional
            If True, keep the reduced dimension with size 1.

        Returns
        -------
        NDArray
            The reduced array.
        """
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out


def array(a: Any, dtype: str = "float32", device: Device | None = None) -> NDArray:
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)
