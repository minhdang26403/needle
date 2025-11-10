import numpy as np

__device_name__ = "numpy"
_dtype = np.float64
_dtype_size = np.dtype(_dtype).itemsize


class Array:
    def __init__(self, size: int):
        # use numpy array as buffer to store the data
        self.buffer = np.empty(size, dtype=np.float64)

    @property
    def size(self) -> int:
        return self.buffer.size

    def ptr(self) -> int:
        return self.buffer.ctypes.data


def to_numpy(
    a: Array, shape: tuple[int, ...], strides: tuple[int, ...], offset: int
) -> np.ndarray:
    """Create a NumPy view into ``a.buffer`` with custom shape/strides and offset.

    Parameters
    ----------
    a : Array
        Source storage.
    shape : tuple of int
        Desired shape of the returned view.
    strides : tuple of int
        Strides expressed in number of elements (not bytes).
    offset : int
        Starting element offset into ``a.buffer``.

    Returns
    -------
    numpy.ndarray
        A view (no copy) that shares memory with ``a.buffer`` and uses
        the requested shape and strides.
    """
    return np.lib.stride_tricks.as_strided(
        a.buffer[offset:],
        shape,
        tuple(s * _dtype_size for s in strides),
    )


def from_numpy(numpy_array: np.ndarray, out: Array) -> None:
    """Copy values from an arbitrary NumPy array into ``out.buffer``.

    Parameters
    ----------
    numpy_array : numpy.ndarray
        Input array. Its flattened size must match ``out.size``.
    out : Array
        Destination storage whose buffer will be overwritten.

    Notes
    -----
    Values are copied in row-major (C-order) via ``numpy_array.flat``.
    A ``ValueError`` will be raised by NumPy if sizes are incompatible.
    """
    out.buffer[:] = numpy_array.flat


def fill(out: Array, val: float) -> None:
    """Fill the entire ``out.buffer`` with a scalar value.

    Parameters
    ----------
    out : Array
        Destination storage to fill.
    val : float
        Scalar value to write into every element.
    """
    out.buffer.fill(val)


def compact(
    a: Array, out: Array, shape: tuple[int, ...], strides: tuple[int, ...], offset: int
) -> None:
    """Materialize a non-compact view of ``a`` into a compact buffer ``out``.

    Parameters
    ----------
    a : Array
        Source storage.
    out : Array
        Destination storage that will receive the compact data.
    shape : tuple of int
        Shape of the logical view into ``a``.
    strides : tuple of int
        Strides of the logical view into ``a`` in elements.
    offset : int
        Starting element offset into ``a.buffer`` for the view.
    """
    out.buffer[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(
    a: Array, out: Array, shape: tuple[int, ...], strides: tuple[int, ...], offset: int
) -> None:
    """Write from compact ``a`` into a non-compact view of ``out``.

    Parameters
    ----------
    a : Array
        Source storage. Expected to be compact and of size ``prod(shape)``.
    out : Array
        Destination storage (may be non-compact via ``shape``/``strides``/``offset``).
    shape : tuple of int
        Logical shape of the output view.
    strides : tuple of int
        Output view strides in elements.
    offset : int
        Starting element offset into ``out.buffer`` for the view.
    """
    to_numpy(out, shape, strides, offset)[:] = a.buffer.reshape(shape)


def scalar_setitem(
    size: int,
    val: float,
    out: Array,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> None:
    """Set every element of a non-compact output view to a scalar value.

    Parameters
    ----------
    size : int
        Number of elements addressed by the output view. Present for API parity;
        not used by this NumPy implementation.
    val : float
        Scalar value to assign.
    out : Array
        Destination storage (may be non-compact via ``shape``/``strides``/``offset``).
    shape : tuple of int
        Logical shape of the output view.
    strides : tuple of int
        Output view strides in elements.
    offset : int
        Starting element offset into ``out.buffer`` for the view.
    """
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a: Array, b: Array, out: Array) -> None:
    """Elementwise addition ``out = a + b`` for compact buffers.

    Parameters
    ----------
    a : Array
        First input (compact).
    b : Array
        Second input (compact). Must have the same size as ``a``.
    out : Array
        Output (compact). Must have the same size as ``a`` and ``b``.
    """
    out.buffer[:] = a.buffer + b.buffer


def scalar_add(a: Array, val: float, out: Array) -> None:
    """Elementwise addition with scalar ``out = a + val`` for compact buffers.

    Parameters
    ----------
    a : Array
        Input (compact).
    val : float
        Scalar to add.
    out : Array
        Output (compact), same size as ``a``.
    """
    out.buffer[:] = a.buffer + val


def ewise_mul(a: Array, b: Array, out: Array) -> None:
    """Elementwise multiplication ``out = a * b`` for compact buffers.

    Parameters
    ----------
    a : Array
        First input (compact).
    b : Array
        Second input (compact). Must have the same size as ``a``.
    out : Array
        Output (compact). Must have the same size as ``a`` and ``b``.
    """
    out.buffer[:] = a.buffer * b.buffer


def scalar_mul(a: Array, val: float, out: Array) -> None:
    """Elementwise multiplication with scalar ``out = a * val`` for compact buffers.

    Parameters
    ----------
    a : Array
        Input (compact).
    val : float
        Scalar multiplier.
    out : Array
        Output (compact), same size as ``a``.
    """
    out.buffer[:] = a.buffer * val


def ewise_div(a: Array, b: Array, out: Array) -> None:
    """Elementwise division ``out = a / b`` for compact buffers.

    Parameters
    ----------
    a : Array
        Numerator (compact).
    b : Array
        Denominator (compact). Must have the same size as ``a``.
    out : Array
        Output (compact). Must have the same size as ``a`` and ``b``.
    """
    out.buffer[:] = a.buffer / b.buffer


def scalar_div(a: Array, val: float, out: Array) -> None:
    """Elementwise division by a scalar ``out = a / val`` for compact buffers.

    Parameters
    ----------
    a : Array
        Numerator (compact).
    val : float
        Scalar denominator.
    out : Array
        Output (compact), same size as ``a``.
    """
    out.buffer[:] = a.buffer / val


def ewise_power(a: Array, b: Array, out: Array) -> None:
    """Elementwise power ``out = a ** b`` for compact buffers.

    Parameters
    ----------
    a : Array
        Base values (compact).
    b : Array
        Exponent values (compact).
    out : Array
        Output (compact), same size as ``a`` and ``b``.
    """
    out.buffer[:] = a.buffer**b.buffer


def scalar_power(a: Array, val: float, out: Array) -> None:
    """Elementwise power with scalar exponent ``out = a ** val`` for compact buffers.

    Parameters
    ----------
    a : Array
        Base values (compact).
    val : float
        Exponent applied to every element.
    out : Array
        Output (compact), same size as ``a``.
    """
    out.buffer[:] = a.buffer**val


def ewise_maximum(a: Array, b: Array, out: Array) -> None:
    """Elementwise maximum ``out = maximum(a, b)`` for compact buffers.

    Parameters
    ----------
    a : Array
        First input (compact).
    b : Array
        Second input (compact). Must have the same size as ``a``.
    out : Array
        Output (compact). Must have the same size as ``a`` and ``b``.
    """
    out.buffer[:] = np.maximum(a.buffer, b.buffer)


def scalar_maximum(a: Array, val: float, out: Array) -> None:
    """Elementwise maximum with scalar ``out = maximum(a, val)`` for compact buffers.

    Parameters
    ----------
    a : Array
        Input (compact).
    val : float
        Scalar to compare against.
    out : Array
        Output (compact), same size as ``a``.
    """
    out.buffer[:] = np.maximum(a.buffer, val)


def ewise_eq(a: Array, b: Array, out: Array) -> None:
    """Elementwise equality test ``out = (a == b)`` stored as float64.

    Parameters
    ----------
    a : Array
        First input (compact).
    b : Array
        Second input (compact). Must have the same size as ``a``.
    out : Array
        Output (compact). Receives ``1.0`` where equal, ``0.0`` otherwise.
    """
    out.buffer[:] = (a.buffer == b.buffer).astype(_dtype)


def scalar_eq(a: Array, val: float, out: Array) -> None:
    """Elementwise equality to scalar ``out = (a == val)`` stored as float64.

    Parameters
    ----------
    a : Array
        Input (compact).
    val : float
        Scalar to compare against.
    out : Array
        Output (compact). Receives ``1.0`` where equal, ``0.0`` otherwise.
    """
    out.buffer[:] = (a.buffer == val).astype(_dtype)


def ewise_ge(a: Array, b: Array, out: Array) -> None:
    """Elementwise comparison ``out = (a >= b)`` stored as float64.

    Parameters
    ----------
    a : Array
        First input (compact).
    b : Array
        Second input (compact). Must have the same size as ``a``.
    out : Array
        Output (compact). Receives ``1.0`` where condition holds, ``0.0`` otherwise.
    """
    out.buffer[:] = (a.buffer >= b.buffer).astype(_dtype)


def scalar_ge(a: Array, val: float, out: Array) -> None:
    """Elementwise comparison to scalar ``out = (a >= val)`` stored as float64.

    Parameters
    ----------
    a : Array
        Input (compact).
    val : float
        Scalar threshold.
    out : Array
        Output (compact). Receives ``1.0`` where condition holds, ``0.0`` otherwise.
    """
    out.buffer[:] = (a.buffer >= val).astype(_dtype)


def ewise_log(a: Array, out: Array) -> None:
    """Elementwise natural logarithm ``out = log(a)`` for compact buffers.

    Parameters
    ----------
    a : Array
        Input (compact).
    out : Array
        Output (compact), same size as ``a``.

    Notes
    -----
    See :func:`numpy.log` for behavior on non-positive inputs (may produce
    ``-inf`` or ``nan``).
    """
    out.buffer[:] = np.log(a.buffer)


def ewise_exp(a: Array, out: Array) -> None:
    """Elementwise exponential ``out = exp(a)`` for compact buffers.

    Parameters
    ----------
    a : Array
        Input (compact).
    out : Array
        Output (compact), same size as ``a``.
    """
    out.buffer[:] = np.exp(a.buffer)


def ewise_tanh(a: Array, out: Array) -> None:
    """Elementwise hyperbolic tangent ``out = tanh(a)`` for compact buffers.

    Parameters
    ----------
    a : Array
        Input (compact).
    out : Array
        Output (compact), same size as ``a``.
    """
    out.buffer[:] = np.tanh(a.buffer)


def matmul(a: Array, b: Array, out: Array, m: int, n: int, p: int) -> None:
    """Matrix multiplication ``out = (A @ B).ravel()`` with compact buffers.

    Parameters
    ----------
    a : Array
        Left matrix storage containing ``A`` flattened with shape ``(m, n)``.
    b : Array
        Right matrix storage containing ``B`` flattened with shape ``(n, p)``.
    out : Array
        Output storage for ``C = A @ B`` flattened with shape ``(m * p,)``.
    m : int
        Number of rows of ``A`` and ``C``.
    n : int
        Shared inner dimension of ``A`` and ``B``.
    p : int
        Number of columns of ``A`` and ``C``.
    """
    out.buffer[:] = (a.buffer.reshape(m, n) @ b.buffer.reshape(n, p)).reshape(-1)


def reduce_max(a: Array, out: Array, reduce_size: int) -> None:
    """Reduce the last logical dimension by maximum.

    Parameters
    ----------
    a : Array
        Input (compact), conceptually reshaped to ``(-1, reduce_size)``.
    out : Array
        Output (compact) receiving one max per group; size ``a.size // reduce_size``.
    reduce_size : int
        Size of the reduced dimension.
    """
    out.buffer[:] = np.max(a.buffer.reshape(-1, reduce_size), axis=1)


def reduce_sum(a: Array, out: Array, reduce_size: int) -> None:
    """Reduce the last logical dimension by sum.

    Parameters
    ----------
    a : Array
        Input (compact), conceptually reshaped to ``(-1, reduce_size)``.
    out : Array
        Output (compact) receiving one sum per group; size ``a.size // reduce_size``.
    reduce_size : int
        Size of the reduced dimension.
    """
    out.buffer[:] = np.sum(a.buffer.reshape(-1, reduce_size), axis=1)
