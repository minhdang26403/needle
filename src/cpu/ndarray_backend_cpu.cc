#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
using scalar_t = float;
constexpr size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * @brief Contiguous heap allocation aligned to @c ALIGNMENT byte boundaries.
 *
 * Owns a @c scalar_t buffer of @c size elements allocated with @c posix_memalign.
 * The alignment is chosen to be friendly to vectorized math and tiled kernels
 * (at least @c TILE * @c ELEM_SIZE).
 *
 * The buffer is freed in the destructor; instances are non-copyable via default
 * generated semantics because the raw pointer is owned.
 */
struct AlignedArray {
  /**
   * @brief Allocate an aligned buffer for @p size elements.
   * @param size Number of scalar elements.
   * @throws std::bad_alloc If aligned allocation fails.
   */
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) {
      throw std::bad_alloc();
    }
    this->size = size;
  }

  /**
   * @brief Free the aligned buffer.
   */
  ~AlignedArray() {
    free(ptr);
  }

  /**
   * @brief Return the raw pointer value as an integer.
   * @return Pointer as integer (useful for interop with Python-side tests).
   */
  size_t ptr_as_int() {
    return (size_t)ptr;
  }

  scalar_t* ptr;
  size_t size;
};


/**
 * @brief Fill every element of @p out with a scalar value.
 * @param out Destination aligned array.
 * @param val Scalar value to write to all elements.
 * @note Complexity: O(out->size).
 */
void Fill(AlignedArray* out, scalar_t val) {
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

/**
 * @brief N-dimensional index traversal over a logical strided view.
 *
 * Applies @p op for each logical element in a strided view described by
 * @p shape, @p strides, and @p offset. The @p op receives the compact
 * (0..num_elements-1) index and the computed strided offset (in elements).
 *
 * @param shape Logical shape of the view (elements).
 * @param strides Element-wise strides for each dimension of the view.
 * @param offset Starting element offset into the base storage.
 * @param num_elements Total logical elements to iterate.
 * @param op Callback invoked with (compact_index, strided_offset).
 *
 * @note This routine is backend-agnostic and used by multiple kernels to
 *       avoid reimplementing strided iteration logic.
 */
void NdApply(
  const std::vector<int32_t>& shape,
  const std::vector<int32_t>& strides,
  size_t offset,
  size_t num_elements,
  std::function<void(size_t, size_t)> op) {

  size_t ndim = shape.size();
  std::vector<size_t> indices(ndim, 0);

  for (size_t cnt = 0; cnt < num_elements; cnt++) {
    size_t strided_offset = offset;
    for (size_t i = 0; i < ndim; i++) {
      strided_offset += strides[i] * indices[i];
    }

    op(cnt, strided_offset);

    int32_t d = ndim - 1;
    indices[d]++;

    while (indices[d] == shape[d]) {
      if (d == 0) {
        break;
      }
      indices[d--] = 0;
      indices[d]++;
    }
  }
}

/**
 * @brief Compact a strided view @p a into contiguous buffer @p out.
 *
 * @param a Source non-compact aligned array (base storage).
 * @param out Destination aligned array with compact layout (size equals product(shape)).
 * @param shape Logical shape of the view to compact.
 * @param strides Element-wise strides describing the view into @p a.
 * @param offset Starting element offset into @p a for the view.
 *
 * @note Copies elements in row-major logical order into @p out.
 */
void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  auto op = [&](size_t compact_idx, size_t strided_offset) {
    out->ptr[compact_idx] = a.ptr[strided_offset];
  };

  NdApply(shape, strides, offset, out->size, op);
}

/**
 * @brief Write values from compact @p a into non-compact view of @p out.
 *
 * @param a Source compact array.
 * @param out Destination non-compact view's base storage.
 * @param shape Logical shape of the destination view.
 * @param strides Strides (elements) of the destination view into @p out.
 * @param offset Starting element offset of the destination view.
 */
void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  auto op = [&](size_t compact_idx, size_t strided_offset) {
    out->ptr[strided_offset] = a.ptr[compact_idx];
  };

  NdApply(shape, strides, offset, a.size, op);
}

/**
 * @brief Set every element of a non-compact view @p out to a scalar @p val.
 *
 * @param size Number of elements addressed by the non-compact view.
 * @param val Scalar to assign.
 * @param out Destination base storage.
 * @param shape Logical shape of the view.
 * @param strides Strides (elements) of the view into @p out.
 * @param offset Starting element offset of the view.
 */
void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  auto op = [&](size_t compact_idx, size_t strided_offset) {
    out->ptr[strided_offset] = val;
  };

  NdApply(shape, strides, offset, size, op);
}

/**
 * @brief Apply an elementwise unary operation.
 *
 * @tparam Op Callable type with signature @c scalar_t(scalar_t)
 * @param a Input compact array.
 * @param out Output compact array (same size as @p a).
 * @param op Operation to apply for each element.
 */
template<typename Op>
void UnaryEwiseOp(const AlignedArray& a, AlignedArray* out, Op op) {
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = op(a.ptr[i]);
  }
}

/**
 * @brief Apply an elementwise binary operation.
 *
 * @tparam Op Callable type with signature @c scalar_t(scalar_t, scalar_t)
 * @param a First input compact array.
 * @param b Second input compact array (same size as @p a).
 * @param out Output compact array (same size as inputs).
 * @param op Operation to apply to each pair of elements.
 */
template<typename Op>
void BinaryEwiseOp(
  const AlignedArray& a,
  const AlignedArray& b,
  AlignedArray* out,
  Op op) {

  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}

/**
 * @brief Elementwise addition: out = a + b.
 */
void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BinaryEwiseOp(a, b, out, [](scalar_t e1, scalar_t e2) {
    return e1 + e2;
  });
}

/**
 * @brief Elementwise scalar addition: out = a + val.
 */
void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  UnaryEwiseOp(a, out, [&](scalar_t e) {
    return e + val;
  });
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/**
 * @brief Elementwise multiplication: out = a * b.
 */
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BinaryEwiseOp(a, b, out, [](scalar_t e1, scalar_t e2) {
    return e1 * e2;
  });
}

/**
 * @brief Elementwise scalar multiplication: out = a * val.
 */
void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  UnaryEwiseOp(a, out, [&](scalar_t e) {
    return e * val;
  });
}

/**
 * @brief Elementwise division: out = a / b.
 */
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BinaryEwiseOp(a, b, out, [](scalar_t e1, scalar_t e2) {
    return e1 / e2;
  });
}

/**
 * @brief Elementwise scalar division: out = a / val.
 */
void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  UnaryEwiseOp(a, out, [&](scalar_t e) {
    return e / val;
  });
}

/**
 * @brief Elementwise power with scalar exponent: out = pow(a, val).
 */
void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  UnaryEwiseOp(a, out, [&](scalar_t e) {
    return std::pow(e, val);
  });
}

/**
 * @brief Elementwise maximum: out = max(a, b).
 */
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BinaryEwiseOp(a, b, out, [](scalar_t e1, scalar_t e2) {
    return std::max(e1, e2);
  });
}

/**
 * @brief Elementwise scalar maximum: out = max(a, val).
 */
void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  UnaryEwiseOp(a, out, [&](scalar_t e) {
    return std::max(e, val);
  });
}

/**
 * @brief Elementwise equality predicate (as 0/1): out = (a == b).
 */
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BinaryEwiseOp(a, b, out, [](scalar_t e1, scalar_t e2) {
    return e1 == e2;
  });
}

/**
 * @brief Elementwise scalar equality predicate (as 0/1): out = (a == val).
 */
void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  UnaryEwiseOp(a, out, [&](scalar_t e) {
    return e == val;
  });
}

/**
 * @brief Elementwise greater-or-equal predicate (as 0/1): out = (a >= b).
 */
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  BinaryEwiseOp(a, b, out, [](scalar_t e1, scalar_t e2) {
    return e1 >= e2;
  });
}

/**
 * @brief Elementwise scalar greater-or-equal predicate (as 0/1): out = (a >= val).
 */
void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  UnaryEwiseOp(a, out, [&](scalar_t e) {
    return e >= val;
  });
}

/**
 * @brief Elementwise natural logarithm: out = log(a).
 * @note Domain errors for non-positive inputs follow @c std::log semantics.
 */
void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  UnaryEwiseOp(a, out, [](scalar_t e) {
    return std::log(e);
  });
}

/**
 * @brief Elementwise exponential: out = exp(a).
 */
void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  UnaryEwiseOp(a, out, [](scalar_t e) {
    return std::exp(e);
  });
}


/**
 * @brief Elementwise hyperbolic tangent: out = tanh(a).
 */
void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  UnaryEwiseOp(a, out, [](scalar_t e) {
    return std::tanh(e);
  });
}


/**
 * @brief Dense matrix multiplication out = a[m x n] @ b[n x p] (compact arrays).
 *
 * @param a Left operand, compact 2D array of size m * n.
 * @param b Right operand, compact 2D array of size n * p.
 * @param out Output, compact 2D array of size m * p (written).
 * @param m Rows of @p a and @p out.
 * @param n Columns of @p a, rows of @p b.
 * @param p Columns of @p b and @p out.
 *
 * @note Naive three-loop multiplication; consider tiled kernel for performance.
 */
void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < p; j++) {
      scalar_t sum = 0;
      for (uint32_t k = 0; k < n; k++) {
        sum += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
      out->ptr[i * p + j] = sum;
    }
  }
}

/**
 * @brief Multiply two TILE x TILE blocks and accumulate into @p out.
 *
 * The function assumes non-overlapping, properly aligned pointers to enable
 * vectorization. The result is added to @p out, not overwritten.
 *
 * @param a Pointer to left TILE x TILE block (row-major).
 * @param b Pointer to right TILE x TILE block (row-major).
 * @param out Pointer to output TILE x TILE block (row-major), updated with +=.
 */
inline void AlignedDot(const scalar_t* __restrict__ a,
                       const scalar_t* __restrict__ b,
                       scalar_t* __restrict__ out) {

  a = (const scalar_t*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const scalar_t*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (scalar_t*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      scalar_t sum = 0;
      for (size_t k = 0; k < TILE; k++) {
        sum += a[i * TILE + k] * b[k * TILE + j];
      }
      out[i * TILE + j] += sum;
    }
  }
}

/**
 * @brief Tiled matrix multiplication on 4D tiled representations.
 *
 * Computes @c out = a @ b where tensors are laid out as:
 * - a: [m/TILE][n/TILE][TILE][TILE]
 * - b: [n/TILE][p/TILE][TILE][TILE]
 * - out: [m/TILE][p/TILE][TILE][TILE]
 *
 * @param a Tiled left operand.
 * @param b Tiled right operand.
 * @param out Tiled output (accumulated into; zero-initialized here).
 * @param m Rows of the untiled result matrix.
 * @param n Shared inner dimension.
 * @param p Columns of the untiled result matrix.
 *
 * @note Assumes @p m, @p n, and @p p are multiples of @c TILE.
 */
void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  for (size_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }

  for (size_t i = 0; i < m / TILE; i++) {
    for (size_t j = 0; j < p / TILE; j++) {
      for (size_t k = 0; k < n / TILE; k++) {
        AlignedDot(
          &a.ptr[i * n * TILE + k * TILE * TILE],
          &b.ptr[k * p * TILE + j * TILE * TILE],
          &out->ptr[i * p * TILE + j * TILE * TILE]
        );
      }
    }
  }
}

/**
 * @brief Reduce contiguous blocks with an associative binary op.
 *
 * @tparam BinaryOp Callable with signature @c scalar_t(scalar_t, scalar_t)
 * @param a Input compact array of size @c out->size * @p reduce_size.
 * @param out Output compact array receiving one reduced value per block.
 * @param reduce_size Size of each contiguous reduction block.
 * @param op Associative reduction operator (e.g., sum, max).
 */
template<typename BinaryOp>
void Reduce(
  const AlignedArray& a,
  AlignedArray* out,
  size_t reduce_size,
  BinaryOp op) {

  size_t idx = 0;
  for (size_t i = 0; i < out->size; i++) {
    size_t end = idx + reduce_size;
    scalar_t init = a.ptr[idx++];
    while (idx < end) {
      init = op(init, a.ptr[idx++]);
    }
    out->ptr[i] = init;
  }
}

/**
 * @brief Reduce by maximum over contiguous blocks.
 *
 * @param a Input compact array of size @c out->size * reduce_size.
 * @param out Output compact array receiving blockwise maxima.
 * @param reduce_size Size of the reduced dimension.
 */
void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  Reduce(a, out, reduce_size, [](scalar_t init, scalar_t v) {
    return std::max(init, v);
  });
}

/**
 * @brief Reduce by sum over contiguous blocks.
 *
 * @param a Input compact array of size @c out->size * reduce_size.
 * @param out Output compact array receiving blockwise sums.
 * @param reduce_size Size of the reduced dimension.
 */
void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  Reduce(a, out, reduce_size, [](scalar_t init, scalar_t v) {
    return init + v;
  });
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  // Python module binding for the CPU backend. Exposes aligned array storage,
  // conversion helpers, elementwise ops, reductions, and matmul kernels.
  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
