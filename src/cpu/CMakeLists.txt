find_package(pybind11 CONFIG REQUIRED)

# Build the pybind11 module for the CPU backend
pybind11_add_module(
  ndarray_backend_cpu
  "${CMAKE_CURRENT_SOURCE_DIR}/ndarray_backend_cpu.cc"
)

target_compile_features(ndarray_backend_cpu PRIVATE cxx_std_17)

# Install the extension into the Python package "needle"
# scikit-build-core will stage this into the wheel under python/needle/
install(
  TARGETS ndarray_backend_cpu
  LIBRARY DESTINATION "needle/backend"
)


