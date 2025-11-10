cmake_minimum_required(VERSION 3.20)
project(needle_dlsys LANGUAGES CXX)

# Configure C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clangd users
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Placeholder: add subdirectories for native backends when implemented
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/CMakeLists.txt")
  add_subdirectory(src/cpu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/CMakeLists.txt")
  add_subdirectory(src/metal)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/CMakeLists.txt")
  add_subdirectory(src/cuda)
endif()

# Convenience: copy compile_commands.json to source dir for clangd
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
  )
endif()

