cmake_minimum_required(VERSION 3.19)
project(fwmc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler warnings and MSVC compatibility
if(MSVC)
  add_compile_options(/W4)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optimization and SIMD
if(MSVC)
  add_compile_options(/arch:AVX2)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optional OpenMP for parallel neuron dynamics
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP enabled: ${OpenMP_CXX_VERSION}")
else()
  message(STATUS "OpenMP not found, single-threaded build")
endif()

# Optional CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_ARCHITECTURES "native")
  add_compile_definitions(FWMC_CUDA)
  message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
else()
  message(STATUS "CUDA not found, CPU-only build")
endif()

include_directories(src brain-model)

# CUDA kernel library (only when CUDA is available)
if(CMAKE_CUDA_COMPILER)
  add_library(fwmc-cuda STATIC
    src/cuda/izhikevich_kernel.cu
    src/cuda/spike_propagation_kernel.cu
    src/cuda/stdp_kernel.cu
  )
  target_include_directories(fwmc-cuda PRIVATE src)
  set_target_properties(fwmc-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
endif()

# Main executable
add_executable(fwmc src/fwmc.cc)
if(NOT MSVC)
  target_link_libraries(fwmc PRIVATE pthread)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(fwmc PRIVATE OpenMP::OpenMP_CXX)
endif()
if(CMAKE_CUDA_COMPILER)
  target_link_libraries(fwmc PRIVATE fwmc-cuda)
endif()

# Tests: each test file is a separate executable
enable_testing()
include_directories(tests)

foreach(test_file tests/test_core.cc tests/test_bridge.cc tests/test_parametric.cc tests/test_tissue.cc)
  get_filename_component(test_name ${test_file} NAME_WE)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}")
    add_executable(${test_name} ${test_file})
    if(NOT MSVC)
      target_link_libraries(${test_name} PRIVATE pthread)
    endif()
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CMAKE_CUDA_COMPILER)
      target_link_libraries(${test_name} PRIVATE fwmc-cuda)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
  endif()
endforeach()

# Benchmarks
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_core.cc")
  add_executable(fwmc-bench tests/bench_core.cc)
  if(NOT MSVC)
    target_link_libraries(fwmc-bench PRIVATE pthread)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(fwmc-bench PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# Viewer (optional, requires OpenGL and git submodules)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/viewer/vendor/glfw/CMakeLists.txt")
  add_subdirectory(viewer)
  message(STATUS "Viewer enabled (OpenGL)")
else()
  message(STATUS "Viewer disabled (run: git submodule update --init --recursive)")
endif()
