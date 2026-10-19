cmake_minimum_required(VERSION 3.20)

# The pinned MLX kernels support Metal 4.0, but SDK 27 defaults to Metal 4.1,
# whose address-space rules reject their thread-qualified member references.
# MLX owns its shader command lines and exposes the deployment target as the
# supported way to select the Metal language version. Targeting macOS 26.2
# selects Metal 4.0 and retains the NAX kernels; older SDKs keep their defaults.
# Set this before project() so C/C++ and Metal agree on the deployment target.
# Respect an explicit CMake or standard macOS deployment-target override.
if(CMAKE_HOST_APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET
   AND "$ENV{MACOSX_DEPLOYMENT_TARGET}" STREQUAL "")
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-version
    OUTPUT_VARIABLE TENSORSHARP_MACOS_SDK_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TENSORSHARP_MACOS_SDK_RESULT)
  if(TENSORSHARP_MACOS_SDK_RESULT EQUAL 0
     AND TENSORSHARP_MACOS_SDK_VERSION VERSION_GREATER_EQUAL "27.0")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.2" CACHE STRING
        "Minimum macOS version supported by the MLX native libraries" FORCE)
  endif()
endif()

project(TensorSharpMlxNative LANGUAGES CXX C)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dist" CACHE PATH "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_RPATH "@loader_path")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

include(FetchContent)

file(READ "${CMAKE_CURRENT_LIST_DIR}/MLX_C_VERSION" MLX_C_GIT_TAG)
string(STRIP "${MLX_C_GIT_TAG}" MLX_C_GIT_TAG)

FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG ${MLX_C_GIT_TAG}
)

FetchContent_MakeAvailable(mlx-c)

# TensorSharp-owned integration: MLX's public as_strided first flattens its
# input into an int-sized axis. Its AsStrided primitive supports size_t-sized
# contiguous backing arrays directly, which large video activations require.
target_sources(mlxc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/TensorSharpMlx.cpp")
target_sources(mlx PRIVATE "${CMAKE_CURRENT_LIST_DIR}/TensorSharpMlxView.cpp")

install(
  TARGETS mlxc
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .
)

if(TARGET mlx)
  install(
    TARGETS mlx
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
  )
endif()

if(TARGET jaccl)
  install(
    TARGETS jaccl
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
  )
endif()
