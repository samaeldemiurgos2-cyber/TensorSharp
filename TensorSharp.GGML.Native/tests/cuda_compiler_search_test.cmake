# cmake -P tests/cuda_compiler_search_test.cmake
#
# Checks that tsg_find_cuda_compiler (tsg_cuda_compiler_search.cmake) finds the
# nvcc that CMake's own compiler search would pick. The CUDA 13 pre-check in
# CMakeLists.txt probes that compiler, so a mismatch fails a fresh configure on an
# nvcc CMake would never use. Only empty fake nvcc files are involved: nothing is
# run, and no toolkit is needed.
cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/../tsg_cuda_compiler_search.cmake")

if (DEFINED TSG_TEST_WORK_DIR)
    set(_root "${TSG_TEST_WORK_DIR}")
else()
    string(RANDOM LENGTH 12 _suffix)
    set(_root "${CMAKE_CURRENT_BINARY_DIR}/tsg-cuda-compiler-search-${_suffix}")
endif()
file(REMOVE_RECURSE "${_root}")

if (CMAKE_HOST_WIN32)
    set(_exe ".exe")
    set(_sep ";")
else()
    set(_exe "")
    set(_sep ":")
endif()

function(fake_nvcc dir)
    file(MAKE_DIRECTORY "${dir}")
    file(WRITE "${dir}/nvcc${_exe}" "")
    if (NOT CMAKE_HOST_WIN32)
        file(CHMOD "${dir}/nvcc" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
    endif()
endfunction()

fake_nvcc("${_root}/cuda12/bin")
fake_nvcc("${_root}/cuda13/bin")
fake_nvcc("${_root}/hint")
file(MAKE_DIRECTORY "${_root}/empty")

# expect(<name> <generator> <PATH> <CUDA_PATH> <CUDA_HOME> <expected path or "">)
function(expect name generator path cuda_path cuda_home expected)
    set(ENV{PATH} "${path}")
    set(ENV{CUDA_PATH} "${cuda_path}")
    set(ENV{CUDA_HOME} "${cuda_home}")
    tsg_find_cuda_compiler(_found "${generator}")
    if (_found)
        file(REAL_PATH "${_found}" _found)
    endif()
    if (expected)
        file(REAL_PATH "${expected}" expected)
    endif()
    if (_found STREQUAL expected)
        message(STATUS "PASS ${name}")
    else()
        message(SEND_ERROR "FAIL ${name}: expected '${expected}', found '${_found}'")
    endif()
endfunction()

set(_nvcc12 "${_root}/cuda12/bin/nvcc${_exe}")
set(_nvcc13 "${_root}/cuda13/bin/nvcc${_exe}")

# The reported failure: CUDA_PATH names 12.x, PATH puts 13.x first. CMake uses the
# PATH one, so the pre-check must too.
expect("PATH beats CUDA_PATH (Ninja)" "Ninja" "${_root}/cuda13/bin" "${_root}/cuda12" "" "${_nvcc13}")
expect("PATH beats CUDA_PATH (Unix Makefiles)" "Unix Makefiles" "${_root}/cuda13/bin" "${_root}/cuda12" "" "${_nvcc13}")
expect("PATH order is kept" "Ninja" "${_root}/cuda12/bin${_sep}${_root}/cuda13/bin" "" "" "${_nvcc12}")
# CUDA_PATH is the last fallback, used only when nothing else has an nvcc.
expect("CUDA_PATH is the last fallback" "Ninja" "${_root}/empty" "${_root}/cuda13" "" "${_nvcc13}")
# CMake's compiler search never reads CUDA_HOME.
expect("CUDA_HOME is not consulted" "Ninja" "${_root}/empty" "" "${_root}/cuda12" "")
# CMAKE_CUDA_COMPILER_HINTS (like the C/C++ compiler directories) beat PATH.
set(CMAKE_CUDA_COMPILER_HINTS "${_root}/hint")
expect("compiler hints beat PATH" "Ninja" "${_root}/cuda13/bin" "${_root}/cuda12" "" "${_root}/hint/nvcc${_exe}")
unset(CMAKE_CUDA_COMPILER_HINTS)
# The Visual Studio generator takes nvcc from its CUDA toolset, which no search predicts.
expect("Visual Studio is not predicted" "Visual Studio 17 2022" "${_root}/cuda13/bin" "${_root}/cuda12" "" "")

file(REMOVE_RECURSE "${_root}")
