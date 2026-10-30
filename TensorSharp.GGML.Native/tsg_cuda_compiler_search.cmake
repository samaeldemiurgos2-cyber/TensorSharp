# tsg_find_cuda_compiler(<out-var> <generator>)
#
# Sets <out-var> to the nvcc that CMake's own CUDA compiler search will pick when
# nothing names one (no CMAKE_CUDA_COMPILER, no CUDACXX), or to "" when that search
# cannot be predicted or finds nothing. CMakeLists.txt runs its CUDA 13 pre-check
# against this compiler. Probing in a different order fails the configure on a
# CUDA 12 nvcc that CMake would never have used: before this helper, CUDA_PATH was
# searched ahead of PATH, so CUDA_PATH=v12.x failed a build whose PATH put v13.4
# first.
#
# The order follows Modules/CMakeDetermineCUDACompiler.cmake and the
# _cmake_find_compiler macro in Modules/CMakeDetermineCompiler.cmake (CMake 3.20
# through 4.1):
#   1. CMAKE_CUDA_COMPILER_HINTS and the directories holding the compilers of the
#      languages already enabled (C, CXX), searched on their own;
#   2. on Windows with a command-line generator (Ninja, MSYS or MinGW Makefiles),
#      PATH alone;
#   3. find_program's default search (CMAKE_PREFIX_PATH, PATH, system prefixes);
#   4. $ENV{CUDA_PATH}/bin, as the very last fallback.
# CMake's compiler search never reads CUDA_HOME, so this helper does not either.
# A Visual Studio generator takes nvcc from the VS CUDA toolset rather than from
# any search path, so the result there is always "".
function(tsg_find_cuda_compiler out_var generator)
    set(${out_var} "" PARENT_SCOPE)
    if (generator MATCHES "Visual Studio")
        return()
    endif()

    set(_hints ${CMAKE_CUDA_COMPILER_HINTS})
    get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    list(REMOVE_ITEM _languages CUDA)
    foreach (_lang IN LISTS _languages)
        if (CMAKE_${_lang}_COMPILER AND IS_ABSOLUTE "${CMAKE_${_lang}_COMPILER}")
            get_filename_component(_hint "${CMAKE_${_lang}_COMPILER}" DIRECTORY)
            if (IS_DIRECTORY "${_hint}")
                list(APPEND _hints "${_hint}")
            endif()
        endif()
    endforeach()

    # find_program() stops at the first call that finds something; every later call
    # is a no-op, which is how CMake chains the same steps.
    unset(TSG_FOUND_NVCC CACHE)
    if (_hints)
        list(REMOVE_DUPLICATES _hints)
        find_program(TSG_FOUND_NVCC NAMES nvcc PATHS ${_hints} NO_DEFAULT_PATH)
    endif()
    if (CMAKE_HOST_WIN32 AND generator MATCHES "Ninja|MSYS Makefiles|MinGW Makefiles")
        find_program(TSG_FOUND_NVCC NAMES nvcc NAMES_PER_DIR
                     NO_PACKAGE_ROOT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
    endif()
    find_program(TSG_FOUND_NVCC NAMES nvcc)
    if (NOT "$ENV{CUDA_PATH}" STREQUAL "")
        find_program(TSG_FOUND_NVCC NAMES nvcc NAMES_PER_DIR PATHS "$ENV{CUDA_PATH}/bin" NO_DEFAULT_PATH)
    endif()
    if (TSG_FOUND_NVCC)
        set(${out_var} "${TSG_FOUND_NVCC}" PARENT_SCOPE)
    endif()
    unset(TSG_FOUND_NVCC CACHE)   # search again on every configure (NO_CACHE needs CMake 3.21)
endfunction()
