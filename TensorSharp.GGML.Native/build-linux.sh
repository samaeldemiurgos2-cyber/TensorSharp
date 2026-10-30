#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ENABLE_CUDA="${TENSORSHARP_GGML_NATIVE_ENABLE_CUDA:-}"
ENABLE_VULKAN="${TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN:-}"
BUILD_TESTS="${TENSORSHARP_GGML_NATIVE_BUILD_TESTS:-OFF}"
CUDA_ARCHITECTURES="${TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES:-}"
BUILD_PARALLEL_LEVEL="${TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL:-${CMAKE_BUILD_PARALLEL_LEVEL:-}}"
EXTRA_CMAKE_ARGS=()
USER_SET_CMAKE_CUDA_ARCHITECTURES=OFF

normalize_bool() {
    local value="${1:-}"
    case "${value}" in
        ON|on|On|TRUE|true|True|YES|yes|Yes|1)
            echo "ON"
            ;;
        OFF|off|Off|FALSE|false|False|NO|no|No|0)
            echo "OFF"
            ;;
        *)
            echo ""
            ;;
    esac
}

has_cuda_toolkit() {
    if command -v nvcc >/dev/null 2>&1; then
        return 0
    fi

    local cuda_home="${CUDA_HOME:-${CUDA_PATH:-}}"
    if [[ -n "${cuda_home}" && -x "${cuda_home}/bin/nvcc" ]]; then
        return 0
    fi

    return 1
}

# The nvcc CMake will use: -DCMAKE_CUDA_COMPILER=..., else CUDACXX, else nvcc on
# PATH, else CUDA_HOME/CUDA_PATH. Prints nothing (and fails) when there is none.
resolve_nvcc() {
    local arg
    for arg in ${EXTRA_CMAKE_ARGS[@]+"${EXTRA_CMAKE_ARGS[@]}"}; do
        case "${arg}" in
            -DCMAKE_CUDA_COMPILER=*|-DCMAKE_CUDA_COMPILER:*=*)
                echo "${arg#*=}"
                return 0
                ;;
        esac
    done
    if [[ -n "${CUDACXX:-}" ]]; then
        echo "${CUDACXX}"
        return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi
    local cuda_home="${CUDA_HOME:-${CUDA_PATH:-}}"
    if [[ -n "${cuda_home}" && -x "${cuda_home}/bin/nvcc" ]]; then
        echo "${cuda_home}/bin/nvcc"
        return 0
    fi
    return 1
}

# "13.4" from `nvcc --version` ("Cuda compilation tools, release 13.4, V13.4.92"),
# or nothing when the compiler cannot be run or prints no release line.
nvcc_release() {
    "$1" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+)\.([0-9]+),.*/\1.\2/p' | head -n 1
}

# Prints the system Vulkan loader library path (preferring the dev symlink
# libvulkan.so over the runtime soname libvulkan.so.1 — CMake accepts a full
# path in Vulkan_LIBRARY either way). Fails when no loader is installed, which
# is the "this machine does not support Vulkan" signal for auto-detection.
# Is a Vulkan BUILD toolchain already present? A loader (libvulkan.so) only says
# the machine can RUN Vulkan -- every NVIDIA driver install ships one -- so it is
# the wrong signal for auto-enabling the backend: on a CUDA-only box it sent the
# build off to download and compile shaderc/SPIRV-Headers before it ever reached
# ggml-cuda. Auto-enable only when the pieces needed to compile the shaders are
# already installed; `--vulkan` still opts in explicitly and will provision them.
have_vulkan_build_toolchain() {
    if [[ -n "${VULKAN_SDK:-}" && -f "${VULKAN_SDK}/include/vulkan/vulkan.h" && -x "${VULKAN_SDK}/bin/glslc" ]]; then
        return 0
    fi
    command -v glslc >/dev/null 2>&1 || return 1
    local dir
    for dir in /usr/include /usr/local/include; do
        [[ -f "${dir}/vulkan/vulkan.h" ]] && return 0
    done
    return 1
}

# Is a Vulkan DRIVER actually registered on this machine? A loader alone proves
# nothing (every NVIDIA install ships one); an ICD manifest means some driver
# claims to implement Vulkan, which is the honest signal for "building the
# backend here is worth the seconds it costs".
have_vulkan_runtime_driver() {
    local dir
    for dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/local/share/vulkan/icd.d; do
        [[ -d "${dir}" ]] || continue
        compgen -G "${dir}/*.json" >/dev/null 2>&1 && return 0
    done
    [[ -n "${VK_ICD_FILENAMES:-}" || -n "${VK_DRIVER_FILES:-}" ]] && return 0
    return 1
}

# The GLVND gap: with an ICD registered and the loader present, instance
# creation still fails with ErrorIncompatibleDriver when libEGL is absent,
# because the NVIDIA Vulkan ICD dlopen()s it. The build succeeds and the
# backend is dead at runtime, which is the worst way to find out - so check
# here and say exactly what is missing.
vulkan_runtime_libs_present() {
    local dir name
    for dir in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib /usr/local/lib; do
        for name in libEGL.so.1 libEGL.so; do
            [[ -e "${dir}/${name}" ]] && return 0
        done
    done
    return 1
}

warn_if_vulkan_runtime_incomplete() {
    have_vulkan_runtime_driver || return 0
    vulkan_runtime_libs_present && return 0
    echo "warning: a Vulkan ICD is registered but libEGL was not found. The NVIDIA Vulkan" >&2
    echo "         driver dlopen()s it, so ggml-vulkan will build fine and then fail at" >&2
    echo "         startup with 'vk::createInstance: ErrorIncompatibleDriver'." >&2
    echo "         Install the GLVND runtime, e.g.: apt-get install -y libegl1 libgl1 libglvnd0" >&2
}

find_vulkan_loader_library() {
    local candidates=()
    if command -v ldconfig >/dev/null 2>&1; then
        while IFS= read -r line; do
            candidates+=("${line}")
        done < <(ldconfig -p 2>/dev/null | awk '$1 == "libvulkan.so" || $1 == "libvulkan.so.1" { print $NF }')
    fi
    local dir name
    for dir in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib /usr/local/lib; do
        for name in libvulkan.so libvulkan.so.1; do
            if [[ -e "${dir}/${name}" ]]; then
                candidates+=("${dir}/${name}")
            fi
        done
    done

    local fallback=""
    local candidate
    for candidate in ${candidates[@]+"${candidates[@]}"}; do
        if [[ "${candidate}" == *libvulkan.so ]]; then
            echo "${candidate}"
            return 0
        fi
        if [[ -z "${fallback}" ]]; then
            fallback="${candidate}"
        fi
    done
    if [[ -n "${fallback}" ]]; then
        echo "${fallback}"
        return 0
    fi
    return 1
}

# Ensures the Vulkan build toolchain (headers, glslc, SPIRV-Headers) exists,
# downloading the portable pieces via eng/fetch-vulkan-toolchain.sh when the
# system does not already provide them, and fills VULKAN_CMAKE_ARGS with
# explicit FindVulkan hints. Returns non-zero when the toolchain cannot be
# provisioned (no loader, no network, ...).
VULKAN_CMAKE_ARGS=()
prepare_vulkan_toolchain() {
    VULKAN_CMAKE_ARGS=()

    # Full LunarG SDK: CMake's FindVulkan discovers everything via VULKAN_SDK.
    if [[ -n "${VULKAN_SDK:-}" && -f "${VULKAN_SDK}/include/vulkan/vulkan.h" && -x "${VULKAN_SDK}/bin/glslc" ]]; then
        return 0
    fi

    local loader
    if ! loader="$(find_vulkan_loader_library)"; then
        echo "warning: no Vulkan loader (libvulkan.so / libvulkan.so.1) found on this system." >&2
        return 1
    fi

    if ! bash "${SCRIPT_DIR}/../eng/fetch-vulkan-toolchain.sh"; then
        return 1
    fi

    local toolchain_dir
    toolchain_dir="$(cd "${SCRIPT_DIR}/.." && pwd)/ExternalProjects/vulkan-toolchain"

    local glslc_path
    if command -v glslc >/dev/null 2>&1; then
        glslc_path="$(command -v glslc)"
    elif [[ -x "${toolchain_dir}/shaderc-linux/bin/glslc" ]]; then
        glslc_path="${toolchain_dir}/shaderc-linux/bin/glslc"
    else
        echo "warning: glslc not found after provisioning the Vulkan toolchain." >&2
        return 1
    fi

    # Point CMake at whichever copy actually exists. The provisioner now prefers
    # distro packages and stages nothing when the system is already complete, so
    # hinting unconditionally at the staged directory handed CMake a
    # non-existent include path ("Imported target Vulkan::Vulkan includes
    # non-existent path .../vulkan-toolchain/Vulkan-Headers/include").
    local include_dir=""
    local d
    for d in "${toolchain_dir}/Vulkan-Headers/include" /usr/include /usr/local/include; do
        if [[ -f "${d}/vulkan/vulkan.h" ]]; then
            include_dir="${d}"
            break
        fi
    done
    if [[ -z "${include_dir}" ]]; then
        echo "warning: Vulkan headers not found after provisioning the toolchain." >&2
        return 1
    fi

    local spirv_dir=""
    for d in "${toolchain_dir}/spirv-headers-install/share/cmake/SPIRV-Headers"              /usr/share/cmake/SPIRV-Headers /usr/lib/cmake/SPIRV-Headers              /usr/local/share/cmake/SPIRV-Headers /usr/local/lib/cmake/SPIRV-Headers              /usr/lib/x86_64-linux-gnu/cmake/SPIRV-Headers              /usr/lib/aarch64-linux-gnu/cmake/SPIRV-Headers; do
        if [[ -f "${d}/SPIRV-HeadersConfig.cmake" ]]; then
            spirv_dir="${d}"
            break
        fi
    done
    if [[ -z "${spirv_dir}" ]]; then
        echo "warning: SPIRV-Headers CMake package not found after provisioning." >&2
        return 1
    fi

    VULKAN_CMAKE_ARGS=(
        "-DVulkan_INCLUDE_DIR=${include_dir}"
        "-DVulkan_LIBRARY=${loader}"
        "-DVulkan_GLSLC_EXECUTABLE=${glslc_path}"
        "-DSPIRV-Headers_DIR=${spirv_dir}"
    )
    return 0
}

read_cached_backend_setting() {
    local cache_variable="$1"
    local cache_file="${BUILD_DIR}/CMakeCache.txt"
    if [[ ! -f "${cache_file}" ]]; then
        echo ""
        return
    fi

    local cached
    cached="$(awk -F= -v var="^${cache_variable}:BOOL=" '$0 ~ var {print $2; exit}' "${cache_file}")"
    normalize_bool "${cached}"
}

detect_local_cuda_architectures() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
        awk '
            BEGIN {
                first = 1
            }
            {
                gsub(/[[:space:]]/, "", $0)
                if ($0 ~ /^[0-9]+\.[0-9]+$/) {
                    split($0, cap, ".")
                    # CUDA 13 compiles nothing below compute capability 7.5 (Turing).
                    if (cap[1] * 10 + cap[2] < 75) {
                        printf("warning: skipping GPU of compute capability %s: CUDA 13 supports 7.5 (Turing) and newer only.\n", $0) > "/dev/stderr"
                        next
                    }
                    arch = cap[1] cap[2] "-real"
                    if (!(arch in seen)) {
                        seen[arch] = 1
                        if (!first) {
                            printf(";")
                        }
                        printf("%s", arch)
                        first = 0
                    }
                }
            }
            END {
                if (first) {
                    exit 1
                }
            }
        '
}

detect_default_build_parallel_level() {
    local cpu_count=1
    if command -v nproc >/dev/null 2>&1; then
        cpu_count="$(nproc)"
    fi

    # Use all CPUs, bounded only by RAM. nvcc is memory-hungry, so allow ~3 GB per
    # parallel job (matches llama.cpp/ggml's own heuristic). The previous hard cap
    # of 4 jobs throttled the CUDA compile to a fraction of available cores even on
    # large machines with plenty of RAM; the memory bound below already prevents
    # OOM. Override with TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL when needed.
    local jobs="${cpu_count}"
    if [[ -r /proc/meminfo ]]; then
        local mem_kb
        mem_kb="$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)"
        if [[ "${mem_kb}" =~ ^[0-9]+$ ]]; then
            local memory_limited_jobs=$(( mem_kb / (3 * 1024 * 1024) ))
            if (( memory_limited_jobs < 1 )); then
                memory_limited_jobs=1
            fi
            if (( memory_limited_jobs < jobs )); then
                jobs="${memory_limited_jobs}"
            fi
        fi
    fi

    echo "${jobs}"
}

while (($# > 0)); do
    case "$1" in
        --cuda)
            ENABLE_CUDA=ON
            ;;
        --no-cuda)
            ENABLE_CUDA=OFF
            ;;
        --vulkan)
            ENABLE_VULKAN=ON
            ;;
        --no-vulkan)
            ENABLE_VULKAN=OFF
            ;;
        --tests)
            BUILD_TESTS=ON
            ;;
        --cuda-arch=*)
            CUDA_ARCHITECTURES="${1#*=}"
            ;;
        --cuda-arch)
            shift
            if (($# == 0)); then
                echo "Missing value for --cuda-arch" >&2
                exit 1
            fi
            CUDA_ARCHITECTURES="$1"
            ;;
        -DCMAKE_CUDA_ARCHITECTURES=*)
            USER_SET_CMAKE_CUDA_ARCHITECTURES=ON
            EXTRA_CMAKE_ARGS+=("$1")
            ;;
        *)
            EXTRA_CMAKE_ARGS+=("$1")
            ;;
    esac
    shift
done

ENABLE_CUDA="$(normalize_bool "${ENABLE_CUDA}")"
if [[ -z "${ENABLE_CUDA}" ]]; then
    ENABLE_CUDA="$(read_cached_backend_setting TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)"
fi
if [[ -z "${ENABLE_CUDA}" ]] && has_cuda_toolkit; then
    ENABLE_CUDA="ON"
fi
if [[ -z "${ENABLE_CUDA}" ]]; then
    ENABLE_CUDA="OFF"
fi

# TensorSharp builds only with CUDA 13 or newer, whether CUDA was requested or
# auto-detected: a CUDA 12 nvcc is an error, never a silent CPU-only build.
NVCC_PATH=""
if [[ "${ENABLE_CUDA}" == "ON" ]] && NVCC_PATH="$(resolve_nvcc)"; then
    # A bare name (CUDACXX=nvcc) resolves against PATH, as CMake resolves it.
    if [[ "${NVCC_PATH}" != */* ]]; then
        NVCC_PATH="$(command -v "${NVCC_PATH}" || echo "${NVCC_PATH}")"
    fi
    NVCC_RELEASE="$(nvcc_release "${NVCC_PATH}" || true)"
    if [[ -z "${NVCC_RELEASE}" ]]; then
        echo "error: CUDA 13+ required, but '${NVCC_PATH} --version' reports no 'release X.Y,' line." >&2
        echo "       Install a CUDA 13 toolkit, point CUDACXX at its nvcc, or pass --no-cuda for a CPU-only build." >&2
        exit 1
    fi
    if (( ${NVCC_RELEASE%%.*} < 13 )); then
        echo "error: CUDA 13+ required (found ${NVCC_RELEASE} at ${NVCC_PATH}); CUDA 12 is not supported." >&2
        echo "       Install a CUDA 13 toolkit and put it first on PATH (or point CUDACXX at its nvcc)," >&2
        echo "       or pass --no-cuda for a CPU-only build." >&2
        exit 1
    fi
    echo "Using nvcc ${NVCC_RELEASE} (${NVCC_PATH})."
    # Hand CMake exactly the compiler that passed the check above.
    if [[ -z "${CUDACXX:-}" ]]; then
        export CUDACXX="${NVCC_PATH}"
    fi

    # CMake caches the full compiler path and never swaps compilers inside a tree,
    # so a tree configured against another toolkit (CUDA 12.8, or 13.2 before a
    # move to 13.4) would keep building with it. Start such a tree over.
    #
    # Comparing paths alone misses a toolkit swapped behind an unchanged path:
    # update-alternatives moving /usr/local/cuda from 12.8 to 13.2 leaves the cached
    # and current realpaths equal. CMake identifies a compiler only once per tree
    # (CMakeFiles/<cmake version>/CMakeCUDACompiler.cmake, read back on every
    # reconfigure), so it would go on reporting 12.8 and the CUDA 13 gate in
    # CMakeLists.txt would reject a working 13.2 nvcc. Compare that recorded toolkit
    # version with the one the nvcc reports now as well.
    if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        stale_reason=""
        cached_nvcc="$(awk -F= '/^CMAKE_CUDA_COMPILER:/ { print $2; exit }' "${BUILD_DIR}/CMakeCache.txt")"
        if [[ -n "${cached_nvcc}" ]]; then
            cached_real="$(readlink -f "${cached_nvcc}" 2>/dev/null || echo "${cached_nvcc}")"
            resolved_real="$(readlink -f "${NVCC_PATH}" 2>/dev/null || echo "${NVCC_PATH}")"
            if [[ "${cached_real}" != "${resolved_real}" ]]; then
                stale_reason="CUDA compiler changed ('${cached_nvcc}' -> '${NVCC_PATH}')"
            fi
        fi
        if [[ -z "${stale_reason}" ]]; then
            for identified in "${BUILD_DIR}"/CMakeFiles/*/CMakeCUDACompiler.cmake; do
                [[ -f "${identified}" ]] || continue
                identified_release="$(awk -F'"' '/^set\(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION / { split($2, v, "."); if (v[1] != "" && v[2] != "") print v[1] "." v[2]; exit }' "${identified}")"
                if [[ -n "${identified_release}" && "${identified_release}" != "${NVCC_RELEASE}" ]]; then
                    stale_reason="the build tree was configured with CUDA ${identified_release}, but ${NVCC_PATH} is now CUDA ${NVCC_RELEASE}"
                    break
                fi
            done
        fi
        if [[ -n "${stale_reason}" ]]; then
            echo "${stale_reason}; removing ${BUILD_DIR} to reconfigure from scratch."
            rm -rf "${BUILD_DIR}"
        fi
    fi
fi

# Vulkan: an explicit choice (--vulkan/--no-vulkan or
# TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN) wins and sticks via the CMake cache;
# TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT records that the cached value was
# explicit, so an auto-detected default never becomes sticky. Otherwise the
# backend is auto-enabled when the machine supports Vulkan (a loader is
# installed) and the build toolchain (headers, glslc, SPIRV-Headers) is
# downloaded automatically by eng/fetch-vulkan-toolchain.sh when missing. An
# auto-enable whose toolchain cannot be provisioned (e.g. no network) degrades
# to a warning and builds without Vulkan; an explicit --vulkan makes it fatal.
VULKAN_EXPLICIT=OFF
ENABLE_VULKAN="$(normalize_bool "${ENABLE_VULKAN}")"
if [[ -n "${ENABLE_VULKAN}" ]]; then
    VULKAN_EXPLICIT=ON
elif [[ "$(read_cached_backend_setting TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT)" == "ON" ]]; then
    ENABLE_VULKAN="$(read_cached_backend_setting TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)"
    if [[ -n "${ENABLE_VULKAN}" ]]; then
        VULKAN_EXPLICIT=ON
    fi
fi
if [[ -z "${ENABLE_VULKAN}" ]]; then
    # Auto-enable when the build toolchain is already here, OR when a Vulkan
    # driver is actually registered and we can provision the toolchain quickly.
    # eng/fetch-vulkan-toolchain.sh now tries the distro packages first, so the
    # second case costs seconds rather than a shaderc source build - which is
    # what used to make auto-enabling on driver-presence a bad trade.
    if have_vulkan_build_toolchain; then
        ENABLE_VULKAN=ON
    elif have_vulkan_runtime_driver; then
        echo "A Vulkan driver is registered on this machine but its build toolchain is not"
        echo "installed; provisioning it (set TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF to skip)."
        ENABLE_VULKAN=ON
    else
        ENABLE_VULKAN=OFF
        if find_vulkan_loader_library >/dev/null; then
            echo "note: a Vulkan loader is present but no build toolchain (glslc + vulkan headers);" >&2
            echo "      building without ggml-vulkan. Pass --vulkan to provision the toolchain and include it." >&2
        fi
    fi
fi
if [[ "${ENABLE_VULKAN}" == "ON" ]] && ! prepare_vulkan_toolchain; then
    if [[ "${VULKAN_EXPLICIT}" == "ON" ]]; then
        echo "error: the ggml-vulkan backend was requested explicitly but its build toolchain could not be provisioned." >&2
        echo "       Install the Vulkan SDK (e.g. apt install libvulkan-dev glslc spirv-headers) or set VULKAN_SDK." >&2
        exit 1
    fi
    echo "warning: this machine supports Vulkan but the ggml-vulkan build toolchain could not be provisioned;" >&2
    echo "         building without the ggml-vulkan backend. Pass --vulkan to make this an error." >&2
    ENABLE_VULKAN=OFF
    VULKAN_DEGRADED=ON
fi

if [[ "${ENABLE_CUDA}" == "ON" && -z "${CUDA_ARCHITECTURES}" && "${USER_SET_CMAKE_CUDA_ARCHITECTURES}" != "ON" ]]; then
    if detected_cuda_architectures="$(detect_local_cuda_architectures)"; then
        CUDA_ARCHITECTURES="${detected_cuda_architectures}"
    fi
fi

CUDA_ARCH_SUMMARY="n/a"
if [[ "${ENABLE_CUDA}" == "ON" ]]; then
    if [[ "${USER_SET_CMAKE_CUDA_ARCHITECTURES}" == "ON" ]]; then
        CUDA_ARCH_SUMMARY="custom via CMAKE_CUDA_ARCHITECTURES"
    elif [[ -n "${CUDA_ARCHITECTURES}" ]]; then
        CUDA_ARCH_SUMMARY="${CUDA_ARCHITECTURES}"
    else
        CUDA_ARCH_SUMMARY="native"
    fi
fi

BUILD_PARALLEL_ARGS=()
if [[ -z "${BUILD_PARALLEL_LEVEL}" ]]; then
    BUILD_PARALLEL_LEVEL="$(detect_default_build_parallel_level)"
fi
if [[ ! "${BUILD_PARALLEL_LEVEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid build parallel level: ${BUILD_PARALLEL_LEVEL}" >&2
    exit 1
fi
BUILD_PARALLEL_ARGS=(--parallel "${BUILD_PARALLEL_LEVEL}")

if [[ "${ENABLE_VULKAN}" == "ON" ]]; then warn_if_vulkan_runtime_incomplete; fi

echo "Configuring TensorSharp.GGML.Native (CUDA=${ENABLE_CUDA}, CUDA_ARCHITECTURES=${CUDA_ARCH_SUMMARY}, VULKAN=${ENABLE_VULKAN}, TESTS=${BUILD_TESTS}, PARALLEL=${BUILD_PARALLEL_LEVEL})"

# Ensure the ggml sources are present (cloned from ggml-org/ggml at build time).
bash "${SCRIPT_DIR}/../eng/fetch-ggml.sh"

# cuDNN for the VAE convolutions. Optional and best-effort: fetch-cudnn.sh always
# exits 0, and CMake keeps ggml's im2col+GEMM lowering when the headers are absent.
# TENSORSHARP_CUDNN=OFF skips it entirely.
if [ "${ENABLE_CUDA}" = "ON" ]; then
    bash "${SCRIPT_DIR}/../eng/fetch-cudnn.sh"
fi

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA="${ENABLE_CUDA}"
    -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN="${ENABLE_VULKAN}"
    -DTENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT="${VULKAN_EXPLICIT}"
    -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS="${BUILD_TESTS}"
    ${VULKAN_CMAKE_ARGS[@]+"${VULKAN_CMAKE_ARGS[@]}"}
)

if [[ "${ENABLE_CUDA}" == "ON" && -n "${CUDA_ARCHITECTURES}" && "${USER_SET_CMAKE_CUDA_ARCHITECTURES}" != "ON" ]]; then
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    "${CMAKE_ARGS[@]}" \
    "${EXTRA_CMAKE_ARGS[@]}"

if [[ "${BUILD_TESTS}" == "ON" ]]; then
    cmake --build "${BUILD_DIR}" --config Release "${BUILD_PARALLEL_ARGS[@]}"
else
    cmake --build "${BUILD_DIR}" --config Release "${BUILD_PARALLEL_ARGS[@]}" --target GgmlOps
fi

# Record whether an auto-enabled Vulkan backend had to be dropped. The build
# itself succeeded either way, so MSBuild would otherwise freshen its
# up-to-date stamp and skip this script on every later build - leaving a
# silently Vulkan-less libGgmlOps.so even after the cause (no network, no
# loader) is fixed. TensorSharp.Backends.GGML.csproj invalidates that stamp
# while this marker exists, so provisioning is retried until it succeeds.
VULKAN_DEGRADED_MARKER="${BUILD_DIR}/vulkan-degraded.marker"
if [[ "${VULKAN_DEGRADED:-OFF}" == "ON" ]]; then
    cat > "${VULKAN_DEGRADED_MARKER}" <<'EOF'
The ggml-vulkan backend was auto-enabled but its build toolchain could not be provisioned,
so this build produced libGgmlOps.so without Vulkan support. Delete this file to stop retrying,
or pass --no-vulkan to disable the backend explicitly.
EOF
else
    rm -f "${VULKAN_DEGRADED_MARKER}"
fi
