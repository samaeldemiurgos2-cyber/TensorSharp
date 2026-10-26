// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_cuda_launch_policy.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const bool unset = !std::strcmp(argv[1], "unset") || !std::strcmp(argv[1], "unaffected");
    const bool affected = std::strcmp(argv[1], "unaffected") != 0;
    if (!unset && std::strcmp(argv[1], "0") && std::strcmp(argv[1], "1")) return 2;
#ifdef _WIN32
    if (_putenv_s("GGML_CUDA_PDL", unset ? "" : argv[1])) return 1;
#else
    if (unset ? unsetenv("GGML_CUDA_PDL") : setenv("GGML_CUDA_PDL", argv[1], 1)) return 1;
#endif
    const auto result = tsg::apply_cuda_pdl_default(affected);
    const char* value = std::getenv("GGML_CUDA_PDL");
    if (unset && affected) {
        if (result != tsg::CudaPdlDefaultResult::Disabled || !value || std::strcmp(value, "0")) return 1;
        if (tsg::apply_cuda_pdl_default(true) != tsg::CudaPdlDefaultResult::Unchanged) return 1;
    } else {
        if (result != tsg::CudaPdlDefaultResult::Unchanged) return 1;
        if (unset ? value != nullptr : (!value || std::strcmp(value, argv[1]))) return 1;
    }
    std::printf("PASS CUDA launch policy %s\n", argv[1]);
    return 0;
}
