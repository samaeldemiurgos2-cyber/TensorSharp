// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "mlx/mlx.h"

namespace mlx::core {
MLX_API array tensorsharp_as_strided(
    const array& input, Shape shape, Strides strides, size_t offset, Stream stream);
}
