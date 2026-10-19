// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <algorithm>
#include <stdexcept>

#include "TensorSharpMlxView.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"

// Keep the input row-contiguous, as MLX's as_strided does, while avoiding its
// flatten step: one int-sized dimension cannot describe a >2^31-element video.
// The unchanged upstream AsStrided primitive uses size_t offsets and 64-bit
// strides and shares a contiguous input's buffer regardless of its rank.
extern "C" int tensorsharp_mlx_as_strided(
    mlx_array* result,
    const mlx_array input,
    const int* shape,
    size_t shape_count,
    const int64_t* strides,
    size_t strides_count,
    size_t offset,
    const mlx_stream stream) {
  try {
    if (shape_count != strides_count) {
      throw std::invalid_argument("Shape and strides must have the same length.");
    }
    mlx::core::Shape output_shape(shape, shape + shape_count);
    if (std::any_of(output_shape.begin(), output_shape.end(), [](int d) { return d < 0; })) {
      throw std::invalid_argument("Negative dimensions are not supported.");
    }
    mlx_array_set_(*result, mlx::core::tensorsharp_as_strided(
        mlx_array_get_(input),
        std::move(output_shape),
        mlx::core::Strides(strides, strides + strides_count),
        offset,
        mlx_stream_get_(stream)));
    return 0;
  } catch (const std::exception& error) {
    mlx_error(error.what());
    return 1;
  }
}
