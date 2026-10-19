// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <memory>

#include "TensorSharpMlxView.h"
#include "mlx/primitives.h"

namespace mlx::core {
array tensorsharp_as_strided(
    const array& input, Shape shape, Strides strides, size_t offset, Stream stream) {
  // AsStrided is internal to libmlx. Compile this owned bridge in that target
  // so its hidden primitive symbols stay internal to the same shared library.
  auto contiguous_input = contiguous(input, false, stream);
  auto copied_shape = shape;
  auto dtype = contiguous_input.dtype();
  return array(
      std::move(copied_shape),
      dtype,
      std::make_shared<AsStrided>(stream, std::move(shape), std::move(strides), offset),
      {std::move(contiguous_input)});
}
}
