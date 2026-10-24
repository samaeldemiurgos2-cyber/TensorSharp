// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstddef>

namespace tsg
{
    // ggml frees the old gallocr buffer before growing it. Avoid addition here:
    // the free and reusable sizes can overflow when supplied by a faulty device.
    inline bool moe_stream_memory_fits(std::size_t required, std::size_t free,
        std::size_t reusable)
    {
        return required <= reusable || required - reusable <= free;
    }
}
