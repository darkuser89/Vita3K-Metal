// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace renderer::metal {
// Keep every guest mip addressable. Downscaling the base below 2^(levels-1)
// would make that chain illegal in Metal, even though its last images are 1x1.
// Grow the storage just enough in that case; per-level sampling grids remain
// separate in TextureMipInfo, including independently rendered native levels.
inline std::array<uint32_t, 2> snapshot_texture_extent(uint64_t width, uint64_t height,
    uint64_t levels, float scale) {
    constexpr uint32_t max_dimension = 16384;
    if (!width || !height || width > max_dimension || height > max_dimension
        || !levels || levels > std::bit_width(std::max(width, height))
        || !std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("Metal: invalid texture snapshot extent or mip chain");
    const uint32_t minimum = 1u << (levels - 1);
    const double factor = std::max(double(scale), double(minimum) / std::max(width, height));
    const double scaled_width = std::max(1.0, std::floor(width * factor));
    const double scaled_height = std::max(1.0, std::floor(height * factor));
    if (scaled_width > max_dimension || scaled_height > max_dimension)
        throw std::length_error("Metal: texture snapshot exceeds native dimensions");
    std::array<uint32_t, 2> result{uint32_t(scaled_width), uint32_t(scaled_height)};
    // Protect the dominant edge from floating-point division rounding down.
    auto &dominant = result[width >= height ? 0 : 1];
    dominant = std::max(dominant, minimum);
    return result;
}
} // namespace renderer::metal
