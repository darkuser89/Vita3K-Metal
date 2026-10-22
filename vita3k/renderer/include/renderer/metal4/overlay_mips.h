// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace renderer::metal {
// Area average in integer coordinates, including every texel at odd edges.
// Used only when an overlay image changes; it never updates an in-flight image.
inline std::vector<uint8_t> overlay_next_mip(std::span<const uint8_t> source,
    uint32_t width, uint32_t height, uint32_t channels) {
    if (!width || !height || width > 16384 || height > 16384
        || (channels != 1 && channels != 4)
        || source.size() != size_t(width) * height * channels)
        throw std::invalid_argument("Metal: invalid overlay mip input");
    const uint32_t next_width = std::max(1u, width / 2);
    const uint32_t next_height = std::max(1u, height / 2);
    std::vector<uint8_t> result(size_t(next_width) * next_height * channels);
    const uint64_t weight = uint64_t(width) * height;
    for (uint32_t y = 0; y < next_height; ++y) {
        const uint32_t top = y * height, bottom = (y + 1) * height;
        for (uint32_t x = 0; x < next_width; ++x) {
            const uint32_t left = x * width, right = (x + 1) * width;
            uint64_t sum[4]{};
            for (uint32_t sy = top / next_height; sy * next_height < bottom; ++sy) {
                const uint32_t wy = std::min(bottom, (sy + 1) * next_height) - std::max(top, sy * next_height);
                for (uint32_t sx = left / next_width; sx * next_width < right; ++sx) {
                    const uint32_t wx = std::min(right, (sx + 1) * next_width) - std::max(left, sx * next_width);
                    for (uint32_t channel = 0; channel < channels; ++channel)
                        sum[channel] += uint64_t(source[(size_t(sy) * width + sx) * channels + channel]) * wx * wy;
                }
            }
            for (uint32_t channel = 0; channel < channels; ++channel)
                result[(size_t(y) * next_width + x) * channels + channel] = uint8_t((sum[channel] + weight / 2) / weight);
        }
    }
    return result;
}
}
