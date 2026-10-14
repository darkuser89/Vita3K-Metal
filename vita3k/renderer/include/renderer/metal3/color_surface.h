// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/gxm_types.h>
#include <cstddef>
#include <cstring>
#include <type_traits>
namespace renderer::metal {
template <typename Surface> struct ColorSurfaceField {
    Surface &surface;
    size_t offset;
    uint32_t shift, mask;
    operator uint32_t() const {
        uint32_t word;
        std::memcpy(&word, reinterpret_cast<const uint8_t *>(&surface) + offset, sizeof(word));
        return (word >> shift) & mask;
    }
    ColorSurfaceField &operator=(uint32_t value) requires (!std::is_const_v<Surface>) {
        uint32_t word;
        auto *data = reinterpret_cast<uint8_t *>(&surface) + offset;
        std::memcpy(&word, data, sizeof(word));
        word = (word & ~(mask << shift)) | ((value & mask) << shift);
        std::memcpy(data, &word, sizeof(word));
        return *this;
    }
};
template <typename Surface> struct ColorSurfaceBits {
    ColorSurfaceField<Surface> dither_enabled, clip_enabled, clip_x_min, clip_y_min;
    ColorSurfaceField<Surface> outputRegisterSize, clip_x_max, clip_y_max;
};
template <typename Surface> ColorSurfaceBits<Surface> color_surface_bits(Surface &surface) {
    constexpr size_t output = offsetof(SceGxmColorSurface, outputRegisterSize);
    return {{surface, 0, 4, 1}, {surface, 0, 5, 1}, {surface, 0, 6, 4095}, {surface, 0, 18, 4095},
        {surface, output, 0, 1}, {surface, output, 1, 4095}, {surface, output, 13, 4095}};
}
}
