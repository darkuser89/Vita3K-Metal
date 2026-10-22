// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <Metal/Metal.h>
#include <array>
#include <bit>
#include <cstdint>

namespace renderer::metal {
// Keep float32 precision. Only omit channels whose reads are guaranteed +0
// for every source view in the assembled mip/face chain. Missing alpha is
// ONE, not zero, so it must be stored too (out-of-bounds reads must stay zero).
inline uint32_t snapshot_live_components(MTLPixelFormat format, MTLTextureSwizzleChannels swizzle) {
    uint32_t components = 0xf;
    switch (format) {
    case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm:
    case MTLPixelFormatR16Float: case MTLPixelFormatR32Float:
        components = 0x9; break; // R, implicit A=1.
    case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRG16Unorm: case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Float: case MTLPixelFormatRG32Float:
        components = 0xb; break; // R, G, implicit A=1.
    case MTLPixelFormatA8Unorm:
        components = 0x8; break;
    default: break; // Includes depth/stencil: their absent channels are undefined.
    }
    const std::array selections{swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha};
    uint32_t mask = 0;
    for (uint32_t i = 0; i < selections.size(); ++i) {
        bool live = true;
        switch (selections[i]) {
        case MTLTextureSwizzleZero: live = false; break;
        case MTLTextureSwizzleOne: break;
        case MTLTextureSwizzleRed: live = components & 1; break;
        case MTLTextureSwizzleGreen: live = components & 2; break;
        case MTLTextureSwizzleBlue: live = components & 4; break;
        case MTLTextureSwizzleAlpha: live = components & 8; break;
        default: return 0xf;
        }
        if (live) mask |= uint32_t(1) << i;
    }
    return mask;
}

struct SnapshotChannels {
    MTLPixelFormat format = MTLPixelFormatRGBA32Float;
    std::array<uint32_t, 4> pack{0, 1, 2, 3};
    MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsDefault;
    bool packed() const { return format != MTLPixelFormatRGBA32Float; }
};

inline SnapshotChannels snapshot_channels(uint32_t live_components) {
    SnapshotChannels result;
    if (live_components & ~0xfu || std::popcount(live_components) > 2) return result;
    result.format = std::popcount(live_components) > 1 ? MTLPixelFormatRG32Float : MTLPixelFormatR32Float;
    result.pack.fill(0);
    std::array<MTLTextureSwizzle, 4> swizzle{};
    uint32_t stored = 0;
    for (uint32_t component = 0; component < 4; ++component) {
        swizzle[component] = MTLTextureSwizzleZero;
        if (live_components & (uint32_t(1) << component)) {
            result.pack[stored] = component;
            swizzle[component] = stored++ ? MTLTextureSwizzleGreen : MTLTextureSwizzleRed;
        }
    }
    result.swizzle = MTLTextureSwizzleChannelsMake(swizzle[0], swizzle[1], swizzle[2], swizzle[3]);
    return result;
}
}
