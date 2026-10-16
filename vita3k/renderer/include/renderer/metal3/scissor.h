// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <renderer/metal3/apple.h>
#include <renderer/types.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace renderer::metal {
// Disjoint scissors preserve blend/occlusion results when excluding an interior rectangle.
inline std::vector<MTLScissorRect> scissors(const GxmRecordState &r, uint32_t w, uint32_t h, float scale) {
    if (r.region_clip_mode == SCE_GXM_REGION_CLIP_ALL) return {};
    if (r.region_clip_mode == SCE_GXM_REGION_CLIP_NONE) return {{0, 0, w, h}};
    const auto clamp_edge = [](double v, uint32_t limit) { return static_cast<uint32_t>(std::clamp(v, 0.0, double(limit))); };
    const double scaled_x = std::trunc(double(r.region_clip_min.x) * scale);
    const double scaled_y = std::trunc(double(r.region_clip_min.y) * scale);
    const auto x0 = clamp_edge(scaled_x, w);
    const auto y0 = clamp_edge(scaled_y, h);
    const bool outside = r.region_clip_mode == SCE_GXM_REGION_CLIP_OUTSIDE;
    // Vulkan scales the tile-aligned scissor origin and extent separately,
    // truncating each. INSIDE must exclude exactly the area OUTSIDE includes.
    const auto x1 = std::max(x0, clamp_edge(scaled_x
        + std::floor((double(r.region_clip_max.x) - r.region_clip_min.x + 1) * scale), w));
    const auto y1 = std::max(y0, clamp_edge(scaled_y
        + std::floor((double(r.region_clip_max.y) - r.region_clip_min.y + 1) * scale), h));
    std::vector<MTLScissorRect> result;
    auto add = [&](uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        if (width && height) result.push_back({x, y, width, height});
    };
    if (outside) add(x0, y0, x1 - x0, y1 - y0);
    else {
        add(0, 0, w, y0); add(0, y1, w, h - y1);
        add(0, y0, x0, y1 - y0); add(x1, y0, w - x1, y1 - y0);
    }
    return result;
}
} // namespace renderer::metal
