// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace renderer::metal {
// Invert depth_patch_coordinate/seed_axis for one axis. extent=0 denotes
// single-sample floor(n*guest/native); MSAA uses 1 or 2 samples on this axis.
// All sample lanes are included, including the clamped final odd guest group.
inline std::pair<uint32_t, uint32_t> depth_patch_bounds(uint32_t native_size,
    uint32_t guest_size, uint32_t extent, uint32_t origin, uint32_t length) {
    if (!native_size || !guest_size || extent > 2)
        throw std::invalid_argument("Metal: invalid depth patch extent");
    if (!length || origin >= guest_size) return {0, 0};
    const auto end = std::min(uint64_t(origin) + length, uint64_t(guest_size));
    if (!origin && end == guest_size) return {0, native_size};
    const auto coordinate = [&](uint32_t native, bool last_sample) -> uint64_t {
        if (!extent) return uint64_t(native) * guest_size / native_size;
        const uint64_t groups = (uint64_t(guest_size) + extent - 1) / extent;
        const auto group = native_size < groups ? uint64_t(native) * groups / native_size
            : ((uint64_t(native) + 1) * groups - 1) / native_size;
        return std::min(group * extent + (last_sample ? extent - 1 : 0), uint64_t(guest_size) - 1);
    };
    const auto lower_bound = [&](uint64_t guest, bool last_sample) {
        uint32_t first = 0, last = native_size;
        while (first < last) {
            const auto middle = first + (last - first) / 2;
            if (coordinate(middle, last_sample) < guest) first = middle + 1;
            else last = middle;
        }
        return first;
    };
    return {lower_bound(origin, true), lower_bound(end, false)};
}
}
