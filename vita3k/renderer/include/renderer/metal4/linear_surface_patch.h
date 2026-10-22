// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace renderer::metal {
struct LinearSurfacePatch {
    size_t x = 0, y = 0, width = 0, height = 0;
};

// Bounding native pixels touched by guest byte intervals. Padding does not
// own pixels; a partial component owns its whole pixel for read/modify/write.
// Only the two edge rows of each interval need inspection. Interior rows
// include the full visible width, irrespective of the guest padding stride.
template <typename Range>
LinearSurfacePatch linear_surface_patch(size_t width, size_t height,
    size_t pixel_bytes, size_t guest_stride, std::span<const Range> ranges) {
    constexpr auto maximum = std::numeric_limits<size_t>::max();
    if (!width || !height || !pixel_bytes || width > maximum / pixel_bytes
        || guest_stride < width * pixel_bytes || height > maximum / guest_stride)
        throw std::invalid_argument("Metal: invalid linear surface patch layout");
    const size_t visible = width * pixel_bytes, size = height * guest_stride;
    size_t left = width, top = height, right = 0, bottom = 0;
    const auto row = [&](size_t y, size_t begin, size_t end) {
        end = std::min(end, visible);
        if (begin >= end) return;
        left = std::min(left, begin / pixel_bytes);
        right = std::max(right, (end - 1) / pixel_bytes + 1);
        top = std::min(top, y);
        bottom = std::max(bottom, y + 1);
    };
    for (const auto &range : ranges) {
        if (!range.size || range.offset >= size || range.size > size - range.offset)
            throw std::invalid_argument("Metal: invalid linear surface patch range");
        const size_t end = range.offset + range.size;
        const size_t first = range.offset / guest_stride, last = (end - 1) / guest_stride;
        if (first == last) {
            row(first, range.offset % guest_stride, end - first * guest_stride);
        } else {
            row(first, range.offset % guest_stride, visible);
            row(last, 0, end - last * guest_stride);
            if (last - first > 1) {
                left = 0; right = width;
                top = std::min(top, first + 1);
                bottom = std::max(bottom, last);
            }
        }
    }
    if (left >= right || top >= bottom) return {};
    return {left, top, right - left, bottom - top};
}

enum class SurfacePatchLayout { Linear, Tiled, Morton };

// Invert byte intervals into a guest bounding rectangle. A covered Morton
// subtree is one rectangle; tiled intervals only inspect touched 32x32 tiles.
// Partial bytes own their entire pixel, while allocation padding owns none.
template <typename Range>
LinearSurfacePatch guest_surface_patch(size_t width, size_t height,
    size_t pixel_bytes, size_t stride, SurfacePatchLayout layout, std::span<const Range> ranges) {
    if (!width || !height || width > 16384 || height > 16384 || !pixel_bytes || pixel_bytes > 16
        || stride < width || stride > std::numeric_limits<size_t>::max() / (16384 * 16)
        || (layout != SurfacePatchLayout::Linear && layout != SurfacePatchLayout::Tiled && layout != SurfacePatchLayout::Morton)
        || (layout == SurfacePatchLayout::Tiled && stride % 32)
        || (layout == SurfacePatchLayout::Morton
            && (stride != width || !std::has_single_bit(width) || !std::has_single_bit(height))))
        throw std::invalid_argument("Metal: invalid surface patch layout");
    if (layout == SurfacePatchLayout::Linear)
        return linear_surface_patch(width, height, pixel_bytes, stride * pixel_bytes, ranges);
    const size_t rows = layout == SurfacePatchLayout::Tiled ? (height + 31) & ~size_t(31) : height;
    const size_t size = rows * stride * pixel_bytes;
    size_t left = width, top = height, right = 0, bottom = 0;
    const auto include = [&](size_t x, size_t y, size_t w, size_t h) {
        if (!w || !h) return;
        left = std::min(left, x); top = std::min(top, y);
        right = std::max(right, x + w); bottom = std::max(bottom, y + h);
    };
    for (const auto &range : ranges) {
        if (!range.size || range.offset >= size || range.size > size - range.offset)
            throw std::invalid_argument("Metal: invalid surface patch range");
        const size_t first = range.offset / pixel_bytes;
        const size_t end = (range.offset + range.size - 1) / pixel_bytes + 1;
        if (layout == SurfacePatchLayout::Tiled) {
            for (size_t tile = first / 1024; tile <= (end - 1) / 1024; ++tile) {
                const size_t x = tile % (stride / 32) * 32, y = tile / (stride / 32) * 32;
                if (x >= width || y >= height) continue;
                const size_t begin = std::max(first, tile * 1024) - tile * 1024;
                const size_t finish = std::min(end, (tile + 1) * 1024) - tile * 1024;
                struct PixelRange { size_t offset, size; } part{begin, finish - begin};
                // The final tile band includes rows beyond the visible image.
                const size_t h = std::min(size_t(32), height - y);
                if (begin >= h * 32) continue;
                part.size = std::min(part.size, h * 32 - begin);
                const auto patch = linear_surface_patch(std::min(size_t(32), width - x), h,
                    1, 32, std::span<const PixelRange>(&part, 1));
                include(x + patch.x, y + patch.y, patch.width, patch.height);
            }
        } else {
            const auto visit = [&](auto &&self, size_t x, size_t y, size_t w, size_t h, size_t offset) -> void {
                const size_t count = w * h;
                if (offset >= end || offset + count <= first) return;
                if (offset >= first && offset + count <= end) {
                    include(x, y, w, h);
                    return;
                }
                // Matches the existing encoder: squares split X before Y.
                if (w >= h) {
                    self(self, x, y, w / 2, h, offset);
                    self(self, x + w / 2, y, w / 2, h, offset + count / 2);
                } else {
                    self(self, x, y, w, h / 2, offset);
                    self(self, x, y + h / 2, w, h / 2, offset + count / 2);
                }
            };
            visit(visit, 0, 0, width, height, 0);
        }
    }
    if (left >= right || top >= bottom) return {};
    return {left, top, right - left, bottom - top};
}
}
