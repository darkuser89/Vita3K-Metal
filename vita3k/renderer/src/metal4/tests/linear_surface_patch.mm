// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Exercise the production linear writer with CPU texture doubles only.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <renderer/metal4/linear_surface_patch.h>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace renderer::metal;
struct SurfaceMemoryRange { size_t offset, size; };
struct SceGxmColorSurface { size_t strideInPixels; };
using Map = std::array<unsigned, 4>;
constexpr Map identity{0, 1, 2, 3};
// FULL_WRITE_HOOK

@interface PatchTexture : NSObject {
@public
    std::vector<uint8_t> pixels;
    size_t width, height, pixel_bytes, read_bytes, write_bytes;
    MTLRegion read_region, write_region;
}
- (void)getBytes:(void *)data bytesPerRow:(NSUInteger)stride fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level;
- (void)replaceRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level withBytes:(const void *)data bytesPerRow:(NSUInteger)stride;
@end
@implementation PatchTexture
- (void)getBytes:(void *)data bytesPerRow:(NSUInteger)stride fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level {
    assert(!level && !region.origin.z && region.size.depth == 1);
    assert(region.origin.x + region.size.width <= width && region.origin.y + region.size.height <= height);
    assert(stride >= region.size.width * pixel_bytes);
    for (size_t y = 0; y < region.size.height; ++y)
        std::memcpy(static_cast<uint8_t *>(data) + y * stride,
            pixels.data() + ((region.origin.y + y) * width + region.origin.x) * pixel_bytes,
            region.size.width * pixel_bytes);
    read_bytes += region.size.width * region.size.height * pixel_bytes;
    read_region = region;
}
- (void)replaceRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level withBytes:(const void *)data bytesPerRow:(NSUInteger)stride {
    assert(!level && !region.origin.z && region.size.depth == 1);
    assert(region.origin.x + region.size.width <= width && region.origin.y + region.size.height <= height);
    assert(stride >= region.size.width * pixel_bytes);
    for (size_t y = 0; y < region.size.height; ++y)
        std::memcpy(pixels.data() + ((region.origin.y + y) * width + region.origin.x) * pixel_bytes,
            static_cast<const uint8_t *>(data) + y * stride, region.size.width * pixel_bytes);
    write_bytes += region.size.width * region.size.height * pixel_bytes;
    write_region = region;
}
@end

static bool apply(PatchTexture *texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges,
    Map mapping, size_t component_bytes, unsigned components) {
    const size_t size = source.size(), pixel_bytes = component_bytes * components;
    const size_t native_width = texture->width, native_height = texture->height;
    const size_t native_stride = native_width * pixel_bytes;
    std::array<size_t, 16> native_byte_indices{};
    for (size_t byte = 0; byte < pixel_bytes; ++byte)
        native_byte_indices[byte] = mapping[byte / component_bytes] * component_bytes + byte % component_bytes;
    const bool direct_linear = true;
    // LINEAR_WRITER_HOOK
    return false;
}

static size_t cases = 0;
static void check(size_t width, size_t height, size_t padding, unsigned components,
    size_t component_bytes, Map mapping, const std::vector<SurfaceMemoryRange> &ranges) {
    @autoreleasepool {
        const auto pixel_bytes = component_bytes * components;
        const auto stride = (width + padding) * pixel_bytes;
        auto texture = [PatchTexture new];
        texture->width = width; texture->height = height; texture->pixel_bytes = pixel_bytes;
        texture->pixels.resize(width * height * pixel_bytes);
        std::vector<uint8_t> source(stride * height);
        for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 59 + (i >> 8));
        for (size_t i = 0; i < texture->pixels.size(); ++i) texture->pixels[i] = uint8_t(i * 71 + 13);
        auto expected = texture->pixels;
        size_t left = width, top = height, right = 0, bottom = 0;
        // Independent per-byte oracle, deliberately not based on patch bounds
        // or row copies. Check bit patterns, including float NaNs, unchanged.
        for (const auto &range : ranges)
            for (size_t offset = range.offset; offset < range.offset + range.size; ++offset) {
                const auto y = offset / stride, x = offset % stride / pixel_bytes;
                if (x >= width) continue;
                const auto byte = offset % pixel_bytes;
                expected[(y * width + x) * pixel_bytes + mapping[byte / component_bytes] * component_bytes + byte % component_bytes]
                    = source[offset];
                left = std::min(left, x); right = std::max(right, x + 1);
                top = std::min(top, y); bottom = std::max(bottom, y + 1);
            }
        assert(apply(texture, {width + padding}, source, ranges, mapping, component_bytes, components));
        assert(texture->pixels == expected);
        const bool empty = left >= right || top >= bottom;
        const auto bytes = empty ? 0 : (right - left) * (bottom - top) * pixel_bytes;
        assert(texture->write_bytes == bytes);
        assert(texture->read_bytes == (full_surface_write(ranges, source.size()) ? 0 : bytes));
        if (!empty) {
            assert(texture->write_region.origin.x == left && texture->write_region.origin.y == top);
            assert(texture->write_region.size.width == right - left && texture->write_region.size.height == bottom - top);
        }
        ++cases;
    }
}

int main() {
    std::mt19937 random(0x50415443);
    for (unsigned components : {1u, 2u, 4u}) {
        Map mapping = identity;
        do {
            for (size_t component_bytes : {1, 2, 4})
                for (size_t width : {1, 3, 17})
                    for (size_t height : {1, 2, 5})
                        for (size_t padding : {0, 1, 7}) {
                            const size_t pixel_bytes = components * component_bytes;
                            const size_t stride = (width + padding) * pixel_bytes, size = height * stride;
                            const auto verify = [&](std::vector<SurfaceMemoryRange> ranges) {
                                check(width, height, padding, components, component_bytes, mapping, ranges);
                            };
                            verify({}); verify({{0, size}});
                            if (size > 1) verify({{0, 1}, {1, size - 1}});
                            // Every byte position, including all component and
                            // row/padding boundaries, is updated individually.
                            for (size_t offset = 0; offset < size; ++offset) verify({{offset, 1}});
                            for (unsigned trial = 0; trial < 20; ++trial) {
                                const size_t first = random() % size;
                                verify({{first, 1 + random() % (size - first)}});
                                std::vector<SurfaceMemoryRange> ranges;
                                for (size_t at = random() % 8; at < size;) {
                                    const size_t count = std::min(size - at, size_t(1 + random() % 23));
                                    ranges.push_back({at, count});
                                    at += count + random() % 13;
                                }
                                verify(ranges);
                            }
                        }
        } while (std::next_permutation(mapping.begin(), mapping.begin() + components));
    }
    // A single byte patch of a 1024x1024 RGBA8 image transfers one pixel in
    // each direction; padding-only changes transfer nothing.
    check(1024, 1024, 8, 4, 1, identity, {{(1032 * 512 + 721) * 4 + 2, 1}});
    check(1024, 1024, 8, 4, 1, identity, {{1024 * 4, 32}});
    bool threw = false;
    try { linear_surface_patch(0, 1, 4, 4, std::span<const SurfaceMemoryRange>{}); }
    catch (const std::invalid_argument &) { threw = true; }
    assert(threw);
    const std::array<SurfaceMemoryRange, 1> invalid{{{0, 5}}};
    threw = false;
    try { linear_surface_patch(1, 1, 4, 4, std::span<const SurfaceMemoryRange>(invalid)); }
    catch (const std::invalid_argument &) { threw = true; }
    assert(threw);
    std::cout << "PASS linear surface patch: " << cases
        << " production-writer cases vs independent byte oracle; all channel permutations/partial components/padding/full writes; 1024x1024 single-byte edit reads/uploads 4+4 bytes; CPU doubles only, ASAN/UBSAN\n";
}
