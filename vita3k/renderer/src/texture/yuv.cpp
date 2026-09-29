// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include <renderer/functions.h>
#include <renderer/texture_cache.h>
#include <utility>
#include <vector>

extern "C" {
#include <libswscale/swscale.h>
}

namespace renderer::texture {

static bool set_yuv_profile(SwsContext *context, SceGxmYuvProfile profile) {
    const int colorspace = (uint32_t(profile) & 1) ? SWS_CS_ITU709 : SWS_CS_ITU601;
    const int source_range = uint32_t(profile) >= SCE_GXM_YUV_PROFILE_BT601_FULL_RANGE ? 1 : 0;
    const int *coefficients = sws_getCoefficients(colorspace);
    return sws_setColorspaceDetails(context, coefficients, source_range, coefficients, 1,
        0, 1 << 16, 1 << 16) >= 0;
}

static SwsContext *get_sws_context(YUVConversionCache &cache, size_t width, size_t height, bool is_p3, bool is_nv21, SceGxmYuvProfile profile) {
    bool recreate = false;
    auto *context = static_cast<SwsContext *>(cache.sws_context);
    if (cache.width != width || cache.height != height || cache.is_p3 != is_p3 || cache.is_nv21 != is_nv21 || cache.profile != profile) {
        recreate = true;
        cache.width = width;
        cache.height = height;
        cache.is_p3 = is_p3;
        cache.is_nv21 = is_nv21;
        cache.profile = profile;
    } else if (context == nullptr) {
        recreate = true;
    }

    if (recreate) {
        if (context != nullptr) {
            sws_freeContext(context);
            context = nullptr;
        }
        const AVPixelFormat format = is_p3 ? AV_PIX_FMT_YUV420P : (is_nv21 ? AV_PIX_FMT_NV21 : AV_PIX_FMT_NV12);
        context = sws_getContext(width, height, format, width, height, AV_PIX_FMT_RGB0,
            0, nullptr, nullptr, nullptr);
        if (context && !set_yuv_profile(context, profile)) {
            sws_freeContext(context);
            context = nullptr;
        }
        cache.sws_context = context;
    }
    return context;
}

void yuv420_texture_to_rgb(YUVConversionCache &cache, uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height, uint32_t layout_width, uint32_t layout_height, bool is_p3, uint32_t swizzle, SceGxmYuvProfile profile) {
    // SDL's Vita backend maps NV12 to YVU420P2 and NV21 to YUV420P2.
    const bool is_nv21 = !is_p3 && !(swizzle & 1);
    SwsContext *context = get_sws_context(cache, width, height, is_p3, is_nv21, profile);
    assert(context);

    const uint8_t *slices[] = {
        src, // Y Slice
        src + layout_width * layout_height, // First chroma plane or interleaved pair
        src + layout_width * layout_height + layout_width * layout_height / 4, // V Slice (for P3)
    };

    if (is_p3 && (swizzle & 1)) std::swap(slices[1], slices[2]);

    int strides[] = {
        static_cast<int>(width),
        static_cast<int>(width / 2),
        static_cast<int>(width / 2),
    };
    if (!is_p3) {
        // src only have two slices
        strides[1] = static_cast<int>(width);
        strides[2] = 0;
    }

    uint8_t *dst_slices[] = {
        dst,
    };

    const int dst_strides[] = {
        static_cast<int>(width * 4),
    };

    int error = sws_scale(context, slices, strides, 0, height, dst_slices, dst_strides);
    assert(error == height);
}

bool yuv422_texture_to_rgb(YUVConversionCache &cache, uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height, uint32_t swizzle, SceGxmYuvProfile profile) {
    if (!dst || !src || !width || !height || (width & 1) || swizzle >= 8) return false;
    const uint32_t byte_mode = swizzle & 3;
    // FFmpeg accepts YUYV, YVYU and UYVY directly. Only VYUY needs a
    // reordered temporary buffer before conversion.
    const uint8_t *input_bytes = src;
    std::vector<uint8_t> yuyv;
    if (byte_mode == 3) {
        yuyv.resize(size_t(width) * height * 2);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; x += 2) {
                const size_t offset = (size_t(y) * width + x) * 2;
                const uint8_t *pair = src + offset;
                uint8_t *normalized = yuyv.data() + offset;
                normalized[0] = pair[1]; normalized[1] = pair[2];
                normalized[2] = pair[3]; normalized[3] = pair[0];
            }
        input_bytes = yuyv.data();
    }
    if (cache.width_422 != width || cache.height_422 != height || cache.profile_422 != profile) {
        for (void *&cached : cache.sws_context_422) {
            if (cached) sws_freeContext(static_cast<SwsContext *>(cached));
            cached = nullptr;
        }
        cache.width_422 = width;
        cache.height_422 = height;
        cache.profile_422 = profile;
    }
    auto *context = static_cast<SwsContext *>(cache.sws_context_422[byte_mode]);
    if (!context) {
        const AVPixelFormat input_format = byte_mode == 1 ? AV_PIX_FMT_YVYU422
            : byte_mode == 2 ? AV_PIX_FMT_UYVY422 : AV_PIX_FMT_YUYV422;
        context = sws_getContext(width, height, input_format,
            width, height, AV_PIX_FMT_RGB0, 0, nullptr, nullptr, nullptr);
        if (context && !set_yuv_profile(context, profile)) {
            sws_freeContext(context);
            context = nullptr;
        }
        cache.sws_context_422[byte_mode] = context;
    }
    if (!context) return false;
    const uint8_t *input[] = {input_bytes};
    const int input_stride[] = {int(width * 2)};
    uint8_t *output[] = {dst};
    const int output_stride[] = {int(width * 4)};
    const bool success = sws_scale(context, input, input_stride, 0, height, output, output_stride) == int(height);
    return success;
}

} // namespace renderer::texture

renderer::TextureCache::~TextureCache() {
    if (auto *context = static_cast<SwsContext *>(yuv_conversion_cache.sws_context)) {
        sws_freeContext(context);
        yuv_conversion_cache.sws_context = nullptr;
    }
    for (void *&cached : yuv_conversion_cache.sws_context_422) {
        if (cached) sws_freeContext(static_cast<SwsContext *>(cached));
        cached = nullptr;
    }
}
