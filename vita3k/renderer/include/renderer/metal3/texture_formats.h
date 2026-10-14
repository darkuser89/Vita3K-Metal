// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <gxm/metal3/types.h>
#include <gxm/metal3/formats.h>
#include <renderer/texture_cache.h>
#include <renderer/types.h>
namespace renderer::metal {
using TextureCacheKey = std::array<uint32_t, 5>;
struct TextureCacheMetadata {
    TextureCacheKey key{};
    uint64_t additional_hash = 0;
    uint32_t protected_size = 0;
    bool import_swap_rb = false;
};
struct YUVConversionCache : renderer::YUVConversionCache {
    std::array<void *, 4> sws_context_422{};
    size_t width_422 = 0, height_422 = 0;
    bool is_nv21 = false;
    uint32_t profile = SCE_GXM_YUV_PROFILE_BT601_STANDARD;
    uint32_t profile_422 = SCE_GXM_YUV_PROFILE_BT601_STANDARD;
};
void release_yuv_cache(YUVConversionCache &);
}
namespace renderer::texture {
void convert_s5s5u6_to_f16f16f16f16(void *dest, const void *data, const uint32_t width, const uint32_t height, const SceGxmTextureFormat format);
void convert_x8s8s8u8_to_f16f16f16f16(void *dest, const void *data, const uint32_t width, const uint32_t height, const SceGxmTextureFormat format);
void yuv420_texture_to_rgb_metal3(metal::YUVConversionCache &, uint8_t *, const uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t, SceGxmYuvProfile);
bool yuv422_texture_to_rgb_metal3(metal::YUVConversionCache &, uint8_t *, const uint8_t *, uint32_t, uint32_t, uint32_t, SceGxmYuvProfile);
bool convert_base_texture_format_to_base_color_format_metal3(SceGxmTextureBaseFormat, SceGxmColorBaseFormat &);
SceGxmTextureBaseFormat get_matching_decompressed_format_metal3(SceGxmTextureBaseFormat);
uint32_t decompress_compressed_texture_metal3(SceGxmTextureBaseFormat, void *, const void *, uint32_t, uint32_t);
}

namespace gxm {
bool convert_color_format_to_texture_format_metal3(SceGxmColorFormat, SceGxmTextureFormat &);
}

namespace renderer::texture {

}
