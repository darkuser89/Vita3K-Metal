// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// cpu_regressions.py prepends actual descriptor initializers and format mapping.
int main() {
    EmuEnvState env;
    size_t cases = 0;
    for (const auto format : {SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,
             SCE_GXM_COLOR_FORMAT_F16_G, SCE_GXM_COLOR_FORMAT_U5U6U5_RGB,
             SCE_GXM_COLOR_FORMAT_F16F16F16F16_RGBA})
    for (const auto type : {SCE_GXM_COLOR_SURFACE_LINEAR,
             SCE_GXM_COLOR_SURFACE_SWIZZLED, SCE_GXM_COLOR_SURFACE_TILED})
    for (const auto scale : {SCE_GXM_COLOR_SURFACE_SCALE_NONE,
             SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE})
    for (const auto registers : {SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT,
             SCE_GXM_OUTPUT_REGISTER_SIZE_64BIT})
    for (const uint32_t stride : {64u, 80u})
    for (const Address address : {0u, 0x1000u}) {
        SceGxmColorSurface surface;
        std::memset(&surface, 0xa5, sizeof(surface));
        assert(export_sceGxmColorSurfaceInit_metal3(env, 0, "test", &surface,
            format, type, scale, registers, 64, 32, stride, Ptr<void>(address)) == 0);
        assert(surface.data.address() == address && surface.disabled == 0);
        assert(surface.width == 64 && surface.height == 32 && surface.strideInPixels == stride);
        assert(surface.colorFormat == format && surface.surfaceType == type);
        assert(surface.downscale == (scale == SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE));
        const auto bits = renderer::metal::color_surface_bits(surface);
        assert(uint32_t(bits.outputRegisterSize) == uint32_t(registers));
        assert(!uint32_t(bits.clip_enabled) && !uint32_t(bits.dither_enabled));
        assert(surface.backgroundTex.data_addr == address / 4);
        SceGxmTextureFormat texture_format;
        assert(gxm::convert_color_format_to_texture_format_metal3(format, texture_format));
        assert(surface.backgroundTex.base_format == ((uint32_t(texture_format) >> 24) & 31));
        const auto texture_type = surface.backgroundTex.texture_type();
        assert(texture_type == (type == SCE_GXM_COLOR_SURFACE_SWIZZLED ? SCE_GXM_TEXTURE_SWIZZLED
            : type == SCE_GXM_COLOR_SURFACE_TILED ? SCE_GXM_TEXTURE_TILED
            : stride == 64 ? SCE_GXM_TEXTURE_LINEAR : SCE_GXM_TEXTURE_LINEAR_STRIDED));
        ++cases;
    }
    const auto invalid = [&](uint32_t width, uint32_t height, uint32_t stride,
                             Address address, SceGxmColorSurfaceType type, uint32_t expected) {
        SceGxmColorSurface surface;
        std::memset(&surface, 0xa5, sizeof(surface));
        const auto before = surface;
        assert(uint32_t(export_sceGxmColorSurfaceInit_metal3(env, 0, "test", &surface,
            SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR, type, SCE_GXM_COLOR_SURFACE_SCALE_NONE,
            SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, width, height, stride, Ptr<void>(address))) == expected);
        assert(std::memcmp(&before, &surface, sizeof(surface)) == 0);
        ++cases;
    };
    for (const Address address : {0u, 0x1000u}) {
        invalid(0, 32, 64, address, SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_ERROR_INVALID_VALUE);
        invalid(4097, 32, 4104, address, SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_ERROR_INVALID_VALUE);
        invalid(64, 0, 64, address, SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_ERROR_INVALID_VALUE);
        invalid(64, 4097, 64, address, SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_ERROR_INVALID_VALUE);
        invalid(64, 32, 63, address, SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_ERROR_INVALID_ALIGNMENT);
        invalid(64, 32, 62, address, SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_ERROR_INVALID_VALUE);
        invalid(63, 32, 64, address, SCE_GXM_COLOR_SURFACE_SWIZZLED, SCE_GXM_ERROR_INVALID_ALIGNMENT);
        invalid(64, 31, 64, address, SCE_GXM_COLOR_SURFACE_SWIZZLED, SCE_GXM_ERROR_INVALID_ALIGNMENT);
        assert(uint32_t(export_sceGxmColorSurfaceInit_metal3(env, 0, "test", nullptr,
            SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR, SCE_GXM_COLOR_SURFACE_LINEAR,
            SCE_GXM_COLOR_SURFACE_SCALE_NONE, SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT,
            64, 32, 64, Ptr<void>(address))) == SCE_GXM_ERROR_INVALID_POINTER);
        ++cases;
    }
    invalid(64, 32, 64, 0x1001, SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_ERROR_INVALID_VALUE);
    assert(cases == 211);
    std::cout << "PASS color surface init: " << cases
              << " actual initializer/format-mapping/background-descriptor cases; null/non-null data, layouts, Metal format/register bits and invalid inputs; ASAN/UBSAN, no GPU/app\n";
}
