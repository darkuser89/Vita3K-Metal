// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct Source {
    SceGxmColorFormat format;
    MTLPixelFormat native;
    unsigned count, bytes;
    std::array<unsigned, 4> memory;
};
struct Target {
    SceGxmTextureBaseFormat format;
    MTLPixelFormat native;
    unsigned bytes;
};
int main() {
    @autoreleasepool {
        try {
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            renderer::metal::SurfaceCaster caster(*device);
            const std::array<unsigned, 4> orders[] = {{0, 1, 2, 3}, {2, 1, 0, 3}, {3, 2, 1, 0}, {3, 0, 1, 2}};
            std::vector<Source> sources;
            for (unsigned order = 0; order < 4; ++order) {
                for (bool srgb : {false, true})
                    sources.push_back(
                        {static_cast<SceGxmColorFormat>(SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR | (order << 20)),
                         srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm, 4, 1, orders[order]});
                sources.push_back({static_cast<SceGxmColorFormat>(SCE_GXM_COLOR_FORMAT_S8S8S8S8_ABGR | (order << 20)),
                                   MTLPixelFormatRGBA8Snorm, 4, 1, orders[order]});
                sources.push_back(
                    {static_cast<SceGxmColorFormat>(SCE_GXM_COLOR_FORMAT_F16F16F16F16_ABGR | (order << 20)),
                     MTLPixelFormatRGBA16Float, 4, 2, orders[order]});
            }
            for (unsigned order = 0; order < 2; ++order) {
                const std::array<unsigned, 4> memory = order ? std::array<unsigned, 4>{1, 0, 2, 3} : orders[0];
                sources.push_back({static_cast<SceGxmColorFormat>(SCE_GXM_COLOR_FORMAT_F32F32_GR | (order << 20)),
                                   MTLPixelFormatRG32Float, 2, 4, memory});
                sources.push_back({static_cast<SceGxmColorFormat>(SCE_GXM_COLOR_FORMAT_U8U8_GR | (order << 20)),
                                   MTLPixelFormatRG8Unorm, 2, 1, memory});
            }
            sources.push_back({SCE_GXM_COLOR_FORMAT_F32_R, MTLPixelFormatR32Float, 1, 4, orders[0]});
            sources.push_back({SCE_GXM_COLOR_FORMAT_F16_R, MTLPixelFormatR16Float, 1, 2, orders[0]});
            sources.push_back({SCE_GXM_COLOR_FORMAT_U16_R, MTLPixelFormatR16Unorm, 1, 2, orders[0]});
            sources.push_back({SCE_GXM_COLOR_FORMAT_U8_R, MTLPixelFormatR8Unorm, 1, 1, orders[0]});
            sources.push_back({SCE_GXM_COLOR_FORMAT_U8_A, MTLPixelFormatR8Unorm, 1, 1, orders[0]});
            const Target targets[] = {
#define TARGET(gxm, mtl, bytes) {SCE_GXM_TEXTURE_BASE_FORMAT_##gxm, MTLPixelFormat##mtl, bytes}
                TARGET(U8, R8Unorm, 1),
                TARGET(S8, R8Snorm, 1),
                TARGET(U8U8, RG8Unorm, 2),
                TARGET(S8S8, RG8Snorm, 2),
                TARGET(U16, R16Unorm, 2),
                TARGET(S16, R16Snorm, 2),
                TARGET(F16, R16Float, 2),
                TARGET(U8U8U8U8, RGBA8Unorm, 4),
                TARGET(S8S8S8S8, RGBA8Snorm, 4),
                TARGET(U16U16, RG16Unorm, 4),
                TARGET(S16S16, RG16Snorm, 4),
                TARGET(F16F16, RG16Float, 4),
                TARGET(F32, R32Float, 4),
                TARGET(F32F32, RG32Float, 8),
                TARGET(U16U16U16U16, RGBA16Unorm, 8),
                TARGET(S16S16S16S16, RGBA16Snorm, 8),
                TARGET(F16F16F16F16, RGBA16Float, 8)
#undef TARGET
            };
            const uint32_t patterns[] = {0,          0x80000000, 0x7f800000, 0x7f800001, 0xffc12345,
                                         0x00000001, 0x7d017c01, 0xfe557fff, 0x3f123456, 0xabcdef01};
            constexpr unsigned width = 17, height = 19;
            unsigned casts = 0, compared = 0, views = 0;
            for (const auto &source : sources) {
                const unsigned pixel_bytes = source.count * source.bytes;
                std::vector<uint8_t> data(width * height * pixel_bytes);
                for (unsigned p = 0; p < width * height; ++p)
                    for (unsigned c = 0; c < source.count; ++c) {
                        // Mix all byte codes with explicit signed zero, infinities,
                        // subnormals and distinct F16/F32 NaN payloads.
                        uint32_t bits =
                            p >= 256 ? patterns[(p + c) % std::size(patterns)] : uint32_t(uint8_t(p * 37 + c * 71)) * 0x01010101u;
                        std::memcpy(data.data() + p * pixel_bytes + c * source.bytes, &bits, source.bytes);
                    }
                auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.native
                                                                               width:width
                                                                              height:height
                                                                           mipmapped:NO];
                desc.storageMode = MTLStorageModeShared;
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
                auto input = [device->native_device() newTextureWithDescriptor:desc];
                check(input != nil, "Source allocation failed");
                [input replaceRegion:MTLRegionMake2D(0, 0, width, height)
                         mipmapLevel:0
                           withBytes:data.data()
                         bytesPerRow:width * pixel_bytes];
                for (const auto &target : targets) {
                    if (target.bytes != pixel_bytes)
                        continue;
                    auto result = caster.surface_format_cast(input, source.format, target.format);
                    check(result && result.pixelFormat == target.native && result.width == width &&
                              result.height == height,
                          "Wrong cast format or native extent");
                    std::vector<uint8_t> actual(data.size());
                    [result getBytes:actual.data()
                         bytesPerRow:width * pixel_bytes
                          fromRegion:MTLRegionMake2D(0, 0, width, height)
                         mipmapLevel:0];
                    for (unsigned p = 0; p < width * height; ++p)
                        for (unsigned c = 0; c < source.count; ++c)
                            for (unsigned b = 0; b < source.bytes; ++b) {
                                const unsigned offset = p * pixel_bytes + c * source.bytes + b;
                                const uint8_t expected = data[p * pixel_bytes + source.memory[c] * source.bytes + b];
                                check(actual[offset] == expected,
                                      "Bit cast mismatch source=" + std::to_string(source.format) + " target=" +
                                          std::to_string(target.format) + " byte=" + std::to_string(offset));
                                ++compared;
                            }
                    bool identity = true;
                    for (unsigned c = 0; c < source.count; ++c)
                        identity &= source.memory[c] == c;
                    if (identity) {
                        check(result == input || result.parentTexture == input,
                              "Identity cast allocated instead of sharing source storage");
                        ++views;
                    }
                    ++casts;
                }
            }
            std::cout << "PASS surface format casts=" << casts << " exact_bytes=" << compared
                      << " shared_views=" << views
                      << " odd extents,1/2/4-byte components,swizzles,sRGB bytes,F16/F32 NaN payloads\n";
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
