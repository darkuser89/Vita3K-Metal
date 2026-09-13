// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/texture_cache.h>
#include <renderer/types.h>
#include <mem/state.h>
#include <gxm/functions.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

struct UploadProbe : renderer::TextureCache {
    std::vector<float> values;
    SceGxmTextureBaseFormat expected_format = SCE_GXM_TEXTURE_BASE_FORMAT_F32;
    uint32_t uploads = 0;
    void select(size_t, const SceGxmTexture &) override {}
    void configure_texture(const SceGxmTexture &) override {}
    void import_configure_impl(SceGxmTextureBaseFormat, uint32_t, uint32_t, bool, uint16_t, uint16_t, bool) override {}
    void upload_texture_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height,
        uint32_t mip, const void *pixels, int face, uint32_t stride) override {
        check(format == expected_format, "Metal texture upload format mismatch");
        check(mip == uploads++ && !face, "Unexpected mip/face in texture test");
        if (format == SCE_GXM_TEXTURE_BASE_FORMAT_F32) {
            auto source = static_cast<const float *>(pixels);
            for (uint32_t y = 0; y < height; ++y)
                values.insert(values.end(), source + y*stride, source + y*stride + width);
        } else {
            auto source = static_cast<const __fp16 *>(pixels);
            for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width*4; ++x)
                values.push_back(source[y*stride*4+x]);
        }
    }
};

int main() {
    try {
        MemState mem;
        check(init(mem, false), "Cannot initialize guest memory");
        Ptr<uint32_t> data(alloc(mem, 8*8*4, "Metal depth texture fixture"));
        check(bool(data), "Cannot allocate depth input");
        for (bool swizzled : {false, true}) for (auto mode : {SCE_GXM_TEXTURE_SWIZZLE2_SD, SCE_GXM_TEXTURE_SWIZZLE2_DS}) {
            const uint32_t width = swizzled ? 8 : 5, height = swizzled ? 8 : 3;
            SceGxmTexture texture{};
            texture.type = (swizzled ? SCE_GXM_TEXTURE_SWIZZLED : SCE_GXM_TEXTURE_LINEAR) >> 29;
            if (swizzled) texture.width_base2 = texture.height_base2 = 3;
            else { texture.width = width - 1; texture.height = height - 1; }
            texture.mip_count = 15;
            texture.base_format = SCE_GXM_TEXTURE_BASE_FORMAT_X8U24 >> 24;
            texture.swizzle_format = uint32_t(mode) >> 12;
            texture.data_addr = data.address() >> 2;
            std::vector<float> expected;
            std::memset(data.get(mem), 0x96, 8*8*4);
            for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
                const uint32_t i = y*width+x;
                const uint32_t depth = i == 0 ? 0 : i == 1 ? 0xffffff : (i*234791) & 0xffffff;
                const uint32_t ignored = (i*37+91) & 255;
                size_t address = y*8+x;
                if (swizzled) {
                    address = 0;
                    for (unsigned b = 0; b < 3; ++b)
                        address |= ((x >> b) & 1) << (2*b+1) | ((y >> b) & 1) << (2*b);
                }
                data.get(mem)[address] = mode == SCE_GXM_TEXTURE_SWIZZLE2_DS ? (depth << 8) | ignored : depth | (ignored << 24);
                expected.push_back(float(depth) / 16777215.0f);
            }
            std::vector<uint32_t> original(data.get(mem), data.get(mem) + 64);
            UploadProbe probe;
            probe.backend = renderer::Backend::Metal;
            probe.upload_texture(texture, mem);
            check(probe.values.size() == expected.size(), "Depth upload size mismatch");
            for (size_t i = 0; i < expected.size(); ++i)
                check(std::abs(probe.values[i] - expected[i]) < 1e-7f, "Depth normalization/layout mismatch");
            check(std::memcmp(original.data(), data.get(mem), 8*8*4) == 0, "Depth upload changed guest input");
            std::cout << "PASS Metal X8U24 -> F32, " << (swizzled ? "Morton layout" : "padded linear rows")
                      << ", " << (mode == SCE_GXM_TEXTURE_SWIZZLE2_DS ? "DS" : "SD") << ", all depths and ignored-byte isolation\n";
        }
        // Two mips catch confusing the expanded 8-byte upload stride with the
        // packed 4-byte guest stride; distinct channels expose layout truncation.
        Ptr<uint32_t> packed(alloc(mem, 1024, "Metal packed float texture"));
        for (bool swizzled : {false, true}) for (auto mode : {SCE_GXM_TEXTURE_SWIZZLE4_ABGR, SCE_GXM_TEXTURE_SWIZZLE4_RGBA}) {
            SceGxmTexture texture{};
            texture.type = (swizzled ? SCE_GXM_TEXTURE_SWIZZLED : SCE_GXM_TEXTURE_LINEAR) >> 29;
            if (swizzled) texture.width_base2 = texture.height_base2 = 3;
            else { texture.width = 7; texture.height = 7; }
            texture.mip_count = 1;
            texture.base_format = (SCE_GXM_TEXTURE_BASE_FORMAT_U2F10F10F10 >> 24) & 31;
            texture.format0 = SCE_GXM_TEXTURE_BASE_FORMAT_U2F10F10F10 >> 31;
            texture.swizzle_format = uint32_t(mode) >> 12;
            texture.data_addr = packed.address() >> 2;
            std::memset(packed.get(mem), 0x96, 1024);
            std::vector<float> expected;
            size_t offset = 0;
            for (uint32_t dim : {8u, 4u}) {
                for (uint32_t y = 0; y < dim; ++y) for (uint32_t x = 0; x < dim; ++x) {
                    const uint32_t i = y*dim+x;
                    const uint32_t components[] = {i % 32, 13*32 + (i%32), 17*32 + ((i*3)%32)};
                    const float rgb[] = {std::ldexp(float(components[0]), -19),
                        std::ldexp(1.0f + float(i%32)/32, -2), std::ldexp(1.0f + float((i*3)%32)/32, 2)};
                    const uint32_t alpha = i%4;
                    const float a = float(__fp16(float(alpha)/3));
                    uint32_t word = components[0] | (components[1] << 10) | (components[2] << 20);
                    if (mode == SCE_GXM_TEXTURE_SWIZZLE4_ABGR) {
                        word |= alpha << 30;
                        expected.insert(expected.end(), {rgb[0], rgb[1], rgb[2], a});
                    } else {
                        word = (word << 2) | alpha;
                        expected.insert(expected.end(), {a, rgb[0], rgb[1], rgb[2]});
                    }
                    size_t address = y*8+x;
                    if (swizzled) {
                        address = 0;
                        for (unsigned b = 0; (1u << b) < dim; ++b)
                            address |= ((x >> b) & 1) << (2*b+1) | ((y >> b) & 1) << (2*b);
                    }
                    packed.get(mem)[offset+address] = word;
                }
                offset += swizzled ? dim*dim : dim*8;
            }
            std::vector<uint32_t> original(packed.get(mem), packed.get(mem)+256);
            UploadProbe probe;
            probe.backend = renderer::Backend::Metal;
            probe.expected_format = SCE_GXM_TEXTURE_BASE_FORMAT_F16F16F16F16;
            probe.upload_texture(texture, mem);
            check(probe.uploads == 2 && probe.values.size() == expected.size(), "Packed float mip size mismatch");
            for (size_t i = 0; i < expected.size(); ++i)
                check(probe.values[i] == expected[i], "Packed float channel, subnormal or mip layout mismatch");
            check(std::memcmp(original.data(), packed.get(mem), 1024) == 0, "Packed float upload changed guest input");
            std::cout << "PASS Metal U2F10F10F10 -> RGBA16F, " << (swizzled ? "Morton" : "padded linear")
                      << ", alpha " << (mode == SCE_GXM_TEXTURE_SWIZZLE4_ABGR ? "upper" : "lower")
                      << ", both mips, HDR and subnormal values\n";
        }
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
