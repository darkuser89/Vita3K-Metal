// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <shader/metal_texture.h>
#include <shader/msl_recompiler.h>
#include <shader/uniform_block.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using Pixel = std::array<float, 4>;
static void check(bool value, const std::string &why) {
    if (!value)
        throw std::runtime_error(why);
}
static Pixel color(unsigned mip, unsigned x, unsigned y) {
    return {float((x * 17 + mip * 37) % 256) / 255, float((y * 23 + mip * 41) % 256) / 255,
            float((x + y + mip * 5) % 3 * 100) / 255, 1};
}
static int address(int value, int size, unsigned mode) {
    if (mode == SCE_GXM_TEXTURE_ADDR_CLAMP)
        return std::clamp(value, 0, size - 1);
    int period = mode == SCE_GXM_TEXTURE_ADDR_MIRROR ? size * 2 : size;
    int phase = (value % period + period) % period;
    return phase < size ? phase : period - phase - 1;
}
// Original Unit 13 material shader, varying UVs with an analytically known
// footprint. The CPU oracle samples original mip grids, without consulting
// shader metadata, assembled pixels or generated Metal expressions.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc >= 3 && argc <= 6, "usage: metal-mip-lod-validation Unit13-80e37b24.gxp output-directory "
                                          "[max-scale 1..3] [corrected|hardware|compare] [exact|fractional]");
            unsigned max_scale = argc >= 4 ? std::stoul(argv[3]) : 3;
            const bool comparing = argc >= 5 && std::string(argv[4]) == "compare";
            bool corrected = argc < 5 || std::string(argv[4]) == "corrected" || comparing;
            check(!comparing || max_scale == 1, "Direct hardware comparison requires identical mip grids at scale 1");
            const bool fractional = argc >= 6 && std::string(argv[5]) == "fractional";
            check(argc < 6 || fractional || std::string(argv[5]) == "exact", "Invalid footprint mode");
            const std::vector<float> footprints =
                fractional ? std::vector<float>{-.9f, -.2f, .2f, .49f, .51f, 1.2f, 1.8f, 2.4f, 3.2f, 4.2f}
                           : std::vector<float>{-2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
            check(max_scale >= 1 && max_scale <= 3, "Invalid scale");
            check(argc < 5 || corrected || std::string(argv[4]) == "hardware", "Invalid sampling path");
            std::filesystem::create_directories(argv[2]);
            std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
            check(bool(input), "Cannot open GXP");
            auto length = input.tellg();
            check(length > 0 && length < 1024 * 1024, "Invalid GXP size");
            std::vector<uint32_t> words((size_t(length) + 3) / 4);
            input.seekg(0);
            check(bool(input.read(reinterpret_cast<char *>(words.data()), length)), "Cannot read GXP");
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            auto native = device->native_device();
            renderer::metal::SurfaceCaster caster(*device);
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *nsError = nil;
            auto vertex = [native newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;
struct O {float4 p [[position]];float4 uv [[user(locn4)]];float4 n [[user(locn5)]];float4 v [[user(locn6)]];};
vertex O v(uint i [[vertex_id]],constant float4 &mapping [[buffer(0)]]) {
 float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};O o;o.p=float4(p[i],0,1);
 o.uv=float4(float2(p[i].x*.5+.5,.5-p[i].y*.5)*mapping.xy+mapping.zw,0,0);
 o.n=float4(0,0,1,0);o.v=float4(0,0,1,0);return o;
})" options:options error:&nsError];
            check(vertex != nil, nsError ? nsError.localizedDescription.UTF8String : "Vertex failed");
            constexpr unsigned side = 16;
            auto make_texture = [&](MTLPixelFormat format, unsigned width, unsigned height, unsigned mips) {
                auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                            width:width
                                                                           height:height
                                                                        mipmapped:NO];
                d.mipmapLevelCount = mips;
                d.storageMode = MTLStorageModeShared;
                d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
                auto t = [native newTextureWithDescriptor:d];
                check(t != nil, "Texture allocation failed");
                return t;
            };
            auto target = make_texture(MTLPixelFormatRG32Float, side, side, 1);
            auto mask = make_texture(MTLPixelFormatRGBA32Float, side, side, 1);
            std::array<Pixel, side * side> white;
            white.fill({1, 1, 1, 1});
            [mask replaceRegion:MTLRegionMake2D(0, 0, side, side)
                    mipmapLevel:0
                      withBytes:white.data()
                    bytesPerRow:side * sizeof(Pixel)];
            auto cube_desc = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                   size:1
                                                                              mipmapped:NO];
            cube_desc.storageMode = MTLStorageModeShared;
            cube_desc.usage = MTLTextureUsageShaderRead;
            auto cube = [native newTextureWithDescriptor:cube_desc];
            check(cube != nil, "Cube allocation failed");
            Pixel zero{};
            for (unsigned face = 0; face < 6; ++face)
                [cube replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                        mipmapLevel:0
                              slice:face
                          withBytes:zero.data()
                        bytesPerRow:16
                      bytesPerImage:16];
            const uint32_t material[4] = {0x3c00, 0, 0x3c000000, 0x3c00};
            auto uniform = [native newBufferWithBytes:material length:16 options:MTLResourceStorageModeShared];
            shader::RenderFragUniformBlockExtended info{};
            info.base_block.res_multiplier = 1;
            info.set_buffer_count(3);
            info.set_buffer_address(2, uniform.gpuAddress);
            info.set_texture_count(2);
            info.set_viewport_ratio(0, {1, 1});
            info.set_viewport_ratio(1, {1, 1});
            std::vector<uint8_t> bytes((info.get_size() + 15) & ~size_t(15));
            info.copy_to(bytes.data());
            SceGxmTexture texture{};
            texture.type = SCE_GXM_TEXTURE_LINEAR >> 29;
            texture.lod_bias = 31;
            auto cube_sampler = renderer::metal::make_sampler(*device, texture, 1);
            unsigned draws = 0, failed_draws = 0;
            uint64_t compared = 0, failures = 0;
            for (float bias : {0.f, .125f, .375f, -.375f}) {
                auto program = compile_gxp_fixture(words, length, "unit13-mip-lod", "gxp-mapped-unit13-mips", bias);
                check(program.cube_texture_mask == 2, "Unexpected shader resources");
                std::ofstream saved(std::filesystem::path(argv[2]) / ("bias-" + std::to_string(bias) + ".metal"));
                saved << program.source;
                saved.close();
                check(bool(saved), "Cannot save MSL");
                auto fragment = device->compile(program, false, error);
                check(bool(fragment), error);
                auto pd = [MTLRenderPipelineDescriptor new];
                pd.vertexFunction = [vertex newFunctionWithName:@"v"];
                pd.fragmentFunction = fragment->function;
                pd.colorAttachments[0].pixelFormat = MTLPixelFormatRG32Float;
                auto pipeline = device->create_pipeline(pd, error);
                check(pipeline != nil, error);
                for (bool npot : {false, true})
                    for (unsigned scale = 1; scale <= max_scale; ++scale) {
                        const unsigned width = npot ? 15 : 16, height = npot ? 9 : 16, levels = npot ? 4 : 5;
                        auto uploaded = make_texture(MTLPixelFormatRGBA32Float, width, height, levels);
                        std::vector<renderer::metal::CubeSurface> sources;
                        for (unsigned mip = 0; mip < levels; ++mip) {
                            unsigned w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
                            std::vector<Pixel> pixels(w * h);
                            for (unsigned y = 0; y < h; ++y)
                                for (unsigned x = 0; x < w; ++x)
                                    pixels[y * w + x] = color(mip, x, y);
                            [uploaded replaceRegion:MTLRegionMake2D(0, 0, w, h)
                                        mipmapLevel:mip
                                          withBytes:pixels.data()
                                        bytesPerRow:w * sizeof(Pixel)];
                            if (mip % 2 == 0) {
                                auto rendered = make_texture(MTLPixelFormatRGBA32Float, w * scale, h * scale, 1);
                                pixels.resize(w * h * scale * scale);
                                for (unsigned y = 0; y < h * scale; ++y)
                                    for (unsigned x = 0; x < w * scale; ++x)
                                        pixels[y * w * scale + x] = color(mip, x, y);
                                [rendered replaceRegion:MTLRegionMake2D(0, 0, w * scale, h * scale)
                                            mipmapLevel:0
                                              withBytes:pixels.data()
                                            bytesPerRow:w * scale * sizeof(Pixel)];
                                sources.push_back({rendered, 0, mip});
                            }
                        }
                        auto assembled = caster.texture_snapshot(uploaded, sources, scale);
                        shader::metal::TextureMipInfos mip_info{};
                        for (unsigned mip = 0; mip < levels; ++mip)
                            mip_info[0].sizes[mip] = {std::max(1u, width >> mip) * (mip % 2 ? 1 : scale),
                                                      std::max(1u, height >> mip) * (mip % 2 ? 1 : scale)};
                        for (unsigned min_filter = 0; min_filter < 2; ++min_filter)
                            for (unsigned mag_filter = 0; mag_filter < 2; ++mag_filter)
                                for (unsigned mip_filter = 0; mip_filter < 2; ++mip_filter)
                                    for (unsigned addr = 0; addr < 3; ++addr)
                                        for (unsigned minimum : {0u, 2u})
                                            for (float footprint : footprints) {
                                                @autoreleasepool {
                                                    texture.mip_count = levels - 1;
                                                    texture.min_filter = min_filter;
                                                    texture.mag_filter = mag_filter;
                                                    texture.mip_filter = mip_filter;
                                                    texture.uaddr_mode = texture.vaddr_mode = addr;
                                                    texture.lod_min0 = minimum & 3;
                                                    texture.lod_min1 = minimum >> 2;
                                                    auto sampler = renderer::metal::make_sampler(*device, texture, 1);
                                                    mip_info[0].control = {corrected ? 1u : 0u,
                                                                           min_filter | (mag_filter << 1) |
                                                                               (mip_filter << 2) | (addr << 3) |
                                                                               (addr << 6) | (minimum << 9),
                                                                           1, 0};
                                                    const float step = std::exp2(footprint);
                                                    const std::array<float, 4> mapping = {
                                                        step * side / (width * scale), step * side / (height * scale),
                                                        -.137f, .217f};
                                                    const auto render = [&](bool use_correction) {
                                                        mip_info[0].control[0] = use_correction;
                                                        auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                                                        pass.colorAttachments[0].texture = target;
                                                        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                                                        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                                                        auto commands = [device->command_queue() commandBuffer];
                                                        auto enc = [commands renderCommandEncoderWithDescriptor:pass];
                                                        [enc setRenderPipelineState:pipeline];
                                                        [enc setVertexBytes:mapping.data()
                                                                     length:sizeof(mapping)
                                                                    atIndex:0];
                                                        [enc setFragmentBytes:bytes.data()
                                                                       length:bytes.size()
                                                                      atIndex:0];
                                                        [enc setFragmentBytes:mip_info.data()
                                                                       length:sizeof(mip_info)
                                                                      atIndex:shader::metal::TEXTURE_INFO_BUFFER];
                                                        [enc useResource:uniform
                                                                   usage:MTLResourceUsageRead
                                                                  stages:MTLRenderStageFragment];
                                                        [enc setFragmentTexture:assembled atIndex:0];
                                                        [enc setFragmentTexture:cube atIndex:1];
                                                        [enc setFragmentTexture:mask atIndex:17];
                                                        [enc setFragmentSamplerState:sampler atIndex:0];
                                                        [enc setFragmentSamplerState:cube_sampler atIndex:1];
                                                        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                                                                vertexStart:0
                                                                vertexCount:3];
                                                        [enc endEncoding];
                                                        check(device->submit_and_wait(commands, error), error);
                                                        std::array<std::array<uint8_t, 8>, side * side> actual;
                                                        [target getBytes:actual.data()
                                                             bytesPerRow:side * 8
                                                              fromRegion:MTLRegionMake2D(0, 0, side, side)
                                                             mipmapLevel:0];
                                                        return actual;
                                                    };
                                                    const auto actual = render(corrected);
                                                    std::array<std::array<uint8_t, 8>, side * side> hardware{};
                                                    if (comparing)
                                                        hardware = render(false);
                                                    const float raw_lod = footprint + bias,
                                                                lod = std::min(float(levels - 1),
                                                                               std::max(raw_lod, float(minimum)));
                                                    const unsigned lower = mip_filter ? unsigned(std::floor(lod))
                                                                                      : unsigned(std::floor(lod + .5f));
                                                    const unsigned upper = std::min(lower + 1, levels - 1);
                                                    const float weight = mip_filter ? lod - std::floor(lod) : 0;
                                                    const bool linear = lod > 0 ? min_filter : mag_filter;
                                                    const auto sample = [&](unsigned mip, float u, float v) {
                                                        const unsigned w = std::max(1u, width >> mip) *
                                                                           (mip % 2 ? 1 : scale),
                                                                       h = std::max(1u, height >> mip) *
                                                                           (mip % 2 ? 1 : scale);
                                                        const float px = u * w - (linear ? .5f : 0.f),
                                                                    py = v * h - (linear ? .5f : 0.f);
                                                        const int x = int(std::floor(px)), y = int(std::floor(py));
                                                        const auto at = [&](int xx, int yy) {
                                                            return color(mip, address(xx, w, addr),
                                                                         address(yy, h, addr));
                                                        };
                                                        auto a = at(x, y);
                                                        if (!linear)
                                                            return a;
                                                        auto b = at(x + 1, y), c = at(x, y + 1), d = at(x + 1, y + 1);
                                                        const float fx = px - x, fy = py - y;
                                                        for (unsigned k = 0; k < 4; ++k)
                                                            a[k] = (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) +
                                                                   (c[k] * (1 - fx) + d[k] * fx) * fy;
                                                        return a;
                                                    };
                                                    unsigned bad = 0;
                                                    for (unsigned y = 0; y < side; ++y)
                                                        for (unsigned x = 0; x < side; ++x) {
                                                            const float u = (x + .5f) / side * mapping[0] + mapping[2],
                                                                        v = (y + .5f) / side * mapping[1] + mapping[3];
                                                            auto a = sample(lower, u, v), b = sample(upper, u, v);
                                                            for (unsigned c = 0; c < 8; ++c) {
                                                                int expected =
                                                                    c < 4    ? int(std::round(
                                                                                   float((__fp16)(a[c] + (b[c] - a[c]) *
                                                                                                             weight)) *
                                                                                   255))
                                                                    : c == 6 ? 127
                                                                             : 0;
                                                                if (comparing)
                                                                    expected = hardware[y * side + x][c];
                                                                bool mismatch = std::abs(int(actual[y * side + x][c]) -
                                                                                         expected) > (c < 3 ? 1 : 0);
                                                                bad += mismatch;
                                                                ++compared;
                                                                if (mismatch && failures + bad <= 12)
                                                                    std::cerr
                                                                        << "Mismatch corrected=" << corrected
                                                                        << " shape=" << npot << " scale=" << scale
                                                                        << " min=" << min_filter
                                                                        << " mag=" << mag_filter
                                                                        << " mipfilter=" << mip_filter
                                                                        << " addr=" << addr << " minimum=" << minimum
                                                                        << " footprint=" << footprint
                                                                        << " bias=" << bias << " pixel=" << x << ","
                                                                        << y << " c=" << c << " actual="
                                                                        << unsigned(actual[y * side + x][c])
                                                                        << " expected=" << expected << '\n';
                                                            }
                                                        }
                                                    if (bad)
                                                        std::cout << "BADCASE scale=" << scale << " npot=" << npot
                                                                  << " min=" << min_filter << " mag=" << mag_filter
                                                                  << " mip=" << mip_filter << " addr=" << addr
                                                                  << " minimum=" << minimum
                                                                  << " footprint=" << footprint << " bias=" << bias
                                                                  << " bytes=" << bad << "\n";
                                                    failures += bad;
                                                    failed_draws += bad != 0;
                                                    ++draws;
                                                }
                                            }
                        std::cout << "PROGRESS bias=" << bias << " npot=" << npot << " scale=" << scale
                                  << " draws=" << draws << " failed_draws=" << failed_draws
                                  << " failed_bytes=" << failures << std::endl;
                    }
            }
            std::cout << "RESULT corrected=" << corrected << " compare=" << comparing << " fractional=" << fractional
                      << " draws=" << draws << " material_bytes=" << compared << " failed_draws=" << failed_draws
                      << " failed_bytes=" << failures << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 2;
        }
    }
}
