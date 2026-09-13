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
#include <set>
#include <stdexcept>
using Pixel = std::array<float, 4>;
static void check(bool ok, const std::string &s) {
    if (!ok)
        throw std::runtime_error(s);
}
static float half(float x) { return float((__fp16)x); }
// Independent directed-rounding reference: start with host nearest-half, then
// move one half ULP toward zero only when it overshot the exact F32 result.
static float half_mad(float x) {
    __fp16 h = (__fp16)x;
    if (std::isfinite(x) && std::abs(float(h)) > std::abs(x)) {
        uint16_t bits;
        std::memcpy(&bits, &h, sizeof(bits));
        --bits;
        std::memcpy(&h, &bits, sizeof(bits));
    }
    return float(h);
}
static Pixel value(unsigned kind, unsigned mip, unsigned x, unsigned y, unsigned w, unsigned h) {
    if (kind == 0)
        return {.125f + float((x * 11 + mip * 37) % 128) / 256, .25f + float((y * 13 + mip * 19) % 64) / 256,
                .125f + float((x + y * 3 + mip) % 7) / 16, 1};
    // Keep cube directions away from face ties: hardware bilinear weights
    // can differ slightly from the CPU. Cycle all positive faces by mip.
    if (kind == 1) {
        Pixel normal{.2f + .1f * x / (w + 1), .1f + .1f * y / (h + 1), .25f, 1};
        // Mip 1 must use +Z: its 0.3 cube channel exposes the observed
        // sampler/half narrowing in the final light accumulation.
        normal[(mip + 1) % 3] += 1;
        return normal;
    }
    return {.125f + .5f * x / (w + 1), -.25f + .5f * y / (h + 1), .125f * mip, 1};
}
static Pixel cube_value(unsigned face) { return {.25f + face / 16.f, .5f - face * .05f, .125f + face / 16.f, 1}; }
static int address(int p, int size, unsigned mode) {
    if (mode == 2)
        return std::clamp(p, 0, size - 1);
    int period = size * (mode == 1 ? 2 : 1);
    int t = (p % period + period) % period;
    return t < size ? t : period - 1 - t;
}
// Original Unit 13 e3ca9d96 light accumulation: three xy/w queries,
// a normal-directed cube query, attenuation, and half framebuffer fetch.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc >= 3 && argc <= 5, "usage: metal-projected-validation e3ca9d96.gxp output-directory [max-scale "
                                          "1..3] [corrected|hardware]");
            unsigned max_scale = argc >= 4 ? std::stoul(argv[3]) : 3;
            bool corrected = argc < 5 || std::string(argv[4]) == "corrected";
            check(max_scale >= 1 && max_scale <= 3, "Invalid scale");
            check(argc < 5 || corrected || std::string(argv[4]) == "hardware", "Invalid path");
            std::filesystem::create_directories(argv[2]);
            std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
            check(bool(f), "Cannot open GXP");
            auto length = f.tellg();
            check(length > 0 && length < 1024 * 1024, "Invalid GXP");
            std::vector<uint32_t> words((size_t(length) + 3) / 4);
            f.seekg(0);
            check(bool(f.read(reinterpret_cast<char *>(words.data()), length)), "Cannot read GXP");
            auto program =
                compile_gxp_fixture(words, length, "unit13-projective-original", "gxp-mapped-projected-half");
            check(program.uses_framebuffer_fetch && program.cube_texture_mask == (1u << 12),
                  "Unexpected shader resources");
            std::ofstream saved(std::filesystem::path(argv[2]) / "original.metal");
            saved << program.source;
            saved.close();
            check(bool(saved), "Cannot save MSL");
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            auto native = device->native_device();
            renderer::metal::SurfaceCaster caster(*device);
            auto fs = device->compile(program, false, error);
            check(bool(fs), error);
            NSError *e = nil;
            auto opts = [MTLCompileOptions new];
            opts.languageVersion = MTLLanguageVersion3_0;
            opts.fastMathEnabled = NO;
            auto vs = [native newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;struct O{float4 p [[position]];float4 uv [[user(locn4)]];};
vertex O v(uint i [[vertex_id]],constant float2 &projection [[buffer(0)]]) {
 float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};float2 uv=float2(p[i].x*.5+.5,.5-p[i].y*.5);
 O o;o.p=float4(p[i],0,1);o.uv=float4(uv*.015625f+float2(.03125f,.0625f),97,projection.x*(1+projection.y*uv.x));return o;
})" options:opts error:&e];
            check(vs != nil, e ? e.localizedDescription.UTF8String : "Vertex failure");
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [vs newFunctionWithName:@"v"];
            pd.fragmentFunction = fs->function;
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
            auto pipeline = device->create_pipeline(pd, error);
            check(pipeline != nil, error);
            constexpr unsigned side = 16;
            const Pixel clear{.125f, .25f, .375f, .5f};
            auto texture = [&](MTLPixelFormat format, unsigned w, unsigned h, unsigned levels) {
                auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
                d.storageMode = MTLStorageModeShared;
                d.mipmapLevelCount = levels;
                d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
                auto t = [native newTextureWithDescriptor:d];
                check(t != nil, "Texture allocation failed");
                return t;
            };
            auto target = texture(MTLPixelFormatRGBA16Float, side, side, 1),
                 mask = texture(MTLPixelFormatRGBA32Float, side, side, 1);
            std::array<Pixel, side * side> mask_pixels;
            for (unsigned p = 0; p < side * side; ++p)
                mask_pixels[p] = p % 7 ? Pixel{1, 1, 1, 1} : Pixel{};
            [mask replaceRegion:MTLRegionMake2D(0, 0, side, side)
                    mipmapLevel:0
                      withBytes:mask_pixels.data()
                    bytesPerRow:side * sizeof(Pixel)];
            auto cd = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                            size:1
                                                                       mipmapped:NO];
            cd.storageMode = MTLStorageModeShared;
            cd.usage = MTLTextureUsageShaderRead;
            auto cube = [native newTextureWithDescriptor:cd];
            check(cube != nil, "Cube allocation failed");
            for (unsigned face = 0; face < 6; ++face) {
                auto p = cube_value(face);
                [cube replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                        mipmapLevel:0
                              slice:face
                          withBytes:p.data()
                        bytesPerRow:16
                      bytesPerImage:16];
            }
            auto uniform = [native newBufferWithLength:48 options:MTLResourceStorageModeShared];
            check(uniform != nil, "Uniform allocation failed");
            shader::RenderFragUniformBlockExtended info{};
            info.base_block.res_multiplier = 1;
            info.set_buffer_count(1);
            info.set_buffer_address(0, uniform.gpuAddress);
            info.set_texture_count(13);
            SceGxmTexture desc{};
            desc.type = SCE_GXM_TEXTURE_LINEAR >> 29;
            desc.lod_bias = 31;
            auto cube_sampler = renderer::metal::make_sampler(*device, desc, 1);
            uint64_t compared = 0, failed = 0, draws = 0, active = 0;
            std::set<unsigned> selected_faces;
            for (bool npot : {false, true})
                for (unsigned scale = 1; scale <= max_scale; ++scale) {
                    const unsigned width = npot ? 15 : 16, height = npot ? 9 : 16, levels = 4;
                    std::array<id<MTLTexture>, 3> images;
                    shader::metal::TextureMipInfos mip_info{};
                    for (unsigned kind = 0; kind < 3; ++kind) {
                        auto uploaded = texture(MTLPixelFormatRGBA32Float, width, height, levels);
                        std::vector<renderer::metal::CubeSurface> sources;
                        for (unsigned mip = 0; mip < levels; ++mip) {
                            unsigned w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
                            std::vector<Pixel> data(w * h);
                            for (unsigned y = 0; y < h; ++y)
                                for (unsigned x = 0; x < w; ++x)
                                    data[y * w + x] = value(kind, mip, x, y, w, h);
                            [uploaded replaceRegion:MTLRegionMake2D(0, 0, w, h)
                                        mipmapLevel:mip
                                          withBytes:data.data()
                                        bytesPerRow:w * sizeof(Pixel)];
                            if (mip % 2 == 0) {
                                w *= scale;
                                h *= scale;
                                auto rendered = texture(MTLPixelFormatRGBA32Float, w, h, 1);
                                data.resize(w * h);
                                for (unsigned y = 0; y < h; ++y)
                                    for (unsigned x = 0; x < w; ++x)
                                        data[y * w + x] = value(kind, mip, x, y, w, h);
                                [rendered replaceRegion:MTLRegionMake2D(0, 0, w, h)
                                            mipmapLevel:0
                                              withBytes:data.data()
                                            bytesPerRow:w * sizeof(Pixel)];
                                sources.push_back({rendered, 0, mip});
                            }
                            mip_info[8 + kind].sizes[mip] = {w, h};
                        }
                        images[kind] = caster.texture_snapshot(uploaded, sources, scale);
                    }
                    for (unsigned mip = 0; mip < levels; ++mip)
                        for (unsigned linear = 0; linear < 2; ++linear)
                            for (unsigned addr = 0; addr < 3; ++addr)
                                for (bool mapped : {false, true})
                                    for (float q : {.25f, .5f, 1.f, -.5f, 4.f})
                                        for (float slope : {0.f, .75f})
                                            for (float attenuation : {0.f, -.25f}) {
                                                @autoreleasepool {
                                                    float uniforms[12] = {.25f, .5f, .75f,        0, 0, 0,
                                                                          0,    0,   attenuation, 0, 0, 0};
                                                    std::memcpy(uniform.contents, uniforms, sizeof(uniforms));
                                                    desc.mip_count = levels - 1;
                                                    desc.min_filter = desc.mag_filter = linear;
                                                    desc.uaddr_mode = desc.vaddr_mode = addr;
                                                    desc.lod_min0 = mip & 3;
                                                    desc.lod_min1 = mip >> 2;
                                                    auto sampler = renderer::metal::make_sampler(*device, desc, 1);
                                                    std::array<std::array<float, 2>, 3> ratios, offsets;
                                                    for (unsigned kind = 0; kind < 3; ++kind) {
                                                        ratios[kind] = mapped ? std::array<float, 2>{1.25f, .75f}
                                                                              : std::array<float, 2>{1, 1};
                                                        offsets[kind] = mapped
                                                                            ? std::array<float, 2>{.125f * kind,
                                                                                                   -.0625f * (kind + 1)}
                                                                            : std::array<float, 2>{0, 0};
                                                        info.set_viewport_ratio(8 + kind, ratios[kind]);
                                                        info.set_viewport_offset(8 + kind, offsets[kind]);
                                                        mip_info[8 + kind].control = {corrected ? 1u : 0u,
                                                                                      linear | (linear << 1) |
                                                                                          (addr << 3) | (addr << 6) |
                                                                                          (mip << 9),
                                                                                      1, 0};
                                                    }
                                                    std::vector<uint8_t> info_bytes((info.get_size() + 15) &
                                                                                    ~size_t(15));
                                                    info.copy_to(info_bytes.data());
                                                    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                                                    pass.colorAttachments[0].texture = target;
                                                    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                                                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                                                    pass.colorAttachments[0].clearColor =
                                                        MTLClearColorMake(clear[0], clear[1], clear[2], clear[3]);
                                                    auto commands = [device->command_queue() commandBuffer];
                                                    auto enc = [commands renderCommandEncoderWithDescriptor:pass];
                                                    [enc setRenderPipelineState:pipeline];
                                                    float projection[2] = {q, slope};
                                                    [enc setVertexBytes:projection length:8 atIndex:0];
                                                    [enc setFragmentBytes:info_bytes.data()
                                                                   length:info_bytes.size()
                                                                  atIndex:0];
                                                    [enc setFragmentBytes:mip_info.data()
                                                                   length:sizeof(mip_info)
                                                                  atIndex:shader::metal::TEXTURE_INFO_BUFFER];
                                                    [enc useResource:uniform
                                                               usage:MTLResourceUsageRead
                                                              stages:MTLRenderStageFragment];
                                                    for (unsigned k = 0; k < 3; ++k) {
                                                        [enc setFragmentTexture:images[k] atIndex:8 + k];
                                                        [enc setFragmentSamplerState:sampler atIndex:8 + k];
                                                    }
                                                    [enc setFragmentTexture:cube atIndex:12];
                                                    [enc setFragmentSamplerState:cube_sampler atIndex:12];
                                                    [enc setFragmentTexture:mask atIndex:17];
                                                    [enc drawPrimitives:MTLPrimitiveTypeTriangle
                                                            vertexStart:0
                                                            vertexCount:3];
                                                    [enc endEncoding];
                                                    check(device->submit_and_wait(commands, error), error);
                                                    std::array<std::array<__fp16, 4>, side * side> actual;
                                                    [target getBytes:actual.data()
                                                         bytesPerRow:side * 8
                                                          fromRegion:MTLRegionMake2D(0, 0, side, side)
                                                         mipmapLevel:0];
                                                    const unsigned w = std::max(1u, width >> mip) *
                                                                       (mip % 2 ? 1 : scale),
                                                                   h = std::max(1u, height >> mip) *
                                                                       (mip % 2 ? 1 : scale);
                                                    auto sample = [&](unsigned kind, float u, float v) {
                                                        float px = u * w - (linear ? .5f : 0),
                                                              py = v * h - (linear ? .5f : 0);
                                                        int x = int(std::floor(px)), y = int(std::floor(py));
                                                        auto at = [&](int xx, int yy) {
                                                            return value(kind, mip, address(xx, w, addr),
                                                                         address(yy, h, addr), w, h);
                                                        };
                                                        auto a = at(x, y);
                                                        if (!linear)
                                                            return a;
                                                        auto b = at(x + 1, y), c = at(x, y + 1), d = at(x + 1, y + 1);
                                                        float fx = px - x, fy = py - y;
                                                        for (unsigned k = 0; k < 4; ++k)
                                                            a[k] = (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) +
                                                                   (c[k] * (1 - fx) + d[k] * fx) * fy;
                                                        return a;
                                                    };
                                                    unsigned bad = 0, changed = 0;
                                                    for (unsigned y = 0; y < side; ++y)
                                                        for (unsigned x = 0; x < side; ++x) {
                                                            unsigned p = y * side + x;
                                                            float sx = (x + .5f) / side, sy = (y + .5f) / side,
                                                                  denom = q * (1 + slope * sx);
                                                            float u = (sx * .015625f + .03125f) / denom,
                                                                  v = (sy * .015625f + .0625f) / denom;
                                                            std::array<Pixel, 3> samples;
                                                            for (unsigned kind = 0; kind < 3; ++kind)
                                                                samples[kind] = sample(
                                                                    kind,
                                                                    std::fma(u, ratios[kind][0], offsets[kind][0]),
                                                                    std::fma(v, ratios[kind][1], offsets[kind][1]));
                                                            unsigned axis = samples[1][0] > samples[1][1] ? 0 : 1;
                                                            if (samples[1][2] > samples[1][axis])
                                                                axis = 2;
                                                            unsigned face = axis * 2;
                                                            selected_faces.insert(face);
                                                            auto environment = cube_value(face);
                                                            float distance = 0;
                                                            for (unsigned c = 0; c < 3; ++c) {
                                                                float delta = half(-half(samples[2][c]));
                                                                distance += delta * delta;
                                                            }
                                                            float weight = std::max(
                                                                0.f,
                                                                half_mad(std::fma(half(distance), half(attenuation), 1.f)));
                                                            weight = half(weight * weight);
                                                            for (unsigned c = 0; c < 4; ++c) {
                                                                float expected = clear[c];
                                                                if (p % 7 && c < 3) {
                                                                    float light = half(half(uniforms[c]) * weight);
                                                                    light = half(light * half(environment[c]));
                                                                    expected = half_mad(
                                                                        std::fma(light, half(samples[0][c]), clear[c]));
                                                                }
                                                                float tolerance = c == 3 || p % 7 == 0 ||
                                                                                          (!linear && attenuation == 0)
                                                                                      ? 0
                                                                                      : 1.f / 1024;
                                                                bool mismatch = !std::isfinite(float(actual[p][c])) ||
                                                                                std::abs(float(actual[p][c]) -
                                                                                         expected) > tolerance;
                                                                bad += mismatch;
                                                                ++compared;
                                                                if (mismatch && failed + bad <= 12)
                                                                    std::cerr << "Mismatch shape=" << npot
                                                                              << " scale=" << scale << " mip=" << mip
                                                                              << " linear=" << linear
                                                                              << " addr=" << addr
                                                                              << " mapped=" << mapped << " q=" << q
                                                                              << " slope=" << slope
                                                                              << " attenuation=" << attenuation
                                                                              << " pixel=" << p << " component=" << c
                                                                              << " actual=" << float(actual[p][c])
                                                                              << " expected=" << expected << '\n';
                                                            }
                                                            changed += float(actual[p][0]) != clear[0];
                                                        }
                                                    check(changed > side,
                                                          "Vacuous output: shader did not add light to enough pixels");
                                                    active += changed;
                                                    failed += bad;
                                                    ++draws;
                                                }
                                            }
                    std::cout << "PROGRESS shape=" << npot << " scale=" << scale << " draws=" << draws
                              << " failed=" << failed << std::endl;
                }
            check(selected_faces.size() >= 2, "Normal texture did not affect cube selection");
            std::cout << "RESULT corrected=" << corrected << " draws=" << draws << " components=" << compared
                      << " lit_pixels=" << active << " cube_faces=" << selected_faces.size() << " failed=" << failed
                      << '\n';
            return failed ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 2;
        }
    }
}
