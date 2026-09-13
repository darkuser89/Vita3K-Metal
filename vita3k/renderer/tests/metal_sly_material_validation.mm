// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <shader/metal_texture.h>
#include <shader/uniform_block.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using Pixel = std::array<float, 4>;
static void check(bool ok, const std::string &why) {
    if (!ok)
        throw std::runtime_error(why);
}
static float half(float x) { return float((__fp16)x); }
static Pixel texel(unsigned kind, unsigned mip, unsigned x, unsigned y) {
    return {.125f + float((x * 7 + y * 3 + kind * 11 + mip * 5) % 16) / 32,
            .25f + float((x * 3 + y * 5 + kind * 7 + mip * 3) % 16) / 32,
            .0625f + float((x + y * 7 + kind * 3 + mip) % 16) / 32, kind ? .75f : float((x + y * 3 + mip) % 4) / 4};
}
static int address(int x, int n, unsigned mode) {
    if (mode == 2)
        return std::clamp(x, 0, n - 1);
    const int period = mode == 1 ? 2 * n : n;
    int p = (x % period + period) % period;
    return p < n ? p : period - 1 - p;
}
// Full original Sly d93185db material shader: alpha predicate, dependent
// texture/light contributions, two rounded fog MADs, final alpha, and depth.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc >= 3 && argc <= 5, "usage: metal-sly-material-validation original-d93185db.gxp output.metal "
                                          "[max-scale 1..3] [corrected|hardware]");
            unsigned max_scale = argc >= 4 ? std::stoul(argv[3]) : 3;
            bool corrected = argc < 5 || std::string(argv[4]) == "corrected";
            check(max_scale >= 1 && max_scale <= 3 && (corrected || max_scale == 1),
                  "Hardware comparison requires scale 1");
            check(argc < 5 || corrected || std::string(argv[4]) == "hardware", "Invalid mode");
            std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
            check(bool(file), "Cannot open GXP");
            auto length = file.tellg();
            check(length > 0 && length < 1024 * 1024, "Invalid GXP");
            std::vector<uint32_t> words((size_t(length) + 3) / 4);
            file.seekg(0);
            check(bool(file.read(reinterpret_cast<char *>(words.data()), length)), "Cannot read GXP");
            auto program = compile_gxp_fixture(words, length, "sly-original-material", "gxp-mapped-mips");
            check(!program.uses_framebuffer_fetch && !program.writes_guest_memory, "Unexpected shader side effects");
            std::ofstream output(argv[2]);
            output << program.source;
            output.close();
            check(bool(output), "Cannot save MSL");
            std::string why;
            auto runtime = renderer::metal::Device::create(why);
            check(bool(runtime), why);
            auto dev = runtime->native_device();
            renderer::metal::SurfaceCaster caster(*runtime);
            auto fragment = runtime->compile(program, false, why);
            check(bool(fragment), why);
            NSError *err = nil;
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            auto vertex = [dev newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;struct Params{float4 tint;float4 light1;float4 light2;float4 uv;};
struct O{float4 p [[position]];float4 tint [[user(locn4)]];float4 light1 [[user(locn5)]];float4 light2 [[user(locn6)]];float4 uv [[user(locn7)]];};
vertex O v(uint i [[vertex_id]],constant Params &params [[buffer(0)]]){
 float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};float2 uv=float2(p[i].x*.5+.5,.5-p[i].y*.5);
 O o;o.p=float4(p[i],.25,1);o.tint=params.tint;o.light1=params.light1;o.light2=params.light2;
 o.uv=float4(uv*params.uv.x+params.uv.yz,params.uv.w,0);return o;}
)" options:options error:&err];
            check(vertex != nil, err ? err.localizedDescription.UTF8String : "Vertex failed");
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [vertex newFunctionWithName:@"v"];
            pd.fragmentFunction = fragment->function;
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
            pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
            auto pipeline = runtime->create_pipeline(pd, why);
            check(pipeline != nil, why);
            auto dd = [MTLDepthStencilDescriptor new];
            dd.depthCompareFunction = MTLCompareFunctionLess;
            dd.depthWriteEnabled = YES;
            auto depth_state = [dev newDepthStencilStateWithDescriptor:dd];
            auto texture = [&](MTLPixelFormat format, unsigned w, unsigned h, unsigned mips = 1) {
                auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
                d.mipmapLevelCount = mips;
                d.storageMode = format == MTLPixelFormatDepth32Float ? MTLStorageModePrivate : MTLStorageModeShared;
                d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
                auto t = [dev newTextureWithDescriptor:d];
                check(t != nil, "Texture allocation failed");
                return t;
            };
            // Prime-denominator UV offsets keep raw interpolants away from exact
            // texel/half ties, where rasterizer precision can change the CPU oracle.
            const Pixel clear{-2, -3, -4, -5}, fog_color{.125, .375, .625, 1}, light1{.25, .5, .75, .5},
                light2{.125, -.25, .5, .75};
            auto uniform = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
            check(uniform != nil, "Uniform allocation failed");
            uint64_t draws = 0, color_checks = 0, depth_checks = 0, failures = 0, kept = 0, killed = 0, equalities = 0;
            for (bool npot : {false, true})
                for (unsigned scale = 1; scale <= max_scale; ++scale) {
                    @autoreleasepool {
                        const unsigned width = npot ? 15 : 16, height = npot ? 9 : 8, levels = 4, side = 16 * scale,
                                       depth_pitch = 256;
                        auto target = texture(MTLPixelFormatRGBA32Float, side, side),
                             depth = texture(MTLPixelFormatDepth32Float, side, side),
                             mask = texture(MTLPixelFormatRGBA8Unorm, side, side);
                        auto depth_buffer = [dev newBufferWithLength:depth_pitch * side
                                                             options:MTLResourceStorageModeShared];
                        check(depth_buffer != nil, "Readback allocation failed");
                        std::vector<uint32_t> masks(side * side);
                        for (unsigned p = 0; p < masks.size(); ++p)
                            masks[p] = p % 7 ? 0xffffffff : 0;
                        [mask replaceRegion:MTLRegionMake2D(0, 0, side, side)
                                mipmapLevel:0
                                  withBytes:masks.data()
                                bytesPerRow:side * 4];
                        std::array<id<MTLTexture>, 3> images;
                        shader::metal::TextureMipInfos mip_info{};
                        for (unsigned kind = 0; kind < 3; ++kind) {
                            auto uploaded = texture(MTLPixelFormatRGBA32Float, width, height, levels);
                            std::vector<renderer::metal::CubeSurface> surfaces;
                            for (unsigned mip = 0; mip < levels; ++mip) {
                                unsigned w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
                                std::vector<Pixel> data(w * h);
                                for (unsigned y = 0; y < h; ++y)
                                    for (unsigned x = 0; x < w; ++x)
                                        data[y * w + x] = texel(kind, mip, x, y);
                                [uploaded replaceRegion:MTLRegionMake2D(0, 0, w, h)
                                            mipmapLevel:mip
                                              withBytes:data.data()
                                            bytesPerRow:w * sizeof(Pixel)];
                                if (corrected && mip % 2) {
                                    w *= scale;
                                    h *= scale;
                                    auto native = texture(MTLPixelFormatRGBA32Float, w, h);
                                    data.resize(w * h);
                                    for (unsigned y = 0; y < h; ++y)
                                        for (unsigned x = 0; x < w; ++x)
                                            data[y * w + x] = texel(kind, mip, x, y);
                                    [native replaceRegion:MTLRegionMake2D(0, 0, w, h)
                                              mipmapLevel:0
                                                withBytes:data.data()
                                              bytesPerRow:w * sizeof(Pixel)];
                                    surfaces.push_back({native, 0, mip});
                                }
                                mip_info[kind].sizes[mip] = {w, h};
                            }
                            images[kind] = corrected ? caster.texture_snapshot(uploaded, surfaces, scale) : uploaded;
                        }
                        for (unsigned linear : {0u, 1u})
                            for (unsigned addr : {0u, 1u, 2u})
                                for (unsigned minimum : {0u, 1u, 3u})
                                    for (float uv_scale : {.5f, 2.f})
                                        for (bool mapped : {false, true})
                                            for (float threshold : {0.f, .25f, .5f, 1.f})
                                                for (float tint_alpha : {.5f, 1.f})
                                                    for (float fog : {0.f, .5f, 1.f}) {
                                                        @autoreleasepool {
                                                            struct Params {
                                                                Pixel tint, light1, light2, uv;
                                                            } params{{.5, .75, .25, tint_alpha},
                                                                     light1,
                                                                     light2,
                                                                     {uv_scale, 1.f / 53, 1.f / 29, fog}};
                                                            std::array<__fp16, 8> uniforms{};
                                                            for (unsigned c = 0; c < 4; ++c)
                                                                uniforms[c] = fog_color[c];
                                                            uniforms[4] = threshold;
                                                            std::memcpy(uniform.contents, uniforms.data(), 16);
                                                            shader::RenderFragUniformBlockExtended info{};
                                                            info.base_block.res_multiplier = scale;
                                                            info.set_buffer_count(1);
                                                            info.set_buffer_address(0, uniform.gpuAddress);
                                                            info.set_texture_count(3);
                                                            std::array<std::array<float, 2>, 3> ratios, offsets;
                                                            std::array<id<MTLSamplerState>, 3> samplers;
                                                            for (unsigned k = 0; k < 3; ++k) {
                                                                ratios[k] = mapped ? std::array<float, 2>{.5, .75}
                                                                                   : std::array<float, 2>{1, 1};
                                                                offsets[k] =
                                                                    mapped ? std::array<float, 2>{k / 8.f, -float(k) / 16.f}
                                                                           : std::array<float, 2>{0, 0};
                                                                info.set_viewport_ratio(k, ratios[k]);
                                                                info.set_viewport_offset(k, offsets[k]);
                                                                // Point alpha deliberately creates exact predicate
                                                                // boundaries and nonuniform discard; dependent light
                                                                // textures also test linear filtering.
                                                                unsigned filter = k ? linear : 0;
                                                                SceGxmTexture desc{};
                                                                desc.type = SCE_GXM_TEXTURE_LINEAR >> 29;
                                                                desc.mip_count = levels - 1;
                                                                desc.lod_bias = 31;
                                                                desc.min_filter = desc.mag_filter = filter;
                                                                desc.uaddr_mode = desc.vaddr_mode = addr;
                                                                desc.lod_min0 = minimum;
                                                                samplers[k] =
                                                                    renderer::metal::make_sampler(*runtime, desc, 1);
                                                                mip_info[k].control = {corrected ? 1u : 0u,
                                                                                       filter | (filter << 1) |
                                                                                           (addr << 3) | (addr << 6) |
                                                                                           (minimum << 9),
                                                                                       1, 0};
                                                            }
                                                            std::vector<uint8_t> bytes((info.get_size() + 15) &
                                                                                       ~size_t(15));
                                                            info.copy_to(bytes.data());
                                                            auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                                                            pass.colorAttachments[0].texture = target;
                                                            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                                                            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                                                            pass.colorAttachments[0].clearColor = MTLClearColorMake(
                                                                clear[0], clear[1], clear[2], clear[3]);
                                                            pass.depthAttachment.texture = depth;
                                                            pass.depthAttachment.loadAction = MTLLoadActionClear;
                                                            pass.depthAttachment.storeAction = MTLStoreActionStore;
                                                            pass.depthAttachment.clearDepth = .75;
                                                            auto cb = [runtime->command_queue() commandBuffer];
                                                            auto enc = [cb renderCommandEncoderWithDescriptor:pass];
                                                            [enc setRenderPipelineState:pipeline];
                                                            [enc setDepthStencilState:depth_state];
                                                            [enc setVertexBytes:&params
                                                                         length:sizeof(params)
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
                                                            for (unsigned k = 0; k < 3; ++k) {
                                                                [enc setFragmentTexture:images[k] atIndex:k];
                                                                [enc setFragmentSamplerState:samplers[k] atIndex:k];
                                                            }
                                                            [enc setFragmentTexture:mask atIndex:17];
                                                            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                                                                    vertexStart:0
                                                                    vertexCount:3];
                                                            [enc endEncoding];
                                                            auto blit = [cb blitCommandEncoder];
                                                            [blit copyFromTexture:depth
                                                                             sourceSlice:0
                                                                             sourceLevel:0
                                                                            sourceOrigin:MTLOriginMake(0, 0, 0)
                                                                              sourceSize:MTLSizeMake(side, side, 1)
                                                                                toBuffer:depth_buffer
                                                                       destinationOffset:0
                                                                  destinationBytesPerRow:depth_pitch
                                                                destinationBytesPerImage:depth_pitch * side];
                                                            [blit endEncoding];
                                                            check(runtime->submit_and_wait(cb, why), why);
                                                            std::vector<Pixel> actual(side * side);
                                                            [target getBytes:actual.data()
                                                                 bytesPerRow:side * sizeof(Pixel)
                                                                  fromRegion:MTLRegionMake2D(0, 0, side, side)
                                                                 mipmapLevel:0];
                                                            auto sample = [&](unsigned k, float u, float v) {
                                                                float footprint =
                                                                    std::max(width * uv_scale * ratios[k][0] / 16,
                                                                             height * uv_scale * ratios[k][1] / 16);
                                                                unsigned mip = std::clamp(
                                                                    int(std::floor(
                                                                        std::max(float(minimum), std::log2(footprint)) +
                                                                        .5f)),
                                                                    0, int(levels - 1));
                                                                unsigned w = mip_info[k].sizes[mip][0],
                                                                         h = mip_info[k].sizes[mip][1];
                                                                bool bilinear = k && linear;
                                                                float px =
                                                                          std::fma(u, ratios[k][0], offsets[k][0]) * w -
                                                                          (bilinear ? .5f : 0),
                                                                      py =
                                                                          std::fma(v, ratios[k][1], offsets[k][1]) * h -
                                                                          (bilinear ? .5f : 0);
                                                                int x = int(std::floor(px)), y = int(std::floor(py));
                                                                auto at = [&](int xx, int yy) {
                                                                    return texel(k, mip, address(xx, w, addr),
                                                                                 address(yy, h, addr));
                                                                };
                                                                auto a = at(x, y);
                                                                if (!bilinear)
                                                                    return a;
                                                                auto b = at(x + 1, y), c = at(x, y + 1),
                                                                     d = at(x + 1, y + 1);
                                                                float fx = px - x, fy = py - y;
                                                                for (unsigned j = 0; j < 4; ++j)
                                                                    a[j] = (a[j] * (1 - fx) + b[j] * fx) * (1 - fy) +
                                                                           (c[j] * (1 - fx) + d[j] * fx) * fy;
                                                                return a;
                                                            };
                                                            for (unsigned y = 0; y < side; ++y)
                                                                for (unsigned x = 0; x < side; ++x) {
                                                                    unsigned p = y * side + x;
                                                                    float u = ((x + .5f) / side) * uv_scale +
                                                                              params.uv[1],
                                                                          v = ((y + .5f) / side) * uv_scale +
                                                                              params.uv[2];
                                                                    auto base = sample(0, u, v);
                                                                    Pixel initial{};
                                                                    for (unsigned c = 0; c < 4; ++c)
                                                                        initial[c] =
                                                                            half(half(base[c]) * half(params.tint[c]));
                                                                    // KILL uses the inverted short predicate in the
                                                                    // existing USSE decoder: this program keeps alpha <
                                                                    // threshold, rejects equality.
                                                                    bool survives =
                                                                        initial[3] < half(threshold) && masks[p];
                                                                    equalities += initial[3] == half(threshold);
                                                                    Pixel expected = clear;
                                                                    if (survives) {
                                                                        ++kept;
                                                                        auto a = sample(1, half(u), half(v)),
                                                                             b = sample(2, half(u), half(v));
                                                                        for (unsigned c = 0; c < 3; ++c) {
                                                                            float value = half(std::fma(
                                                                                half(half(a[c]) * half(light1[3])),
                                                                                half(light1[c]), initial[c]));
                                                                            value = half(std::fma(
                                                                                half(half(b[c]) * half(light2[3])),
                                                                                half(light2[c]), value));
                                                                            float first = half(std::fma(
                                                                                half(fog), half(fog_color[c]), value));
                                                                            expected[c] = half(
                                                                                std::fma(half(fog), -value, first));
                                                                        }
                                                                        expected[3] = initial[3];
                                                                    } else
                                                                        ++killed;
                                                                    auto mismatch = [&](const char *stage, float got,
                                                                                        float want) {
                                                                        if (failures++ < 12)
                                                                            std::cerr << "Mismatch " << stage
                                                                                      << " shape=" << npot
                                                                                      << " scale=" << scale
                                                                                      << " linear=" << linear
                                                                                      << " addr=" << addr
                                                                                      << " min=" << minimum
                                                                                      << " uvscale=" << uv_scale
                                                                                      << " mapped=" << mapped
                                                                                      << " threshold=" << threshold
                                                                                      << " tintalpha=" << tint_alpha
                                                                                      << " fog=" << fog
                                                                                      << " pixel=" << p
                                                                                      << " actual=" << got
                                                                                      << " expected=" << want << '\n';
                                                                    };
                                                                    for (unsigned c = 0; c < 4; ++c) {
                                                                        float tolerance = !survives || c == 3 || !linear
                                                                                              ? 0
                                                                                              : 1.f / 256;
                                                                        if (!std::isfinite(actual[p][c]) ||
                                                                            std::abs(actual[p][c] - expected[c]) >
                                                                                tolerance)
                                                                            mismatch("color", actual[p][c],
                                                                                     expected[c]);
                                                                        ++color_checks;
                                                                    }
                                                                    float z = reinterpret_cast<const float *>(
                                                                        static_cast<const uint8_t *>(
                                                                            depth_buffer.contents) +
                                                                        y * depth_pitch)[x];
                                                                    float want_z = survives ? .25f : .75f;
                                                                    if (z != want_z)
                                                                        mismatch("depth", z, want_z);
                                                                    ++depth_checks;
                                                                }
                                                            ++draws;
                                                        }
                                                    }
                        std::cout << "PROGRESS shape=" << npot << " scale=" << scale << " draws=" << draws
                                  << " failed=" << failures << std::endl;
                    }
                }
            check(kept && killed && equalities, "Insufficient alpha coverage");
            std::cout << "RESULT corrected=" << corrected << " draws=" << draws << " color_components=" << color_checks
                      << " depth_values=" << depth_checks << " kept=" << kept << " killed=" << killed
                      << " equalities=" << equalities << " failed=" << failures << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 2;
        }
    }
}
