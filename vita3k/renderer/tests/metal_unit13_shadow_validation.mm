// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <shader/uniform_block.h>
#include <shader/metal_texture.h>
#include <stdexcept>

static void check(bool ok, const std::string &why) {
    if (!ok)
        throw std::runtime_error(why);
}
static float h(float x) { return float((__fp16)x); }
using V4 = std::array<float, 4>;

// Original Unit 13 0e2b773d cascade-selection and gather shader. The CPU
// reference follows its projection, split distances, PCF and F16 fade math.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 3 || argc == 5 || argc == 6,
                  "usage: metal-unit13-shadow-validation original-0e2b773d.gxp output.metal "
                  "[shadow-side address-mode(0=clamp,1=repeat,2=mirror) [reconstruction-scale(1..3)]]");
            const int shadow_side = argc >= 5 ? std::stoi(argv[3]) : 16;
            const int address_mode = argc >= 5 ? std::stoi(argv[4]) : 0;
            check(shadow_side > 0 && shadow_side <= 64 && address_mode >= 0 && address_mode <= 2,
                  "Invalid texture parameters");
            const unsigned reconstruction_scale = argc == 6 ? std::stoul(argv[5]) : 0;
            check(argc != 6 || (reconstruction_scale >= 1 && reconstruction_scale <= 3),
                  "Invalid reconstruction scale");
            std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
            check(bool(file), "Cannot open GXP");
            auto size = file.tellg();
            check(size > 0 && size < 1024 * 1024, "Invalid GXP size");
            std::vector<uint32_t> words((size_t(size) + 3) / 4);
            file.seekg(0);
            check(bool(file.read(reinterpret_cast<char *>(words.data()), size)), "Cannot read GXP");
            auto program = compile_gxp_fixture(words, size, "unit13-shadow-original",
                                               reconstruction_scale ? "gxp-mapped-unit13-shadow-mips"
                                                                    : "gxp-mapped-unit13-shadow");
            check(program.stage == shader::metal::Stage::Fragment && !program.writes_guest_memory, "Unexpected shader");
            std::ofstream output(argv[2]);
            output << program.source;
            check(bool(output), "Cannot save generated MSL");
            output.close();
            std::string device_error;
            auto runtime = renderer::metal::Device::create(device_error);
            check(bool(runtime), device_error);
            renderer::metal::SurfaceCaster caster(*runtime);
            auto device = runtime->native_device();
            check(device != nil, "No Metal device");
            auto queue = [device newCommandQueue];
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            auto library = [&](NSString *source) {
                NSError *error = nil;
                auto lib = [device newLibraryWithSource:source options:options error:&error];
                check(lib != nil, error ? error.localizedDescription.UTF8String : "Shader compilation failed");
                return lib;
            };
            auto fragment = library([NSString stringWithUTF8String:program.source.c_str()]);
            auto vertex = library(
                @"#include <metal_stdlib>\nusing namespace metal;struct O{float4 p [[position]];"
                 "float4 uv [[user(locn4)]];float4 ray [[user(locn5)]];};"
                 "vertex O v(uint i [[vertex_id]],constant float4 &phase [[buffer(0)]]){"
                 "float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};float2 uv=float2(p[i].x*.5+.5,.5-p[i].y*.5);"
                 "O o;o.p=float4(p[i],0,1);o.uv=float4(uv,0,0);o.ray=float4(uv+phase.xy,1,0);return o;}");
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [vertex newFunctionWithName:@"v"];
            pd.fragmentFunction = [fragment newFunctionWithName:@"main_fs"];
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
            NSError *error = nil;
            auto pipeline = [device newRenderPipelineStateWithDescriptor:pd error:&error];
            check(pipeline != nil, error ? error.localizedDescription.UTF8String : "Pipeline failed");
            auto texture = [&](MTLPixelFormat format, unsigned side, unsigned levels = 1) {
                auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                             width:side
                                                                            height:side
                                                                         mipmapped:NO];
                td.mipmapLevelCount = levels;
                td.storageMode = MTLStorageModeShared;
                td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
                return [device newTextureWithDescriptor:td];
            };
            auto depth = texture(MTLPixelFormatR32Float, 1),
                 shadow = texture(MTLPixelFormatR32Float, shadow_side, reconstruction_scale && shadow_side > 1 ? 2 : 1);
            auto target = texture(MTLPixelFormatR32Float, 8), mask = texture(MTLPixelFormatRGBA8Unorm, 8);
            std::array<uint8_t, 8 * 8 * 4> maskbytes;
            maskbytes.fill(255);
            [mask replaceRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0 withBytes:maskbytes.data() bytesPerRow:32];
            std::vector<float> depths(shadow_side * shadow_side);
            for (int y = 0; y < shadow_side; y++)
                for (int x = 0; x < shadow_side; x++)
                    depths[y * shadow_side + x] = ((x * 13 + y * 7 + x * y) % 5 < 3) ? .75f : .25f;
            [shadow replaceRegion:MTLRegionMake2D(0, 0, shadow_side, shadow_side)
                      mipmapLevel:0
                        withBytes:depths.data()
                      bytesPerRow:shadow_side * sizeof(float)];
            // Exercise the renderer's actual reconstruction, with an original
            // RAM base mip replicated into the larger native allocation.
            shader::metal::TextureMipInfos mip_info{};
            if (reconstruction_scale) {
                std::vector<renderer::metal::CubeSurface> surfaces;
                if (shadow.mipmapLevelCount > 1) {
                    const unsigned native_side = (shadow_side / 2) * reconstruction_scale;
                    auto rendered = texture(MTLPixelFormatR32Float, native_side);
                    std::vector<float> native_values(native_side * native_side, .875f);
                    [rendered replaceRegion:MTLRegionMake2D(0, 0, native_side, native_side)
                                mipmapLevel:0
                                  withBytes:native_values.data()
                                bytesPerRow:native_side * sizeof(float)];
                    surfaces.push_back({rendered, 0, 1});
                    mip_info[11].sizes[1] = {native_side, native_side};
                }
                shadow = caster.texture_snapshot(shadow, surfaces, reconstruction_scale);
                const unsigned mode = address_mode == 0 ? 2 : address_mode == 1 ? 0 : 1;
                mip_info[11].control = {1, (mode << 3) | (mode << 6) | 3u, 1, 0};
                mip_info[11].sizes[0] = {unsigned(shadow_side), unsigned(shadow_side)};
            }
            auto sd = [MTLSamplerDescriptor new];
            sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
            sd.sAddressMode = sd.tAddressMode = address_mode == 0   ? MTLSamplerAddressModeClampToEdge
                                                : address_mode == 1 ? MTLSamplerAddressModeRepeat
                                                                    : MTLSamplerAddressModeMirrorRepeat;
            const auto address = [&](int i) {
                if (address_mode == 0)
                    return std::clamp(i, 0, shadow_side - 1);
                const int period = shadow_side * (address_mode == 2 ? 2 : 1);
                i = (i % period + period) % period;
                return i < shadow_side ? i : period - 1 - i;
            };
            auto sampler = [device newSamplerStateWithDescriptor:sd];
            auto uniform = [device newBufferWithLength:80 * 4 options:MTLResourceStorageModeShared];
            unsigned failures = 0, draws = 0, comparisons = 0;
            std::array<unsigned, 4> cascade_pixels{};
            for (unsigned projection = 0; projection < 2; projection++)
                for (unsigned cap : {0u, 1u, 3u})
                    for (unsigned depth_case = 0; depth_case < 3; depth_case++)
                        for (unsigned mapping = 0; mapping < 4; mapping++)
                            for (unsigned phase = 0; phase < 6; phase++)
                                for (unsigned fade = 0; fade < 4; fade++) {
                                    std::array<float, 80> u{};
                                    u[0] = 1.125f;
                                    u[1] = 1.25f;
                                    u[2] = 1.5f;
                                    u[3] = 2.f;
                                    u[4] = fade == 0 ? 0 : fade == 1 ? .125f : -.25f;
                                    u[5] = fade == 2 ? .5f : fade == 3 ? .125f : 0;
                                    u[7] = float(cap);
                                    if (projection) {
                                        u[8] = .125f;
                                        u[9] = -.25f;
                                        u[10] = .5f;
                                    }
                                    u[12] = 1;
                                    u[13] = 1.5f;
                                    for (unsigned cascade = 0; cascade < 4; cascade++) {
                                        const unsigned at = 14 + cascade * 16;
                                        u[at] = .5f;
                                        u[at + 5] = .5f;
                                        u[at + 10] = .5f;
                                        u[at + 12] = cascade / 16.f;
                                        u[at + 13] = (3 - cascade) / 16.f;
                                        u[at + 15] = 1;
                                        if (projection) {
                                            u[at + 3] = .125f;
                                            u[at + 7] = -.0625f;
                                            u[at + 11] = .25f;
                                        }
                                    }
                                    std::memcpy(uniform.contents, u.data(), sizeof(u));
                                    const float depth_value = depth_case / 8.f;
                                    [depth replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                                             mipmapLevel:0
                                               withBytes:&depth_value
                                             bytesPerRow:4];
                                    const V4 transform = phase < 4    ? V4{phase / 64.f, (3 - phase) / 64.f, 0, 0}
                                                         : phase == 4 ? V4{-.5f, -.5f, 0, 0}
                                                                      : V4{1, 1, 0, 0};
                                    const float rx = mapping == 0 ? 1 : .5f, ry = mapping < 2 ? 1 : .75f,
                                                ox = mapping < 2 ? 0 : .125f, oy = mapping == 3 ? .0625f : 0;
                                    shader::RenderFragUniformBlockExtended info{};
                                    info.base_block.res_multiplier = 1;
                                    info.set_buffer_count(1);
                                    info.set_texture_count(12);
                                    info.set_buffer_address(0, uniform.gpuAddress);
                                    for (unsigned i = 0; i < 12; i++)
                                        info.set_viewport_ratio(i, {1, 1});
                                    info.set_viewport_ratio(11, {rx, ry});
                                    info.set_viewport_offset(11, {ox, oy});
                                    std::vector<uint8_t> info_bytes((info.get_size() + 15) & ~size_t(15));
                                    info.copy_to(info_bytes.data());
                                    auto cb = [queue commandBuffer];
                                    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                                    pass.colorAttachments[0].texture = target;
                                    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                                    auto enc = [cb renderCommandEncoderWithDescriptor:pass];
                                    [enc setRenderPipelineState:pipeline];
                                    [enc setVertexBytes:transform.data() length:sizeof(transform) atIndex:0];
                                    [enc setFragmentBytes:info_bytes.data() length:info_bytes.size() atIndex:0];
                                    if (reconstruction_scale)
                                        [enc setFragmentBytes:mip_info.data()
                                                       length:sizeof(mip_info)
                                                      atIndex:shader::metal::TEXTURE_INFO_BUFFER];
                                    [enc useResource:uniform usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
                                    [enc setFragmentTexture:depth atIndex:10];
                                    [enc setFragmentTexture:shadow atIndex:11];
                                    [enc setFragmentTexture:mask atIndex:17];
                                    [enc setFragmentSamplerState:sampler atIndex:10];
                                    [enc setFragmentSamplerState:sampler atIndex:11];
                                    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                                    [enc endEncoding];
                                    [cb commit];
                                    [cb waitUntilCompleted];
                                    check(cb.status == MTLCommandBufferStatusCompleted,
                                          cb.error ? cb.error.localizedDescription.UTF8String : "GPU failure");
                                    std::array<float, 64> actual;
                                    [target getBytes:actual.data()
                                         bytesPerRow:32
                                          fromRegion:MTLRegionMake2D(0, 0, 8, 8)
                                         mipmapLevel:0];
                                    for (unsigned y = 0; y < 8; y++)
                                        for (unsigned x = 0; x < 8; x++) {
                                            const float d = 1 / std::fma(depth_value, -2.f, u[13]);
                                            const float wx = std::fma((x + .5f) / 8 + transform[0], d, u[8]),
                                                        wy = std::fma((y + .5f) / 8 + transform[1], d, u[9]),
                                                        wz = d + u[10];
                                            const float dx = wx - u[8], dy = wy - u[9], dz = wz - u[10];
                                            const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                                            unsigned cascade = 0;
                                            for (unsigned i = 0; i < 4; i++)
                                                cascade += distance > u[i];
                                            cascade = std::min(cascade, cap);
                                            ++cascade_pixels[cascade];
                                            const unsigned at = 14 + cascade * 16;
                                            const float qw =
                                                std::fma(wz, u[at + 11],
                                                         std::fma(wy, u[at + 7], std::fma(wx, u[at + 3], u[at + 15])));
                                            const float tu = std::fma(wx, u[at], u[at + 12]) / qw,
                                                        tv = std::fma(wy, u[at + 5], u[at + 13]) / qw,
                                                        tz = wz * .5f / qw;
                                            const float px = std::fma(tu, rx, ox) * shadow_side - .5f,
                                                        py = std::fma(tv, ry, oy) * shadow_side - .5f;
                                            const int ix = std::floor(px), iy = std::floor(py);
                                            const float fx = px - ix, fy = py - iy;
                                            float lit = 0;
                                            for (unsigned dy = 0; dy < 2; dy++)
                                                for (unsigned dx = 0; dx < 2; dx++) {
                                                    const int sx = address(ix + int(dx)), sy = address(iy + int(dy));
                                                    if (depths[sy * shadow_side + sx] > tz)
                                                        lit += h((dx ? fx : 1 - fx) * (dy ? fy : 1 - fy));
                                                }
                                            const float fade_value =
                                                std::max(h(std::fma(h(distance), h(u[4]), h(u[5]))), 0.f);
                                            const float expected = std::min(h(h(lit) + fade_value), 1.f);
                                            const float value = actual[y * 8 + x];
                                            ++comparisons;
                                            if (!std::isfinite(value) || std::abs(value - expected) > .0015f) {
                                                if (failures++ < 12)
                                                    std::cerr << "Mismatch draw=" << draws << " x=" << x << " y=" << y
                                                              << " cascade=" << cascade << " actual=" << value
                                                              << " expected=" << expected << '\n';
                                            }
                                        }
                                    ++draws;
                                }
            for (unsigned i = 0; i < 4; i++) {
                check(cascade_pixels[i] > 0, "Unvisited shadow cascade");
                std::cout << "CASCADE " << i << " pixels=" << cascade_pixels[i] << '\n';
            }
            std::cout << "RESULT reconstruction_scale=" << reconstruction_scale << " shadow_side=" << shadow_side
                      << " address_mode=" << address_mode << " draws=" << draws
                      << " compared_red_values=" << comparisons << " failed=" << failures << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
