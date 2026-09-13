// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <shader/uniform_block.h>
#include <stdexcept>
using Pixel = std::array<uint8_t, 4>;
static void check(bool value, const std::string &s) {
    if (!value)
        throw std::runtime_error(s);
}
static float halfround(float x) { return float((__fp16)x); }
static int byte(float x) { return int(std::round(std::clamp(x, 0.f, 1.f) * 255.f)); }
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 2, "usage: metal-stealth-validation GXP-directory");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            check(device != nil, "No Metal device");
            auto queue = [device newCommandQueue];
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *err = nil;
            NSString *vsSource =
                @"#include <metal_stdlib>\nusing namespace metal; struct O{float4 p [[position]];float4 c "
                @"[[user(locn1)]];float4 uv [[user(locn4)]];};vertex O v(uint i [[vertex_id]],constant float4 &c "
                @"[[buffer(0)]]){float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};O "
                @"o;o.p=float4(p[i],0,1);o.c=c;o.uv=float4(p[i].x*.5+.5,.5-p[i].y*.5,0,0);return o;}";
            auto vsLibrary = [device newLibraryWithSource:vsSource options:options error:&err];
            check(vsLibrary != nil, err ? err.localizedDescription.UTF8String : "vertex");
            auto tex = [&](MTLPixelFormat format = MTLPixelFormatRGBA8Unorm) {
                auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:8 height:8 mipmapped:NO];
                d.storageMode = MTLStorageModeShared;
                d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
                return [device newTextureWithDescriptor:d];
            };
            auto source = tex(), secondary = tex(), mask = tex();
            std::array<Pixel, 64> white;
            white.fill({255, 255, 255, 255});
            [mask replaceRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0 withBytes:white.data() bytesPerRow:32];
            auto sd = [MTLSamplerDescriptor new];
            sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
            auto sampler = [device newSamplerStateWithDescriptor:sd];
            // Original Stealth Inc. programs: texture*tint, YUV video conversion,
            // source-alpha blending, and destination-modulated blending. The CPU
            // oracle uses the color equations, independent of the generated MSL.
            // The last program is also tested with both tile register widths:
            // its raw uniform payload is U8 for 32-bit, F16 for 64-bit tiles.
            const char *prefixes[] = {"3b6ed0699c3b", "1b4ea77bce5c", "10bfcb3976d1", "692579507865",
                                      "72c63c37adb0", "72c63c37adb0", "e860d1b8927d", "90e43f99eee7",
                                      "bedcd9b98047", "e860d1b8927d", "90e43f99eee7"};
            int failed = 0, draws = 0;
            for (int kind = 0; kind < 11; ++kind) {
                const bool nativeUniform = kind == 4 || kind == 5;
                const bool paired = kind >= 9;
                const bool twoTextures = kind == 6 || kind == 7 || paired;
                const auto targetFormat = paired ? MTLPixelFormatRG32Float : MTLPixelFormatRGBA8Unorm;
                auto target = tex(targetFormat);
                std::filesystem::path file;
                for (auto &f : std::filesystem::directory_iterator(argv[1]))
                    if (f.path().extension() == ".gxp" &&
                        f.path().filename().string().find(prefixes[kind]) != std::string::npos)
                        file = f.path();
                check(!file.empty(), "shader missing");
                std::ifstream in(file, std::ios::binary | std::ios::ate);
                const auto size = in.tellg();
                check(size > 0 && size < 1024 * 1024, "Invalid GXP size");
                std::vector<uint32_t> words((size_t(size) + 3) / 4);
                in.seekg(0);
                check(bool(in.read(reinterpret_cast<char *>(words.data()), size)), "GXP read failed");
                auto program = compile_gxp_fixture(words, size, file.stem().string(),
                                                   paired      ? "gxp-mapped-gbuffer"
                                                   : kind == 5 ? "gxp-native64"
                                                               : "gxp-mapped-viewport");
                check(!program.writes_guest_memory, "Unexpected guest memory writer");
                auto str = program.source;
                auto lib = [device newLibraryWithSource:[NSString stringWithUTF8String:str.c_str()]
                                                options:options
                                                  error:&err];
                check(lib != nil, err ? err.localizedDescription.UTF8String : "fragment");
                auto pd = [MTLRenderPipelineDescriptor new];
                pd.vertexFunction = [vsLibrary newFunctionWithName:@"v"];
                pd.fragmentFunction = [lib newFunctionWithName:@"main_fs"];
                pd.colorAttachments[0].pixelFormat = targetFormat;
                auto pipeline = [device newRenderPipelineStateWithDescriptor:pd error:&err];
                check(pipeline != nil, err ? err.localizedDescription.UTF8String : "pipeline");
                int shaderFailed = 0;
                for (int pass = 0; pass < 4; ++pass) {
                    std::array<Pixel, 64> inputs, auxiliary, previous, actual, expected;
                    std::array<Pixel, 64> previousAux, expectedAux;
                    std::array<uint32_t, 128> storedPairs{}, actualPairs{};
                    float tint[4] = {pass == 1 ? .5f : 1.f, pass == 2 ? .25f : 1.f, pass == 3 ? .75f : 1.f, 1.f};
                    for (int i = 0; i < 64; ++i) {
                        inputs[i] = {uint8_t((i * 37 + pass * 17) % 256), uint8_t((i * 71 + 31) % 256),
                                     uint8_t((i * 19 + 73) % 256),
                                     uint8_t(pass == 0   ? 255
                                             : pass == 1 ? 0
                                                         : (i * 29) % 256)};
                        previous[i] = {uint8_t(i * 13), uint8_t(i * 43), uint8_t(i * 7), uint8_t(255 - i * 3)};
                        auxiliary[i] = {uint8_t(255 - i * 3), uint8_t(i * 7 + pass * 13), uint8_t(191 - i * 2),
                                        uint8_t(i * 41 + pass * 29)};
                        previousAux[i] = {uint8_t(pass * 21 + i * 11), uint8_t(255 - i * 3), uint8_t(i * 13),
                                          uint8_t(63 + i * 7)};
                        // The packed-alpha program encodes a four-bit opacity
                        // plus a fractional attribute. Keep opacity in [0,15];
                        // alpha=1 produces an out-of-range unscaled U8 pack.
                        if (kind == 8 && inputs[i][3] == 255)
                            inputs[i][3] = 254;
                        float rgba[4];
                        for (int c = 0; c < 4; ++c)
                            rgba[c] = halfround(halfround(inputs[i][c] / 255.f) * halfround(tint[c]));
                        if (kind == 1) {
                            float y = inputs[i][0] / 255.f, u = inputs[i][1] / 255.f, v = inputs[i][2] / 255.f;
                            float l = std::fma(y, 1.1643f, -.07276875f);
                            rgba[0] = halfround(l + std::fma(v, 1.5958f, -.7979f));
                            rgba[1] = halfround(std::fma(v - .5f, -.8129f, std::fma(u - .5f, -.39173f, l)));
                            rgba[2] = halfround(l + std::fma(u, 2.017f, -1.0085f));
                            rgba[3] = 1;
                        }
                        for (int c = 0; c < 4; ++c)
                            expected[i][c] = byte(rgba[c]);
                        if (kind == 2 || kind == 3 || twoTextures) {
                            const auto src = expected[i];
                            float a = src[3] / 255.f;
                            for (int c = 0; c < 4; ++c) {
                                float d = previous[i][c] / 255.f, s = src[c] / 255.f;
                                expected[i][c] = byte(kind == 3 && c < 3 ? d * s + a * d : a * s + (1 - a) * d);
                            }
                        }
                        if (kind == 8) {
                            // From the original shader's packed-alpha encoding:
                            // high four bits drive RGB blending; the decimal
                            // digit in the fractional remainder drives alpha.
                            const float encoded = float(inputs[i][3]) / 255.f * 16.f;
                            const float opacity = std::floor(encoded);
                            const float attribute = std::floor((encoded - opacity) * 10.f) / 10.f;
                            const float weight = opacity * 16.f / 255.f;
                            for (int c = 0; c < 3; ++c)
                                expected[i][c] =
                                    byte(weight * (inputs[i][c] / 255.f) + (1 - weight) * (previous[i][c] / 255.f));
                            const float backgroundAlpha = previous[i][3] / 255.f;
                            expected[i][3] = byte(backgroundAlpha + (attribute - backgroundAlpha) * (opacity / 16.f));
                        }
                        if (paired) {
                            std::memcpy(&storedPairs[i * 2], previous[i].data(), 4);
                            std::memcpy(&storedPairs[i * 2 + 1], previousAux[i].data(), 4);
                            expectedAux[i] = previousAux[i];
                            if (kind == 9) {
                                // Emission strength updates only the second
                                // word's alpha; the other bytes survive.
                                const float emission =
                                    halfround(halfround(tint[3]) * halfround(auxiliary[i][0] / 255.f));
                                const float delta = halfround(std::fma(emission, 255.f, -float(previousAux[i][3])));
                                const float result = halfround(std::fma(delta, rgba[3], float(previousAux[i][3])));
                                expectedAux[i][3] = uint8_t(std::clamp(int(result), 0, 255));
                            } else {
                                // Normal RGB is blended independently of diffuse
                                // color, then its alpha is cleared.
                                const float normalAlpha = auxiliary[i][3] / 255.f;
                                for (int c = 0; c < 3; ++c)
                                    expectedAux[i][c] = byte(normalAlpha * (auxiliary[i][c] / 255.f) +
                                                             (1 - normalAlpha) * (previousAux[i][c] / 255.f));
                                expectedAux[i][3] = 0;
                            }
                        }
                    }
                    id<MTLBuffer> colorBuffer = nil;
                    std::vector<uint8_t> renderInfo;
                    shader::RenderFragUniformBlockExtended info{};
                    info.base_block.res_multiplier = 1;
                    if (nativeUniform) {
                        Pixel color{uint8_t(pass * 53), uint8_t(31 + pass * 29), uint8_t(73 + pass * 17),
                                    uint8_t(255 - pass * 41)};
                        std::array<uint8_t, 8> payload{};
                        if (kind == 4) {
                            std::copy(color.begin(), color.end(), payload.begin());
                        } else {
                            std::array<__fp16, 4> values;
                            for (int c = 0; c < 4; ++c)
                                values[c] = color[c] / 255.f;
                            std::memcpy(payload.data(), values.data(), payload.size());
                        }
                        colorBuffer = [device newBufferWithBytes:payload.data()
                                                          length:payload.size()
                                                         options:MTLResourceStorageModeShared];
                        info.set_buffer_count(1);
                        info.set_buffer_address(0, colorBuffer.gpuAddress);
                        expected.fill(color);
                    } else {
                        info.set_texture_count(twoTextures ? 2 : 1);
                        for (unsigned slot = 0; slot < info.texture_count; ++slot)
                            info.set_viewport_ratio(slot, {1, 1});
                    }
                    renderInfo.resize(info.get_size());
                    info.copy_to(renderInfo.data());
                    [source replaceRegion:MTLRegionMake2D(0, 0, 8, 8)
                              mipmapLevel:0
                                withBytes:inputs.data()
                              bytesPerRow:32];
                    [target replaceRegion:MTLRegionMake2D(0, 0, 8, 8)
                              mipmapLevel:0
                                withBytes:paired ? static_cast<const void *>(storedPairs.data()) : previous.data()
                              bytesPerRow:paired ? 64 : 32];
                    [secondary replaceRegion:MTLRegionMake2D(0, 0, 8, 8)
                                 mipmapLevel:0
                                   withBytes:auxiliary.data()
                                 bytesPerRow:32];
                    auto cb = [queue commandBuffer];
                    auto rp = [MTLRenderPassDescriptor renderPassDescriptor];
                    rp.colorAttachments[0].texture = target;
                    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
                    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                    auto e = [cb renderCommandEncoderWithDescriptor:rp];
                    [e setRenderPipelineState:pipeline];
                    [e setVertexBytes:tint length:16 atIndex:0];
                    [e setFragmentBytes:renderInfo.data() length:renderInfo.size() atIndex:0];
                    if (nativeUniform) {
                        [e useResource:colorBuffer usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
                    }
                    [e setFragmentTexture:source atIndex:0];
                    [e setFragmentTexture:mask atIndex:17];
                    [e setFragmentSamplerState:sampler atIndex:0];
                    if (twoTextures) {
                        [e setFragmentTexture:secondary atIndex:1];
                        [e setFragmentSamplerState:sampler atIndex:1];
                    }
                    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                    [e endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    check(cb.status == MTLCommandBufferStatusCompleted,
                          cb.error ? cb.error.localizedDescription.UTF8String : "GPU failure");
                    if (paired) {
                        [target getBytes:actualPairs.data()
                             bytesPerRow:64
                              fromRegion:MTLRegionMake2D(0, 0, 8, 8)
                             mipmapLevel:0];
                        for (int i = 0; i < 64; ++i)
                            std::memcpy(actual[i].data(), &actualPairs[i * 2], 4);
                    } else {
                        [target getBytes:actual.data()
                             bytesPerRow:32
                              fromRegion:MTLRegionMake2D(0, 0, 8, 8)
                             mipmapLevel:0];
                    }
                    int bad = 0;
                    for (int i = 0; i < 64; ++i)
                        for (int c = 0; c < 4; ++c)
                            if (std::abs(int(actual[i][c]) - int(expected[i][c])) > 1) {
                                if (bad == 0)
                                    std::cout << "MISMATCH " << prefixes[kind] << " pass=" << pass << " pixel=" << i
                                              << " channel=" << c << " actual=" << int(actual[i][c])
                                              << " expected=" << int(expected[i][c]) << '\n';
                                ++bad;
                            }
                    if (paired)
                        for (int i = 0; i < 64; ++i) {
                            Pixel aux;
                            std::memcpy(aux.data(), &actualPairs[i * 2 + 1], 4);
                            for (int c = 0; c < 4; ++c)
                                if (std::abs(int(aux[c]) - int(expectedAux[i][c])) > 1) {
                                    if (bad == 0)
                                        std::cout << "MISMATCH auxiliary " << prefixes[kind] << " pass=" << pass
                                                  << " pixel=" << i << " channel=" << c << " actual=" << int(aux[c])
                                                  << " expected=" << int(expectedAux[i][c]) << '\n';
                                    ++bad;
                                }
                        }
                    shaderFailed += bad;
                    ++draws;
                }
                std::cout << (shaderFailed ? "FAIL " : "PASS ") << prefixes[kind]
                          << " register_bits=" << (kind == 5 || paired ? 64 : 32) << " paired=" << paired
                          << " mismatched_channels=" << shaderFailed << " /" << (paired ? 2048 : 1024) << '\n';
                failed += shaderFailed;
            }
            std::cout << "RESULT draws=" << draws << " mismatched_channels=" << failed << '\n';
            return failed ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            return 2;
        }
    }
}
