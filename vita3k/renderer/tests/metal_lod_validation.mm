// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <shader/uniform_block.h>
#include <stdexcept>
using Pixel = std::array<float, 4>;
static void check(bool value, const std::string &message) {
    if (!value)
        throw std::runtime_error(message);
}
static float halfround(float value) { return float((__fp16)value); }

// Original Stealth texture/tint fragment. Every mip has a unique constant
// color, so the CPU oracle checks level selection independently of UV filtering.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 2, "usage: metal-lod-validation original-3b6ed069.gxp");
            std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
            check(bool(file), "Cannot open GXP");
            auto length = file.tellg();
            check(length > 0 && length < 1024 * 1024, "Invalid GXP");
            std::vector<uint32_t> words((size_t(length) + 3) / 4);
            file.seekg(0);
            check(bool(file.read(reinterpret_cast<char *>(words.data()), length)), "Cannot read GXP");
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            auto native = device->native_device();
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *nsError = nil;
            auto vertex = [native
                newLibraryWithSource:
                    @"#include <metal_stdlib>\nusing namespace metal;struct O{float4 p [[position]];float4 c "
                    @"[[user(locn1)]];float4 uv [[user(locn4)]];};vertex O v(uint i [[vertex_id]]){float2 "
                    @"p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};O "
                    @"o;o.p=float4(p[i],0,1);o.c=float4(1);o.uv=float4(p[i].x*.5+.5,.5-p[i].y*.5,0,0);return o;}"
                             options:options
                               error:&nsError];
            check(vertex != nil, nsError ? nsError.localizedDescription.UTF8String : "Vertex failed");
            auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                         width:64
                                                                        height:64
                                                                     mipmapped:YES];
            td.storageMode = MTLStorageModeShared;
            td.usage = MTLTextureUsageShaderRead;
            auto source = [native newTextureWithDescriptor:td];
            check(source != nil, "Texture failed");
            for (int level = 0; level < 7; ++level) {
                const int size = 64 >> level;
                std::vector<Pixel> values(size * size, Pixel{level / 8.f, (6 - level) / 8.f, .25f, 1});
                [source replaceRegion:MTLRegionMake2D(0, 0, size, size)
                          mipmapLevel:level
                            withBytes:values.data()
                          bytesPerRow:size * sizeof(Pixel)];
            }
            td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                    width:8
                                                                   height:8
                                                                mipmapped:NO];
            td.storageMode = MTLStorageModeShared;
            td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            auto target = [native newTextureWithDescriptor:td], mask = [native newTextureWithDescriptor:td];
            std::array<Pixel, 64> white;
            white.fill({1, 1, 1, 1});
            [mask replaceRegion:MTLRegionMake2D(0, 0, 8, 8)
                    mipmapLevel:0
                      withBytes:white.data()
                    bytesPerRow:8 * sizeof(Pixel)];
            shader::RenderFragUniformBlockExtended info{};
            info.set_texture_count(1);
            info.set_viewport_ratio(0, {1, 1});
            info.base_block.res_multiplier = 1;
            std::vector<uint8_t> bytes((info.get_size() + 15) & ~size_t(15));
            info.copy_to(bytes.data());
            int failures = 0, draws = 0, components = 0;
            for (unsigned encodedBias = 0; encodedBias < 64; ++encodedBias) {
                SceGxmTexture texture{};
                texture.type = SCE_GXM_TEXTURE_SWIZZLED >> 29;
                texture.lod_bias = encodedBias;
                texture.mip_count = 6;
                texture.min_filter = texture.mag_filter = SCE_GXM_TEXTURE_FILTER_LINEAR;
                texture.uaddr_mode = texture.vaddr_mode = SCE_GXM_TEXTURE_ADDR_CLAMP;
                const float bias = renderer::metal::sampler_lod_bias(texture);
                check(bias == (float(encodedBias) - 31.f) / 8.f, "Encoded bias mismatch");
                auto program = compile_gxp_fixture(words, length, "stealth-original-lod", "gxp-mapped-viewport", bias);
                auto fragment = [native newLibraryWithSource:[NSString stringWithUTF8String:program.source.c_str()]
                                                     options:options
                                                       error:&nsError];
                check(fragment != nil, nsError ? nsError.localizedDescription.UTF8String : "Fragment failed");
                auto pd = [MTLRenderPipelineDescriptor new];
                pd.vertexFunction = [vertex newFunctionWithName:@"v"];
                pd.fragmentFunction = [fragment newFunctionWithName:@"main_fs"];
                pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
                auto pipeline = [native newRenderPipelineStateWithDescriptor:pd error:&nsError];
                check(pipeline != nil, nsError ? nsError.localizedDescription.UTF8String : "Pipeline failed");
                for (unsigned linear = 0; linear < 2; ++linear)
                    for (unsigned minimum : {0u, 2u, 5u, 15u}) {
                        texture.mip_filter = linear;
                        texture.lod_min0 = minimum & 3;
                        texture.lod_min1 = minimum >> 2;
                        auto sampler = renderer::metal::make_sampler(*device, texture, 1);
                        auto commands = [device->command_queue() commandBuffer];
                        auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                        pass.colorAttachments[0].texture = target;
                        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                        auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
                        [encoder setRenderPipelineState:pipeline];
                        [encoder setFragmentBytes:bytes.data() length:bytes.size() atIndex:0];
                        [encoder setFragmentTexture:source atIndex:0];
                        [encoder setFragmentTexture:mask atIndex:17];
                        [encoder setFragmentSamplerState:sampler atIndex:0];
                        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        [encoder endEncoding];
                        check(device->submit_and_wait(commands, error), error);
                        ++draws;
                        std::array<Pixel, 64> actual;
                        [target getBytes:actual.data()
                             bytesPerRow:8 * sizeof(Pixel)
                              fromRegion:MTLRegionMake2D(0, 0, 8, 8)
                             mipmapLevel:0];
                        // 64 texels across an 8-pixel output give exactly LOD3.
                        const float lod =
                            std::clamp(std::max(3.f + (float(encodedBias) - 31.f) / 8.f, float(minimum)), 0.f, 6.f);
                        const bool tie = !linear && lod - std::floor(lod) == .5f;
                        const float selected = linear ? lod : std::floor(lod + .5f),
                                    other = tie ? std::floor(lod) : selected;
                        Pixel expected = {halfround(selected / 8), halfround((6 - selected) / 8), .25f, 1};
                        Pixel alternative = {halfround(other / 8), halfround((6 - other) / 8), .25f, 1};
                        for (int p = 0; p < 64; ++p) {
                            bool match = true, alternate = true;
                            for (int c = 0; c < 4; ++c) {
                                ++components;
                                match &= std::isfinite(actual[p][c]) && std::abs(actual[p][c] - expected[c]) <= .001f;
                                alternate &=
                                    std::isfinite(actual[p][c]) && std::abs(actual[p][c] - alternative[c]) <= .001f;
                            }
                            if (!match && !alternate) {
                                if (failures++ < 12)
                                    std::cerr << "Mismatch bias=" << bias << " min=" << minimum << " linear=" << linear
                                              << " pixel=" << p << " actual=" << actual[p][0]
                                              << " expected=" << expected[0] << '\n';
                            }
                        }
                    }
            }
            // Strided descriptors reuse these bits as stride; they must not
            // produce a shader bias even for non-neutral raw bit patterns.
            SceGxmTexture strided{};
            strided.type = SCE_GXM_TEXTURE_LINEAR_STRIDED >> 29;
            for (unsigned bits = 0; bits < 64; ++bits) {
                strided.lod_bias = bits;
                check(renderer::metal::sampler_lod_bias(strided) == 0, "Strided bias leaked");
            }
            std::cout << "RESULT draws=" << draws << " components=" << components << " failed_pixels=" << failures
                      << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
