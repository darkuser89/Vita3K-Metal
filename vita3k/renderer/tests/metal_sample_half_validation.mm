// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#include <algorithm>
#include <bit>
#include <limits>
#include <shader/metal_texture.h>
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

// Original Stealth texture/tint fragment, tint fixed to one. The CPU oracle
// uses the platform half conversion, independent of the shader integer logic.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 3, "usage: metal-sample-half-validation original-3b6ed069.gxp output.metal");
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
            // Exercise every finite adjacent half pair at the midpoint and its
            // two neighboring float32 values, using the unchanged game shader.
            std::vector<float> values;
            for (uint16_t bits = 0; bits < 0x7bff; ++bits) {
                __fp16 a = std::bit_cast<__fp16>(bits), b = std::bit_cast<__fp16>(uint16_t(bits + 1));
                float midpoint = (float(a) + float(b)) * .5f;
                for (float x : {std::nextafter(midpoint, -INFINITY), midpoint, std::nextafter(midpoint, INFINITY)}) {
                    values.push_back(x);
                    values.push_back(-x);
                }
            }
            for (float x : {0.f, -0.f, 65504.f, 65519.f, 65520.f, 65536.f, INFINITY, -INFINITY,
                            std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::quiet_NaN()}) {
                values.push_back(x);
                values.push_back(-x);
            }
            auto input_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                 width:8
                                                                                height:8
                                                                             mipmapped:NO];
            input_desc.storageMode = MTLStorageModeShared;
            input_desc.usage = MTLTextureUsageShaderRead;
            auto source = [native newTextureWithDescriptor:input_desc];
            check(source != nil, "Input allocation failed");
            SceGxmTexture texture{};
            texture.type = SCE_GXM_TEXTURE_LINEAR >> 29;
            texture.lod_bias = 31;
            texture.uaddr_mode = texture.vaddr_mode = SCE_GXM_TEXTURE_ADDR_CLAMP;
            auto sampler = renderer::metal::make_sampler(*device, texture, 1);
            auto program = compile_gxp_fixture(words, length, "stealth-original-half-rounding", "gxp-mapped-mips");
            std::ofstream saved(argv[2]);
            saved << program.source;
            saved.close();
            check(bool(saved), "Cannot save MSL");
            auto fragment = device->compile(program, false, error);
            check(bool(fragment), error);
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [vertex newFunctionWithName:@"v"];
            pd.fragmentFunction = fragment->function;
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
            auto pipeline = device->create_pipeline(pd, error);
            check(pipeline != nil, error);
            unsigned failed = 0, draws = 0, components = 0;
            for (unsigned corrected : {0u, 1u}) {
                shader::metal::TextureMipInfos mip_info{};
                mip_info[0].control = {corrected, (2u << 3) | (2u << 6), 1, 0};
                mip_info[0].sizes[0] = {8, 8};
                for (size_t start = 0; start < values.size(); start += 256) {
                    @autoreleasepool {
                        std::array<Pixel, 64> input{};
                        for (size_t k = 0; k < 256 && start + k < values.size(); ++k)
                            input[k / 4][k % 4] = values[start + k];
                        [source replaceRegion:MTLRegionMake2D(0, 0, 8, 8)
                                  mipmapLevel:0
                                    withBytes:input.data()
                                  bytesPerRow:8 * sizeof(Pixel)];
                        auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                        pass.colorAttachments[0].texture = target;
                        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                        auto commands = [device->command_queue() commandBuffer];
                        auto enc = [commands renderCommandEncoderWithDescriptor:pass];
                        [enc setRenderPipelineState:pipeline];
                        [enc setFragmentBytes:bytes.data() length:bytes.size() atIndex:0];
                        [enc setFragmentBytes:mip_info.data()
                                       length:sizeof(mip_info)
                                      atIndex:shader::metal::TEXTURE_INFO_BUFFER];
                        [enc setFragmentTexture:source atIndex:0];
                        [enc setFragmentTexture:mask atIndex:17];
                        [enc setFragmentSamplerState:sampler atIndex:0];
                        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        [enc endEncoding];
                        check(device->submit_and_wait(commands, error), error);
                        std::array<Pixel, 64> actual;
                        [target getBytes:actual.data()
                             bytesPerRow:8 * sizeof(Pixel)
                              fromRegion:MTLRegionMake2D(0, 0, 8, 8)
                             mipmapLevel:0];
                        for (size_t k = 0; k < 256 && start + k < values.size(); ++k) {
                            float expected = halfround(values[start + k]), got = actual[k / 4][k % 4];
                            bool match = std::isnan(expected)
                                             ? std::isnan(got)
                                             : std::bit_cast<uint32_t>(expected) == std::bit_cast<uint32_t>(got);
                            if (!match && failed++ < 12)
                                std::cerr << "Mismatch corrected=" << corrected << " index=" << start + k
                                          << " input=" << values[start + k] << " got=" << got
                                          << " expected=" << expected << "\n";
                            ++components;
                        }
                        ++draws;
                    }
                }
            }
            std::cout << "RESULT draws=" << draws << " components=" << components << " failures=" << failed << "\n";
            return failed ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << "\n";
            return 2;
        }
    }
}
