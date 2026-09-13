// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <shader/uniform_block.h>
#include <stdexcept>

static void check(bool value, const std::string &message) {
    if (!value)
        throw std::runtime_error(message);
}

// The original Stealth 10bfcb39 shader blends four packed color bytes and
// copies the second tile word. Test the actual framebuffer-fetch interface,
// not a synthetic replacement for the guest instructions.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 2, "usage: metal-half-fetch-validation original-10bfcb39.gxp");
            std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
            check(bool(file), "Cannot open GXP");
            const auto size = file.tellg();
            check(size > 0 && size < 1024 * 1024, "Invalid GXP size");
            std::vector<uint32_t> words((size_t(size) + 3) / 4);
            file.seekg(0);
            check(bool(file.read(reinterpret_cast<char *>(words.data()), size)), "Cannot read GXP");
            auto program = compile_gxp_fixture(words, size, "half-fetch", "gxp-mapped-half");
            check(program.uses_framebuffer_fetch && !program.writes_guest_memory, "Expected framebuffer reader");
            check(program.source.find("half4 last_frag_data") != std::string::npos,
                  "Framebuffer fetch must retain half precision");
            auto device = MTLCreateSystemDefaultDevice();
            check(device != nil, "No Metal device");
            auto queue = [device newCommandQueue];
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *error = nil;
            auto library = [device newLibraryWithSource:[NSString stringWithUTF8String:program.source.c_str()]
                                                options:options
                                                  error:&error];
            check(library != nil, error ? error.localizedDescription.UTF8String : "Fragment compile failed");
            NSString *vertex =
                @"#include <metal_stdlib>\nusing namespace metal;"
                @"struct O {float4 p [[position]];float4 color [[user(locn1)]];float4 uv [[user(locn4)]];};"
                @"vertex O v(uint i [[vertex_id]]){float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};"
                @"O o;o.p=float4(p[i],0,1);o.color=float4(1);o.uv=float4(p[i].x*.5+.5,.5-p[i].y*.5,0,0);return o;}";
            auto vertex_library = [device newLibraryWithSource:vertex options:options error:&error];
            check(vertex_library != nil, error ? error.localizedDescription.UTF8String : "Vertex compile failed");
            auto descriptor = [MTLRenderPipelineDescriptor new];
            descriptor.vertexFunction = [vertex_library newFunctionWithName:@"v"];
            descriptor.fragmentFunction = [library newFunctionWithName:@"main_fs"];
            descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
            auto pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
            check(pipeline != nil, error ? error.localizedDescription.UTF8String : "Pipeline failed");
            constexpr unsigned width = 256, height = 256, pixels = width * height;
            auto texture = [&](MTLPixelFormat format) {
                auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                             width:width
                                                                            height:height
                                                                         mipmapped:NO];
                td.storageMode = MTLStorageModeShared;
                td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
                return [device newTextureWithDescriptor:td];
            };
            auto target = texture(MTLPixelFormatRGBA16Float);
            auto input = texture(MTLPixelFormatRGBA8Unorm);
            auto mask = texture(MTLPixelFormatR8Unorm);
            std::vector<uint8_t> mask_bytes(pixels, 255);
            [mask replaceRegion:MTLRegionMake2D(0, 0, width, height)
                    mipmapLevel:0
                      withBytes:mask_bytes.data()
                    bytesPerRow:width];
            auto sd = [MTLSamplerDescriptor new];
            sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
            auto sampler = [device newSamplerStateWithDescriptor:sd];
            shader::RenderFragUniformBlockExtended info{};
            info.base_block.res_multiplier = 1;
            info.set_texture_count(1);
            info.set_viewport_ratio(0, {1, 1});
            std::vector<uint8_t> info_bytes((info.get_size() + 15) & ~size_t(15));
            info.copy_to(info_bytes.data());
            size_t failures = 0, comparisons = 0;
            for (unsigned alpha : {0u, 255u, 127u}) {
                std::vector<uint16_t> previous(pixels * 4);
                std::vector<uint8_t> samples(pixels * 4), actual(pixels * 8);
                for (unsigned i = 0; i < pixels; ++i) {
                    previous[i * 4] = uint16_t(i);
                    previous[i * 4 + 1] = uint16_t(i + 0x1234);
                    previous[i * 4 + 2] = uint16_t(i ^ 0xa5a5);
                    previous[i * 4 + 3] = uint16_t(i ^ 0x5a5a);
                    for (unsigned c = 0; c < 3; ++c)
                        samples[i * 4 + c] = uint8_t(i * (c * 7 + 3) + c * 31);
                    samples[i * 4 + 3] = alpha;
                }
                [target replaceRegion:MTLRegionMake2D(0, 0, width, height)
                          mipmapLevel:0
                            withBytes:previous.data()
                          bytesPerRow:width * 8];
                [input replaceRegion:MTLRegionMake2D(0, 0, width, height)
                         mipmapLevel:0
                           withBytes:samples.data()
                         bytesPerRow:width * 4];
                auto commands = [queue commandBuffer];
                auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = target;
                pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
                [encoder setRenderPipelineState:pipeline];
                [encoder setFragmentBytes:info_bytes.data() length:info_bytes.size() atIndex:0];
                [encoder setFragmentTexture:input atIndex:0];
                [encoder setFragmentTexture:mask atIndex:17];
                [encoder setFragmentSamplerState:sampler atIndex:0];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                [encoder endEncoding];
                [commands commit];
                [commands waitUntilCompleted];
                check(commands.status == MTLCommandBufferStatusCompleted,
                      commands.error ? commands.error.localizedDescription.UTF8String : "GPU failed");
                [target getBytes:actual.data()
                     bytesPerRow:width * 8
                      fromRegion:MTLRegionMake2D(0, 0, width, height)
                     mipmapLevel:0];
                const auto *before = reinterpret_cast<const uint8_t *>(previous.data());
                for (unsigned i = 0; i < pixels; ++i)
                    for (unsigned c = 0; c < 8; ++c) {
                        const unsigned expected =
                            c >= 4 ? before[i * 8 + c]
                                   : unsigned(std::round(
                                         (samples[i * 4 + c] * alpha + before[i * 8 + c] * (255u - alpha)) / 255.f));
                        const unsigned tolerance = c < 4 && alpha == 127 ? 1 : 0;
                        ++comparisons;
                        if (std::abs(int(actual[i * 8 + c]) - int(expected)) <= int(tolerance))
                            continue;
                        if (failures++ < 8)
                            std::cerr << "Mismatch alpha=" << alpha << " pixel=" << i << " byte=" << c
                                      << " actual=" << unsigned(actual[i * 8 + c]) << " expected=" << expected << '\n';
                    }
            }
            std::cout << "RESULT draws=3 pixels=" << pixels * 3 << " compared_bytes=" << comparisons
                      << " failed=" << failures << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &error) {
            std::cerr << "FAIL " << error.what() << '\n';
            return 1;
        }
    }
}
