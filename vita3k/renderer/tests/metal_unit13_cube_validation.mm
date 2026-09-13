// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <shader/uniform_block.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const std::string &why) {
    if (!ok)
        throw std::runtime_error(why);
}
static float h(float x) { return float((__fp16)x); }
static uint32_t halves(float x, float y) {
    return uint32_t(std::bit_cast<uint16_t>((__fp16)x)) | (uint32_t(std::bit_cast<uint16_t>((__fp16)y)) << 16);
}
using V3 = std::array<float, 3>;
using Pixel = std::array<uint8_t, 4>;
static Pixel color(unsigned face, unsigned mip, unsigned x, unsigned y) {
    return {uint8_t(16 + face * 32), uint8_t(x * 11 + mip * 19), uint8_t(y * 17 + mip * 23), 255};
}
// Original Unit 13 80e37b24 reflection/material shader. Constant varyings
// isolate cube orientation, sampler minimum LOD and packed material output.
// The cube has distinct texels on all six faces and seven mip levels.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc == 3, "usage: metal-unit13-cube-validation original-80e37b24.gxp output.metal");
            std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
            check(bool(input), "Cannot open GXP");
            auto length = input.tellg();
            check(length > 0 && length < 1024 * 1024, "Invalid GXP length");
            std::vector<uint32_t> words((size_t(length) + 3) / 4);
            input.seekg(0);
            check(bool(input.read(reinterpret_cast<char *>(words.data()), length)), "Cannot read GXP");
            auto program = compile_gxp_fixture(words, length, "unit13-cube-original", "gxp-mapped-unit13-cube");
            check(program.cube_texture_mask == 2 && !program.writes_guest_memory, "Unexpected cube reflection");
            std::ofstream saved(argv[2]);
            saved << program.source;
            saved.close();
            check(bool(saved), "Cannot save generated MSL");
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            auto native = device->native_device();
            auto fragment = device->compile(program, false, error);
            check(bool(fragment), error);
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *nsError = nil;
            auto vertex = [native newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;
struct O { float4 p [[position]]; float4 uv [[user(locn4)]]; float4 n [[user(locn5)]]; float4 v [[user(locn6)]]; };
vertex O v(uint i [[vertex_id]], constant float4 *inputs [[buffer(0)]]) {
    float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    O o;o.p=float4(p[i],0,1);o.uv=float4(.5,.5,0,0);o.n=inputs[0];o.v=inputs[1];return o;
})" options:options error:&nsError];
            check(vertex != nil, nsError ? nsError.localizedDescription.UTF8String : "Vertex compile failed");
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [vertex newFunctionWithName:@"v"];
            pd.fragmentFunction = fragment->function;
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatRG32Float;
            auto pipeline = device->create_pipeline(pd, error);
            check(pipeline != nil, error);
            auto td = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                            size:64
                                                                       mipmapped:YES];
            td.storageMode = MTLStorageModeShared;
            td.usage = MTLTextureUsageShaderRead;
            auto cube = [native newTextureWithDescriptor:td];
            check(cube != nil, "Cannot allocate cube");
            for (unsigned face = 0; face < 6; ++face)
                for (unsigned mip = 0; mip < 7; ++mip) {
                    unsigned size = 64 >> mip;
                    std::vector<Pixel> values(size * size);
                    for (unsigned y = 0; y < size; ++y)
                        for (unsigned x = 0; x < size; ++x)
                            values[y * size + x] = color(face, mip, x, y);
                    [cube replaceRegion:MTLRegionMake2D(0, 0, size, size)
                            mipmapLevel:mip
                                  slice:face
                              withBytes:values.data()
                            bytesPerRow:size * 4
                          bytesPerImage:size * size * 4];
                }
            auto texture = [&](MTLPixelFormat format, unsigned size) {
                auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                            width:size
                                                                           height:size
                                                                        mipmapped:NO];
                d.storageMode = MTLStorageModeShared;
                d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
                return [native newTextureWithDescriptor:d];
            };
            auto target = texture(MTLPixelFormatRG32Float, 8), mask = texture(MTLPixelFormatRGBA8Unorm, 8);
            auto base = texture(MTLPixelFormatRGBA8Unorm, 1);
            std::array<uint8_t, 256> white;
            white.fill(255);
            [mask replaceRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0 withBytes:white.data() bytesPerRow:32];
            auto uniform = [native newBufferWithLength:16 options:MTLResourceStorageModeShared];
            shader::RenderFragUniformBlockExtended info{};
            info.base_block.res_multiplier = 1;
            info.set_buffer_count(3);
            info.set_buffer_address(2, uniform.gpuAddress);
            info.set_texture_count(2);
            info.set_viewport_ratio(0, {1, 1});
            // Cube directions must ignore 2D surface viewport mapping.
            info.set_viewport_ratio(1, {.25f, .75f});
            info.set_viewport_offset(1, {.125f, .25f});
            std::vector<uint8_t> bytes((info.get_size() + 15) & ~size_t(15));
            info.copy_to(bytes.data());
            SceGxmTexture desc{};
            desc.type = SCE_GXM_TEXTURE_CUBE >> 29;
            desc.mip_count = 6;
            desc.lod_bias = 31;
            desc.min_filter = desc.mag_filter = SCE_GXM_TEXTURE_FILTER_POINT;
            desc.uaddr_mode = desc.vaddr_mode = SCE_GXM_TEXTURE_ADDR_CLAMP;
            auto base_sampler = renderer::metal::make_sampler(*device, desc, 1);
            const std::array<V3, 4> normals = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 2}}};
            const std::array<std::array<float, 2>, 3> coordinates = {{{.125f, .375f}, {.375f, .75f}, {.75f, .125f}}};
            unsigned draws = 0, components = 0, failures = 0;
            for (unsigned mip : {0u, 1u, 3u, 6u}) {
                desc.lod_min0 = mip & 3;
                desc.lod_min1 = mip >> 2;
                auto sampler = renderer::metal::make_sampler(*device, desc, 16);
                for (unsigned face = 0; face < 6; ++face)
                    for (auto uv : coordinates)
                        for (auto n : normals)
                            for (float weight : {0.f, .25f, .75f, 1.f})
                                for (uint8_t alpha : {uint8_t(0), uint8_t(127), uint8_t(255)}) {
                                    float s = 2 * uv[0] - 1, t = 2 * uv[1] - 1;
                                    const std::array<V3, 6> directions = {
                                        {{1, -t, -s}, {-1, -t, s}, {s, 1, t}, {s, -1, -t}, {s, -t, 1}, {-s, -t, -1}}};
                                    auto direction = directions[face];
                                    V3 normal = n;
                                    float scale = h(1 / std::sqrt(h(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])));
                                    for (auto &v : normal)
                                        v = h(v * scale);
                                    float dn =
                                        direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2];
                                    std::array<float, 8> varyings{n[0],
                                                                  n[1],
                                                                  n[2],
                                                                  0,
                                                                  2 * dn * normal[0] - direction[0],
                                                                  2 * dn * normal[1] - direction[1],
                                                                  2 * dn * normal[2] - direction[2],
                                                                  0};
                                    auto *u = static_cast<uint32_t *>(uniform.contents);
                                    u[0] = halves(1, 0);
                                    u[1] = 0;
                                    u[2] = halves(0, weight);
                                    u[3] = halves(1, 0);
                                    Pixel base_color{51, 102, 153, alpha};
                                    [base replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                                            mipmapLevel:0
                                              withBytes:base_color.data()
                                            bytesPerRow:4];
                                    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                                    pass.colorAttachments[0].texture = target;
                                    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                                    auto commands = [device->command_queue() commandBuffer];
                                    auto enc = [commands renderCommandEncoderWithDescriptor:pass];
                                    [enc setRenderPipelineState:pipeline];
                                    [enc setVertexBytes:varyings.data() length:sizeof(varyings) atIndex:0];
                                    [enc setFragmentBytes:bytes.data() length:bytes.size() atIndex:0];
                                    [enc useResource:uniform usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
                                    [enc setFragmentTexture:base atIndex:0];
                                    [enc setFragmentTexture:cube atIndex:1];
                                    [enc setFragmentTexture:mask atIndex:17];
                                    [enc setFragmentSamplerState:base_sampler atIndex:0];
                                    [enc setFragmentSamplerState:sampler atIndex:1];
                                    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                                    [enc endEncoding];
                                    check(device->submit_and_wait(commands, error), error);
                                    std::array<std::array<uint8_t, 8>, 64> actual{};
                                    [target getBytes:actual.data()
                                         bytesPerRow:64
                                          fromRegion:MTLRegionMake2D(0, 0, 8, 8)
                                         mipmapLevel:0];
                                    unsigned size = 64 >> mip;
                                    auto reflected = color(face, mip, unsigned(uv[0] * size), unsigned(uv[1] * size));
                                    std::array<uint8_t, 8> expected{};
                                    for (unsigned c = 0; c < 3; ++c) {
                                        float reflection = h(reflected[c] / 255.f), a = h(alpha / 255.f),
                                              base_value = h(base_color[c] / 255.f);
                                        float delta = h(std::fma(reflection, -a, base_value));
                                        float mixed = h(std::fma(weight, delta, h(reflection * a)));
                                        expected[c] = uint8_t(std::round(std::clamp(mixed, 0.f, 1.f) * 255));
                                        expected[c + 4] = uint8_t(int8_t(std::round(normal[c] * 127)));
                                    }
                                    expected[3] = alpha;
                                    expected[7] = 0;
                                    for (unsigned pixel = 0; pixel < 64; ++pixel)
                                        for (unsigned c = 0; c < 8; ++c) {
                                            ++components;
                                            const int tolerance = c < 3 ? 1 : 0;
                                            if (std::abs(int(actual[pixel][c]) - int(expected[c])) > tolerance) {
                                                if (failures++ < 12)
                                                    std::cerr << "Mismatch draw=" << draws << " mip=" << mip
                                                              << " face=" << face << " pixel=" << pixel << " byte=" << c
                                                              << " actual=" << unsigned(actual[pixel][c])
                                                              << " expected=" << unsigned(expected[c]) << '\n';
                                            }
                                        }
                                    ++draws;
                                }
            }
            std::cout << "RESULT draws=" << draws << " compared_material_bytes=" << components << " failed=" << failures
                      << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
