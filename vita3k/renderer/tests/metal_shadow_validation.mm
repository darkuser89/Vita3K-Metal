// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "../../shader/tests/metal_shader_fixture.h"
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <shader/metal_texture.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <shader/uniform_block.h>
#include <stdexcept>
using Pixel = std::array<float, 4>;
static void check(bool value, const std::string &message) {
    if (!value)
        throw std::runtime_error(message);
}
static float halfround(float value) { return float((__fp16)value); }
// Original SAO 95840b43 shadow program, unmodified. White albedo and zero
// fog/ambient isolate its bilinear depth comparison and [0.6,1] shadow clamp.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc >= 2 && argc <= 4,
                  "usage: metal-shadow-validation original-95840b43.gxp [reconstruction-scale(1..3) [output.metal]]");
            const unsigned scale = argc >= 3 ? std::stoul(argv[2]) : 0;
            check(argc < 3 || (scale >= 1 && scale <= 3), "Invalid scale");
            // Mipmapped cases verify that weights describe the base-level
            // texels read by the emitted OpImageGather. They do not establish
            // complete SGX raw-sample LOD selection (bias/explicit/gradient).
            std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
            check(bool(file), "Cannot open GXP");
            auto length = file.tellg();
            check(length > 0 && length < 1024 * 1024, "Invalid GXP length");
            std::vector<uint32_t> words((size_t(length) + 3) / 4);
            file.seekg(0);
            check(bool(file.read(reinterpret_cast<char *>(words.data()), length)), "Cannot read GXP");
            auto program = compile_gxp_fixture(words, length, "sao-shadow-original",
                                               scale ? "gxp-mapped-shadow-mips" : "gxp-mapped-shadow");
            check(program.stage == shader::metal::Stage::Fragment && !program.writes_guest_memory, "Unexpected GXP");
            if (argc == 4) {
                std::ofstream output(argv[3]);
                output << program.source;
                output.close();
                check(bool(output), "Cannot save MSL");
            }
            std::string reason;
            auto runtime = renderer::metal::Device::create(reason);
            check(bool(runtime), reason);
            renderer::metal::SurfaceCaster caster(*runtime);
            id<MTLDevice> device = runtime->native_device();
            check(device != nil, "No Metal device");
            auto queue = [device newCommandQueue];
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *error = nil;
            auto fragment = [device newLibraryWithSource:[NSString stringWithUTF8String:program.source.c_str()]
                                                 options:options
                                                   error:&error];
            check(fragment != nil, error ? error.localizedDescription.UTF8String : "Fragment compilation failed");
            auto vertex =
                [device newLibraryWithSource:
                            @"#include <metal_stdlib>\nusing namespace metal;struct O{float4 p [[position]];float4 uv "
                            @"[[user(locn4)]];float4 tint [[user(locn5)]];float4 fog [[user(locn6)]];float4 shadow "
                            @"[[user(locn7)]];};vertex O v(uint id [[vertex_id]],constant float4 &phase "
                            @"[[buffer(0)]]){float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};O "
                            @"o;o.p=float4(p[id],0,1);o.uv=float4(0);o.tint=float4(1);o.fog=float4(0);o.shadow=float4("
                            @"float2(p[id].x*.5+.5,.5-p[id].y*.5)*phase.z+phase.xy,.5,1);return o;}"
                                     options:options
                                       error:&error];
            check(vertex != nil, error ? error.localizedDescription.UTF8String : "Vertex compilation failed");
            auto pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [vertex newFunctionWithName:@"v"];
            pd.fragmentFunction = [fragment newFunctionWithName:@"main_fs"];
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
            auto pipeline = [device newRenderPipelineStateWithDescriptor:pd error:&error];
            check(pipeline != nil, error ? error.localizedDescription.UTF8String : "Pipeline failed");
            auto texture = [&](MTLPixelFormat format, int size, bool mips = false) {
                auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                             width:size
                                                                            height:size
                                                                         mipmapped:mips];
                td.storageMode = MTLStorageModeShared;
                td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
                return [device newTextureWithDescriptor:td];
            };
            auto color = texture(MTLPixelFormatRGBA32Float, 1), fog = texture(MTLPixelFormatRGBA32Float, 1);
            auto mask = texture(MTLPixelFormatRGBA32Float, 8), target = texture(MTLPixelFormatRGBA32Float, 8);
            Pixel white = {1, 1, 1, 1}, zero = {0, 0, 0, 0};
            std::array<Pixel, 64> maskPixels;
            maskPixels.fill(white);
            [color replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:white.data() bytesPerRow:16];
            [fog replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:zero.data() bytesPerRow:16];
            [mask replaceRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0 withBytes:maskPixels.data() bytesPerRow:128];
            std::array<Pixel, 2> uniforms{};
            auto uniform = [device newBufferWithBytes:uniforms.data()
                                               length:sizeof(uniforms)
                                              options:MTLResourceStorageModeShared];
            int failures = 0, comparisons = 0;
            for (int mip = 0; mip < 3; ++mip) {
                auto shadow = texture(MTLPixelFormatR32Float, 16, mip != 0);
                for (NSUInteger level = 0; level < shadow.mipmapLevelCount; ++level) {
                    const int size = 16 >> level;
                    std::vector<float> depths(size * size);
                    for (int y = 0; y < size; ++y)
                        for (int x = 0; x < size; ++x)
                            depths[y * size + x] = level ? .125f : ((x * 13 + y * 7 + x * y) % 5 < 3 ? .75f : .25f);
                    [shadow replaceRegion:MTLRegionMake2D(0, 0, size, size)
                              mipmapLevel:level
                                withBytes:depths.data()
                              bytesPerRow:size * 4];
                }
                shader::metal::TextureMipInfos mip_info{};
                if (scale) {
                    for (unsigned level = 0; level < shadow.mipmapLevelCount; ++level)
                        mip_info[14].sizes[level] = {16u >> level, 16u >> level};
                    // Gather is unfiltered even with anisotropy/minimum LOD set.
                    mip_info[14].control = {1, 3u | (2u << 3) | (2u << 6) | ((mip == 2 ? 2u : 0u) << 9), 16, 0};
                    shadow = caster.texture_snapshot(shadow, {}, scale);
                }
                auto sd = [MTLSamplerDescriptor new];
                sd.maxAnisotropy = scale ? 16 : 1;
                sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
                sd.mipFilter = mip ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterNotMipmapped;
                // Native gather's base-level behavior must remain consistent
                // when ordinary sampling receives a nonzero minimum LOD.
                sd.lodMinClamp = mip == 2 ? 2.f : 0.f;
                sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
                auto sampler = [device newSamplerStateWithDescriptor:sd];
                for (int mapping = 0; mapping < 4; ++mapping)
                    for (int phase = 0; phase < 4; ++phase) {
                        Pixel transform = {phase / 64.f, (3 - phase) / 64.f, 1, 0};
                        const float rx = mapping == 0 ? 1.f : .5f, ry = mapping < 2 ? 1.f : .75f;
                        const float ox = mapping < 2 ? 0.f : .125f, oy = mapping == 3 ? .0625f : 0.f;
                        shader::RenderFragUniformBlockExtended info{};
                        info.set_buffer_count(1);
                        info.set_texture_count(16);
                        info.base_block.res_multiplier = 1;
                        info.set_buffer_address(0, uniform.gpuAddress);
                        for (int slot = 0; slot < 16; ++slot)
                            info.set_viewport_ratio(slot, {1, 1});
                        info.set_viewport_ratio(14, {rx, ry});
                        info.set_viewport_offset(14, {ox, oy});
                        std::vector<uint8_t> bytes((info.get_size() + 15) & ~size_t(15));
                        info.copy_to(bytes.data());
                        auto commands = [queue commandBuffer];
                        auto rp = [MTLRenderPassDescriptor renderPassDescriptor];
                        rp.colorAttachments[0].texture = target;
                        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
                        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                        auto encoder = [commands renderCommandEncoderWithDescriptor:rp];
                        [encoder setRenderPipelineState:pipeline];
                        [encoder setVertexBytes:transform.data() length:sizeof(transform) atIndex:0];
                        [encoder setFragmentBytes:bytes.data() length:bytes.size() atIndex:0];
                        if (scale)
                            [encoder setFragmentBytes:mip_info.data()
                                               length:sizeof(mip_info)
                                              atIndex:shader::metal::TEXTURE_INFO_BUFFER];
                        [encoder setFragmentTexture:color atIndex:0];
                        [encoder setFragmentTexture:shadow atIndex:14];
                        [encoder setFragmentTexture:fog atIndex:15];
                        [encoder setFragmentTexture:mask atIndex:17];
                        for (int slot : {0, 14, 15})
                            [encoder setFragmentSamplerState:sampler atIndex:slot];
                        [encoder useResource:uniform usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
                        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        [encoder endEncoding];
                        [commands commit];
                        [commands waitUntilCompleted];
                        check(commands.status == MTLCommandBufferStatusCompleted,
                              commands.error ? commands.error.localizedDescription.UTF8String : "GPU failed");
                        std::array<Pixel, 64> actual;
                        [target getBytes:actual.data()
                             bytesPerRow:128
                              fromRegion:MTLRegionMake2D(0, 0, 8, 8)
                             mipmapLevel:0];
                        int caseFailures = 0;
                        for (int y = 0; y < 8; ++y)
                            for (int x = 0; x < 8; ++x) {
                                const float u = halfround((x + .5f) / 8 + transform[0]),
                                            v = halfround((y + .5f) / 8 + transform[1]);
                                const float tx = std::fma(u, rx, ox) * 16 - .5f, ty = std::fma(v, ry, oy) * 16 - .5f;
                                const int ix = int(std::floor(tx)), iy = int(std::floor(ty));
                                const float fx = tx - ix, fy = ty - iy;
                                float lit = 0;
                                for (int dy = 0; dy < 2; ++dy)
                                    for (int dx = 0; dx < 2; ++dx) {
                                        const int sx = std::clamp(ix + dx, 0, 15), sy = std::clamp(iy + dy, 0, 15);
                                        if ((sx * 13 + sy * 7 + sx * sy) % 5 < 3)
                                            lit += halfround((dx ? fx : 1 - fx) * (dy ? fy : 1 - fy));
                                    }
                                if (u <= 0 || v <= 0 || u >= 1 || v >= 1)
                                    lit = 1;
                                const float expected = halfround(std::clamp(lit, .6f, 1.f));
                                for (int c = 0; c < 4; ++c) {
                                    ++comparisons;
                                    float want = c == 3 ? .5f : expected;
                                    if (!std::isfinite(actual[y * 8 + x][c]) ||
                                        std::abs(actual[y * 8 + x][c] - want) > .001f) {
                                        if (failures++ < 12)
                                            std::cerr << "Mismatch mip=" << mip << " mapping=" << mapping
                                                      << " phase=" << phase << " x=" << x << " y=" << y << " c=" << c
                                                      << " actual=" << actual[y * 8 + x][c] << " expected=" << want
                                                      << '\n';
                                        ++caseFailures;
                                    }
                                }
                            }
                        std::cout << "CASE mip=" << mip << " mapping=" << mapping << " phase=" << phase
                                  << " failed=" << caseFailures << '\n';
                    }
            }
            std::cout << "RESULT reconstruction_scale=" << scale << " draws=48 components=" << comparisons
                      << " failed=" << failures << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
