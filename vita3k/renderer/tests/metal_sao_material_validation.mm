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
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
using Pixel = std::array<float, 4>;
static void check(bool ok, const std::string &why) {
    if (!ok)
        throw std::runtime_error(why);
}
static float h(float x) { return float((__fp16)x); }
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

static int address(int x, int n, unsigned mode) {
    if (mode == 2)
        return std::clamp(x, 0, n - 1);
    const int period = mode == 1 ? n * 2 : n;
    int p = (x % period + period) % period;
    return p < n ? p : period - 1 - p;
}
static Pixel texel(unsigned kind, unsigned x, unsigned y) {
    return {.125f + float((x * 7 + y * 3 + kind * 11) % 16) / 16, .25f + float((x * 3 + y * 5 + kind * 7) % 16) / 16,
            .0625f + float((x + y * 7 + kind * 3) % 16) / 16, .75};
}
static float shadow_texel(unsigned x, unsigned y) { return float((x * 13 + y * 7 + x * y) % 7 + 1) / 8; }
// Full original SAO ebac8695: two material textures/tints, projected shadow
// comparisons, upper color clamp and two separately rounded fog MADs.
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            check(argc >= 3 && argc <= 5, "usage: metal-sao-material-validation original-ebac8695.gxp output.metal "
                                          "[max-scale 1..3] [corrected|hardware]");
            unsigned max_scale = argc >= 4 ? std::stoul(argv[3]) : 3;
            bool corrected = argc < 5 || std::string(argv[4]) == "corrected";
            check(max_scale >= 1 && max_scale <= 3 && (corrected || max_scale == 1), "Invalid scale for mode");
            check(argc < 5 || corrected || std::string(argv[4]) == "hardware", "Invalid mode");
            std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
            check(bool(file), "Cannot open GXP");
            auto length = file.tellg();
            check(length > 0 && length < 1024 * 1024, "Invalid GXP length");
            std::vector<uint32_t> words((size_t(length) + 3) / 4);
            file.seekg(0);
            check(bool(file.read(reinterpret_cast<char *>(words.data()), length)), "Cannot read GXP");
            auto program = compile_gxp_fixture(words, length, "sao-full-original-material", "gxp-mapped-shadow-mips");
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
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            options.fastMathEnabled = NO;
            NSError *error = nil;
            auto vertex = [dev newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;
struct Params {float4 tintA,tintB,phase,control;};
struct O {float4 p [[position]];float4 uv0 [[user(locn4)]];float4 uv1 [[user(locn5)]];float4 tintA [[user(locn6)]];float4 tintB [[user(locn7)]];float4 fog [[user(locn8)]];float4 shadow [[user(locn9)]];};
vertex O v(uint i [[vertex_id]],constant Params &a [[buffer(0)]]) {
 float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
 float2 uv=float2(p[i].x*.5+.5,.5-p[i].y*.5);O o;o.p=float4(p[i],.25,1);
 o.uv0=float4(fma(uv,float2(.75),float2(1.0/53,1.0/29)),0,0);
 o.uv1=float4(fma(uv,float2(1.25),float2(-1.0/31,1.0/47)),0,0);
 o.fog=float4(fma(uv,float2(.5),float2(1.0/37,1.0/19)),0,0);
 o.tintA=a.tintA;o.tintB=a.tintB;
 o.shadow=float4(fma(uv,float2(a.phase.z),a.phase.xy)*a.phase.w,a.control.x*a.phase.w,a.phase.w);return o;
})" options:options error:&error];
            check(vertex != nil, error ? error.localizedDescription.UTF8String : "Vertex compile failed");
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
            auto texture = [&](MTLPixelFormat format, unsigned w, unsigned ht, unsigned levels = 1) {
                auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                             width:w
                                                                            height:ht
                                                                         mipmapped:NO];
                td.mipmapLevelCount = levels;
                td.storageMode = format == MTLPixelFormatDepth32Float ? MTLStorageModePrivate : MTLStorageModeShared;
                td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
                auto t = [dev newTextureWithDescriptor:td];
                check(t != nil, "Texture allocation failed");
                return t;
            };
            auto uniform = [dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
            check(uniform != nil, "Uniform allocation failed");
            const Pixel clear{-2, -3, -4, -5}, fog_rgb{.125, .75, .375, 0}, ambient{.125, .0625, -.125, 0};
            const std::array<Pixel, 4> tintsA = {
                {{.5, .75, .25, 1}, {1.5, .5, -.25, 1}, {.25, 1.5, .75, 1}, {1, .25, .5, 1}}};
            const std::array<Pixel, 4> tintsB = {
                {{.75, .25, .5, 0}, {.5, 1.25, .75, .5}, {1, .5, .25, 1}, {.25, .75, 1.25, 1.5}}};
            uint64_t draws = 0, colors = 0, depths = 0, failures = 0, outside = 0, equality = 0, clamped = 0,
                     negative = 0, masked = 0;
            for (unsigned shape = 0; shape < 3; ++shape)
                for (unsigned scale = 1; scale <= max_scale; ++scale) {
                    @autoreleasepool {
                        unsigned width = shape == 0   ? 16
                                         : shape == 1 ? 15
                                                      : 1,
                                 height = shape == 0   ? 8
                                          : shape == 1 ? 9
                                                       : 1,
                                 levels = shape == 2 ? 1 : 4, side = 16 * scale;
                        auto target = texture(MTLPixelFormatRGBA32Float, side, side),
                             depth = texture(MTLPixelFormatDepth32Float, side, side),
                             mask = texture(MTLPixelFormatRGBA8Unorm, side, side);
                        auto depth_buffer = [dev newBufferWithLength:256 * side options:MTLResourceStorageModeShared];
                        check(depth_buffer != nil, "Depth readback allocation failed");
                        std::vector<uint32_t> masks(side * side);
                        for (unsigned i = 0; i < masks.size(); ++i)
                            masks[i] = i % 11 ? 0xffffffff : 0;
                        [mask replaceRegion:MTLRegionMake2D(0, 0, side, side)
                                mipmapLevel:0
                                  withBytes:masks.data()
                                bytesPerRow:side * 4];
                        id<MTLTexture> images[16] = {};
                        shader::metal::TextureMipInfos mip_info{};
                        for (unsigned slot : {0u, 1u, 15u}) {
                            auto image = texture(MTLPixelFormatRGBA32Float, 7, 5);
                            std::vector<Pixel> data(35);
                            for (unsigned y = 0; y < 5; ++y)
                                for (unsigned x = 0; x < 7; ++x)
                                    data[y * 7 + x] = texel(slot, x, y);
                            [image replaceRegion:MTLRegionMake2D(0, 0, 7, 5)
                                     mipmapLevel:0
                                       withBytes:data.data()
                                     bytesPerRow:7 * sizeof(Pixel)];
                            images[slot] = corrected ? caster.texture_snapshot(image, {}, scale) : image;
                            mip_info[slot].sizes[0] = {7, 5};
                        }
                        auto shadow = texture(MTLPixelFormatR32Float, width, height, levels);
                        for (unsigned level = 0; level < levels; ++level) {
                            unsigned w = std::max(1u, width >> level), ht = std::max(1u, height >> level);
                            std::vector<float> data(w * ht);
                            for (unsigned y = 0; y < ht; ++y)
                                for (unsigned x = 0; x < w; ++x)
                                    data[y * w + x] = level ? .125f : shadow_texel(x, y);
                            [shadow replaceRegion:MTLRegionMake2D(0, 0, w, ht)
                                      mipmapLevel:level
                                        withBytes:data.data()
                                      bytesPerRow:w * 4];
                            mip_info[14].sizes[level] = {w, ht};
                        }
                        images[14] = corrected ? caster.texture_snapshot(shadow, {}, scale) : shadow;
                        for (unsigned addr = 0; addr < 3; ++addr)
                            for (unsigned mapping = 0; mapping < 2; ++mapping)
                                for (unsigned phase = 0; phase < 2; ++phase)
                                    for (float q : {.25f, 1.f, 2.f, -.5f, 1.5f})
                                        for (unsigned profile = 0; profile < 4; ++profile)
                                            for (float fog_strength : {0.f, .5f, 1.5f}) {
                                                @autoreleasepool {
                                                    struct Params {
                                                        Pixel tintA, tintB, phase, control;
                                                    } params{
                                                        tintsA[profile],
                                                        tintsB[profile],
                                                        {phase ? 1.f / 53 : -.125f, phase ? 1.f / 29 : -.25f, 1.25f, q},
                                                        {profile == 0   ? 0.f
                                                         : profile == 1 ? .5f
                                                         : profile == 2 ? 1.f
                                                                        : .375f,
                                                         0, 0, 0}};
                                                    Pixel fog_color = fog_rgb;
                                                    fog_color[3] = fog_strength;
                                                    std::array<Pixel, 2> uniforms{fog_color, ambient};
                                                    std::memcpy(uniform.contents, uniforms.data(), 32);
                                                    shader::RenderFragUniformBlockExtended info{};
                                                    info.base_block.res_multiplier = scale;
                                                    info.set_buffer_count(1);
                                                    info.set_buffer_address(0, uniform.gpuAddress);
                                                    info.set_texture_count(16);
                                                    std::array<id<MTLSamplerState>, 16> samplers{};
                                                    float rx = mapping ? .5f : 1, ry = mapping ? .75f : 1,
                                                          ox = mapping ? 1.f / 17 : 0, oy = mapping ? -1.f / 31 : 0;
                                                    for (unsigned slot : {0u, 1u, 14u, 15u}) {
                                                        info.set_viewport_ratio(slot, {rx, ry});
                                                        info.set_viewport_offset(slot, {ox, oy});
                                                        auto sd = [MTLSamplerDescriptor new];
                                                        sd.minFilter = sd.magFilter =
                                                            slot == 14 ? MTLSamplerMinMagFilterLinear
                                                                       : MTLSamplerMinMagFilterNearest;
                                                        sd.sAddressMode = sd.tAddressMode =
                                                            addr == 0   ? MTLSamplerAddressModeRepeat
                                                            : addr == 1 ? MTLSamplerAddressModeMirrorRepeat
                                                                        : MTLSamplerAddressModeClampToEdge;
                                                        sd.maxAnisotropy = slot == 14 ? 16 : 1;
                                                        sd.mipFilter = MTLSamplerMipFilterNearest;
                                                        sd.lodMinClamp = slot == 14 && mapping ? 2 : 0;
                                                        samplers[slot] = [dev newSamplerStateWithDescriptor:sd];
                                                        mip_info[slot].control = {
                                                            corrected ? 1u : 0u,
                                                            (slot == 14 ? 3u : 0u) | (addr << 3) | (addr << 6) |
                                                                ((slot == 14 && mapping ? 2u : 0u) << 9),
                                                            slot == 14 ? 16u : 1u, 0};
                                                    }
                                                    std::vector<uint8_t> bytes((info.get_size() + 15) & ~size_t(15));
                                                    info.copy_to(bytes.data());
                                                    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                                                    pass.colorAttachments[0].texture = target;
                                                    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                                                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                                                    pass.colorAttachments[0].clearColor =
                                                        MTLClearColorMake(clear[0], clear[1], clear[2], clear[3]);
                                                    pass.depthAttachment.texture = depth;
                                                    pass.depthAttachment.loadAction = MTLLoadActionClear;
                                                    pass.depthAttachment.storeAction = MTLStoreActionStore;
                                                    pass.depthAttachment.clearDepth = .75;
                                                    auto cb = [runtime->command_queue() commandBuffer];
                                                    auto enc = [cb renderCommandEncoderWithDescriptor:pass];
                                                    [enc setRenderPipelineState:pipeline];
                                                    [enc setDepthStencilState:depth_state];
                                                    [enc setVertexBytes:&params length:sizeof(params) atIndex:0];
                                                    [enc setFragmentBytes:bytes.data() length:bytes.size() atIndex:0];
                                                    [enc setFragmentBytes:mip_info.data()
                                                                   length:sizeof(mip_info)
                                                                  atIndex:shader::metal::TEXTURE_INFO_BUFFER];
                                                    [enc useResource:uniform
                                                               usage:MTLResourceUsageRead
                                                              stages:MTLRenderStageFragment];
                                                    for (unsigned slot : {0u, 1u, 14u, 15u}) {
                                                        [enc setFragmentTexture:images[slot] atIndex:slot];
                                                        [enc setFragmentSamplerState:samplers[slot] atIndex:slot];
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
                                                          destinationBytesPerRow:256
                                                        destinationBytesPerImage:256 * side];
                                                    [blit endEncoding];
                                                    check(runtime->submit_and_wait(cb, why), why);
                                                    std::vector<Pixel> actual(side * side);
                                                    [target getBytes:actual.data()
                                                         bytesPerRow:side * sizeof(Pixel)
                                                          fromRegion:MTLRegionMake2D(0, 0, side, side)
                                                         mipmapLevel:0];
                                                    for (unsigned y = 0; y < side; ++y)
                                                        for (unsigned x = 0; x < side; ++x) {
                                                            float u = (x + .5f) / side, v = (y + .5f) / side;
                                                            bool keep = masks[y * side + x] != 0;
                                                            Pixel expected = clear;
                                                            if (keep) {
                                                                float sx = h(std::fma(u, params.phase[2],
                                                                                      params.phase[0]) *
                                                                             q),
                                                                      sy = h(std::fma(v, params.phase[2],
                                                                                      params.phase[1]) *
                                                                             q),
                                                                      sw = h(q), sz = h(params.control[0] * q);
                                                                // The original VRCP followed by VMAD/VMUL rounds the
                                                                // reciprocal first.
                                                                float reciprocal = 1.f / sw;
                                                                float projected_x = std::fma(sx, reciprocal, 0.f),
                                                                      projected_y = std::fma(sy, reciprocal, 0.f),
                                                                      projected_z = sz * reciprocal;
                                                                float tx = std::fma(projected_x, rx, ox) * width - .5f,
                                                                      ty = std::fma(projected_y, ry, oy) * height - .5f;
                                                                int ix = int(std::floor(tx)), iy = int(std::floor(ty));
                                                                float fx = tx - ix, fy = ty - iy, lit = 0;
                                                                for (unsigned dy = 0; dy < 2; ++dy)
                                                                    for (unsigned dx = 0; dx < 2; ++dx) {
                                                                        float sample = shadow_texel(
                                                                            address(ix + dx, width, addr),
                                                                            address(iy + dy, height, addr));
                                                                        if (sample == projected_z)
                                                                            ++equality;
                                                                        if (sample > projected_z)
                                                                            lit += h((dx ? fx : 1 - fx) *
                                                                                     (dy ? fy : 1 - fy));
                                                                    }
                                                                bool out = std::min(sx, sy) <= 0 ||
                                                                           h(std::max(sx, sy) - sw) >= 0;
                                                                outside += out;
                                                                lit = std::clamp(lit + (out ? 1.f : 0.f), .6f, 1.f);
                                                                auto sample = [&](unsigned slot, float a, float b) {
                                                                    int xx = address(
                                                                            int(std::floor(std::fma(a, rx, ox) * 7)), 7,
                                                                            addr),
                                                                        yy = address(
                                                                            int(std::floor(std::fma(b, ry, oy) * 5)), 5,
                                                                            addr);
                                                                    return texel(slot, xx, yy);
                                                                };
                                                                auto a = sample(0, std::fma(u, .75f, 1.f / 53),
                                                                                std::fma(v, .75f, 1.f / 29));
                                                                auto b = sample(1, std::fma(u, 1.25f, -1.f / 31),
                                                                                std::fma(v, 1.25f, 1.f / 47));
                                                                auto fog = sample(15, std::fma(u, .5f, 1.f / 37),
                                                                                  std::fma(v, .5f, 1.f / 19));
                                                                float factor = h(h(fog_strength) * h(fog[0]));
                                                                for (unsigned c = 0; c < 3; ++c) {
                                                                    float base = h(h(params.tintA[c]) * h(a[c]));
                                                                    float difference =
                                                                        half_mad(std::fma(h(params.tintB[c]), h(b[c]), -base));
                                                                    float mixed = half_mad(std::fma(
                                                                        difference, h(params.tintB[3]), h(ambient[c])));
                                                                    float material =
                                                                        half_mad(std::fma(h(a[c]), h(params.tintA[c]), mixed));
                                                                    clamped += material > 1;
                                                                    negative += material < 0;
                                                                    float lighting = std::min(material, 1.f) * lit;
                                                                    float fogged =
                                                                        half_mad(std::fma(factor, h(fog_color[c]), lighting));
                                                                    expected[c] =
                                                                        half_mad(std::fma(factor, -lighting, fogged));
                                                                }
                                                                expected[3] = .5f;
                                                            } else
                                                                ++masked;
                                                            for (unsigned c = 0; c < 4; ++c) {
                                                                ++colors;
                                                                if (actual[y * side + x][c] != expected[c]) {
                                                                    if (failures++ < 16)
                                                                        std::cerr
                                                                            << "MISMATCH shape=" << shape
                                                                            << " scale=" << scale << " addr=" << addr
                                                                            << " mapping=" << mapping
                                                                            << " phase=" << phase << " q=" << q
                                                                            << " profile=" << profile
                                                                            << " fog=" << fog_strength << " x=" << x
                                                                            << " y=" << y << " c=" << c
                                                                            << " actual=" << actual[y * side + x][c]
                                                                            << " expected=" << expected[c] << '\n';
                                                                }
                                                            }
                                                            ++depths;
                                                            float got_depth;
                                                            std::memcpy(&got_depth,
                                                                        (char *)depth_buffer.contents + y * 256 + x * 4,
                                                                        4);
                                                            if (got_depth != (keep ? .25f : .75f))
                                                                ++failures;
                                                        }
                                                    ++draws;
                                                }
                                            }
                        std::cout << "PROGRESS shape=" << shape << " scale=" << scale << " draws=" << draws
                                  << " failed=" << failures << std::endl;
                    }
                }
            std::cout << "RESULT corrected=" << corrected << " draws=" << draws << " color_components=" << colors
                      << " depth_values=" << depths << " outside_shadow=" << outside
                      << " equal_depth_samples=" << equality << " clamped_components=" << clamped
                      << " negative_components=" << negative << " masked=" << masked << " failed=" << failures << '\n';
            return failures ? 1 : 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
