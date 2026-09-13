// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gxm/functions.h>
#include <iostream>
#include <renderer/metal/device.h>
#include <shader/msl_recompiler.h>
#include <shader/spirv_recompiler.h>
#include <stdexcept>
static void check(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
static std::vector<uint32_t> read(const std::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    check(bool(f), "Cannot open fixture");
    size_t n = f.tellg();
    check(n > 0 && n < 1024 * 1024, "Invalid fixture size");
    std::vector<uint32_t> b((n + 3) / 4);
    f.seekg(0);
    check(bool(f.read(reinterpret_cast<char *>(b.data()), n)), "Cannot read fixture");
    return b;
}
int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: metal-interface-validation <DOA vertex.gxp> <DOA fragment.gxp> <video draw directory>\n";
        return 2;
    }
    @autoreleasepool {
        try {
            auto vb = read(argv[1]), fb = read(argv[2]);
            const auto &v = *reinterpret_cast<const SceGxmProgram *>(vb.data()), &f = *reinterpret_cast<const SceGxmProgram *>(fb.data());
            check(v.is_vertex() && f.is_fragment(), "Wrong shader stages");
            FeatureState features;
            features.direct_fragcolor = true;
            features.support_unknown_format = true;
            features.support_scaled_attribute_formats = false;
            features.use_mask_bit = true;
            features.enable_memory_mapping = true;
            features.use_texture_viewport = true;
            std::vector<SceGxmVertexAttribute> attributes;
            for (const auto a : std::array<std::array<uint32_t, 4>, 5>{ { { 0, 9, 3, 0 }, { 23, 8, 4, 4 }, { 12, 5, 3, 8 }, { 15, 4, 4, 12 }, { 19, 0, 4, 16 } } }) {
                SceGxmVertexAttribute item{};
                item.offset = a[0];
                item.format = static_cast<SceGxmAttributeFormat>(a[1]);
                item.componentCount = a[2];
                item.regIndex = a[3];
                attributes.push_back(item);
            }
            shader::Hints hints{};
            hints.attributes = &attributes;
            hints.color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
            std::fill_n(hints.vertex_textures, 16, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR);
            std::fill_n(hints.fragment_textures, 16, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR);
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error);
            const uint32_t missing = uint32_t(gxp::get_fragment_inputs(f)) & ~uint32_t(gxp::get_vertex_outputs(v)) & 0x3ffe;
            check(missing == ((1u << 9) | (1u << 13)), "Captured pair's missing interface changed");
            auto fs = device->compile(shader::metal::convert_gxp(f, "captured-fragment", features, hints), false, error);
            check(bool(fs), error);
            auto layout = [MTLVertexDescriptor vertexDescriptor];
            const MTLVertexFormat formats[] = { MTLVertexFormatFloat3, MTLVertexFormatHalf4, MTLVertexFormatChar3Normalized, MTLVertexFormatUChar4Normalized, MTLVertexFormatUChar4 };
            for (unsigned i = 0; i < 5; ++i) {
                layout.attributes[i].format = formats[i];
                layout.attributes[i].offset = attributes[i].offset;
                layout.attributes[i].bufferIndex = 4;
            }
            layout.layouts[4].stride = 32;
            layout.layouts[4].stepFunction = MTLVertexStepFunctionPerVertex;
            layout.layouts[4].stepRate = 1;
            for (bool bridge : { false, true }) {
                hints.metal_missing_vertex_outputs = bridge ? missing : 0;
                auto vs = device->compile(shader::metal::convert_gxp(v, "captured-vertex", features, hints), false, error);
                check(bool(vs), error);
                auto pd = [MTLRenderPipelineDescriptor new];
                pd.vertexFunction = vs->function;
                pd.fragmentFunction = fs->function;
                pd.vertexDescriptor = layout;
                pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
                auto pipeline = device->create_pipeline(pd, error);
                if (bridge)
                    check(pipeline != nil, error);
                else
                    check(!pipeline && error.find("mismatching vertex shader") != std::string::npos, "Original captured pair did not reproduce interface rejection");
            }
            std::cout << "PASS exact DOA pair: original interface rejected, paired missing-output variant links\n";
            const std::filesystem::path draw(argv[3]);
            auto logo = read(draw / "vertex.gxp"), stream = read(draw / "stream-4.bin"), indices = read(draw / "indices.bin"), info = read(draw / "vertex-render-info.bin");
            const auto &gxp = *reinterpret_cast<const SceGxmProgram *>(logo.data());
            attributes.clear();
            for (unsigned i = 0; i < 2; ++i) {
                SceGxmVertexAttribute a{};
                a.offset = i * 8;
                a.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
                a.componentCount = 2;
                a.regIndex = i * 4;
                attributes.push_back(a);
            }
            auto geometry = [device->native_device() newBufferWithBytes:stream.data() length:stream.size() * 4 options:MTLResourceStorageModeShared];
            auto index = [device->native_device() newBufferWithBytes:indices.data() length:indices.size() * 4 options:MTLResourceStorageModeShared];
            layout = [MTLVertexDescriptor vertexDescriptor];
            for (unsigned i = 0; i < 2; ++i) {
                layout.attributes[i].format = MTLVertexFormatFloat2;
                layout.attributes[i].offset = i * 8;
                layout.attributes[i].bufferIndex = 4;
            }
            layout.layouts[4].stride = 16;
            layout.layouts[4].stepFunction = MTLVertexStepFunctionPerVertex;
            layout.layouts[4].stepRate = 1;
            auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:16 height:16 mipmapped:NO];
            td.storageMode = MTLStorageModeShared;
            td.usage = MTLTextureUsageRenderTarget;
            auto target = [device->native_device() newTextureWithDescriptor:td];
            std::array<float, 1024> baseline{};
            for (bool bridge : { false, true }) {
                hints.metal_missing_vertex_outputs = bridge ? ((1u << 1) | (1u << 2) | (1u << 9) | (1u << 13)) : 0;
                auto vs = device->compile(shader::metal::convert_gxp(gxp, "video-vertex", features, hints), false, error);
                check(bool(vs), error);
                std::string inputs = "struct In {float4 uv [[user(locn4)]];";
                if (bridge)
                    inputs += "float4 color [[user(locn1)]];float4 fog [[user(locn3)]];float4 a [[user(locn9)]];float4 b [[user(locn13)]];";
                inputs += "}; fragment float4 probe(In in [[stage_in]]) {return float4(in.uv.xy,0,1)";
                if (bridge)
                    inputs += "+abs(in.color)+abs(in.fog)+abs(in.a)+abs(in.b)";
                inputs += ";}";
                shader::metal::Program probe{ .source = "#include <metal_stdlib>\nusing namespace metal;\n" + inputs, .entry_point = "probe", .stage = shader::metal::Stage::Fragment };
                auto fragment = device->compile(probe, false, error);
                check(bool(fragment), error);
                auto pd = [MTLRenderPipelineDescriptor new];
                pd.vertexFunction = vs->function;
                pd.fragmentFunction = fragment->function;
                pd.vertexDescriptor = layout;
                pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
                auto pipeline = device->create_pipeline(pd, error);
                check(pipeline != nil, error);
                auto commands = [device->command_queue() commandBuffer];
                auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = target;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
                auto enc = [commands renderCommandEncoderWithDescriptor:pass];
                [enc setRenderPipelineState:pipeline];
                [enc setViewport:MTLViewport{ 0, 0, 16, 16, 0, 1 }];
                [enc setVertexBuffer:geometry offset:0 atIndex:4];
                [enc setVertexBytes:info.data() length:info.size() * 4 atIndex:0];
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:indices.size() indexType:MTLIndexTypeUInt32 indexBuffer:index indexBufferOffset:0];
                [enc endEncoding];
                check(device->submit_and_wait(commands, error), error);
                std::array<float, 1024> pixels;
                [target getBytes:pixels.data() bytesPerRow:256 fromRegion:MTLRegionMake2D(0, 0, 16, 16) mipmapLevel:0];
                for (unsigned y = 0; y < 16; ++y)
                    for (unsigned x = 0; x < 16; ++x)
                        for (unsigned c = 0; c < 4; ++c) {
                            float expected = c == 0 ? (x + .5f) / 16 : c == 1 ? (y + .5f) / 16
                                : c == 2                                      ? 0
                                                                              : 1;
                            check(std::abs(pixels[(y * 16 + x) * 4 + c] - expected) < 1e-6f, "Missing-output bridge changed UV interpolation or did not provide zero");
                        }
                if (bridge)
                    check(pixels == baseline, "Missing-output bridge shifted existing output registers");
                else
                    baseline = pixels;
            }
            std::cout << "PASS native GPU interpolation: 2048 CPU-reference channels; added color/fog/TC5/TC9 are zero, existing TC0 and position unchanged\n";
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
