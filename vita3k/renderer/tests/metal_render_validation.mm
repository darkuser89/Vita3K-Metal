// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const std::string &error) {
    if (!ok)
        throw std::runtime_error(error);
}

static void render_and_verify(renderer::metal::Device &device, const shader::metal::Program &fragment,
    bool fetch, const std::array<uint8_t, 4> &expected, bool gamma_correction = false,
    bool srgb_target = false, bool blend = false) {
    std::string error;
    shader::metal::Program vertex{
        .source = R"(#include <metal_stdlib>
using namespace metal;
vertex float4 fullscreen(uint i [[vertex_id]]) {
    const float2 p[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
    return float4(p[i], 0, 1);
})",
        .entry_point = "fullscreen",
        .stage = shader::metal::Stage::Vertex
    };
    auto vs = device.compile(vertex, false, error);
    require(bool(vs), error);
    auto fs = device.compile(fragment, gamma_correction, error);
    require(bool(fs), error);
    MTLRenderPipelineDescriptor *pipeline = [MTLRenderPipelineDescriptor new];
    pipeline.vertexFunction = vs->function;
    pipeline.fragmentFunction = fs->function;
    const auto format = srgb_target ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
    pipeline.colorAttachments[0].pixelFormat = format;
    pipeline.colorAttachments[0].blendingEnabled = blend;
    pipeline.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pipeline.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
    pipeline.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipeline.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
    id<MTLRenderPipelineState> pso = device.create_pipeline(pipeline, error);
    require(pso != nil, error);
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                 width:8 height:8 mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device.native_device() newTextureWithDescriptor:desc];
    require(texture != nil, "Cannot allocate Metal readback target");
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    id<MTLCommandBuffer> commands = [device.command_queue() commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
    require(encoder != nil, "Cannot create Metal render encoder");
    [encoder setRenderPipelineState:pso];
    const float color[] = {0.25f, 0.5f, 0.75f, 1.0f};
    // Also exercises the uploaded GXM uniform buffer slot used by the fixture.
    [encoder setFragmentBytes:color length:sizeof(color) atIndex:shader::metal::UNIFORM_BUFFER];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    if (fetch)
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    require(device.submit_and_wait(commands, error), error);
    std::array<uint8_t, 8 * 8 * 4> pixels{};
    [texture getBytes:pixels.data() bytesPerRow:8 * 4 fromRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0];
    for (size_t i = 0; i < pixels.size(); ++i)
        require(std::abs(int(pixels[i]) - int(expected[i % 4])) <= 1,
            "Metal pixel mismatch at byte " + std::to_string(i) + ": " + std::to_string(pixels[i]));
    require(!device.submit_and_wait(commands, error), "Already submitted command buffer was accepted");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            std::string error;
            auto device = renderer::metal::Device::create(error);
            require(bool(device), error);
            shader::metal::Program color{
                .source = R"(#include <metal_stdlib>
using namespace metal;
fragment float4 color(constant float4 &value [[buffer(1)]]) { return value; })",
                .entry_point = "color",
                .stage = shader::metal::Stage::Fragment
            };
            render_and_verify(*device, color, false, {64, 128, 191, 255});
            std::cout << "PASS native uniform upload, pipeline, draw, completion and pixel readback\n";
            shader::metal::Program fetch{
                .source = R"(#include <metal_stdlib>
using namespace metal;
fragment float4 fetch(float4 previous [[color(0)]]) { return previous + float4(0.25,0,0,0); })",
                .entry_point = "fetch",
                .stage = shader::metal::Stage::Fragment,
                .uses_framebuffer_fetch = true
            };
            render_and_verify(*device, fetch, true, {128, 0, 0, 255});
            std::cout << "PASS framebuffer fetch across two overlapping draws\n";
            render_and_verify(*device, color, false, {137,188,225,255}, false, true);
            render_and_verify(*device, color, true, {188,255,255,255}, false, true, true);
            render_and_verify(*device, fetch, true, {188,0,0,255}, false, true);
            std::cout << "PASS sRGB target encoding, linear fixed-function blending and linear framebuffer fetch\n";
            shader::metal::Program specialized{
                .source = R"(#include <metal_stdlib>
using namespace metal;
constant bool gamma [[function_constant(0)]];
fragment float4 specialized() { return float4(gamma ? 1.0 : 0.0, 0, 0, 1); })",
                .entry_point = "specialized",
                .stage = shader::metal::Stage::Fragment
            };
            render_and_verify(*device, specialized, false, {0, 0, 0, 255}, false);
            render_and_verify(*device, specialized, false, {255, 0, 0, 255}, true);
            std::cout << "PASS both gamma function-constant specializations\n";
            if (argc == 2) {
                // Optional local GXP-derived constant-color fragment fixture:
                // Sly PCSF00209, vk13-0285820e... (four float uniforms in buffer 0).
                std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
                require(bool(input), "Cannot open local SPIR-V fixture");
                std::streamsize size = input.tellg();
                require(size >= 20 && size % 4 == 0, "Invalid local SPIR-V fixture length");
                std::vector<uint32_t> words(size / 4);
                input.seekg(0);
                require(bool(input.read(reinterpret_cast<char *>(words.data()), size)), "Cannot read local SPIR-V fixture");
                auto translated = shader::metal::convert_spirv(words);
                render_and_verify(*device, translated, false, {64, 128, 191, 255});
                std::cout << "PASS GXP-derived Sly fragment shader: native GPU pixels match uniforms\n";
            }
            return 0;
        } catch (const std::exception &error) {
            std::cerr << "FAIL " << error.what() << '\n';
            return 1;
        }
    }
}
