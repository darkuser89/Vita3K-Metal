// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/screen.h>
#include <stdexcept>

namespace renderer::metal {
ScreenRenderer::ScreenRenderer(Device &device) : device(device) {
    const std::string source = R"(#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; float2 uv; };
vertex V screen_vertex(uint i [[vertex_id]]) {
    const float2 p[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
    V v; v.position=float4(p[i],0,1); v.uv=float2((p[i].x+1)*0.5,(1-p[i].y)*0.5); return v;
}
fragment float4 screen_fragment(V v [[stage_in]], texture2d<float> image [[texture(0)]], sampler s [[sampler(0)]]) { return image.sample(s,v.uv); })";
    std::string error;
    auto vert = device.compile({source, "screen_vertex", shader::metal::Stage::Vertex}, false, error);
    if (!vert) throw std::runtime_error(error);
    auto frag = device.compile({source, "screen_fragment", shader::metal::Stage::Fragment}, false, error);
    if (!frag) throw std::runtime_error(error);
    auto desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = vert->function;
    desc.fragmentFunction = frag->function;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipeline = device.create_pipeline(desc, error);
    if (!pipeline) throw std::runtime_error(error);
    set_filter(true);
}
void ScreenRenderer::set_filter(bool linear) {
    auto desc = [MTLSamplerDescriptor new];
    desc.minFilter = desc.magFilter = linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    desc.sAddressMode = desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler = [device.native_device() newSamplerStateWithDescriptor:desc];
    if (!sampler) throw std::runtime_error("Metal: screen sampler allocation failed");
}
void ScreenRenderer::render(id<MTLRenderCommandEncoder> encoder, id<MTLTexture> source, MTLViewport viewport) {
    if (!source) return;
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:viewport];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder setFragmentSamplerState:sampler atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}
}
