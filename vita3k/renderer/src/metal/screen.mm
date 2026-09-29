// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/screen.h>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace renderer::metal {
namespace {
std::vector<uint32_t> load_spirv(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Metal: cannot open built-in shader " + path.string());
    const std::streamsize size = input.tellg();
    if (size < 20 || size % sizeof(uint32_t))
        throw std::runtime_error("Metal: invalid built-in SPIR-V " + path.string());
    std::vector<uint32_t> words(size / sizeof(uint32_t));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(words.data()), size))
        throw std::runtime_error("Metal: cannot read built-in shader " + path.string());
    return words;
}
}
ScreenRenderer::ScreenRenderer(Device &device, std::filesystem::path static_assets)
    : device(device), assets(std::move(static_assets)) {
    const std::string source = R"(#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; float2 uv; };
vertex V screen_vertex(uint i [[vertex_id]]) {
    const float2 p[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
    V v; v.position=float4(p[i],0,1); v.uv=float2((p[i].x+1)*0.5,(1-p[i].y)*0.5); return v;
}
fragment float4 screen_fragment(V v [[stage_in]], texture2d<float> image [[texture(0)]], sampler s [[sampler(0)]]) { return image.sample(s,v.uv); })";
    bicubic_source = R"(#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; float2 uv; };
float4 cubic(float v) {
    float4 n = float4(1.0, 2.0, 3.0, 4.0) - v;
    float4 powers = n * n * n;
    float x = powers.x;
    float y = powers.y - 4.0 * powers.x;
    float z = powers.z - 4.0 * powers.y + 6.0 * powers.x;
    return float4(x, y, z, 6.0 - x - y - z) / 6.0;
}
fragment float4 screen_bicubic(V v [[stage_in]], texture2d<float> image [[texture(0)]], sampler s [[sampler(0)]]) {
    float2 size = float2(image.get_width(), image.get_height());
    float2 texel = 1.0 / size;
    float2 coord = v.uv * size - 0.5;
    float2 fxy = fract(coord);
    coord -= fxy;
    float4 xcubic = cubic(fxy.x);
    float4 ycubic = cubic(fxy.y);
    float4 c = coord.xxyy + float4(-0.5, 1.5, -0.5, 1.5);
    float4 weights = float4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    float4 offset = (c + float4(xcubic.yw, ycubic.yw) / weights) * texel.xxyy;
    float4 a = image.sample(s, offset.xz);
    float4 b = image.sample(s, offset.yz);
    float4 c0 = image.sample(s, offset.xw);
    float4 d = image.sample(s, offset.yw);
    float sx = weights.x / (weights.x + weights.y);
    float sy = weights.z / (weights.z + weights.w);
    return mix(mix(d, c0, sx), mix(b, a, sx), sy);
})";
    fxaa_source = R"(#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; float2 uv; };
fragment float4 screen_fxaa(V v [[stage_in]], texture2d<float> image [[texture(0)]], sampler s [[sampler(0)]]) {
    float2 texel = 1.0 / float2(image.get_width(), image.get_height());
    float3 nw = image.sample(s, v.uv + float2(-1.0, -1.0) * texel).rgb;
    float3 ne = image.sample(s, v.uv + float2( 1.0, -1.0) * texel).rgb;
    float3 sw = image.sample(s, v.uv + float2(-1.0,  1.0) * texel).rgb;
    float3 se = image.sample(s, v.uv + float2( 1.0,  1.0) * texel).rgb;
    float3 center = image.sample(s, v.uv).rgb;
    float3 luma = float3(0.299, 0.587, 0.114);
    float luma_nw = dot(nw, luma), luma_ne = dot(ne, luma);
    float luma_sw = dot(sw, luma), luma_se = dot(se, luma);
    float luma_center = dot(center, luma);
    float luma_min = min(luma_center, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    float luma_max = max(luma_center, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
    float2 direction = float2(-((luma_nw + luma_ne) - (luma_sw + luma_se)),
                              ((luma_nw + luma_sw) - (luma_ne + luma_se)));
    float reduction = max((luma_nw + luma_ne + luma_sw + luma_se) / 32.0, 1.0 / 128.0);
    float inverse = 1.0 / (min(abs(direction.x), abs(direction.y)) + reduction);
    direction = clamp(direction * inverse, float2(-8.0), float2(8.0)) * texel;
    float3 rgb_a = 0.5 * (image.sample(s, v.uv + direction * (1.0 / 3.0 - 0.5)).rgb
                          + image.sample(s, v.uv + direction * (2.0 / 3.0 - 0.5)).rgb);
    float3 rgb_b = rgb_a * 0.5 + 0.25 * (image.sample(s, v.uv - direction * 0.5).rgb
                                          + image.sample(s, v.uv + direction * 0.5).rgb);
    float3 result = dot(rgb_b, luma) < luma_min || dot(rgb_b, luma) > luma_max ? rgb_a : rgb_b;
    return float4(result, 1.0);
})";
    std::string error;
    auto vert = device.compile({source, "screen_vertex", shader::metal::Stage::Vertex}, false, error);
    if (!vert) throw std::runtime_error(error);
    vertex_function = vert->function;
    auto frag = device.compile({source, "screen_fragment", shader::metal::Stage::Fragment}, false, error);
    if (!frag) throw std::runtime_error(error);
    auto desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = vertex_function;
    desc.fragmentFunction = frag->function;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipeline = device.create_pipeline(desc, error);
    if (!pipeline) throw std::runtime_error(error);
    set_filter(true);
}
id<MTLRenderPipelineState> ScreenRenderer::create_filter_pipeline(const std::string &source, const char *entry) {
    std::string error;
    auto fragment = device.compile({source, entry, shader::metal::Stage::Fragment}, false, error);
    if (!fragment) throw std::runtime_error(error);
    auto desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = vertex_function;
    desc.fragmentFunction = fragment->function;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    auto result = device.create_pipeline(desc, error);
    if (!result) throw std::runtime_error(error);
    return result;
}
void ScreenRenderer::initialize_fsr() {
    if (easu_pipeline && rcas_pipeline) return;
    const auto directory = assets / "shaders-builtin/vulkan";
    auto easu = shader::metal::convert_builtin_compute_spirv(load_spirv(directory / "fsr_filter_easu.comp.spv"));
    auto rcas = shader::metal::convert_builtin_compute_spirv(load_spirv(directory / "fsr_filter_rcas.comp.spv"));
    std::string error;
    auto easu_program = device.compile(easu, false, error);
    if (!easu_program) throw std::runtime_error(error);
    auto rcas_program = device.compile(rcas, false, error);
    if (!rcas_program) throw std::runtime_error(error);
    easu_pipeline = device.create_compute_pipeline(easu_program->function, error);
    if (!easu_pipeline) throw std::runtime_error(error);
    rcas_pipeline = device.create_compute_pipeline(rcas_program->function, error);
    if (!rcas_pipeline) throw std::runtime_error(error);
}
void ScreenRenderer::set_filter(bool linear) {
    filter = linear ? Filter::Bilinear : Filter::Nearest;
    auto desc = [MTLSamplerDescriptor new];
    desc.minFilter = desc.magFilter = linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    desc.sAddressMode = desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler = [device.native_device() newSamplerStateWithDescriptor:desc];
    if (!sampler) throw std::runtime_error("Metal: screen sampler allocation failed");
}
void ScreenRenderer::set_filter(std::string_view name) {
    if (name == "Bicubic") {
        if (!bicubic_pipeline) bicubic_pipeline = create_filter_pipeline(bicubic_source, "screen_bicubic");
        set_filter(false);
        filter = Filter::Bicubic;
    } else if (name == "FXAA") {
        if (!fxaa_pipeline) fxaa_pipeline = create_filter_pipeline(fxaa_source, "screen_fxaa");
        set_filter(true);
        filter = Filter::FXAA;
    } else if (name == "FSR") {
        initialize_fsr();
        set_filter(true);
        filter = Filter::FSR;
    } else {
        set_filter(name != "Nearest");
    }
}
id<MTLTexture> ScreenRenderer::prepare(id<MTLCommandBuffer> commands, id<MTLTexture> source,
    uint32_t width, uint32_t height) {
    if (filter != Filter::FSR || !source || !width || !height) return source;
    if (!fsr_intermediate || fsr_intermediate.width != width || fsr_intermediate.height != height) {
        auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                       width:width height:height mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        fsr_intermediate = [device.native_device() newTextureWithDescriptor:desc];
        fsr_output = [device.native_device() newTextureWithDescriptor:desc];
        if (!fsr_intermediate || !fsr_output)
            throw std::runtime_error("Metal: cannot allocate FSR textures");
    }
    struct EasuConstants {
        uint32_t offset[2], dimensions[2], texture_dimensions[2], output_dimensions[2];
    } easu{{0, 0}, {uint32_t(source.width), uint32_t(source.height)},
        {uint32_t(source.width), uint32_t(source.height)}, {width, height}};
    struct RcasConstants {
        uint32_t offset[2];
        float sharpening;
        uint32_t padding;
    } rcas{{0, 0}, 0.2f, 0};
    static_assert(sizeof(EasuConstants) == 32 && sizeof(RcasConstants) == 16);
    const MTLSize groups = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
    const MTLSize threads = MTLSizeMake(64, 1, 1);
    auto encoder = [commands computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal: cannot begin FSR upscaling pass");
    [encoder setComputePipelineState:easu_pipeline];
    [encoder setBytes:&easu length:sizeof(easu) atIndex:0];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:fsr_intermediate atIndex:1];
    [encoder setSamplerState:sampler atIndex:0];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    encoder = [commands computeCommandEncoder];
    if (!encoder) throw std::runtime_error("Metal: cannot begin FSR sharpening pass");
    [encoder setComputePipelineState:rcas_pipeline];
    [encoder setBytes:&rcas length:sizeof(rcas) atIndex:0];
    [encoder setTexture:fsr_intermediate atIndex:0];
    [encoder setTexture:fsr_output atIndex:1];
    [encoder setSamplerState:sampler atIndex:0];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return fsr_output;
}
void ScreenRenderer::render(id<MTLRenderCommandEncoder> encoder, id<MTLTexture> source, MTLViewport viewport) {
    if (!source) return;
    const auto selected = filter == Filter::Bicubic ? bicubic_pipeline : filter == Filter::FXAA ? fxaa_pipeline : pipeline;
    [encoder setRenderPipelineState:selected];
    [encoder setViewport:viewport];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder setFragmentSamplerState:sampler atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}
}
