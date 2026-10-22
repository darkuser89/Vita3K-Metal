// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/screen.h>
#include <renderer/metal4/fsr_presentation.h>
#include <renderer/metal4/capabilities.h>
#include <util/hash.h>
#include <util/log.h>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace renderer::metal {
bool supports_metalfx_spatial() {
    static const bool supported = [] {
        @autoreleasepool {
            if (@available(macOS 26.0, *)) {
            } else return false;
            auto device = MTLCreateSystemDefaultDevice();
            return device && [device supportsFamily:MTLGPUFamilyMetal4]
                && [MTLFXSpatialScalerDescriptor supportsDevice:device];
        }
    }();
    return supported;
}
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
    // Both screen entry points live in the same fixed source. Compile it once
    // instead of invoking the Metal source compiler separately for each stage.
    auto options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_0;
    options.fastMathEnabled = NO;
    NSError *native_error = nil;
    auto library = device.compile_library([NSString stringWithUTF8String:source.c_str()], options, &native_error);
    if (!library) throw std::runtime_error(native_error.localizedDescription.UTF8String ?: "Metal: screen shader compilation failed");
    vertex_function = metal4_function(library, @"screen_vertex");
    auto fragment_function = metal4_function(library, @"screen_fragment");
    if (!vertex_function || vertex_function.functionType != MTLFunctionTypeVertex
        || !fragment_function || fragment_function.functionType != MTLFunctionTypeFragment)
        throw std::runtime_error("Metal: invalid screen shader entry points");
    std::string error;
    auto desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = vertex_function;
    desc.fragmentFunction = fragment_function;
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
    if (easu_pipeline && rcas_pipeline && rcas_present_pipeline) return;
    const auto directory = assets / "shaders-builtin/vulkan";
    const auto rcas_words = load_spirv(directory / "fsr_filter_rcas.comp.spv");
    auto rcas = shader::metal::convert_builtin_compute_spirv(rcas_words);
    if (!easu_pipeline || !rcas_pipeline) {
        auto easu = shader::metal::convert_builtin_compute_spirv(load_spirv(directory / "fsr_filter_easu.comp.spv"));
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
    if (hex_string(sha256(rcas_words.data(), rcas_words.size() * sizeof(uint32_t))) != fsr_rcas_module_sha256)
        return;
    try {
        // Preserve the original half-precision RCAS functions verbatim. Only
        // the entry point changes: write directly to the BGRA render target.
        rcas_present_pipeline = create_filter_pipeline(rcas.source + std::string(fsr_rcas_fragment), "screen_rcas");
    } catch (const std::exception &error) {
        // Optional optimization: the existing FSR compute path remains usable.
        // Selecting FSR again retries; never retry compilation every frame.
        LOG_WARN("Metal FSR fused presentation unavailable: {}; using compute RCAS", error.what());
    }
}
void ScreenRenderer::disable_fsr(std::string_view reason) {
    // FSR uses the bilinear presentation sampler already. Stop retrying the
    // failed path each frame; selecting FSR again retries initialization.
    filter = Filter::Bilinear;
    LOG_WARN("Metal FSR unavailable: {}; using Bilinear", reason);
}
void ScreenRenderer::set_filter(bool linear) {
    fsr_fused_source = nil;
    metalfx_scaler = nil;
    metalfx_input = nil;
    metalfx_output = nil;
    metalfx_dimensions = {};
    filter = linear ? Filter::Bilinear : Filter::Nearest;
    auto desc = [MTLSamplerDescriptor new];
    desc.minFilter = desc.magFilter = linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    desc.sAddressMode = desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.supportArgumentBuffers = YES;
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
        set_filter(true);
        try {
            initialize_fsr();
            filter = Filter::FSR;
        } catch (const std::exception &error) {
            disable_fsr(error.what());
        }
    } else if (name == "MetalFX Spatial") {
        set_filter(true);
        if ([MTLFXSpatialScalerDescriptor supportsDevice:device.native_device()])
            filter = Filter::MetalFX;
        else
            LOG_WARN("MetalFX Spatial is unavailable on this device; using Bilinear");
    } else {
        set_filter(name != "Nearest");
    }
}
id<MTLTexture> ScreenRenderer::prepare_metalfx(V3KMetal4Commands * commands, id<MTLTexture> source,
    uint32_t width, uint32_t height) {
    // Spatial reconstruction is useful only when enlarging the image. Mixed-axis
    // downscaling and pixel-perfect cropping use the normal presentation sampler.
    if (width < source.width || height < source.height || (width == source.width && height == source.height))
        return source;
    const std::array<NSUInteger, 4> dimensions{source.width, source.height, width, height};
    if (metalfx_dimensions != dimensions) {
        metalfx_dimensions = dimensions;
        metalfx_scaler = nil;
        metalfx_input = nil;
        metalfx_output = nil;
        auto desc = [MTLFXSpatialScalerDescriptor new];
        desc.inputWidth = source.width;
        desc.inputHeight = source.height;
        desc.outputWidth = width;
        desc.outputHeight = height;
        desc.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
        desc.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
        // Presentation receives encoded framebuffer values, including UNORM
        // views of sRGB surfaces. Do not linearize these bytes a second time.
        desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
        auto scaler = [desc newSpatialScalerWithDevice:device.native_device() compiler:device.native_compiler()];
        if (scaler) {
            auto texture = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:desc.colorTextureFormat
                width:source.width height:source.height mipmapped:NO];
            texture.storageMode = MTLStorageModePrivate;
            texture.usage = scaler.colorTextureUsage | MTLTextureUsageRenderTarget;
            metalfx_input = [device.native_device() newTextureWithDescriptor:texture];
            texture.width = width;
            texture.height = height;
            texture.usage = scaler.outputTextureUsage | MTLTextureUsageShaderRead;
            metalfx_output = [device.native_device() newTextureWithDescriptor:texture];
            if (metalfx_input && metalfx_output) {
                metalfx_input.label = @"MetalFX Spatial input";
                metalfx_output.label = @"MetalFX Spatial output";
                metalfx_scaler = scaler;
            }
        }
        if (!metalfx_scaler) {
            metalfx_input = nil;
            metalfx_output = nil;
            // Keep the failed dimensions to avoid allocations and log spam on
            // every frame. Resizing or selecting the filter again retries.
            LOG_WARN("MetalFX Spatial could not initialize {}x{} -> {}x{}; using Bilinear",
                source.width, source.height, width, height);
        }
    }
    if (!metalfx_scaler) return source;
    // Normalize arbitrary guest surface formats and texture usage to the
    // scaler's contract. Producers, conversion, upscale and presentation all
    // remain ordered on the same command buffer with tracked resources.
    auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = metalfx_input;
    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) return source;
    encoder.label = @"MetalFX Spatial input conversion";
    render(encoder, source, {0, 0, double(source.width), double(source.height), 0, 1});
    [encoder endEncoding];
    metalfx_scaler.inputContentWidth = source.width;
    metalfx_scaler.inputContentHeight = source.height;
    metalfx_scaler.colorTexture = metalfx_input;
    metalfx_scaler.outputTexture = metalfx_output;
    [commands retainResource:metalfx_input];
    [commands retainResource:metalfx_output];
    [commands retainObject:metalfx_scaler];
    [metalfx_scaler encodeToCommandBuffer:commands.native];
    return metalfx_output;
}
id<MTLTexture> ScreenRenderer::prepare(V3KMetal4Commands * commands, id<MTLTexture> source,
    MTLViewport viewport) {
    fsr_fused_source = nil;
    if (!std::isfinite(viewport.width) || !std::isfinite(viewport.height)
        || viewport.width < 0.5 || viewport.height < 0.5
        || viewport.width > UINT32_MAX || viewport.height > UINT32_MAX) return source;
    const auto width = uint32_t(std::lround(viewport.width));
    const auto height = uint32_t(std::lround(viewport.height));
    if (!commands || !source || !width || !height) return source;
    if (filter == Filter::MetalFX) return prepare_metalfx(commands, source, width, height);
    if (filter != Filter::FSR) return source;
    const bool fused = rcas_present_pipeline
        && fsr_pixel_aligned(viewport.originX, viewport.originY, viewport.width, viewport.height);
    if (!fsr_intermediate || fsr_intermediate.width != width || fsr_intermediate.height != height
        || (!fused && (!fsr_output || fsr_output.width != width || fsr_output.height != height))) {
        auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                       width:width height:height mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        // Publish only after every required allocation succeeds. A pixel-aligned
        // frame needs just EASU's RGBA16F texture, preserving its precision.
        auto intermediate = fsr_intermediate;
        if (!intermediate || intermediate.width != width || intermediate.height != height)
            intermediate = [device.native_device() newTextureWithDescriptor:desc];
        auto output = fsr_output;
        if (!fused && (!output || output.width != width || output.height != height))
            output = [device.native_device() newTextureWithDescriptor:desc];
        if (!intermediate || (!fused && !output)) {
            disable_fsr("cannot allocate FSR textures");
            return source;
        }
        fsr_intermediate = intermediate;
        fsr_output = output;
    }
    // Encoders retain previously used textures through GPU completion. Keeping
    // an idle RCAS output here would defeat the one-intermediate memory saving.
    if (fused) fsr_output = nil;
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
    if (!encoder) {
        disable_fsr("cannot begin FSR upscaling pass");
        return source;
    }
    [encoder setComputePipelineState:easu_pipeline];
    [encoder setBytes:&easu length:sizeof(easu) atIndex:0];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:fsr_intermediate atIndex:1];
    [encoder setSamplerState:sampler atIndex:0];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    if (fused) {
        fsr_fused_source = fsr_intermediate;
        return fsr_intermediate;
    }
    encoder = [commands computeCommandEncoder];
    if (!encoder) {
        // EASU may already be encoded, but presentation can still sample the
        // original image rather than an unfinished sharpening destination.
        disable_fsr("cannot begin FSR sharpening pass");
        return source;
    }
    [encoder setComputePipelineState:rcas_pipeline];
    [encoder setBytes:&rcas length:sizeof(rcas) atIndex:0];
    [encoder setTexture:fsr_intermediate atIndex:0];
    [encoder setTexture:fsr_output atIndex:1];
    [encoder setSamplerState:sampler atIndex:0];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return fsr_output;
}
void ScreenRenderer::render(V3KMetal4RenderEncoder * encoder, id<MTLTexture> source, MTLViewport viewport) {
    if (!source) return;
    const bool fused = filter == Filter::FSR && fsr_fused_source && source == fsr_fused_source;
    const auto selected = fused ? rcas_present_pipeline
        : filter == Filter::Bicubic ? bicubic_pipeline : filter == Filter::FXAA ? fxaa_pipeline : pipeline;
    [encoder setRenderPipelineState:selected];
    if (fused) {
        const std::array<float, 4> constants{float(viewport.originX), float(viewport.originY), 0.2f, 0};
        [encoder setFragmentBytes:constants.data() length:sizeof(constants) atIndex:0];
    }
    [encoder setViewport:viewport];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setFragmentTexture:source atIndex:0];
    [encoder setFragmentSamplerState:sampler atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}
}
