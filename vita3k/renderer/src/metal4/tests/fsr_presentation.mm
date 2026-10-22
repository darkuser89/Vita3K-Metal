// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// CPU API doubles and pixel mapping proof. Never constructs an MTLDevice.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <renderer/metal4/fsr_presentation.h>
#include <shader/metal4/msl_recompiler.h>
#include <util/hash.h>
#include <array>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace renderer::metal;
static size_t allocations, fail_allocation, compute_compiles, fragment_compiles;
static bool fail_fragment;
static std::string fragment_source;
Sha256Hash sha256(const void *data, size_t size) {
    Sha256Hash result; CC_SHA256(data, CC_LONG(size), result.data()); return result;
}
void hex_buf(const uint8_t *hash, char *dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[2*i] = "0123456789abcdef"[hash[i] >> 4];
        dst[2*i+1] = "0123456789abcdef"[hash[i] & 15];
    }
}
namespace shader::metal {
Program convert_builtin_compute_spirv(const std::vector<uint32_t> &) {
    return {"original-compute-source", "main0", Stage::Compute};
}
}

@interface FsrTexture : NSObject
@property(strong) MTLTextureDescriptor *descriptor;
- (NSUInteger)width;
- (NSUInteger)height;
@end
@implementation FsrTexture
- (NSUInteger)width { return _descriptor.width; }
- (NSUInteger)height { return _descriptor.height; }
@end
@interface FsrDevice : NSObject
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)desc;
@end
@implementation FsrDevice
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)desc {
    ++allocations;
    if (allocations == fail_allocation) return nil;
    assert(desc.pixelFormat == MTLPixelFormatRGBA16Float && desc.storageMode == MTLStorageModePrivate);
    assert(desc.usage == (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite));
    assert(desc.mipmapLevelCount == 1 && desc.sampleCount == 1);
    auto t = [FsrTexture new]; t.descriptor = [desc copy]; return (id<MTLTexture>)t;
}
@end
@interface FsrEncoder : NSObject {
@public
    MTLViewport viewport;
    MTLSize groups, threads;
}
@property(strong) id selected;
@property(strong) NSMutableArray *textures;
@property(strong) NSData *constants;
@property NSUInteger draws;
@property BOOL ended;
- (void)setComputePipelineState:(id)p;
- (void)setRenderPipelineState:(id)p;
- (void)setBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i;
- (void)setFragmentBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i;
- (void)setTexture:(id)t atIndex:(NSUInteger)i;
- (void)setFragmentTexture:(id)t atIndex:(NSUInteger)i;
- (void)setSamplerState:(id)s atIndex:(NSUInteger)i;
- (void)setFragmentSamplerState:(id)s atIndex:(NSUInteger)i;
- (void)dispatchThreadgroups:(MTLSize)g threadsPerThreadgroup:(MTLSize)t;
- (void)endEncoding;
- (void)setViewport:(MTLViewport)v;
- (void)setCullMode:(MTLCullMode)c;
- (void)drawPrimitives:(MTLPrimitiveType)t vertexStart:(NSUInteger)s vertexCount:(NSUInteger)n;
@end
@implementation FsrEncoder
- (instancetype)init {
    if ((self = [super init])) _textures = [NSMutableArray arrayWithObjects:[NSNull null], [NSNull null], nil];
    return self;
}
- (void)setComputePipelineState:(id)p { _selected = p; }
- (void)setRenderPipelineState:(id)p { _selected = p; }
- (void)setBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i {
    assert(i == 0); _constants = [NSData dataWithBytes:p length:n];
}
- (void)setFragmentBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i { [self setBytes:p length:n atIndex:i]; }
- (void)setTexture:(id)t atIndex:(NSUInteger)i { assert(i < 2); _textures[i] = t; }
- (void)setFragmentTexture:(id)t atIndex:(NSUInteger)i { [self setTexture:t atIndex:i]; }
- (void)setSamplerState:(id)s atIndex:(NSUInteger)i { assert(s && i == 0); }
- (void)setFragmentSamplerState:(id)s atIndex:(NSUInteger)i { [self setSamplerState:s atIndex:i]; }
- (void)dispatchThreadgroups:(MTLSize)g threadsPerThreadgroup:(MTLSize)t { groups = g; threads = t; }
- (void)endEncoding { assert(!_ended); _ended = YES; }
- (void)setViewport:(MTLViewport)v { viewport = v; }
- (void)setCullMode:(MTLCullMode)c { assert(c == MTLCullModeNone); }
- (void)drawPrimitives:(MTLPrimitiveType)t vertexStart:(NSUInteger)s vertexCount:(NSUInteger)n {
    assert(t == MTLPrimitiveTypeTriangle && !s && n == 3); ++_draws;
}
@end
@interface FsrCommands : NSObject
@property(strong) NSMutableArray *encoders;
@property NSUInteger fail_encoder;
@property NSUInteger attempts;
- (FsrEncoder *)computeCommandEncoder;
@end
@implementation FsrCommands
- (instancetype)init { if ((self = [super init])) _encoders = [NSMutableArray new]; return self; }
- (FsrEncoder *)computeCommandEncoder {
    if (++_attempts == _fail_encoder) return nil;
    auto e = [FsrEncoder new]; [_encoders addObject:e]; return e;
}
@end
using V3KMetal4Commands = FsrCommands;
using V3KMetal4RenderEncoder = FsrEncoder;
namespace renderer::metal {
struct Compiled { id function = @"compute"; };
struct Device {
    FsrDevice *native_device() { static auto device = [FsrDevice new]; return device; }
    std::unique_ptr<Compiled> compile(const shader::metal::Program &, bool, std::string &) {
        ++compute_compiles; return std::make_unique<Compiled>();
    }
    id<MTLComputePipelineState> create_compute_pipeline(id f, std::string &) { return f; }
};
struct ScreenRenderer {
    Device device;
    std::filesystem::path assets;
    enum class Filter { Nearest, Bilinear, Bicubic, FXAA, FSR, MetalFX } filter = Filter::FSR;
    id<MTLComputePipelineState> easu_pipeline = (id)@"EASU", rcas_pipeline = (id)@"RCAS";
    id<MTLRenderPipelineState> pipeline = (id)@"plain", bicubic_pipeline = (id)@"bicubic",
        fxaa_pipeline = (id)@"fxaa", rcas_present_pipeline = (id)@"RCAS-present";
    id<MTLSamplerState> sampler = (id)@"sampler";
    id<MTLTexture> fsr_intermediate, fsr_output, fsr_fused_source;
    unsigned warnings = 0, metalfx_calls = 0;
    void initialize_fsr();
    void disable_fsr(std::string_view reason);
    id<MTLTexture> prepare(V3KMetal4Commands *, id<MTLTexture>, MTLViewport);
    void render(V3KMetal4RenderEncoder *, id<MTLTexture>, MTLViewport);
    id<MTLTexture> prepare_metalfx(V3KMetal4Commands *, id<MTLTexture> s, uint32_t, uint32_t) { ++metalfx_calls; return s; }
    id<MTLRenderPipelineState> create_filter_pipeline(const std::string &source, const char *entry) {
        ++fragment_compiles; fragment_source = source; assert(std::string(entry) == "screen_rcas");
        if (fail_fragment) throw std::runtime_error("injected compiler error");
        return (id)@"RCAS-present";
    }
};
#define LOG_WARN(...) ++warnings
// LOAD_HOOK
// INITIALIZE_HOOK
// DISABLE_HOOK
// PREPARE_HOOK
// RENDER_HOOK
#undef LOG_WARN
}
static id<MTLTexture> input(MTLPixelFormat format) {
    auto t = [FsrTexture new];
    t.descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:960 height:544 mipmapped:NO];
    return (id<MTLTexture>)t;
}
static void check_dispatch(FsrEncoder *e, unsigned width, unsigned height, id pipeline) {
    assert(e.ended && e.selected == pipeline);
    assert(e->groups.width == (width + 15) / 16 && e->groups.height == (height + 15) / 16 && e->groups.depth == 1);
    assert(e->threads.width == 64 && e->threads.height == 1 && e->threads.depth == 1);
}
int main(int argc, char **argv) {
    assert(argc == 3);
    @autoreleasepool {
        // Validate original initialization code, unknown shader fallback and
        // optional compiler failure. The asset path is a temporary test folder.
        ScreenRenderer init; init.assets = argv[1];
        init.easu_pipeline = nil; init.rcas_pipeline = nil; init.rcas_present_pipeline = nil;
        fail_fragment = true; init.initialize_fsr();
        assert(init.easu_pipeline && init.rcas_pipeline && !init.rcas_present_pipeline && init.warnings == 1);
        assert(compute_compiles == 2 && fragment_compiles == 1);
        assert(fragment_source == "original-compute-source" + std::string(fsr_rcas_fragment));
        fail_fragment = false; init.initialize_fsr(); init.initialize_fsr();
        assert(compute_compiles == 2 && fragment_compiles == 2 && init.rcas_present_pipeline);
        init.assets = argv[2]; init.rcas_present_pipeline = nil; init.initialize_fsr();
        assert(compute_compiles == 2 && fragment_compiles == 2 && !init.rcas_present_pipeline);

        size_t cases = 0, mappings = 0;
        for (auto format : {MTLPixelFormatRGBA8Unorm, MTLPixelFormatBGRA8Unorm,
                 MTLPixelFormatRGBA8Unorm_sRGB, MTLPixelFormatRGBA16Float})
            for (unsigned width : {1u, 2u, 15u, 16u, 17u, 960u, 1920u})
                for (unsigned height : {1u, 3u, 15u, 16u, 17u, 544u, 1088u})
                    for (double origin : {-4., 0., 1., 8., 0.5, 1.25}) {
                        ScreenRenderer s; auto commands = [FsrCommands new]; auto src = input(format);
                        const MTLViewport viewport{origin, origin, double(width), double(height), 0, 1};
                        const bool fused = fsr_pixel_aligned(origin, origin, width, height);
                        const auto before = allocations;
                        auto result = s.prepare(commands, src, viewport);
                        assert(allocations - before == (fused ? 1 : 2));
                        assert(commands.encoders.count == (fused ? 1 : 2));
                        auto easu = (FsrEncoder *)commands.encoders[0];
                        check_dispatch(easu, width, height, s.easu_pipeline);
                        assert(easu.textures[0] == src && easu.textures[1] == s.fsr_intermediate);
                        const std::array<uint32_t, 8> expected{0,0,960,544,960,544,width,height};
                        assert(easu.constants.length == sizeof(expected) && !memcmp(easu.constants.bytes, expected.data(), sizeof(expected)));
                        if (fused) assert(result == s.fsr_intermediate && s.fsr_fused_source == result && !s.fsr_output);
                        else {
                            auto rcas = (FsrEncoder *)commands.encoders[1];
                            check_dispatch(rcas, width, height, s.rcas_pipeline);
                            assert(rcas.textures[0] == s.fsr_intermediate && rcas.textures[1] == result && result == s.fsr_output);
                            assert(rcas.constants.length == 16);
                            const auto p = static_cast<const uint32_t *>(rcas.constants.bytes);
                            float sharpness; memcpy(&sharpness, p + 2, 4);
                            assert(!p[0] && !p[1] && sharpness == 0.2f && !p[3]);
                        }
                        auto render = [FsrEncoder new]; s.render(render, result, viewport);
                        assert(render.selected == (fused ? s.rcas_present_pipeline : s.pipeline));
                        assert(render.textures[0] == result && render.draws == 1);
                        assert(!memcmp(&render->viewport, &viewport, sizeof(viewport)));
                        if (fused) {
                            const std::array<float, 4> expected{float(origin), float(origin), 0.2f, 0};
                            assert(render.constants.length == sizeof(expected) && !memcmp(render.constants.bytes, expected.data(), sizeof(expected)));
                        } else assert(!render.constants);
                        ++cases;
                    }
        // Warm reuse and fallback/fused transitions retain the existing EASU
        // image; commands independently hold a removed RCAS destination.
        ScreenRenderer s; auto src = input(MTLPixelFormatBGRA8Unorm);
        const MTLViewport aligned{0, -4, 1920, 1088, 0, 1}, fractional{0.5, -4, 1920, 1088, 0, 1};
        auto previous = [FsrCommands new];
        auto old_output = s.prepare(previous, src, fractional); auto intermediate = s.fsr_intermediate;
        const auto before = allocations;
        for (unsigned i = 0; i < 10000; ++i) {
            auto c = [FsrCommands new]; assert(s.prepare(c, src, aligned) == intermediate);
            assert(c.encoders.count == 1 && !s.fsr_output);
        }
        assert(allocations == before && ((FsrEncoder *)previous.encoders[1]).textures[1] == old_output);
        auto c = [FsrCommands new]; assert(s.prepare(c, src, fractional) != old_output);
        assert(allocations == before + 1 && s.fsr_intermediate == intermediate && c.encoders.count == 2);
        for (bool fused : {false, true}) {
            ScreenRenderer resized;
            auto old_commands = [FsrCommands new];
            const auto viewport = fused ? aligned : fractional;
            auto held = resized.prepare(old_commands, src, viewport);
            auto held_intermediate = resized.fsr_intermediate, held_output = resized.fsr_output;
            const MTLViewport next{fused ? 0. : 0.5, 0, 17, 15, 0, 1};
            fail_allocation = allocations + (fused ? 1 : 2);
            auto new_commands = [FsrCommands new];
            assert(resized.prepare(new_commands, src, next) == src);
            assert(resized.fsr_intermediate == held_intermediate && resized.fsr_output == held_output);
            fail_allocation = 0; resized.filter = ScreenRenderer::Filter::FSR;
            auto result = resized.prepare(new_commands, src, next);
            assert(result != held && result.width == 17 && result.height == 15);
            assert(resized.fsr_intermediate != held_intermediate);
            assert(((FsrEncoder *)old_commands.encoders[0]).textures[1] == held_intermediate);
            if (!fused) assert(((FsrEncoder *)old_commands.encoders[1]).textures[1] == held_output);
            assert(new_commands.encoders.count == (fused ? 1 : 2));
        }
        // Both allocation positions and both compute-encoder failures preserve
        // presentation from the original source; no partial RCAS image leaks.
        for (bool fused : {false, true}) for (unsigned failure : {1u, 2u, 3u, 4u}) {
            if (fused && (failure == 2 || failure == 4)) continue;
            ScreenRenderer s; auto c = [FsrCommands new];
            auto v = fused ? aligned : fractional;
            if (failure <= 2) fail_allocation = allocations + failure;
            else c.fail_encoder = failure - 2;
            assert(s.prepare(c, src, v) == src && s.filter == ScreenRenderer::Filter::Bilinear && !s.fsr_fused_source);
            assert(s.warnings == 1); fail_allocation = 0;
            s.filter = ScreenRenderer::Filter::FSR; c = [FsrCommands new];
            assert(s.prepare(c, src, v) != src);
        }
        // Unknown/missing fragment PSO retains ordinary compute RCAS even for
        // aligned output. Other filters and invalid input never inherit fusion.
        ScreenRenderer fallback; fallback.rcas_present_pipeline = nil;
        c = [FsrCommands new]; fallback.prepare(c, src, aligned); assert(c.encoders.count == 2);
        for (auto filter : {ScreenRenderer::Filter::Nearest, ScreenRenderer::Filter::Bilinear,
                 ScreenRenderer::Filter::Bicubic, ScreenRenderer::Filter::FXAA, ScreenRenderer::Filter::MetalFX}) {
            s.filter = filter; c = [FsrCommands new];
            assert(s.prepare(c, src, aligned) == src && !c.encoders.count && !s.fsr_fused_source);
            auto render = [FsrEncoder new]; s.render(render, src, aligned);
            assert(render.selected == (filter == ScreenRenderer::Filter::Bicubic ? s.bicubic_pipeline
                : filter == ScreenRenderer::Filter::FXAA ? s.fxaa_pipeline : s.pipeline));
        }
        s.filter = ScreenRenderer::Filter::FSR;
        for (auto invalid : {0., -1., double(INFINITY), double(NAN), double(UINT32_MAX) + 1}) {
            c = [FsrCommands new]; assert(s.prepare(c, src, {0,0,invalid,1,0,1}) == src && !c.encoders.count);
        }
        c = [FsrCommands new]; assert(!s.prepare(c, nil, aligned)); assert(s.prepare(nil, src, aligned) == src);

        // Independent mapping: the old final sampler's texel center is exactly
        // the integer RCAS pixel computed by the fragment wrapper. Include
        // cropped/letterboxed origins, odd extents and FP32 precision limits.
        std::mt19937 random(0x46535234);
        for (unsigned i = 0; i < 200000; ++i) {
            const double extent = 1 + random() % 32767, origin = int(random() % 2097153) - 1048576;
            const unsigned pixel = random() % unsigned(extent);
            const double window = origin + pixel + 0.5;
            assert(fsr_pixel_aligned(origin, origin, extent, extent));
            const double old_texel = ((window - origin) / extent) * extent - 0.5;
            const unsigned direct_pixel = unsigned(float(window) - float(origin));
            assert(direct_pixel == pixel && std::abs(old_texel - double(pixel)) < 1e-10);
            ++mappings;
        }
        for (auto v : {0.25, -0.25, double(INFINITY), double(NAN), 1048577.}) assert(!fsr_pixel_aligned(v, 0, 960, 544));
        assert(!fsr_pixel_aligned(0, 0, 960.25, 544) && !fsr_pixel_aligned(0, 0, 32768, 544));
        std::cout << "PASS FSR presentation: " << cases << " production encoding/filter/format/odd-size cases; "
                  << mappings << " aligned pixel mappings; 10000 single-texture reuses; allocation/encoder/compiler/unknown-module fallbacks\n";
    }
}
