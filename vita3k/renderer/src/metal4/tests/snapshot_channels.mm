// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Float-bit/channel proof and production snapshot encoding with CPU doubles.
#include <renderer/metal4/snapshot_channels.h>
#include <renderer/metal4/texture_layout.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace renderer::metal;
using Words = std::array<uint32_t, 4>;
static constexpr uint32_t one = 0x3f800000;
static Words swizzle_words(Words words, MTLTextureSwizzleChannels s) {
    Words result{};
    std::array selections{s.red, s.green, s.blue, s.alpha};
    for (size_t i = 0; i < 4; ++i) {
        switch (selections[i]) {
        case MTLTextureSwizzleZero: result[i] = 0; break;
        case MTLTextureSwizzleOne: result[i] = one; break;
        case MTLTextureSwizzleRed: result[i] = words[0]; break;
        case MTLTextureSwizzleGreen: result[i] = words[1]; break;
        case MTLTextureSwizzleBlue: result[i] = words[2]; break;
        case MTLTextureSwizzleAlpha: result[i] = words[3]; break;
        default: assert(false);
        }
    }
    return result;
}
static Words roundtrip(Words value, const SnapshotChannels &layout, bool outside = false) {
    if (!layout.packed()) return outside ? Words{} : value;
    // Model writes to physical R/RG storage, followed by Metal's defaults and
    // the returned reconstruction view. Missing A=1 must never leak out.
    Words stored{0, 0, 0, one};
    if (!outside) {
        stored[0] = value[layout.pack[0]];
        if (layout.format == MTLPixelFormatRG32Float) stored[1] = value[layout.pack[1]];
    }
    return swizzle_words(stored, layout.swizzle);
}

static size_t prove_channels() {
    struct Format { MTLPixelFormat value; unsigned real; };
    const Format formats[] = {
        {MTLPixelFormatR8Unorm, 1}, {MTLPixelFormatR8Snorm, 1},
        {MTLPixelFormatR16Unorm, 1}, {MTLPixelFormatR16Snorm, 1},
        {MTLPixelFormatR16Float, 1}, {MTLPixelFormatR32Float, 1},
        {MTLPixelFormatRG8Unorm, 3}, {MTLPixelFormatRG8Snorm, 3},
        {MTLPixelFormatRG16Unorm, 3}, {MTLPixelFormatRG16Snorm, 3},
        {MTLPixelFormatRG16Float, 3}, {MTLPixelFormatRG32Float, 3},
        {MTLPixelFormatA8Unorm, 8}, {MTLPixelFormatRGBA32Float, 15},
        {MTLPixelFormatRGBA16Float, 15}, {MTLPixelFormatRGBA16Unorm, 15},
        {MTLPixelFormatRGBA8Unorm_sRGB, 15}, {MTLPixelFormatDepth32Float, 15},
        {MTLPixelFormatStencil8, 15}, {MTLPixelFormatBC1_RGBA, 15},
    };
    const MTLTextureSwizzle options[] = {MTLTextureSwizzleZero, MTLTextureSwizzleOne,
        MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha};
    std::mt19937 random(0x4348414e);
    size_t cases = 0;
    for (const auto &format : formats)
        for (auto r : options) for (auto g : options) for (auto b : options) for (auto a : options) {
            const auto s = MTLTextureSwizzleChannelsMake(r, g, b, a);
            const auto mask = snapshot_live_components(format.value, s);
            const auto layout = snapshot_channels(mask);
            for (unsigned pattern = 0; pattern < 16; ++pattern) {
                Words original{};
                // Arbitrary FP32 bits exercise signs, infinities, denormals,
                // quiet/signaling NaN payloads; packing must never quantize.
                for (unsigned i = 0; i < 4; ++i)
                    original[i] = format.real & (1u << i) ? uint32_t(random()) : i == 3 ? one : 0;
                constexpr uint32_t special[] = {0, 0x80000000, 0x7f800000, 0xff800000,
                    0x7fc12345, 0x7f812345, 1, 0x80000001};
                if (pattern < std::size(special)) for (unsigned i = 0; i < 4; ++i)
                    if (format.real & (1u << i)) original[i] = special[(pattern + i) % std::size(special)];
                if (pattern == 0 && (format.real & 1)) original[0] = 0x80000000; // -0 in a live R.
                const auto expected = swizzle_words(original, s);
                for (unsigned i = 0; i < 4; ++i) if (!(mask & (1u << i))) assert(expected[i] == 0);
                assert(roundtrip(expected, layout) == expected);
                assert(roundtrip(expected, layout, true) == Words{});
                ++cases;
            }
        }
    // A chain uses the union, not just its base level or the first override.
    for (uint32_t first = 0; first < 16; ++first)
        for (uint32_t second = 0; second < 16; ++second) {
            const auto plan = snapshot_channels(first | second);
            for (auto mask : {first, second}) {
                Words value{};
                for (unsigned i = 0; i < 4; ++i) if (mask & (1u << i)) value[i] = random();
                assert(roundtrip(value, plan) == value);
            }
            ++cases;
        }
    assert(snapshot_channels(0x10).format == MTLPixelFormatRGBA32Float);
    assert(snapshot_channels(9).format == MTLPixelFormatRG32Float);
    assert(snapshot_channels(11).format == MTLPixelFormatRGBA32Float);
    assert(snapshot_channels(8).format == MTLPixelFormatR32Float);
    const auto unknown = MTLTextureSwizzleChannelsMake(static_cast<MTLTextureSwizzle>(100),
        MTLTextureSwizzleZero, MTLTextureSwizzleZero, MTLTextureSwizzleZero);
    assert(snapshot_live_components(MTLPixelFormatR32Float, unknown) == 15);
    return cases;
}

static size_t libraries, pipelines, submissions, dispatches;
static bool fail_library, fail_pipeline, fail_allocation, fail_view;
@interface SnapshotTexture : NSObject
@property MTLPixelFormat pixelFormat;
@property MTLTextureType textureType;
@property NSUInteger width, height, mipmapLevelCount, face, level, sampleCount;
@property MTLTextureSwizzleChannels swizzle, childSwizzle;
@property(strong) SnapshotTexture *parent;
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format textureType:(MTLTextureType)type levels:(NSRange)levels slices:(NSRange)slices;
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format textureType:(MTLTextureType)type levels:(NSRange)levels slices:(NSRange)slices swizzle:(MTLTextureSwizzleChannels)swizzle;
@end
@implementation SnapshotTexture
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format textureType:(MTLTextureType)type levels:(NSRange)levels slices:(NSRange)slices {
    return [self newTextureViewWithPixelFormat:format textureType:type levels:levels slices:slices swizzle:self.childSwizzle];
}
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format textureType:(MTLTextureType)type levels:(NSRange)levels slices:(NSRange)slices swizzle:(MTLTextureSwizzleChannels)s {
    if (fail_view) { fail_view = false; return nil; }
    assert(levels.length && levels.location + levels.length <= self.mipmapLevelCount);
    assert(slices.length && slices.location + slices.length <= (self.textureType == MTLTextureTypeCube ? 6 : 1));
    auto view = [SnapshotTexture new];
    view.pixelFormat = format; view.textureType = type;
    view.width = std::max(NSUInteger(1), self.width >> levels.location);
    view.height = std::max(NSUInteger(1), self.height >> levels.location);
    view.mipmapLevelCount = levels.length; view.swizzle = s; view.childSwizzle = s;
    view.sampleCount = 1;
    view.face = slices.location; view.level = levels.location; view.parent = self;
    return (id<MTLTexture>)view;
}
@end

@interface SnapshotEncoder : NSObject {
@public
    std::vector<id<MTLTexture>> sources, outputs;
    std::array<uint32_t, 4> pack;
}
@property(strong) NSString *pipeline;
@property(strong) id<MTLTexture> input, output;
@property bool packed;
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline;
- (void)setBytes:(const void *)data length:(NSUInteger)size atIndex:(NSUInteger)index;
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)dispatchThreads:(MTLSize)size threadsPerThreadgroup:(MTLSize)group;
- (void)endEncoding;
@end
@implementation SnapshotEncoder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline { self.pipeline = (NSString *)pipeline; }
- (void)setBytes:(const void *)data length:(NSUInteger)size atIndex:(NSUInteger)index {
    assert(size == sizeof(pack) && index == 0); std::memcpy(pack.data(), data, size); self.packed = true;
}
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index {
    assert(index < 2); if (index == 0) self.input = texture; else self.output = texture;
}
- (void)dispatchThreads:(MTLSize)size threadsPerThreadgroup:(MTLSize)group {
    assert(size.width == self.output.width && size.height == self.output.height && size.depth == 1);
    assert(group.width == 8 && group.height == 8 && group.depth == 1);
    assert(self.packed == [self.pipeline isEqualToString:@"copy_cube_face_packed"]);
    sources.push_back(self.input); outputs.push_back(self.output); ++dispatches;
}
- (void)endEncoding {}
@end
@interface V3KMetal4Commands : NSObject
@property MTLCommandBufferStatus status;
@property(strong) SnapshotEncoder *encoder;
- (SnapshotEncoder *)computeCommandEncoder;
@end
@implementation V3KMetal4Commands
- (SnapshotEncoder *)computeCommandEncoder { self.encoder = [SnapshotEncoder new]; return self.encoder; }
@end
@interface SnapshotDevice : NSObject
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)desc;
@end
@implementation SnapshotDevice
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)desc {
    if (fail_allocation) { fail_allocation = false; return nil; }
    assert(desc.storageMode == MTLStorageModeShared && desc.mipmapLevelCount);
    auto result = [SnapshotTexture new]; result.pixelFormat = desc.pixelFormat;
    result.textureType = desc.textureType; result.width = desc.width; result.height = desc.height;
    result.mipmapLevelCount = desc.mipmapLevelCount;
    result.sampleCount = 1;
    result.swizzle = result.childSwizzle = MTLTextureSwizzleChannelsDefault;
    return (id<MTLTexture>)result;
}
@end
struct Device {
    SnapshotDevice *native = [SnapshotDevice new];
    SnapshotDevice *native_device() { return native; }
    id<MTLLibrary> compile_library(NSString *, id, NSError **) {
        ++libraries;
        if (fail_library) { fail_library = false; return nil; }
        return (id<MTLLibrary>)[NSObject new];
    }
    bool submit_and_wait(V3KMetal4Commands *, std::string &) { ++submissions; return true; }
};
static id<MTLComputePipelineState> cached_compute(Device &, id<MTLLibrary>, NSString *name) {
    ++pipelines;
    if (fail_pipeline) { fail_pipeline = false; throw std::runtime_error("test pipeline error"); }
    return (id<MTLComputePipelineState>)name;
}
static V3KMetal4Commands *last_commands;
static V3KMetal4Commands *surface_command_buffer(Device &, NSString *) {
    last_commands = [V3KMetal4Commands new]; return last_commands;
}
struct CubeSurface { id<MTLTexture> texture; uint32_t face, mip; };
struct SurfaceCaster {
    Device &device;
    id<MTLLibrary> snapshot_library;
    id<MTLComputePipelineState> cube_pipeline, cube_packed_pipeline, depth_pipeline;
    id<MTLTexture> texture_snapshot(id<MTLTexture>, std::span<const CubeSurface>, float, V3KMetal4Commands *);
    id<MTLTexture> sampling_snapshot(id<MTLTexture>, uint32_t, uint32_t);
    id<MTLComputePipelineState> compute_pipeline(id<MTLComputePipelineState> __strong &, NSString *name) {
        return (id<MTLComputePipelineState>)name;
    }
};
// SNAPSHOT_HOOK
// CAPTURE_HOOK

static SnapshotTexture *texture(MTLPixelFormat format, bool cube = false) {
    auto t = [SnapshotTexture new]; t.width = 9; t.height = cube ? 9 : 5;
    t.mipmapLevelCount = 4; t.textureType = cube ? MTLTextureTypeCube : MTLTextureType2D;
    t.pixelFormat = format; t.swizzle = t.childSwizzle = MTLTextureSwizzleChannelsDefault;
    t.sampleCount = 1;
    return t;
}
static size_t prove_encoding() {
    Device device; SurfaceCaster caster{device}; size_t cases = 0;
    for (bool cube : {false, true}) for (float scale : {0.5f, 0.75f, 1.f, 1.5f, 2.f})
        for (unsigned mode = 0; mode < 6; ++mode) {
            @autoreleasepool {
                auto uploaded = texture(mode == 0 ? MTLPixelFormatR16Float : MTLPixelFormatRGBA32Float, cube);
                std::vector<CubeSurface> overrides;
                if (mode) for (uint32_t face = 0; face < (cube ? 6u : 1u); ++face)
                    for (uint32_t mip = 0; mip < 4; ++mip) {
                        auto source = texture(mode == 5 ? MTLPixelFormatA8Unorm : MTLPixelFormatR32Float);
                        source.mipmapLevelCount = 1; source.width = std::max(1u, 9u >> mip);
                        source.height = std::max(1u, (cube ? 9u : 5u) >> mip);
                        overrides.push_back({(id<MTLTexture>)source, face, mip});
                    }
                if (mode == 2) overrides.back().texture = (id<MTLTexture>)texture(MTLPixelFormatRGBA16Unorm);
                if (mode == 3) {
                    overrides.insert(overrides.begin(), {(id<MTLTexture>)texture(MTLPixelFormatRGBA32Float), 0, 0});
                }
                if (mode == 4) overrides.push_back({nil, 0, 0}); // Last nil restores uploaded face/mip.
                auto commands = [V3KMetal4Commands new]; const auto before = submissions;
                auto result = caster.texture_snapshot((id<MTLTexture>)uploaded, overrides, scale, commands);
                const auto expected_format = mode == 2 || mode == 4 ? MTLPixelFormatRGBA32Float
                    : mode == 5 ? MTLPixelFormatR32Float : MTLPixelFormatRG32Float;
                assert(result.pixelFormat == expected_format && submissions == before);
                assert(result.mipmapLevelCount == 4 && result.textureType == uploaded.textureType);
                const auto extent = snapshot_texture_extent(uploaded.width, uploaded.height, 4, scale);
                assert(result.width == extent[0] && result.height == extent[1]);
                auto encoder = commands.encoder; assert(encoder->sources.size() == (cube ? 24 : 4));
                for (size_t i = 0; i < encoder->sources.size(); ++i) {
                    const auto face = i / 4, mip = i % 4;
                    id<MTLTexture> expected = nil;
                    for (const auto &override : overrides) if (override.face == face && override.mip == mip) expected = override.texture;
                    auto source = (SnapshotTexture *)encoder->sources[i];
                    if (expected) assert((id<MTLTexture>)source == expected);
                    else assert(source.parent == uploaded && source.face == face && source.level == mip);
                    auto output = (SnapshotTexture *)encoder->outputs[i];
                    assert(output.face == face && output.level == mip && output.pixelFormat == expected_format);
                }
                if (expected_format == MTLPixelFormatRG32Float) {
                    assert(encoder->pack[0] == 0 && encoder->pack[1] == 3);
                    assert(result.swizzle.alpha == MTLTextureSwizzleGreen);
                }
                if (expected_format == MTLPixelFormatR32Float) assert(encoder->pack[0] == 3);
                ++cases;
            }
        }
    assert(libraries == 1 && pipelines == 2); // One library, two lazy variants.
    auto t = texture(MTLPixelFormatR32Float);
    // Inspect the effective fallback view rather than assuming its parent's
    // swizzle. This double intentionally supplies a different child swizzle.
    t.childSwizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed,
        MTLTextureSwizzleRed, MTLTextureSwizzleAlpha);
    assert(caster.texture_snapshot((id<MTLTexture>)t, {}, 1, nil).pixelFormat == MTLPixelFormatRGBA32Float);
    assert(submissions == 1);
    // Existing diagnostic capture must reconstruct the sampled RGBA32F
    // image, including the packed view's swizzle on every mip and cube face.
    for (bool cube : {false, true}) {
        auto original = texture(MTLPixelFormatR32Float, cube);
        auto packed = caster.texture_snapshot((id<MTLTexture>)original, {}, 1, [V3KMetal4Commands new]);
        assert(packed.pixelFormat == MTLPixelFormatRG32Float);
        for (unsigned face = 0; face < (cube ? 6u : 1u); ++face)
            for (unsigned mip = 0; mip < 4; ++mip) {
                auto capture = caster.sampling_snapshot(packed, mip, face);
                assert(capture.pixelFormat == MTLPixelFormatRGBA32Float && capture.textureType == MTLTextureType2D);
                auto input = last_commands.encoder->sources.front();
                assert(input.swizzle.red == packed.swizzle.red && input.swizzle.green == packed.swizzle.green
                    && input.swizzle.blue == packed.swizzle.blue && input.swizzle.alpha == packed.swizzle.alpha);
                assert(capture.width == std::max(NSUInteger(1), packed.width >> mip)
                    && capture.height == std::max(NSUInteger(1), packed.height >> mip));
                ++cases;
            }
    }
    for (bool *failure : {&fail_library, &fail_pipeline, &fail_allocation, &fail_view}) {
        SurfaceCaster fresh{device}; *failure = true; bool threw = false;
        try { (void)fresh.texture_snapshot((id<MTLTexture>)t, {}, 1, [V3KMetal4Commands new]); }
        catch (const std::runtime_error &) { threw = true; }
        assert(threw);
        assert(fresh.texture_snapshot((id<MTLTexture>)t, {}, 1, [V3KMetal4Commands new]));
    }
    return cases;
}

int main() {
    @autoreleasepool {
        const auto channel_cases = prove_channels(), encoding_cases = prove_encoding();
        std::cout << "PASS snapshot channels: " << channel_cases << " bit-exact swizzle/format/chain-mask cases; "
                  << encoding_cases << " production cube/2D/mip/NPOT/scale/override encodings; explicit stored alpha, zero borders, lazy pipelines and failure retries; CPU doubles only, ASAN/UBSAN\n";
    }
}
