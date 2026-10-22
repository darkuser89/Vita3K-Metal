// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// CPU API boundary doubles, no actual Metal device or presentation.
#include <renderer/metal4/presentation_scratch.h>
#include <cassert>
#include <cstring>
#include <iostream>

static size_t allocations, copies;
static bool fail_allocation;
static size_t pixel_size(MTLPixelFormat format) {
    return format == MTLPixelFormatRGBA16Float ? 8 : 4;
}
@interface ScratchTexture : NSObject
@property(strong) MTLTextureDescriptor *descriptor;
@property(strong) NSMutableData *pixels;
- (NSUInteger)width;
- (NSUInteger)height;
- (MTLPixelFormat)pixelFormat;
- (MTLTextureUsage)usage;
- (MTLTextureType)textureType;
- (NSUInteger)sampleCount;
@end
@implementation ScratchTexture
- (NSUInteger)width { return _descriptor.width; }
- (NSUInteger)height { return _descriptor.height; }
- (MTLPixelFormat)pixelFormat { return _descriptor.pixelFormat; }
- (MTLTextureUsage)usage { return _descriptor.usage; }
- (MTLTextureType)textureType { return _descriptor.textureType; }
- (NSUInteger)sampleCount { return _descriptor.sampleCount; }
@end
@interface ScratchDevice : NSObject
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)desc;
@end
@implementation ScratchDevice
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)desc {
    if (fail_allocation) return nil;
    assert(desc.textureType == MTLTextureType2D && desc.sampleCount == 1);
    assert(desc.storageMode == MTLStorageModeShared && desc.mipmapLevelCount == 1);
    auto texture = [ScratchTexture new]; texture.descriptor = [desc copy];
    texture.pixels = [NSMutableData dataWithLength:desc.width * desc.height * pixel_size(desc.pixelFormat)];
    ++allocations;
    return (id<MTLTexture>)texture;
}
@end
@interface ScratchBlit : NSObject
- (void)copyFromTexture:(id<MTLTexture>)source sourceSlice:(NSUInteger)slice sourceLevel:(NSUInteger)level
    sourceOrigin:(MTLOrigin)origin sourceSize:(MTLSize)size toTexture:(id<MTLTexture>)target
    destinationSlice:(NSUInteger)targetSlice destinationLevel:(NSUInteger)targetLevel destinationOrigin:(MTLOrigin)dest;
- (void)endEncoding;
@end
@implementation ScratchBlit
- (void)copyFromTexture:(id<MTLTexture>)source sourceSlice:(NSUInteger)slice sourceLevel:(NSUInteger)level
    sourceOrigin:(MTLOrigin)origin sourceSize:(MTLSize)size toTexture:(id<MTLTexture>)target
    destinationSlice:(NSUInteger)targetSlice destinationLevel:(NSUInteger)targetLevel destinationOrigin:(MTLOrigin)dest {
    assert(!slice && !level && !targetSlice && !targetLevel && !origin.z);
    assert(size.depth == 1 && !dest.x && !dest.y && !dest.z);
    assert(target != source && target.pixelFormat == source.pixelFormat);
    assert(size.width == target.width && size.height == target.height);
    assert(origin.x + size.width <= source.width && origin.y + size.height <= source.height);
    const auto bytes = pixel_size(source.pixelFormat);
    for (NSUInteger y = 0; y < size.height; ++y)
        std::memcpy(static_cast<uint8_t *>(((ScratchTexture *)target).pixels.mutableBytes) + y * size.width * bytes,
            static_cast<const uint8_t *>(((ScratchTexture *)source).pixels.bytes)
                + ((y + origin.y) * source.width + origin.x) * bytes, size.width * bytes);
    ++copies;
}
- (void)endEncoding {}
@end
@interface ScratchCommands : NSObject
@property MTLCommandBufferStatus status;
@property(strong) NSMutableArray *resources;
- (void)retainResource:(id)resource;
- (V3KMetal4ComputeEncoder *)blitCommandEncoder;
@end
@implementation ScratchCommands
- (instancetype)init {
    if ((self = [super init])) { _status = MTLCommandBufferStatusNotEnqueued; _resources = [NSMutableArray new]; }
    return self;
}
- (void)retainResource:(id)resource { [_resources addObject:resource]; }
- (V3KMetal4ComputeEncoder *)blitCommandEncoder { return (V3KMetal4ComputeEncoder *)[ScratchBlit new]; }
@end
int main() {
    @autoreleasepool {
        using renderer::metal::PresentationScratch;
        auto device = (id<MTLDevice>)[ScratchDevice new];
        PresentationScratch scratch;
        auto commands = [ScratchCommands new];
        auto first = scratch.acquire(device, MTLPixelFormatRGBA8Unorm, 17, 9,
            MTLTextureUsageShaderRead, (V3KMetal4Commands *)commands);
        for (size_t frame = 0; frame < 10000; ++frame) {
            commands.status = MTLCommandBufferStatusCompleted;
            scratch.completed((V3KMetal4Commands *)commands);
            commands = [ScratchCommands new];
            auto texture = scratch.acquire(device, MTLPixelFormatRGBA8Unorm, 17, 9,
                MTLTextureUsageShaderRead, (V3KMetal4Commands *)commands);
            assert(texture == first && [commands.resources containsObject:texture]);
        }
        assert(allocations == 1);
        // Unsubmitted, committed, scheduled and failed commands must not grant
        // reuse, even if completed() is called at an incorrect boundary.
        for (auto status : {MTLCommandBufferStatusNotEnqueued, MTLCommandBufferStatusCommitted,
                 MTLCommandBufferStatusScheduled, MTLCommandBufferStatusError}) {
            commands.status = status;
            scratch.completed((V3KMetal4Commands *)commands);
            auto next_commands = [ScratchCommands new];
            auto next = scratch.acquire(device, MTLPixelFormatRGBA8Unorm, 17, 9,
                MTLTextureUsageShaderRead, (V3KMetal4Commands *)next_commands);
            assert(next != first && [commands.resources containsObject:first]);
            first = next; commands = next_commands;
        }
        // Every descriptor field is part of the key; failed replacement leaves
        // the previous reusable texture intact. No size/format history is held.
        for (size_t key = 0; key < 4; ++key) {
            commands.status = MTLCommandBufferStatusCompleted;
            commands = [ScratchCommands new];
            first = scratch.acquire(device, MTLPixelFormatRGBA8Unorm, 17, 9,
                MTLTextureUsageShaderRead, (V3KMetal4Commands *)commands);
            commands.status = MTLCommandBufferStatusCompleted;
            auto next_commands = [ScratchCommands new];
            auto next = scratch.acquire(device, key == 0 ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA8Unorm,
                key == 1 ? 18 : 17, key == 2 ? 10 : 9,
                key == 3 ? MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget : MTLTextureUsageShaderRead,
                (V3KMetal4Commands *)next_commands);
            assert(next != first); first = next; commands = next_commands;
        }
        commands.status = MTLCommandBufferStatusCompleted;
        auto retry = [ScratchCommands new]; fail_allocation = true;
        bool failed = false;
        try { scratch.acquire(device, MTLPixelFormatRGBA8Unorm, 100, 100, MTLTextureUsageShaderRead, (V3KMetal4Commands *)retry); }
        catch (const std::runtime_error &) { failed = true; }
        assert(failed); fail_allocation = false;
        assert(scratch.acquire(device, first.pixelFormat, first.width, first.height, first.usage, (V3KMetal4Commands *)retry) == first);
        // Exhaust native crop rectangles against CPU row extraction. Bytewise
        // comparison also protects half-float payloads and gamma/raw carriers.
        size_t rectangles = 0;
        for (auto format : {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB,
                 MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float}) {
            auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:9 height:7 mipmapped:NO];
            desc.storageMode = MTLStorageModeShared;
            auto source = [device newTextureWithDescriptor:desc];
            auto pixels = ((ScratchTexture *)source).pixels;
            for (size_t i = 0; i < pixels.length; ++i) static_cast<uint8_t *>(pixels.mutableBytes)[i] = uint8_t(i * 31 + 7);
            auto original = [pixels copy];
            for (uint32_t y = 0; y < 7; ++y) for (uint32_t x = 0; x < 9; ++x)
                for (uint32_t h = 1; h <= 7 - y; ++h) for (uint32_t w = 1; w <= 9 - x; ++w) {
                    auto crop_commands = [ScratchCommands new];
                    auto result = scratch.crop(device, source, x, y, w, h, (V3KMetal4Commands *)crop_commands);
                    const auto bpp = pixel_size(format);
                    auto bytes = ((ScratchTexture *)result).pixels;
                    assert(result.width == w && result.height == h && result.pixelFormat == format);
                    for (uint32_t row = 0; row < h; ++row)
                        assert(!std::memcmp(static_cast<const uint8_t *>(bytes.bytes) + row * w * bpp,
                            static_cast<const uint8_t *>(pixels.bytes) + ((y + row) * 9 + x) * bpp, w * bpp));
                    assert([original isEqualToData:pixels]);
                    crop_commands.status = MTLCommandBufferStatusCompleted;
                    scratch.completed((V3KMetal4Commands *)crop_commands);
                    ++rectangles;
                }
        }
        assert(copies == rectangles);
        __weak id old_texture;
        @autoreleasepool {
            auto last_commands = [ScratchCommands new];
            auto last = scratch.acquire(device, MTLPixelFormatRGBA8Unorm, 1, 1, MTLTextureUsageShaderRead, (V3KMetal4Commands *)last_commands);
            old_texture = last; last_commands.status = MTLCommandBufferStatusCompleted;
            scratch.completed((V3KMetal4Commands *)last_commands);
        }
        assert(old_texture); scratch.clear(); assert(!old_texture);
        std::cout << "PASS presentation scratch: 10000 completed-frame reuses, busy/error isolation, complete keys, allocation recovery, "
            << rectangles << " byte-exact native crop rectangles and cleanup\n";
    }
}
