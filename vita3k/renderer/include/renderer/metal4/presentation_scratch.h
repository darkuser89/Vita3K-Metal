// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal4/metal4_commands.h>
#include <cstdint>
#include <stdexcept>

namespace renderer::metal {
// One current presentation image, never a history of display sizes/formats.
// CPU uploads must not overwrite a texture referenced by unfinished work.
class PresentationScratch {
    id<MTLTexture> texture;
    V3KMetal4Commands *owner;
public:
    id<MTLTexture> acquire(id<MTLDevice> device, MTLPixelFormat format,
        uint32_t width, uint32_t height, MTLTextureUsage usage, V3KMetal4Commands *commands) {
        if (!width || !height || !commands || commands.status != MTLCommandBufferStatusNotEnqueued)
            throw std::runtime_error("Metal: invalid presentation scratch request");
        const bool reusable = !owner || owner.status == MTLCommandBufferStatusCompleted;
        if (!reusable || !texture || texture.width != width || texture.height != height
            || texture.pixelFormat != format || texture.usage != usage) {
            auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                width:width height:height mipmapped:NO];
            desc.storageMode = MTLStorageModeShared;
            desc.usage = usage;
            auto replacement = [device newTextureWithDescriptor:desc];
            if (!replacement) throw std::runtime_error("Metal: presentation scratch allocation failed");
            texture = replacement;
        }
        // Retain even before the first encoder uses it. Replacing this cache
        // entry cannot release a prior submission's texture prematurely.
        [commands retainResource:texture];
        owner = commands;
        return texture;
    }
    id<MTLTexture> crop(id<MTLDevice> device, id<MTLTexture> source, uint32_t x, uint32_t y,
        uint32_t width, uint32_t height, V3KMetal4Commands *commands) {
        if (!source || source.textureType != MTLTextureType2D || source.sampleCount != 1
            || uint64_t(x) + width > source.width || uint64_t(y) + height > source.height)
            throw std::runtime_error("Metal: invalid presentation crop");
        auto result = acquire(device, source.pixelFormat, width, height,
            MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, commands);
        auto encoder = [commands blitCommandEncoder];
        if (!encoder) throw std::runtime_error("Metal: cannot begin presentation crop");
        [encoder copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x,y,0)
            sourceSize:MTLSizeMake(width,height,1) toTexture:result destinationSlice:0
            destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [encoder endEncoding];
        return result;
    }
    void completed(V3KMetal4Commands *commands) {
        // Drop the completed command's other resources as soon as swap ends.
        if (owner == commands && commands.status == MTLCommandBufferStatusCompleted) owner = nil;
    }
    // Caller first retires/discards the corresponding presentation commands.
    void clear() { texture = nil; owner = nil; }
};
}
