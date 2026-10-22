// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <Metal/Metal.h>
#include <array>
#include <list>
#include <stdexcept>

namespace renderer::metal {
struct TextureViewDescription {
    MTLPixelFormat format;
    MTLTextureType type;
    NSUInteger levels, slices;
    std::array<MTLTextureSwizzle, 4> swizzle;
    bool format_only = false;
    bool operator==(const TextureViewDescription &) const = default;
};

// A view retains its source in Metal. Holding either strongly in a global
// cache would extend large surfaces' lifetime. Weak entries reuse views owned
// by a texture binding or in-flight commands without keeping either alive.
// Zeroing weak references also prevent a recycled object address from hitting.
class TextureViewCache {
    struct Entry {
        __weak id<MTLTexture> source;
        __weak id<MTLTexture> view;
        TextureViewDescription description;
    };
    std::list<Entry> entries;
public:
    static constexpr size_t capacity = 128;
    id<MTLTexture> get(id<MTLTexture> source, const TextureViewDescription &description) {
        if (!source) throw std::invalid_argument("Metal: null texture view source");
        for (auto it = entries.begin(); it != entries.end();) {
            id<MTLTexture> owner = it->source, view = it->view;
            if (!owner || !view) { it = entries.erase(it); continue; }
            if (owner == source && it->description == description) {
                entries.splice(entries.begin(), entries, it);
                return view;
            }
            ++it;
        }
        id<MTLTexture> view = description.format_only
            ? [source newTextureViewWithPixelFormat:description.format]
            : [source newTextureViewWithPixelFormat:description.format
                textureType:description.type levels:NSMakeRange(0, description.levels)
                slices:NSMakeRange(0, description.slices)
                swizzle:MTLTextureSwizzleChannelsMake(description.swizzle[0], description.swizzle[1],
                    description.swizzle[2], description.swizzle[3])];
        if (!view) throw std::runtime_error("Metal: cannot create texture view");
        entries.push_front({source, view, description});
        if (entries.size() > capacity) entries.pop_back();
        return view;
    }
    size_t size() const { return entries.size(); }
};

inline id<MTLTexture> cached_texture_view(id<MTLTexture> source, const TextureViewDescription &description) {
    // Texture preparation is render-thread owned. Separate caches also avoid
    // sharing mutable cache state with future shader/upload workers.
    static thread_local TextureViewCache cache;
    return cache.get(source, description);
}
}
