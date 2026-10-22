// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#import <Metal/Metal.h>
#include <array>
#include <memory>
#include <stdexcept>

namespace renderer::metal {
struct ReadbackBuffer {
    id<MTLBuffer> buffer;
};
// Render-thread acquisition; leases can be released by a completion worker.
// A returned payload must retain its lease through GPU completion AND CPU
// conversion/publication. ARC alone cannot detect that the bytes are in use.
class ReadbackBufferPool {
    static constexpr size_t max_bytes = 64 * 1024 * 1024;
    std::array<std::shared_ptr<const ReadbackBuffer>, 2> cached{};
public:
    std::shared_ptr<const ReadbackBuffer> acquire(id<MTLDevice> device, size_t length) {
        size_t best = cached.size();
        for (size_t i = 0; i < cached.size(); ++i)
            if (cached[i] && cached[i].use_count() == 1 && cached[i]->buffer.length >= length
                && (best == cached.size() || cached[i]->buffer.length < cached[best]->buffer.length)) best = i;
        if (best != cached.size()) return cached[best];
        id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (!buffer) throw std::runtime_error("Metal: readback buffer allocation failed");
        auto lease = std::make_shared<const ReadbackBuffer>(ReadbackBuffer{buffer});
        if (buffer.length > max_bytes) return lease;
        size_t bytes = 0, empty = cached.size();
        for (size_t i = 0; i < cached.size(); ++i) {
            if (cached[i]) bytes += cached[i]->buffer.length;
            else empty = i;
        }
        for (size_t i = 0; i < cached.size() && (empty == cached.size() || bytes + buffer.length > max_bytes); ++i)
            if (cached[i] && cached[i].use_count() == 1) {
                bytes -= cached[i]->buffer.length;
                cached[i].reset(); empty = i;
            }
        if (empty != cached.size() && bytes + buffer.length <= max_bytes) cached[empty] = lease;
        return lease;
    }
};
}
