// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#import <Metal/Metal.h>
#include <array>
#include <cstring>
#include <stdexcept>

namespace renderer::metal {
// Render-thread only. A checked-out buffer belongs exclusively to its pass.
// Return it only after GPU completion AND publication of every CPU query.
// Deferred submissions with no query results deliberately bypass recycling.
class VisibilityBufferPool {
    static constexpr size_t max_bytes = 1024 * 1024;
    std::array<id<MTLBuffer>, 8> available{};
    size_t bytes = 0;
public:
    id<MTLBuffer> acquire(id<MTLDevice> device, size_t length) {
        size_t best = available.size();
        for (size_t i = 0; i < available.size(); ++i)
            if (available[i] && available[i].length >= length
                && (best == available.size() || available[i].length < available[best].length))
                best = i;
        id<MTLBuffer> buffer;
        if (best != available.size()) {
            buffer = available[best];
            bytes -= buffer.length;
            available[best] = nil;
        } else {
            buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        }
        if (!buffer) throw std::runtime_error("Metal: visibility result buffer allocation failed");
        // A query with no passing samples must not retain an earlier result.
        std::memset(buffer.contents, 0, buffer.length);
        return buffer;
    }
    void recycle_after_publication(id<MTLBuffer> buffer) {
        if (!buffer || buffer.length > max_bytes - bytes) return;
        for (const auto &entry : available) if (entry == buffer) return;
        for (auto &entry : available) if (!entry) {
            entry = buffer;
            bytes += buffer.length;
            return;
        }
    }
};
}
