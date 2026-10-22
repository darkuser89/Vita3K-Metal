// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace gxm::metal3 {
// GXM map mutations use the existing API ordering. This mutex protects the
// derived cache from concurrent readers; it does not synchronize the source map.
class UniformMappingBounds {
    struct Range { uint64_t begin, end; };
    std::vector<Range> ranges;
    std::mutex mutex;
    const void *source = nullptr;
    size_t source_size = 0;
    bool valid = false;

public:
    void invalidate() {
        const std::lock_guard lock(mutex);
        valid = false;
    }

    template <typename Mappings>
    uint32_t size(const Mappings &mappings, uint32_t address, uint32_t requested) {
        const std::lock_guard lock(mutex);
        if (!valid || source != &mappings || source_size != mappings.size()) {
            // Publish a complete snapshot only. Allocation failure leaves it
            // invalid so a later read retries against the authoritative map.
            valid = false;
            std::vector<Range> updated;
            for (const auto &[base, info] : mappings) {
                if (!info.size) continue;
                const uint64_t end = uint64_t(base) + info.size;
                if (!updated.empty() && base <= updated.back().end)
                    updated.back().end = std::max(updated.back().end, end);
                else updated.push_back({base, end});
            }
            ranges.swap(updated);
            source = &mappings;
            source_size = mappings.size();
            valid = true;
        }
        auto next = std::upper_bound(ranges.begin(), ranges.end(), uint64_t(address),
            [](uint64_t value, const Range &range) { return value < range.begin; });
        if (next == ranges.begin() || uint64_t(address) >= (--next)->end)
            return requested; // Preserve the no-GXM-mapping fallback, including endpoints.
        return static_cast<uint32_t>(std::min<uint64_t>(requested, next->end - address));
    }
};
}
