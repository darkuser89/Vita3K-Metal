// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <gxm/types.h>
#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <tuple>
#include <vector>

namespace renderer::metal {
// Conservative guest-address bound for both layout owners and rectangular
// subimages. All current guest color/texture formats have at most 64 bits per
// pixel; native decoded formats do not determine guest addressing. Full rows
// and 32-row bands deliberately include padding. Exact predicates still decide
// ownership. Saturate before multiplication at the end of the 32-bit space.
template <typename GuestSurface>
uint64_t surface_lookup_end(const GuestSurface &surface) {
    const uint64_t base = surface.data.address(), limit = uint64_t(1) << 32;
    if (!base || !surface.width || !surface.height || surface.strideInPixels < surface.width)
        return base;
    const uint64_t pitch = uint64_t(surface.strideInPixels) * 8;
    const uint64_t rows = (uint64_t(surface.height) + 31) & ~uint64_t(31);
    return rows > (limit - base) / pitch ? limit : base + rows * pitch;
}

// Persistent acceleration of descending map scans. Only address bounds and
// stable map iterators are retained, never eligibility, native format or a
// chosen owner. Invalidate before erasure and after a bound change; insertion
// also invalidates. Rebuild lazily, once, on the next lookup. A failed rebuild
// leaves the index invalid and is retried rather than exposing partial data.
template <typename SurfaceMap>
class SurfaceAddressIndex {
    using Iterator = typename SurfaceMap::iterator;
    std::vector<Iterator> entries;
    std::vector<uint64_t> maximum_end;
    size_t leaf_count = 0;
    bool valid = false;

public:
    void invalidate() { valid = false; }
    void clear() {
        entries.clear(); maximum_end.clear(); leaf_count = 0; valid = false;
    }

    template <typename End, typename Accept>
    Iterator find(SurfaceMap &surfaces, uint64_t address, End end, Accept accept) {
        if (!valid || entries.size() != surfaces.size()) {
            valid = false;
            entries.clear();
            entries.reserve(surfaces.size());
            leaf_count = std::bit_ceil(std::max(size_t(1), surfaces.size()));
            maximum_end.assign(leaf_count * 2, 0);
            for (auto it = surfaces.begin(); it != surfaces.end(); ++it) {
                maximum_end[leaf_count + entries.size()] = end(it->second);
                entries.push_back(it);
            }
            for (size_t n = leaf_count; --n;)
                maximum_end[n] = std::max(maximum_end[2 * n], maximum_end[2 * n + 1]);
            valid = true;
        }
        const size_t upper = std::upper_bound(entries.begin(), entries.end(), address,
            [](uint64_t value, Iterator it) { return value < it->first; }) - entries.begin();
        const auto search = [&](auto &&self, size_t node, size_t first, size_t last) -> Iterator {
            if (first >= upper || maximum_end[node] <= address) return surfaces.end();
            if (last - first == 1)
                return accept(entries[first]) ? entries[first] : surfaces.end();
            const size_t middle = first + (last - first) / 2;
            const auto right = self(self, 2 * node + 1, middle, last);
            return right != surfaces.end() ? right : self(self, 2 * node, first, middle);
        };
        return search(search, 1, 0, leaf_count);
    }
};

// Draw-local only: owners and guest layouts must remain stable for its lifetime.
// The complete descriptor distinguishes pitch/tiling, word phase and formats;
// negative results are cached too. Extra descriptors safely bypass the bound.
template <typename Result>
class DrawTextureLookup {
    using Key = std::array<uint32_t, 4>;
    std::map<Key, Result> entries;

public:
    template <typename Resolve>
    Result get(const SceGxmTexture &texture, Resolve resolve) {
        const auto key = std::bit_cast<Key>(texture);
        if (const auto found = entries.find(key); found != entries.end())
            return found->second;
        auto result = resolve();
        if (entries.size() < SCE_GXM_MAX_TEXTURE_UNITS * 2)
            entries.emplace(key, result);
        return result;
    }
};

// Geometry is pure for one guest-surface object and descriptor during a draw.
// Include the object identity: overlapping owners at different addresses (or
// two objects describing the same address) must never share rectangle results.
// The operation distinguishes ordinary/word/halfword/etc. geometry. Eligibility
// and native texture formats are deliberately evaluated outside this cache.
template <typename Result>
class DrawSurfaceLookup {
    using Key = std::tuple<uintptr_t, std::array<uint32_t, 4>, uint32_t>;
    std::map<Key, Result> entries;

public:
    static constexpr size_t capacity = SCE_GXM_MAX_TEXTURE_UNITS * 2 * 8;

    template <typename Resolve>
    Result get(const void *owner, const SceGxmTexture &texture, uint32_t operation, Resolve resolve) {
        const Key key{reinterpret_cast<uintptr_t>(owner),
            std::bit_cast<std::array<uint32_t, 4>>(texture), operation};
        if (const auto found = entries.find(key); found != entries.end())
            return found->second;
        auto result = resolve();
        if (entries.size() < capacity) entries.emplace(key, result);
        return result;
    }
};
}
