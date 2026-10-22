// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal4/surface_write_watch.h>
#include <renderer/metal4/writable_guest_alias.h>
#include <renderer/metal4/publication_edge_pages.h>
#include <cstring>
#include <span>

namespace renderer::metal {
class GuestMemoryPublication {
    Address address = 0, protected_begin = 0, protected_size = 0;
    Address watched_begin = 0, watched_size = 0;
    size_t size = 0;
    SurfaceWriteStamp watch;
    std::unique_ptr<WritableGuestAlias> alias;
    static bool pure_watches(const MemState &mem, Address address, size_t size, bool require_coverage) {
        const uint64_t end = uint64_t(address) + size;
        uint64_t covered = end;
        // protect_tree is in descending address order.
        for (const auto &[base, segment] : mem.protect_tree) {
            if (uint64_t(base) >= end || uint64_t(base) + segment.size <= address) continue;
            if (segment.perm != MemPerm::ReadOnly || segment.write_watch_ranges_only
                || std::any_of(segment.blocks.begin(), segment.blocks.end(), [](const auto &entry) {
                       return !entry.second.retain_on_read;
                   })) return false;
            if (require_coverage && uint64_t(base) + segment.size < covered) return false;
            covered = std::min<uint64_t>(covered, base);
        }
        return !require_coverage || covered <= address;
    }
public:
    template<class Function>
    bool adopt(MemState &mem, const SurfaceWriteStamp &source, std::span<const uint8_t> published,
        Function &&function) const {
        const std::lock_guard generation(mem.generation_mutex);
        const std::lock_guard protection(mem.protect_mutex);
        if (published.size() != size || source.changed() || watch.changed()
            || (watch.token && !watch.token->armed.load())
            || !is_valid_addr_range(mem, address, Address(uint64_t(address) + size))
            || !pure_watches(mem, protected_begin, protected_size, false)
            || (watched_size && !pure_watches(mem, watched_begin, watched_size, true))) return false;
        PublicationEdgePages edges(mem.memory.get() + address, size);
        if (!WritableGuestAlias::original_is_read_only(mem.memory.get() + protected_begin, protected_size)) return false;
        // A completed job no longer blocks free. The original allocation may
        // have been replaced before render-thread adoption: inspect a fresh
        // mapping, never the job's retained alias of an old VM object.
        WritableGuestAlias current(mem.memory.get() + address, size);
        if (!current.contents() || std::memcmp(current.contents(), published.data(), size)) return false;
        function(published);
        return true;
    }
    // A queue reservation must already protect the MemState against free.
    // Persistent watches cover only fully owned interior host pages. Edge
    // pages are protected briefly under the allocator/protection locks, so a
    // neighboring allocation cannot unprotect them during the actual copy.
    static std::shared_ptr<GuestMemoryPublication> prepare(MemState &mem,
        SurfaceWriteTracker &tracker, Address address, size_t size) {
        const std::lock_guard generation(mem.generation_mutex);
        const uint64_t page = mem.host_page_size, end = uint64_t(address) + size;
        if (mem.use_page_table || page != vm_page_size || !address || !size
            || size > 64u * 1024u * 1024u || end > uint64_t(UINT32_MAX) - page) return {};
        const uint64_t begin = uint64_t(address) / page * page;
        const uint64_t limit = (end + page - 1) / page * page;
        if (begin < page || !is_valid_addr_range(mem, address, Address(end))) return {};
        {
            const std::lock_guard protection(mem.protect_mutex);
            if (!pure_watches(mem, Address(begin), size_t(limit - begin), false)) return {};
        }
        auto result = std::make_shared<GuestMemoryPublication>();
        result->address = address; result->size = size;
        result->protected_begin = Address(begin); result->protected_size = Address(limit - begin);
        const uint64_t inside_begin = (uint64_t(address) + page - 1) / page * page;
        const uint64_t inside_end = end / page * page;
        if (inside_end > inside_begin) {
            result->watched_begin = Address(inside_begin);
            result->watched_size = Address(inside_end - inside_begin);
            result->watch = tracker.capture(mem, Address(inside_begin), size_t(inside_end - inside_begin));
            if (!result->watch.token) return {};
        }
        result->alias = std::make_unique<WritableGuestAlias>(mem.memory.get() + address, size);
        return result->alias->contents() ? result : nullptr;
    }
    template<class Range>
    bool publish(MemState &mem, const SurfaceWriteStamp &source,
        std::span<const uint8_t> expected, std::span<const uint8_t> converted, std::span<const Range> ranges) const {
        if (expected.size() != size || converted.size() != size)
            throw std::runtime_error("Metal: publication storage size changed");
        size_t previous_end = 0;
        for (const auto &range : ranges) {
            if (!range.size || range.offset < previous_end || range.offset > size || range.size > size - range.offset)
                throw std::runtime_error("Metal: invalid publication write ranges");
            previous_end = range.offset + range.size;
        }
        const std::lock_guard generation(mem.generation_mutex);
        const std::lock_guard protection(mem.protect_mutex);
        if (!is_valid_addr_range(mem, address, Address(uint64_t(address) + size)))
            throw std::runtime_error("Metal: pending publication lost its allocation");
        // A consumed watch means the CPU could already be storing into these
        // bytes. Never infer ownership just from their current byte values.
        if (source.changed() || watch.changed() || (watch.token && !watch.token->armed.load())) return false;
        if (!pure_watches(mem, protected_begin, protected_size, false)
            || (watched_size && !pure_watches(mem, watched_begin, watched_size, true)))
            throw std::runtime_error("Metal: pending publication acquired a foreign memory trap");
        PublicationEdgePages edges(mem.memory.get() + address, size);
        if (!WritableGuestAlias::original_is_read_only(mem.memory.get() + protected_begin, protected_size))
            throw std::runtime_error("Metal: pending publication lost kernel write protection");
        if (std::memcmp(alias->contents(), expected.data(), size) != 0) return false;
        if (ranges.empty()) return true;
        struct WritebackScope {
            WritebackScope() { surface_writeback_depth.fetch_add(1); }
            ~WritebackScope() { surface_writeback_depth.fetch_sub(1); }
        } scope;
        // Original guest pages remain read-only until every owned byte is
        // written. A faulting CPU store waits on protect_mutex and runs after
        // publication, so its value cannot be overwritten by this job.
        for (const auto &range : ranges)
            std::memcpy(alias->contents() + range.offset, converted.data() + range.offset, range.size);
        for (auto it = mem.protect_tree.begin(); it != mem.protect_tree.end();) {
            if (uint64_t(it->first) >= uint64_t(address) + size || uint64_t(it->first) + it->second.size <= address) { ++it; continue; }
            const auto touched = std::max(address, it->first);
            for (auto &[base, block] : it->second.blocks) block.callback(touched, true);
            unprotect_inner(mem, it->first, it->second.size);
            it = mem.protect_tree.erase(it);
        }
        edges.published();
        return true;
    }
};
}
