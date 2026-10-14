// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <mem/functions.h>
#include <mem/state.h>
#include <atomic>
#include <algorithm>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>

namespace renderer::metal {
// Signal callbacks must see this even when the guarded stores are inlined.
inline thread_local std::atomic<uint32_t> surface_writeback_depth{0};
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<bool>::is_always_lock_free);

struct SurfaceWriteToken {
    Address address;
    uint32_t size;
    std::atomic<uint64_t> epoch{0};
    std::atomic<bool> armed{false};
    bool gpu_write_pending = false; // Renderer thread only.
    SurfaceWriteToken(Address address, uint32_t size) : address(address), size(size) {}
};

struct SurfaceWriteStamp {
    std::shared_ptr<SurfaceWriteToken> token;
    uint64_t imported_epoch = 0;
    bool changed() const {
        return token && token->epoch.load() != imported_epoch;
    }
};

class SurfaceWriteTracker {
    // A callback may outlive its surface. It owns only its atomic token; it
    // never accesses this registry or a renderer object from a signal handler.
    std::map<std::pair<Address, uint32_t>, std::weak_ptr<SurfaceWriteToken>> tokens;

    static void arm(MemState &mem, const std::shared_ptr<SurfaceWriteToken> &token) {
        if (token->gpu_write_pending) return;
        if (token->armed.exchange(true)) return;
        try {
            if (!add_write_watch(mem, token->address, token->size, [token](Address, bool write) {
                    if (write && !surface_writeback_depth.load()) token->epoch.fetch_add(1);
                    // Last callback action: a new registration cannot have
                    // its armed state overwritten by this consumed callback.
                    token->armed.store(false);
                    return true;
                })) throw std::runtime_error("Metal: cannot arm surface CPU write watch");
        } catch (...) {
            token->armed.store(false);
            throw;
        }
    }

    static void release_cpu_writeback_protection(MemState &mem, Address address, size_t size) {
        if (mem.use_page_table) return;
        const uint64_t end = uint64_t(address) + size;
        if (!address || !size || end > UINT32_MAX)
            throw std::runtime_error("Metal: invalid CPU writeback range");
        for (;;) {
            Address protected_address = 0;
            {
                const std::lock_guard lock(mem.protect_mutex);
                for (const auto &[base, segment] : mem.protect_tree)
                    if (uint64_t(base) < end && uint64_t(base) + segment.size > address) {
                        protected_address = std::max(base, address);
                        break;
                    }
            }
            if (!protected_address) break;
            // getBytes can write from inside the Metal driver. Release the
            // watched guest page here so its host fault does not first enter
            // Dynarmic's unrelated Mach exception handler.
            if (!handle_access_violation(mem, mem.memory.get() + protected_address, true))
                throw std::runtime_error("Metal: cannot release protection for CPU writeback");
        }
    }

public:
    SurfaceWriteStamp capture(MemState &mem, Address address, size_t size) {
        const uint64_t page = mem.host_page_size;
        const uint64_t end = uint64_t(address) + size;
        if (mem.use_page_table || !page || !address || !size || size > UINT32_MAX
            || end > uint64_t(UINT32_MAX) - page) return {};
        // Never protect a partial host page belonging to another allocation.
        const uint64_t begin = ((uint64_t(address) + page - 1) / page) * page;
        const uint64_t limit = (end / page) * page;
        if (begin >= limit) return {};
        auto &weak = tokens[{Address(begin), uint32_t(limit - begin)}];
        auto token = weak.lock();
        if (!token) {
            token = std::make_shared<SurfaceWriteToken>(Address(begin), uint32_t(limit - begin));
            weak = token;
        }
        arm(mem, token);
        return {token, token->epoch.load()};
    }

    void rearm(MemState &mem) {
        for (auto it = tokens.begin(); it != tokens.end();) {
            if (auto token = it->second.lock()) {
                // A retired/unmapped surface must not protect a dead range.
                if (!token->armed.load() && is_valid_addr_range(mem, token->address,
                        token->address + token->size)) arm(mem, token);
                ++it;
            } else it = tokens.erase(it);
        }
    }

    template <typename Function>
    auto writeback(MemState &mem, Function &&function) {
        struct Guard {
            Guard() { surface_writeback_depth.fetch_add(1); }
            ~Guard() { surface_writeback_depth.fetch_sub(1); }
        } guard;
        try {
            auto result = function();
            // Rearm every live token: overlapping callbacks may all have
            // been consumed by this write, including another surface's watch.
            rearm(mem);
            return result;
        } catch (...) {
            const auto failure = std::current_exception();
            try { rearm(mem); } catch (...) {}
            std::rethrow_exception(failure);
        }
    }

    template <typename Function>
    auto writeback(MemState &mem, Address address, size_t size, Function &&function) {
        return writeback(mem, [&] {
            release_cpu_writeback_protection(mem, address, size);
            return function();
        });
    }
    // Apple no-copy buffers created over CPU-ReadOnly pages can receive GPU
    // stores in a separate mapping. Make shader-store ranges writable before
    // creating their resource, then restore watches after GPU completion.
    void prepare_gpu_writes(MemState &mem, Address address, size_t size) {
        const uint64_t page = mem.host_page_size;
        const uint64_t requested_end = uint64_t(address) + size;
        if (mem.use_page_table || !page || !address || !size || size > UINT32_MAX || requested_end > UINT32_MAX)
            throw std::runtime_error("Metal: invalid GPU write-watch range");
        address = Address((uint64_t(address) / page) * page);
        const uint64_t end = ((requested_end + page - 1) / page) * page;
        if (!address || end > UINT32_MAX)
            throw std::runtime_error("Metal: invalid aligned GPU write-watch range");
        for (const auto &[range, weak] : tokens)
            if (uint64_t(range.first) < end && uint64_t(range.first) + range.second > address)
                if (auto token = weak.lock()) token->gpu_write_pending = true;
        writeback(mem, [&] {
            for (;;) {
                Address protected_address = 0;
                {
                    const std::lock_guard lock(mem.protect_mutex);
                    for (const auto &[base, segment] : mem.protect_tree)
                        if (uint64_t(base) < end && uint64_t(base) + segment.size > address) {
                            protected_address = std::max(base, address);
                            break;
                        }
                }
                if (!protected_address) break;
                // Notify legacy texture callbacks too: GPU stores invalidate
                // their bytes just as a CPU store would. Surface tokens are
                // suppressed by the writeback scope and remain pending.
                if (!handle_access_violation(mem, mem.memory.get() + protected_address, true))
                    throw std::runtime_error("Metal: cannot release protection for GPU stores");
            }
            return true;
        });
    }

    void complete_gpu_writes(MemState &mem) {
        for (const auto &[range, weak] : tokens)
            if (auto token = weak.lock()) token->gpu_write_pending = false;
        rearm(mem);
    }


};
}
