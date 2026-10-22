// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace renderer::metal {
// Variant keys can differ even when their emitted native shader is identical.
// Share only the native result; each variant keeps its own guest metadata.
// Weak ownership avoids extending library lifetime beyond its resident users.
template <class Value> class LibraryReuse {
    using Owner = std::shared_ptr<const Value>;
    std::mutex mutex;
    std::map<std::string, std::weak_ptr<const Value>> resident;
    std::map<std::string, std::shared_future<Owner>> pending;
    static constexpr size_t max_entries = 512;

public:
    template <class Build> Owner get(const std::string &key, Build build, bool &reused) {
        std::shared_future<Owner> future;
        std::promise<Owner> promise;
        bool producer = false;
        {
            std::lock_guard lock(mutex);
            reused = true;
            if (auto found = resident.find(key); found != resident.end()) {
                if (auto value = found->second.lock()) return value;
                resident.erase(found);
            }
            if (auto found = pending.find(key); found != pending.end()) future = found->second;
            else {
                future = promise.get_future().share();
                pending.emplace(key, future);
                producer = true;
                reused = false;
            }
        }
        // Independent shaders may compile concurrently. Only equal identities
        // wait on one another; no Metal call or future wait holds the mutex.
        if (!producer) return future.get();
        try {
            Owner value = build();
            {
                std::lock_guard lock(mutex);
                if (value) {
                    for (auto it = resident.begin(); it != resident.end();)
                        if (it->second.expired()) it = resident.erase(it); else ++it;
                    if (resident.size() >= max_entries) resident.erase(resident.begin());
                    resident[key] = value;
                }
                pending.erase(key);
                promise.set_value(value);
            }
            return value;
        } catch (...) {
            std::lock_guard lock(mutex);
            pending.erase(key);
            promise.set_exception(std::current_exception());
            throw;
        }
    }
    void clear() {
        std::lock_guard lock(mutex);
        resident.clear();
    }
};
} // namespace renderer::metal
