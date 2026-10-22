// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <chrono>
#include <cstdint>
#include <future>
#include <map>
#include <optional>
#include <string>

namespace renderer::metal {
// A skipped mask update would change the result of every later scene draw.
// Such draws must wait for their shader and pipeline, even with async enabled.
inline bool allow_async_draw_compile(bool enabled, bool mask_update,
    bool capture_geometry, size_t count, uint32_t instances) {
    return enabled && !mask_update && !capture_geometry && (count > 6 || instances > 1);
}
// Accessed by the render thread only. Workers own their inputs and return a
// value; they never mutate this map or the renderer's resident caches.
template <typename Result> class CompileJobs {
    std::map<std::string, std::future<Result>> jobs;
public:
    size_t size() const { return jobs.size(); }
    bool contains(const std::string &key) const { return jobs.contains(key); }
    template <typename Work> bool submit(const std::string &key, size_t limit, Work work) {
        if (jobs.contains(key) || jobs.size() >= limit) return false;
        auto entry = jobs.try_emplace(key).first;
        try { entry->second = std::async(std::launch::async, std::move(work)); }
        catch (...) { jobs.erase(entry); throw; }
        return true;
    }
    std::optional<std::future<Result>> take(const std::string &key, bool wait) {
        auto entry = jobs.find(key);
        if (entry == jobs.end() || (!wait && entry->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready))
            return std::nullopt;
        auto future = std::move(entry->second);
        jobs.erase(entry); // even get() throwing cannot leave an invalid future
        return future;
    }
    template <typename Publish> void collect(bool wait, Publish publish) {
        for (auto entry = jobs.begin(); entry != jobs.end();) {
            if (!wait && entry->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                ++entry;
                continue;
            }
            auto key = entry->first;
            auto future = std::move(entry->second);
            entry = jobs.erase(entry);
            publish(key, std::move(future));
        }
    }
    // std::launch::async futures join before captured inputs/device ownership
    // can be released or the game's persistent cache can be reconfigured.
    void clear() { jobs.clear(); }
};
}
