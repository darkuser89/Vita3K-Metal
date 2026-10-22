// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>

namespace renderer::metal {
// Caller serializes access and owns the corresponding native allocations.
// Cache ownership may outlive a command only for GPU-owned native storage.
class ResidencyCachePolicy {
    struct Entry { size_t bytes, users; uint64_t age; };
    std::map<uintptr_t, Entry> entries;
    size_t budget, count_limit, used = 0;
    uint64_t age = 0;
public:
    ResidencyCachePolicy(size_t bytes, size_t count) : budget(bytes), count_limit(count) {}
    template<class Evict>
    bool pin(uintptr_t key, size_t bytes, Evict evict) {
        if (!key || !bytes || bytes > budget || !count_limit) return false;
        if (auto found=entries.find(key); found!=entries.end()) {
            if (found->second.users==std::numeric_limits<size_t>::max())
                throw std::overflow_error("Metal residency user count overflow");
            ++found->second.users;found->second.age=++age;return true;
        }
        while (entries.size()>=count_limit || used>budget-bytes) {
            auto victim=entries.end();
            for (auto it=entries.begin();it!=entries.end();++it)
                if (!it->second.users && (victim==entries.end() || it->second.age<victim->second.age)) victim=it;
            // Fall back to submission-local residency if all room is in use.
            if (victim==entries.end()) return false;
            evict(victim->first);used-=victim->second.bytes;entries.erase(victim);
        }
        entries.emplace(key,Entry{bytes,1,++age});used+=bytes;return true;
    }
    void unpin(uintptr_t key) {
        auto found=entries.find(key);
        if (found==entries.end() || !found->second.users)
            throw std::logic_error("Metal residency lease retired twice");
        --found->second.users;
    }
    size_t bytes() const { return used; }
    size_t size() const { return entries.size(); }
};
}
