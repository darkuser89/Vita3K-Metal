// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace renderer::metal {
// Accounting identities refer to underlying allocations, not texture views.
// A pinned root remains counted exactly once even when many cache views use it.
class SurfaceCacheBudget {
    struct Root { uint64_t bytes = 0; size_t owners = 0; bool pinned = false; };
    struct Group { uint64_t age, payload; size_t entries; std::vector<uintptr_t> roots; };
    std::map<uintptr_t, Root> roots;
    std::vector<Group> groups;
    uint64_t pinned_payload = 0;
    static uint64_t add(uint64_t a, uint64_t b) {
        if (b > std::numeric_limits<uint64_t>::max() - a)
            throw std::overflow_error("Metal surface cache accounting overflow");
        return a + b;
    }
public:
    struct Allocation { uintptr_t identity; uint64_t bytes; };
    struct Plan {
        uint64_t pinned_bytes = 0, derived_before = 0, derived_after = 0;
        size_t entries_before = 0, entries_after = 0;
        std::vector<size_t> evict;
    };
    void pin(Allocation allocation) {
        if (!allocation.identity) return;
        auto &root = roots[allocation.identity];
        root.bytes = std::max(root.bytes, allocation.bytes); root.pinned = true;
    }
    void pin_payload(uint64_t bytes) { pinned_payload = add(pinned_payload, bytes); }
    size_t add_group(uint64_t age, std::vector<Allocation> allocations, uint64_t payload, size_t entries) {
        Group group{age, payload, entries, {}};
        for (const auto &allocation : allocations) {
            if (!allocation.identity) continue;
            auto &root = roots[allocation.identity];
            root.bytes = std::max(root.bytes, allocation.bytes);
            group.roots.push_back(allocation.identity);
        }
        std::sort(group.roots.begin(), group.roots.end());
        group.roots.erase(std::unique(group.roots.begin(), group.roots.end()), group.roots.end());
        for (auto identity : group.roots) ++roots.at(identity).owners;
        groups.push_back(std::move(group));
        return groups.size() - 1;
    }
    // A one-shot plan, oldest groups first. Removing one of several owners
    // never claims the allocation was released. No Metal/resource mutation.
    Plan plan(uint64_t derived_budget, size_t entry_budget) const {
        auto remaining = roots;
        Plan result; result.pinned_bytes = pinned_payload;
        for (const auto &[identity, root] : roots) {
            if (root.pinned) result.pinned_bytes = add(result.pinned_bytes, root.bytes);
            else result.derived_before = add(result.derived_before, root.bytes);
        }
        for (const auto &group : groups) {
            result.derived_before = add(result.derived_before, group.payload);
            if (group.entries > std::numeric_limits<size_t>::max() - result.entries_before)
                throw std::overflow_error("Metal surface cache entry accounting overflow");
            result.entries_before += group.entries;
        }
        result.derived_after = result.derived_before; result.entries_after = result.entries_before;
        std::vector<size_t> order(groups.size()); std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return groups[a].age < groups[b].age; });
        for (auto index : order) {
            if (result.derived_after <= derived_budget && result.entries_after <= entry_budget) break;
            const auto &group = groups[index];
            result.evict.push_back(index);
            result.derived_after -= group.payload; result.entries_after -= group.entries;
            for (auto identity : group.roots) {
                auto &root = remaining.at(identity);
                if (!--root.owners && !root.pinned) result.derived_after -= root.bytes;
            }
        }
        return result;
    }
};
}
