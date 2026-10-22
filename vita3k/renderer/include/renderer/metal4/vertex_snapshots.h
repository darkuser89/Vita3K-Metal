// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace renderer::metal {
using OwnedVertexSnapshot = std::shared_ptr<const std::vector<uint8_t>>;

// Called under the existing ring mutex, after validating that the monotonic
// handle still names this live ring record. Never return a cached owner for
// an expired/disabled ring record: that must retain the old live-RAM fallback.
class VertexSnapshotOwners {
    std::map<std::pair<uint64_t, size_t>, OwnedVertexSnapshot> snapshots;
public:
    OwnedVertexSnapshot get(uint64_t handle, std::span<const uint8_t> bytes) {
        const auto key = std::pair{handle, bytes.size()};
        if (const auto found = snapshots.find(key); found != snapshots.end()) return found->second;
        auto owner = std::make_shared<const std::vector<uint8_t>>(bytes.begin(), bytes.end());
        // Existing ring records are at most 16 KiB: retain at most 1 MiB here.
        // Outstanding draws/upload batches may independently own their bytes.
        if (snapshots.size() >= 64) snapshots.erase(snapshots.begin());
        snapshots.emplace(key, owner);
        return owner;
    }
    void clear() { snapshots.clear(); }
};
}
