// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace renderer::metal {
// Whole texture roots/heaps; byte ranges for buffers. CPU and GPU address
// spaces are separate: distinct no-copy buffers can alias only in CPU space.
struct CommandAccess {
    uintptr_t texture = 0, heap = 0;
    uint64_t gpu_begin = 0, gpu_end = 0, cpu_begin = 0, cpu_end = 0;
    bool opaque = true, write = true;

    bool conflicts(const CommandAccess &other) const {
        if (!write && !other.write) return false;
        const auto overlap = [](uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
            return a < d && c < b && a < b && c < d;
        };
        return opaque || other.opaque || (texture && texture == other.texture)
            || (heap && heap == other.heap)
            || overlap(gpu_begin, gpu_end, other.gpu_begin, other.gpu_end)
            || overlap(cpu_begin, cpu_end, other.cpu_begin, other.cpu_end);
    }
};

class CommandDependencies {
public:
    enum Stage : uint8_t { dispatch = 1, blit = 2 };
    // Returns source stages for a Device-visibility barrier targeting BOTH
    // dispatch and blit. Only that all-target barrier permits forgetting the
    // old epoch, including accesses in a stage different from the next command.
    uint8_t prepare(uint8_t stage, std::span<const CommandAccess> accesses) {
        const CommandAccess unknown{};
        if (accesses.empty() || accesses.size() > pending.size()) accesses = {&unknown, 1};
        bool barrier = accesses.size() > pending.size() - count;
        for (const auto &access : accesses)
            for (size_t i = 0; i < count && !barrier; ++i)
                barrier = access.conflicts(pending[i]);
        const auto source = barrier ? stages : uint8_t(0);
        if (barrier) { count = 0; stages = 0; }
        for (const auto &access : accesses) pending[count++] = access;
        stages |= stage;
        return source;
    }

private:
    std::array<CommandAccess, 64> pending{};
    size_t count = 0;
    uint8_t stages = 0;
};
}
