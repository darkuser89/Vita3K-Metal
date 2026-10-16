// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <mem/ptr.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
namespace renderer { struct CommandList; }
enum class MetalGxmProgramKind {
    Vertex,
    Fragment,
};

namespace gxm::metal3 {
struct Bookkeeping {
    std::mutex metal_shader_caches_mutex;
    std::unordered_map<const void *, std::shared_ptr<void>> metal_shader_caches;
    std::mutex metal_programs_mutex;
    std::unordered_map<Address, MetalGxmProgramKind> metal_live_programs;
    uint64_t metal_stale_program_warnings = 0;

    std::mutex metal_command_lists_mutex;
    std::unordered_set<const renderer::CommandList *> metal_command_lists;

    void clear_metal_programs() {
        const std::lock_guard<std::mutex> lock(metal_programs_mutex);
        metal_live_programs.clear();
        metal_stale_program_warnings = 0;
    }

};
}
