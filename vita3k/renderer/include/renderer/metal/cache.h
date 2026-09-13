// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <filesystem>
#include <memory>
#include <optional>
#include <renderer/metal/apple.h>
#include <shader/msl_recompiler.h>
#include <string_view>

namespace renderer::metal {
class Device;
struct CacheStats {
    uint64_t program_hits = 0, program_misses = 0, program_writes = 0;
    uint64_t pipeline_hits = 0, pipeline_misses = 0, archive_writes = 0;
    uint64_t rejected_files = 0, io_errors = 0;
    bool archive_loaded = false, archive_writable = false;
};
class PersistentCache {
    struct Impl;
    std::unique_ptr<Impl> impl;

public:
    PersistentCache(Device &, const std::filesystem::path &root);
    ~PersistentCache();
    std::optional<shader::metal::Program> load_program(std::string_view key);
    void store_program(std::string_view key, const shader::metal::Program &);
    id<MTLRenderPipelineState> create_render_pipeline(MTLRenderPipelineDescriptor *, std::string &error);
    id<MTLComputePipelineState> create_compute_pipeline(id<MTLFunction>, std::string &error);
    void flush();
    CacheStats stats() const;
    std::filesystem::path directory() const;
};
} // namespace renderer::metal
