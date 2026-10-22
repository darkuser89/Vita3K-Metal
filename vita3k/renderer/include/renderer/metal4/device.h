// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// Native API boundary. Include only in Objective-C++ implementation files;
// guest GXM headers use Ptr, which collides with Apple's legacy MacTypes.h.
#include <renderer/metal4/metal4_commands.h>
#include <renderer/metal4/metal4_pipeline.h>
#include <renderer/metal4/cache.h>

#include <shader/metal4/msl_recompiler.h>
#include <memory>
#include <string>

namespace renderer::metal {

struct NativeShaderLibrary;
struct CompiledProgram {
    std::shared_ptr<const NativeShaderLibrary> native_owner;
    id<MTLLibrary> library;
    id<MTLFunction> function;
    shader::metal::Stage stage;
    bool uses_framebuffer_fetch;
    bool writes_guest_memory = true;
    uint32_t cube_texture_mask = 0;
    // Retained only when draw capture is configured, including cache warmup.
    std::unique_ptr<const shader::metal::Program> capture_source;
};

class Device {
public:
    static std::unique_ptr<Device> create(std::string &error);
    ~Device();
    void configure_cache(const std::filesystem::path &root);
    std::optional<shader::metal::Program> load_cached_program(std::string_view key) const;
    void store_cached_program(std::string_view key, const shader::metal::Program &,
        std::string_view guest_hash = {}, bool gamma_correction = false) const;
    std::vector<CachedVariant> cached_variants(std::string_view guest_hash) const;
    void store_cached_pipeline_template(std::string_view fragment_hash, std::string_view vertex_hash,
        std::string_view key, std::string_view vertex_key, std::string_view fragment_key,
        MTLRenderPipelineDescriptor *descriptor, std::string_view descriptor_key = {}) const;
    std::vector<CachedPipeline> cached_pipeline_templates(std::string_view fragment_hash, std::string_view vertex_hash) const;
    void flush_cache() const;
    CacheStats cache_stats() const;
    std::filesystem::path cache_directory() const;
    id<MTLComputePipelineState> create_compute_pipeline(id<MTLFunction>, std::string &error) const;
    id<MTLLibrary> compile_library(NSString *source, MTLCompileOptions *options, NSError **error) const;
    std::unique_ptr<CompiledProgram> compile(const shader::metal::Program &program,
        bool gamma_correction, std::string &error, bool retain_capture_source = false) const;
    id<MTLRenderPipelineState> create_pipeline(MTLRenderPipelineDescriptor *descriptor,
        std::string &error, std::string *descriptor_key = nullptr) const;
    bool submit_and_wait(V3KMetal4Commands * commands, std::string &error) const;

    id<MTLDevice> native_device() const { return device; }
    id<MTL4Compiler> native_compiler() const { return compiler; }
    V3KMetal4Queue * command_queue() const { return queue; }
    bool supports_raster_order_groups() const { return device.rasterOrderGroupsSupported; }

private:
    Device();
    struct LibraryCache;
    std::unique_ptr<LibraryCache> libraries;
    id<MTLDevice> device;
    id<MTL4Compiler> compiler;
    V3KMetal4Queue * queue;
    std::unique_ptr<PersistentCache> cache;
};

} // namespace renderer::metal
