// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <renderer/state.h>
#include <renderer/texture_cache.h>
#include <renderer/metal3/texture_formats.h>
#include <shader/metal3/spirv_recompiler.h>
#include <shader/metal3/usse_program_analyzer.h>
#include <array>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace renderer::metal {
struct MetalState;
struct MetalGuestContext : renderer::Context {
    struct GuestState {
        std::array<SceGxmYuvProfile, 2> yuv_profiles{SCE_GXM_YUV_PROFILE_BT601_STANDARD, SCE_GXM_YUV_PROFILE_BT601_STANDARD};
        bool back_visibility_enable = false;
        uint32_t back_visibility_index = 0;
        bool back_visibility_is_increment = false;
    } guest_state;
    struct RecordMetadata {
        bool has_depth_stencil_surface = true;
        uint32_t back_line_width = 1;
        int back_depth_bias_unit = 0, back_depth_bias_slope = 0;
    } native_record;
    // Guest-thread bookkeeping for Metal's deferred GXM callbacks. Keep it
    // here so the guest-allocated SceGxmContext retains its size and layout.
    struct GxmMetadata {
        bool enabled = false;
        bool recording_failed = false;
        // Non-owning: GXM destroys its contexts before its host state.
        GxmState *owner = nullptr;
        size_t vertex_reserved_bytes = 0;
        size_t fragment_reserved_bytes = 0;
    } metal_gxm;
};

struct MetalContext : MetalGuestContext {
    struct Impl;
    std::unique_ptr<Impl> impl;
    shader::metal3::Hints shader_hints{};
    struct UniformBinding { uint8_t *data = nullptr; size_t size = 0; Address address{}; };
    std::array<std::array<UniformBinding, SCE_GXM_REAL_MAX_UNIFORM_BUFFER>, 2> uniforms;
    std::array<SceGxmTexture, 2 * SCE_GXM_MAX_TEXTURE_UNITS> textures{};
    std::array<uint64_t, SCE_GXM_MAX_VERTEX_STREAMS> vertex_stream_snapshots;
    std::shared_ptr<const ProgramBinding> vertex_program, fragment_program;
    std::array<float, 4> viewport = {0, 0, 960, 544};
    MetalContext();
    ~MetalContext() override;
};
bool is_native_context(renderer::Context *context);
MetalGuestContext::GxmMetadata &gxm_metadata(renderer::Context &context);
struct MetalRenderTarget : renderer::RenderTarget {
    uint32_t width = 0, height = 0;
    uint32_t guest_width = 0, guest_height = 0;
    uint32_t multisample_locations = 0;
    bool custom_multisample_locations = false;
};
struct MetalFragmentProgram : renderer::FragmentProgram {
    SceGxmBlendInfo blend{};
    SceGxmOutputRegisterFormat output_register_format = SCE_GXM_OUTPUT_REGISTER_FORMAT_DECLARED;
};
struct ProgramBinding {
    std::vector<uint8_t> gxp;
    std::shared_ptr<const VertexProgram> vertex_program;
    std::shared_ptr<const MetalFragmentProgram> fragment_program;
    std::vector<SceGxmVertexStream> streams;
    std::vector<SceGxmVertexAttribute> attributes;
    shader::metal3::usse::AttributeInformationMap attribute_infos;
    uint32_t dynamic_uniform_buffers = 0;
    bool is_maskupdate = false;
    const SceGxmProgram *program() const { return reinterpret_cast<const SceGxmProgram *>(gxp.data()); }
};
void freeze_program(SceGxmVertexProgram &, const MemState &);
void freeze_program(SceGxmFragmentProgram &, const MemState &);

struct MetalTextureCache : renderer::TextureCache {
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit MetalTextureCache(MetalState &state);
    ~MetalTextureCache() override;
    YUVConversionCache yuv_conversion_cache;
    std::array<SceGxmYuvProfile, 2> yuv_profiles{SCE_GXM_YUV_PROFILE_BT601_STANDARD, SCE_GXM_YUV_PROFILE_BT601_STANDARD};
    unordered_map_fast<TextureCacheKey, TextureCacheInfo *> metal_texture_lookup;
    std::array<TextureCacheMetadata, TextureCacheSize> native_metadata{};
    TextureCacheMetadata &metadata(const TextureCacheInfo &info) { return native_metadata.at(info.index); }
    bool support_dxt_software_import = false;
    bool support_astc_software_import = false;
    void upload_texture(const SceGxmTexture &, MemState &) override;
    void cache_and_bind_texture(const SceGxmTexture &, MemState &) override;
    int cache_and_bind_sampler(const SceGxmTexture &, bool is_depth = false) override;
    bool import_configure_texture() override;
    void import_upload_texture() override;
    void cache_and_bind_image(const SceGxmTexture &texture, MemState &mem);
    void select(size_t index, const SceGxmTexture &texture) override;
    void configure_texture(const SceGxmTexture &texture) override;
    uint32_t texture_size_to_protect(const SceGxmTexture &, uint32_t first_mip_size) const;
    uint64_t additional_texture_hash(const SceGxmTexture &, const MemState &, bool storage_protected) const;
    void upload_texture_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height,
        uint32_t mip, const void *pixels, int face, uint32_t stride) override;
    void import_configure_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height,
        bool srgb, uint16_t components, uint16_t mips, bool swap_rb) override;
};

struct MetalState : renderer::State {
    struct Impl;
    std::unique_ptr<Impl> impl;
    MetalTextureCache texture_cache;
    uint32_t pipelines_count_precompiled = 0;
    uint64_t inline_depth_snapshots = 0;
    uint64_t inline_depth_crops = 0;
    uint64_t inline_depth_stores = 0;
    uint64_t deferred_scene_submits = 0;
    MetalState();
    ~MetalState() override;
    bool init() override;
    void cleanup() override;
    void set_app(const char *title_id, const char *self_name) override;
    void late_init(const Config &, std::string_view game_id, MemState &) override;
    bool map_memory(MemState &, Ptr<void> address, uint32_t size) override;
    void unmap_memory(MemState &, Ptr<void> address) override;
    TextureCache *get_texture_cache() override { return &texture_cache; }
    void render_frame(DisplayState &, const GxmState &, MemState &) override;
    void swap_window() override;
    std::vector<uint32_t> dump_frame(DisplayState &, uint32_t &, uint32_t &) override;
    std::vector<uint32_t> read_display_frame(DisplayState &, uint32_t &, uint32_t &);
    uint32_t get_features_mask() override;
    int get_supported_filters() override;
    void set_screen_filter(const std::string_view &) override;
    int get_max_anisotropic_filtering() override { return 16; }
    void set_anisotropic_filtering(int value) override;
    int get_max_2d_texture_width() override { return 16384; }
    std::string_view get_gpu_name() override;
    void precompile_shader(const ShadersHash &) override;
    void set_async_compilation(bool enable) override;
    void preclose_action() override;
    void snapshot_vertex_streams(MemState &, CommandList &);
    void disable_vertex_stream_snapshots();
    void reset_vertex_stream_snapshots();
    void set_context(MetalContext &, MemState &);
    void new_frame();
    void draw(MetalContext &, MemState &, SceGxmPrimitiveType, SceGxmIndexFormat,
        const void *indices, uint32_t count, uint32_t instances);
    bool finish(MetalContext &, bool publish_color = true, bool wait_without_publication = false);
    // After the render thread has joined, retire submitted work even if the
    // normal finish/publication path fails, before releasing context storage.
    void finish_for_shutdown(MetalContext &);
    // Diagnostic readback of exact native RGBA8 MSAA samples, before guest
    // sample packing or resolution scaling can duplicate/drop output pixels.
    uint64_t count_nonzero_color_samples(MetalContext &);
    void mid_scene_flush(MetalContext &, bool wait_for_completion);
    bool end_scene(MetalContext &, bool allow_deferred = false);
    void set_visibility_buffer(MetalContext &, Ptr<uint32_t> buffer, uint32_t stride);
    void set_visibility_index(MetalContext &, bool enable, uint32_t index, bool increment);
    void set_back_visibility_index(MetalContext &, bool enable, uint32_t index, bool increment);
    bool sync_surface(MemState &, const SceGxmColorSurface &);
    int sync_surfaces_for_cpu_read(MemState &, Address address, uint32_t size);
    bool transfer_fill(MemState &, const SceGxmTransferImage &, uint32_t color);
    bool transfer_copy(MemState &, const SceGxmTransferImage &source, const SceGxmTransferImage &destination,
        SceGxmTransferType source_type, SceGxmTransferType destination_type,
        SceGxmTransferColorKeyMode mode, uint32_t key, uint32_t mask);
    bool transfer_downscale(MemState &, const SceGxmTransferImage &source, const SceGxmTransferImage &destination);
private:
    bool try_transfer_depth_gpu(MemState &, const SceGxmTransferImage &source, const SceGxmTransferImage &destination,
        SceGxmTransferType source_type, SceGxmTransferType destination_type,
        std::span<const std::pair<uint64_t, uint64_t>> reads,
        std::span<const std::pair<uint64_t, uint64_t>> writes);
    std::vector<uint8_t> vertex_stream_snapshot(uint64_t handle, size_t size);
    void retire_cpu_overwritten_surfaces(MemState &, std::span<const std::pair<uint64_t, uint64_t>> ranges);
    bool transfer_image(MemState &, const SceGxmTransferImage &source, const SceGxmTransferImage &destination,
        SceGxmTransferType source_type, SceGxmTransferType destination_type,
        SceGxmTransferColorKeyMode mode, uint32_t key, uint32_t mask, bool downscale);
};

void set_uniform_buffer(MetalContext &, const ShaderProgram &, bool vertex, int block,
    uint32_t size, const uint8_t *data, Address address);
void set_viewport(MetalContext &, float x, float y, float sx, float sy);
} // namespace renderer::metal
