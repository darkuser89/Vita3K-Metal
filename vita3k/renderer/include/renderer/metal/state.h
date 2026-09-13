// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <renderer/state.h>
#include <renderer/texture_cache.h>
#include <array>
#include <memory>

namespace renderer::metal {
struct MetalState;
struct MetalContext : renderer::Context {
    struct Impl;
    std::unique_ptr<Impl> impl;
    struct UniformBinding { uint8_t *data = nullptr; size_t size = 0; };
    std::array<std::array<UniformBinding, SCE_GXM_REAL_MAX_UNIFORM_BUFFER>, 2> uniforms;
    std::array<SceGxmTexture, 2 * SCE_GXM_MAX_TEXTURE_UNITS> textures{};
    std::array<float, 4> viewport = {0, 0, 960, 544};
    MetalContext();
    ~MetalContext() override;
};
struct MetalRenderTarget : renderer::RenderTarget {
    uint32_t width = 0, height = 0;
    uint32_t multisample_locations = 0;
    bool custom_multisample_locations = false;
};
struct MetalFragmentProgram : renderer::FragmentProgram {
    SceGxmBlendInfo blend{};
};

struct MetalTextureCache : renderer::TextureCache {
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit MetalTextureCache(MetalState &state);
    ~MetalTextureCache() override;
    void cache_and_bind_image(const SceGxmTexture &texture, MemState &mem);
    void select(size_t index, const SceGxmTexture &texture) override;
    void configure_texture(const SceGxmTexture &texture) override;
    uint64_t additional_texture_hash(const SceGxmTexture &, const MemState &) const override;
    void upload_texture_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height,
        uint32_t mip, const void *pixels, int face, uint32_t stride) override;
    void import_configure_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height,
        bool srgb, uint16_t components, uint16_t mips, bool swap_rb) override;
};

struct MetalState : renderer::State {
    struct Impl;
    std::unique_ptr<Impl> impl;
    MetalTextureCache texture_cache;
    MetalState();
    ~MetalState() override;
    bool init() override;
    void cleanup() override;
    void set_app(const char *title_id, const char *self_name) override;
    void late_init(const Config &, std::string_view game_id, MemState &) override;
    TextureCache *get_texture_cache() override { return &texture_cache; }
    void render_frame(DisplayState &, const GxmState &, MemState &) override;
    void swap_window() override;
    std::vector<uint32_t> dump_frame(DisplayState &, uint32_t &, uint32_t &) override;
    int get_supported_filters() override;
    void set_screen_filter(const std::string_view &) override;
    int get_max_anisotropic_filtering() override { return 16; }
    void set_anisotropic_filtering(int value) override;
    int get_max_2d_texture_width() override { return 16384; }
    std::string_view get_gpu_name() override;
    void precompile_shader(const ShadersHash &) override;
    void preclose_action() override;
    void set_context(MetalContext &, MemState &);
    void draw(MetalContext &, MemState &, SceGxmPrimitiveType, SceGxmIndexFormat,
        const void *indices, uint32_t count, uint32_t instances);
    void finish(MetalContext &);
    void end_scene(MetalContext &);
    bool sync_surface(MemState &, const SceGxmColorSurface &);
    bool transfer_fill(MemState &, const SceGxmTransferImage &, uint32_t color);
    bool transfer_copy(MemState &, const SceGxmTransferImage &source, const SceGxmTransferImage &destination,
        SceGxmTransferType source_type, SceGxmTransferType destination_type,
        SceGxmTransferColorKeyMode mode, uint32_t key, uint32_t mask);
    bool transfer_downscale(MemState &, const SceGxmTransferImage &source, const SceGxmTransferImage &destination);
private:
    bool transfer_image(MemState &, const SceGxmTransferImage &source, const SceGxmTransferImage &destination,
        SceGxmTransferType source_type, SceGxmTransferType destination_type,
        SceGxmTransferColorKeyMode mode, uint32_t key, uint32_t mask, bool downscale);
};

void set_uniform_buffer(MetalContext &, const ShaderProgram &, bool vertex, int block,
    uint32_t size, const uint8_t *data);
void set_viewport(MetalContext &, float x, float y, float sx, float sy);
} // namespace renderer::metal
