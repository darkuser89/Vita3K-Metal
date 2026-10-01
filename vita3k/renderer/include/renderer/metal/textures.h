// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal/apple.h>
#include <gxm/types.h>
#include <span>
#include <array>
#include <optional>
#include <map>
#include <vector>
struct SceGxmColorSurface;
struct SceGxmDepthStencilSurface;

namespace renderer::metal {
inline uint32_t packed_10_native_word(uint32_t guest, uint32_t mode) {
    if (mode == 0) return guest;
    if (mode == 1) return (guest & 0xc00ffc00u)
        | ((guest & 0x000003ffu) << 20) | ((guest & 0x3ff00000u) >> 20);
    const uint32_t first = (guest >> 2) & 0x3ff;
    const uint32_t green = (guest >> 12) & 0x3ff;
    const uint32_t third = (guest >> 22) & 0x3ff;
    return (mode == 2 ? first : third) | (green << 10)
        | ((mode == 2 ? third : first) << 20) | ((guest & 3) << 30);
}
// Linear-strided descriptors reuse LOD bits for stride and have no LOD bias.
inline float sampler_lod_bias(const SceGxmTexture &texture) {
    return texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED ? 0.f : (float(texture.lod_bias) - 31.f) / 8.f;
}
inline uint32_t metal_lod_min(const SceGxmTexture &texture) {
    return (texture.lod_min0 << 2) | texture.lod_min1;
}
inline uint32_t sampler_metadata_flags(const SceGxmTexture &texture) {
    const auto linear=[](uint32_t filter) {
        return filter==SCE_GXM_TEXTURE_FILTER_LINEAR || filter==SCE_GXM_TEXTURE_FILTER_MIPMAP_LINEAR;
    };
    const bool strided=texture.texture_type()==SCE_GXM_TEXTURE_LINEAR_STRIDED;
    return uint32_t(linear(strided ? texture.mag_filter : texture.min_filter))
        |(uint32_t(linear(texture.mag_filter))<<1)
        |((strided ? 0u : texture.mip_filter)<<2)
        |(texture.uaddr_mode<<3)|(texture.vaddr_mode<<6)
        |((strided ? 0u : metal_lod_min(texture))<<9);
}
class Device;
struct MetalTextureCache;
struct ImportedTextureView {
    SceGxmTextureBaseFormat base_format{};
    uint16_t components = 0;
    bool active = false, swap_rb = false;
};
struct CubeSurface {
    id<MTLTexture> texture;
    uint32_t face, mip;
};
struct SurfaceRect { uint32_t x, y, width, height; };
struct PublicationClip { SurfaceRect guest, source; };
struct SurfaceMemoryRange { size_t offset, size; };
std::optional<SurfaceRect> surface_subrectangle(const SceGxmColorSurface &, const SceGxmTexture &);
std::optional<SurfaceRect> surface_word_subrectangle(const SceGxmColorSurface &, const SceGxmTexture &);
bool surface_texture_layout_overlap(const SceGxmColorSurface &, const SceGxmTexture &);
bool surface_texture_needs_native_resolution(SceGxmColorFormat, const SceGxmTexture &, float scale, bool use_texture_viewport);
// Select the owner by overlap/byte pitch/tiling before checking the complete
// subimage, as in Plus. Failed bounds or format conversion must not reveal an
// older enclosing owner. Retain native representable Morton subimages, which
// can have a smaller pitch than their parent allocation.
template <typename SurfaceMap, typename Eligible>
auto find_color_subrectangle(SurfaceMap &surfaces, const SceGxmTexture &texture, Eligible eligible) {
    auto it = surfaces.upper_bound(texture.data_addr << 2);
    while (it != surfaces.begin()) {
        --it;
        const auto rectangle = surface_subrectangle(it->second.guest, texture);
        if (surface_texture_layout_overlap(it->second.guest, texture) || rectangle)
            return eligible(it->second) && rectangle ? it : surfaces.end();
    }
    return surfaces.end();
}
// Establish cube storage layout before checking ownership. A dirty face in
// the first complete layout must not expose a different set of six images.
template <typename SurfaceMap, typename Matches, typename Eligible>
auto find_color_cube_faces(SurfaceMap &surfaces, uint64_t address,
    std::span<const uint64_t> steps, Matches matches, Eligible eligible)
    -> std::optional<std::pair<uint64_t, std::array<typename SurfaceMap::mapped_type *, 6>>> {
    constexpr uint64_t limit = uint64_t(UINT32_MAX) - 4095;
    if (!address || address > limit) return std::nullopt;
    for (const uint64_t step : steps) {
        if (!step || step > (limit - address) / 5) continue;
        std::array<typename SurfaceMap::mapped_type *, 6> faces{};
        bool complete = true;
        for (uint32_t face = 0; face < faces.size(); ++face) {
            const auto found = surfaces.find(uint32_t(address + face * step));
            if (found == surfaces.end() || !matches(found->second)) {
                complete = false;
                break;
            }
            faces[face] = &found->second;
        }
        if (!complete) continue;
        for (const auto *face : faces)
            if (!eligible(*face)) return std::nullopt;
        return std::pair{step, faces};
    }
    return std::nullopt;
}
// A larger linear descriptor can sample only the rendered prefix of its storage.
std::optional<std::pair<float,float>> surface_texture_viewport(const SceGxmColorSurface &, const SceGxmTexture &);
// Match a representable depth view in the allocation's guest sample coordinates.
std::optional<SurfaceRect> depth_subrectangle(const SceGxmDepthStencilSurface &, uint32_t width, uint32_t height,
    SceGxmMultisampleMode, const SceGxmTexture &);
// Match a complete depth allocation in guest sample coordinates.
bool depth_texture_matches(const SceGxmDepthStencilSurface &, uint32_t width, uint32_t height,
    SceGxmMultisampleMode, const SceGxmTexture &);
id<MTLTexture> current_texture(const MetalTextureCache &cache);
id<MTLTexture> current_texture_view(const MetalTextureCache &cache, SceGxmTextureFormat format);
struct DepthMemoryLayout {
    uint32_t width, height, stride, depth_bytes;
    size_t depth_size, stencil_size;
    bool tiled, packed;
};
std::optional<DepthMemoryLayout> depth_memory_layout(const SceGxmDepthStencilSurface &, uint32_t width, uint32_t height,
    SceGxmMultisampleMode);
struct DepthStoreReadback {
    id<MTLBuffer> depth = nil;
    id<MTLBuffer> mask = nil;
    // The shared buffer already contains final guest words (S8D24 or DF32).
    bool packed_direct = false;
};
struct DepthMemoryWrite {
    bool depth = false, stencil = false, mask = false;
};
class SurfaceCaster {
    Device &device;
    std::map<uint32_t,id<MTLRenderPipelineState>> depth_seed_pipelines;
    std::map<std::pair<uint32_t,bool>,id<MTLRenderPipelineState>> depth_copy_pipelines;
    std::map<std::pair<uint32_t,uint32_t>,id<MTLRenderPipelineState>> depth_patch_pipelines;
    std::map<std::pair<uint32_t,bool>,id<MTLRenderPipelineState>> depth_clear_pipelines;
    std::map<std::pair<uint32_t,bool>,id<MTLRenderPipelineState>> depth_resample_pipelines;
    std::map<uint32_t,id<MTLRenderPipelineState>> mask_seed_pipelines;
    id<MTLComputePipelineState> depth_store_pipeline, depth_store_ms_pipeline;
    id<MTLComputePipelineState> packed_depth_store_pipeline, packed_depth_store_ms_pipeline;
    id<MTLComputePipelineState> mask_store_pipeline, mask_store_ms_pipeline;
    id<MTLBuffer> depth_store_buffer, mask_store_buffer;
    id<MTLComputePipelineState> pipeline, halfword_unpack_pipeline, byte_halfword_pipeline,
        byte_unpack_pipeline;
    id<MTLComputePipelineState> word_buffer_pipeline, halfword_buffer_pipeline;
    id<MTLComputePipelineState> x8_word_decode_pipeline;
    id<MTLComputePipelineState> cube_pipeline, component_cast_pipeline;
    id<MTLComputePipelineState> depth_pipeline, stencil_pipeline, stencil_ms_pipeline, rg_gamma_pipeline;
    id<MTLComputePipelineState> scaled_snapshot_pipeline;
    id<MTLComputePipelineState> packed_depth_snapshot_pipeline;
    id<MTLComputePipelineState> multisample_pipeline, multisample_depth_pipeline;
    id<MTLComputePipelineState> multisample_integer_pipeline;
    id<MTLComputePipelineState> raw_multisample_resolve_pipeline;
    id<MTLLibrary> multisample_library;
    std::map<std::pair<uint32_t,uint32_t>,id<MTLRenderPipelineState>> seed_pipelines;
    std::map<std::pair<uint32_t,uint32_t>,id<MTLRenderPipelineState>> clip_pipelines;
    std::map<uint32_t,id<MTLRenderPipelineState>> publication_pipelines;
public:
    explicit SurfaceCaster(Device &device);
    // Scissored clear that preserves storage outside the active scene extent.
    void clear_depth_region(id<MTLTexture>, SurfaceRect, float depth, uint32_t stencil,
        DepthMemoryWrite aspects, id<MTLCommandBuffer>, const MTLSamplePosition *sample_positions = nullptr);
    // Reconstruct the Plus nearest sample-rate view from an expanded source
    // grid. The scene receives one selected value per pixel in all samples;
    // the original depth/stencil image and optional DF32M mask stay intact.
    bool resample_depth(id<MTLTexture> source, id<MTLTexture> destination,
        uint32_t grid_width, uint32_t grid_height, uint32_t guest_width, uint32_t guest_height,
        id<MTLCommandBuffer>, id<MTLTexture> source_mask = nil, id<MTLTexture> destination_mask = nil,
        const MTLSamplePosition *sample_positions = nullptr);
    // Copy exactly one native aspect; the other aspect and pixels outside the
    // destination rectangle remain in the attachment, including MSAA samples.
    bool copy_depth_stencil_region(id<MTLTexture> source, id<MTLTexture> destination,
        SurfaceRect source_rect, SurfaceRect destination_rect, bool stencil,
        id<MTLCommandBuffer> commands);
    // Empty change lists validate the representation without encoding work.
    bool patch_depth_memory(id<MTLTexture>, const SceGxmDepthStencilSurface &, const DepthMemoryLayout &, float scale,
        std::span<const uint8_t> depth, std::span<const uint8_t> stencil,
        std::span<const SurfaceMemoryRange> depth_changes, std::span<const SurfaceMemoryRange> stencil_changes,
        id<MTLTexture> mask = nil, id<MTLCommandBuffer> commands = nil, DepthMemoryWrite *written = nullptr);
    // separate_word duplicates the selected word instead of shifting an
    // interleaved byte view; callers use it for non-unit-scale paired views.
    id<MTLTexture> rgba8_from_rg32(id<MTLTexture> source, bool swap_words,
        uint32_t word_offset = 0, bool signed_normalized = false,
        uint32_t guest_width = 0, uint32_t guest_height = 0,
        id<MTLCommandBuffer> pending_commands = nil, bool separate_word = false);
    id<MTLTexture> word_texture_from_rg32(id<MTLTexture> source, SceGxmTextureBaseFormat texture,
        uint32_t texture_swizzle, bool swap_words, uint32_t word_offset, uint32_t guest_width = 0,
        uint32_t guest_height = 0, id<MTLCommandBuffer> pending_commands = nil,
        bool separate_word = false);
    id<MTLTexture> word_texture_from_rgba16(id<MTLTexture> source, SceGxmColorFormat color,
        SceGxmTextureBaseFormat texture, uint32_t texture_swizzle, uint32_t word_offset, uint32_t guest_width = 0,
        uint32_t guest_height = 0, id<MTLCommandBuffer> pending_commands = nil,
        bool separate_word = false);
    id<MTLTexture> halfword_texture_from_rgba8(id<MTLTexture> source, SceGxmColorFormat color,
        SceGxmTextureBaseFormat texture, id<MTLCommandBuffer> pending_commands = nil);
    id<MTLTexture> byte_texture_from_16bit_surface(id<MTLTexture> source, SceGxmColorFormat color,
        SceGxmTextureBaseFormat texture, id<MTLCommandBuffer> pending_commands = nil);
    bool load_depth_memory(id<MTLTexture>, const SceGxmDepthStencilSurface &, const DepthMemoryLayout &, float scale,
        std::span<const uint8_t> depth, std::span<const uint8_t> stencil,
        id<MTLCommandBuffer> commands = nil);
    bool load_mask_memory(id<MTLTexture>, const SceGxmDepthStencilSurface &, const DepthMemoryLayout &, float scale,
        std::span<const uint8_t> depth, id<MTLCommandBuffer> commands = nil);
    bool store_depth_memory(id<MTLTexture>, const SceGxmDepthStencilSurface &, const DepthMemoryLayout &, float scale,
        std::span<uint8_t> depth, std::span<uint8_t> stencil, id<MTLTexture> mask = nil);
    bool enqueue_depth_store(id<MTLTexture>, const SceGxmDepthStencilSurface &, const DepthMemoryLayout &, float scale,
        size_t depth_size, size_t stencil_size, id<MTLTexture> mask, id<MTLCommandBuffer>, DepthStoreReadback &);
    void finish_depth_store(const DepthStoreReadback &, const SceGxmDepthStencilSurface &, const DepthMemoryLayout &,
        std::span<uint8_t> depth, std::span<uint8_t> stencil);
    id<MTLTexture> depth_snapshot(id<MTLTexture> source, bool normalized16 = false, float scale = 1,
        uint32_t guest_width = 0, uint32_t guest_height = 0, bool wait_for_completion = true,
        id<MTLCommandBuffer> pending_commands = nil);
    id<MTLTexture> stencil_snapshot(id<MTLTexture> source, bool signed_normalized, float scale = 1,
        uint32_t guest_width = 0, uint32_t guest_height = 0, bool wait_for_completion = true,
        id<MTLCommandBuffer> pending_commands = nil);
    id<MTLTexture> packed_depth_snapshot(id<MTLTexture> source, bool wait_for_completion = true,
        id<MTLCommandBuffer> pending_commands = nil);
    // Native sample IDs in a 1x2 (2X) or 2x2 (4X) image. Each guest sample
    // maps to the native extent using the actual guest and render dimensions.
    void expand_multisample(id<MTLTexture> source, id<MTLTexture> destination, float scale,
        uint32_t guest_width = 0, uint32_t guest_height = 0, id<MTLCommandBuffer> pending_commands = nil);
    void seed_multisample(id<MTLTexture> source, id<MTLTexture> destination, float scale, bool expanded,
        uint32_t guest_width = 0, uint32_t guest_height = 0, id<MTLCommandBuffer> pending_commands = nil);
    // Keep identical F16 sample words; use the normal float resolve for mixed samples.
    void resolve_raw_multisample(id<MTLTexture> source, id<MTLTexture> resolved,
        id<MTLTexture> destination, id<MTLCommandBuffer> commands);
    void restore_clipped_multisample(id<MTLTexture> source, id<MTLTexture> destination,
        const SceGxmColorSurface &surface, id<MTLCommandBuffer> commands,
        const MTLSamplePosition *sample_positions = nullptr);
    bool patch_multisample(id<MTLTexture> texture, const SceGxmColorSurface &, float scale,
        std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges);
    // Diagnostic canonical RGBA values after the texture view's swizzle.
    // Returns nil for integer/unsupported formats; caller must complete producers.
    id<MTLTexture> sampling_snapshot(id<MTLTexture> source, uint32_t mip = 0, uint32_t face = 0);
    // Repack equal-sized guest pixels using integer views, preserving every
    // native pixel and floating payload. Returns nil for unsupported layouts.
    id<MTLTexture> surface_format_cast(id<MTLTexture> source, SceGxmColorFormat, SceGxmTextureBaseFormat,
        uint32_t texture_swizzle = UINT32_MAX, id<MTLCommandBuffer> pending_commands = nil,
        bool raw_bits = false);
    // Input is unswizzled guest storage. Preserve all mips/faces as UNORM16
    // halves before any float-valued snapshot or texture sampling occurs.
    id<MTLTexture> raw_texture_snapshot(id<MTLTexture> source, id<MTLCommandBuffer> pending_commands = nil);
    id<MTLTexture> rgba8_memory_snapshot(id<MTLTexture> source, SceGxmColorFormat);
    // Convert encoded guest channels before filtering or texture swizzling.
    // Gamma 0: linear bytes, 1: sRGB RGB, 3: sRGB RG with unchanged BA.
    id<MTLTexture> rgba8_surface_sampling(id<MTLTexture> source, SceGxmColorFormat, uint32_t gamma,
        id<MTLCommandBuffer> pending_commands = nil);
    id<MTLTexture> color_snapshot(id<MTLTexture> source, id<MTLCommandBuffer> pending_commands = nil);
    // Copy six complete, identically formatted 2D faces without a float
    // round-trip. Inputs must already have identical memory-channel order.
    id<MTLTexture> rendered_cube(std::span<const id<MTLTexture>> faces,
        id<MTLCommandBuffer> pending_commands);
    // Inputs already have their guest channel/gamma sampling views applied.
    // Assemble all levels of a 2D image or cube from RAM uploads and rendered subresources.
    id<MTLTexture> texture_snapshot(id<MTLTexture> uploaded, std::span<const CubeSurface> surfaces, float scale,
        id<MTLCommandBuffer> pending_commands = nil);
    id<MTLTexture> cube_snapshot(id<MTLTexture> uploaded, std::span<const CubeSurface> surfaces, float scale,
        id<MTLCommandBuffer> pending_commands = nil);
    id<MTLTexture> snapshot_subrectangle(id<MTLTexture> source, uint32_t width, uint32_t height, SurfaceRect);
    // A smaller same-aspect depth view samples the whole source grid. Copy
    // storage words with nearest coordinates so signed bytes and float bits survive.
    id<MTLTexture> scaled_snapshot(id<MTLTexture> source, uint32_t width, uint32_t height,
        id<MTLCommandBuffer> pending_commands = nil);
    // Only destination_rect is initialized. With a command buffer, queue the
    // blit after its producers; otherwise submit and wait for guest publication.
    id<MTLTexture> resample_publication(id<MTLTexture> source, uint32_t width, uint32_t height,
        SurfaceRect source_rect, SurfaceRect destination_rect, bool raw_words, id<MTLCommandBuffer> commands = nil,
        const PublicationClip *clip = nullptr);
    id<MTLTexture> enqueue_subrectangle(id<MTLTexture> source, uint32_t width, uint32_t height,
        SurfaceRect, id<MTLCommandBuffer> commands);
    id<MTLTexture> color_subrectangle(id<MTLTexture> source, const SceGxmColorSurface &, SurfaceRect);
};
// Uploaded textures contain guest memory channels. Rendered surfaces instead
// contain logical shader RGBA; undo their color swizzle before sampling them.
id<MTLTexture> sampling_view(id<MTLTexture> texture, SceGxmTextureFormat format,
    const SceGxmColorFormat *rendered_format = nullptr,
    const ImportedTextureView *imported = nullptr);
id<MTLTexture> rgba8_gamma_view(id<MTLTexture> texture, bool srgb);
// Effective hardware setting after point-filter and strided-descriptor rules.
uint32_t effective_sampler_anisotropy(const SceGxmTexture &texture, uint32_t requested);
id<MTLSamplerState> make_sampler(Device &device, const SceGxmTexture &texture, uint32_t anisotropy);
bool surface_format_cast_supported(SceGxmColorFormat, SceGxmTextureBaseFormat,
    uint32_t texture_swizzle = UINT32_MAX);
bool surface_word_target_supported(SceGxmTextureBaseFormat);
bool surface_halfword_target_supported(SceGxmTextureBaseFormat);
std::optional<SurfaceRect> surface_halfword_subrectangle(const SceGxmColorSurface &,
    const SceGxmTexture &);
std::optional<SurfaceRect> surface_byte_subrectangle(const SceGxmColorSurface &,
    const SceGxmTexture &);
bool surface_format_cast_enqueueable(SceGxmColorFormat, SceGxmTextureBaseFormat,
    uint32_t texture_swizzle = UINT32_MAX);
// Plus carries 64-bit float surface aliases through four UNORM16 halves to
// keep the sampler from canonicalizing the reinterpreted F32 NaN words.
bool surface_raw_cast_required(SceGxmColorFormat, SceGxmTextureBaseFormat);
bool raw_texture_snapshot_supported(id<MTLTexture>);
float packed_alias_x_offset(float scale, uint32_t native_alias_width);
// GPU work must be complete before reading. Preserves guest row/tile padding
// and copies raw component bits, including floating-point NaN payloads.
size_t surface_memory_size(const SceGxmColorSurface &surface);
// Guest byte ownership for a clipped rectangle, in memory order. Excludes
// linear pitch and edge-tile padding; an empty rectangle produces no ranges.
bool surface_memory_ranges(const SceGxmColorSurface &surface, SurfaceRect rectangle,
    std::vector<SurfaceMemoryRange> &ranges);
bool read_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<uint8_t> destination);
// Update only written guest bytes, including partial components. Empty ranges
// validate the mapping without modifying the texture.
bool write_surface_memory(id<MTLTexture> texture, const SceGxmColorSurface &surface,
    std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges);
// A cube sampler addresses six consecutive swizzled faces. Resolve that view
// from the shader when the guest supplies the generic swizzled descriptor.
// Sampling controls do not change uploaded pixels. Strided fields still encode pitch.
SceGxmTexture texture_image_descriptor(const SceGxmTexture &texture);
SceGxmTexture cube_texture_descriptor(const SceGxmTexture &texture);
size_t cube_texture_storage_size(const SceGxmTexture &texture);
size_t texture_storage_size(const SceGxmTexture &texture);
}
