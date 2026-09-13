// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal/apple.h>
#include <gxm/types.h>
#include <span>
#include <optional>
#include <map>
struct SceGxmColorSurface;
struct SceGxmDepthStencilSurface;

namespace renderer::metal {
// Linear-strided descriptors reuse LOD bits for stride and have no LOD bias.
inline float sampler_lod_bias(const SceGxmTexture &texture) {
    return texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED ? 0.f : (float(texture.lod_bias) - 31.f) / 8.f;
}
class Device;
struct MetalTextureCache;
struct CubeSurface {
    id<MTLTexture> texture;
    uint32_t face, mip;
};
struct SurfaceRect { uint32_t x, y, width, height; };
struct SurfaceMemoryRange { size_t offset, size; };
std::optional<SurfaceRect> surface_subrectangle(const SceGxmColorSurface &, const SceGxmTexture &);
// Match a representable depth view in the allocation's guest sample coordinates.
std::optional<SurfaceRect> depth_subrectangle(const SceGxmDepthStencilSurface &, uint32_t width, uint32_t height,
    SceGxmMultisampleMode, const SceGxmTexture &);
// Match a complete depth allocation in guest sample coordinates.
bool depth_texture_matches(const SceGxmDepthStencilSurface &, uint32_t width, uint32_t height,
    SceGxmMultisampleMode, const SceGxmTexture &);
id<MTLTexture> current_texture(const MetalTextureCache &cache);
struct DepthMemoryLayout {
    uint32_t width, height, stride, depth_bytes;
    size_t depth_size, stencil_size;
    bool tiled, packed;
};
std::optional<DepthMemoryLayout> depth_memory_layout(const SceGxmDepthStencilSurface &, uint32_t width, uint32_t height,
    SceGxmMultisampleMode);
class SurfaceCaster {
    Device &device;
    std::map<uint32_t,id<MTLRenderPipelineState>> depth_seed_pipelines;
    id<MTLComputePipelineState> depth_store_pipeline, depth_store_ms_pipeline;
    id<MTLBuffer> depth_store_buffer;
    id<MTLComputePipelineState> pipeline;
    id<MTLComputePipelineState> cube_pipeline, component_cast_pipeline;
    id<MTLComputePipelineState> depth_pipeline, rg_gamma_pipeline;
    id<MTLComputePipelineState> multisample_pipeline, multisample_depth_pipeline;
    id<MTLComputePipelineState> multisample_integer_pipeline;
    id<MTLLibrary> multisample_library;
    std::map<std::pair<uint32_t,uint32_t>,id<MTLRenderPipelineState>> seed_pipelines;
public:
    explicit SurfaceCaster(Device &device);
    id<MTLTexture> rgba8_from_rg32(id<MTLTexture> source, uint32_t scale, bool swap_words,
        uint32_t word_offset = 0, bool signed_normalized = false);
    bool load_depth_memory(id<MTLTexture>, const SceGxmDepthStencilSurface &, const DepthMemoryLayout &, uint32_t scale,
        std::span<const uint8_t> depth, std::span<const uint8_t> stencil);
    bool store_depth_memory(id<MTLTexture>, const SceGxmDepthStencilSurface &, const DepthMemoryLayout &, uint32_t scale,
        std::span<uint8_t> depth, std::span<uint8_t> stencil);
    id<MTLTexture> depth_snapshot(id<MTLTexture> source, bool normalized16 = false, uint32_t scale = 1);
    // Native sample IDs in a 1x2 (2X) or 2x2 (4X) image. Each guest sample
    // occupies a scale-by-scale block when rendering at higher resolution.
    void expand_multisample(id<MTLTexture> source, id<MTLTexture> destination, uint32_t scale);
    void seed_multisample(id<MTLTexture> source, id<MTLTexture> destination, uint32_t scale, bool expanded);
    bool patch_multisample(id<MTLTexture> texture, const SceGxmColorSurface &, uint32_t scale,
        std::span<const uint8_t> source, std::span<const SurfaceMemoryRange> ranges);
    // Diagnostic canonical RGBA values after the texture view's swizzle.
    // Returns nil for integer/unsupported formats; caller must complete producers.
    id<MTLTexture> sampling_snapshot(id<MTLTexture> source, uint32_t mip = 0, uint32_t face = 0);
    // Repack equal-sized guest pixels using integer views, preserving every
    // native pixel and floating payload. Returns nil for unsupported layouts.
    id<MTLTexture> surface_format_cast(id<MTLTexture> source, SceGxmColorFormat, SceGxmTextureBaseFormat);
    id<MTLTexture> rgba8_memory_snapshot(id<MTLTexture> source, SceGxmColorFormat);
    // Convert encoded guest channels before filtering or texture swizzling.
    // Gamma 0: linear bytes, 1: sRGB RGB, 3: sRGB RG with unchanged BA.
    id<MTLTexture> rgba8_surface_sampling(id<MTLTexture> source, SceGxmColorFormat, uint32_t gamma);
    id<MTLTexture> color_snapshot(id<MTLTexture> source);
    // Inputs already have their guest channel/gamma sampling views applied.
    // Assemble all levels of a 2D image or cube from RAM uploads and rendered subresources.
    id<MTLTexture> texture_snapshot(id<MTLTexture> uploaded, std::span<const CubeSurface> surfaces, uint32_t scale);
    id<MTLTexture> cube_snapshot(id<MTLTexture> uploaded, std::span<const CubeSurface> surfaces, uint32_t scale);
    id<MTLTexture> snapshot_subrectangle(id<MTLTexture> source, uint32_t width, uint32_t height, SurfaceRect);
    id<MTLTexture> color_subrectangle(id<MTLTexture> source, const SceGxmColorSurface &, SurfaceRect);
};
// Uploaded textures contain guest memory channels. Rendered surfaces instead
// contain logical shader RGBA; undo their color swizzle before sampling them.
id<MTLTexture> sampling_view(id<MTLTexture> texture, SceGxmTextureFormat format,
    const SceGxmColorFormat *rendered_format = nullptr);
id<MTLTexture> rgba8_gamma_view(id<MTLTexture> texture, bool srgb);
// Effective hardware setting after point-filter and strided-descriptor rules.
uint32_t effective_sampler_anisotropy(const SceGxmTexture &texture, uint32_t requested);
id<MTLSamplerState> make_sampler(Device &device, const SceGxmTexture &texture, uint32_t anisotropy);
bool surface_format_cast_supported(SceGxmColorFormat, SceGxmTextureBaseFormat);
float packed_alias_x_offset(float scale, uint32_t native_alias_width);
// GPU work must be complete before reading. Preserves guest row/tile padding
// and copies raw component bits, including floating-point NaN payloads.
size_t surface_memory_size(const SceGxmColorSurface &surface);
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
