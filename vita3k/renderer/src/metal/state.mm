// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <renderer/metal/buffers.h>
#include <renderer/metal/overlay.h>
#include <renderer/metal/polygon_clip.h>
#include <renderer/metal/screen.h>
#include <renderer/metal/textures.h>
#include <renderer/metal/state.h>
#include <renderer/metal/vertex_layout.h>
#include <renderer/metal/surface_write_watch.h>
#include <renderer/functions.h>
#include <renderer/shaders.h>
#include <shader/msl_recompiler.h>
#include <shader/metal_texture.h>
#include <shader/uniform_block.h>
#include <gxm/functions.h>
#include <mem/functions.h>
#include <config/state.h>
#include <display/state.h>
#include <overlay/display_manager.h>
#include <util/log.h>
#include <util/bytes.h>
#include <astcenc.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#define XXH_INLINE_ALL
#include <xxhash.h>
extern "C" {
#include <libswscale/swscale.h>
}

namespace renderer::metal {
namespace {
void require(bool condition, const std::string &error) {
    if (!condition) throw std::runtime_error(error);
}
id<MTLCommandBuffer> scene_command_buffer(Device &device) {
    id<MTLCommandBuffer> commands = [device.command_queue() commandBuffer];
    commands.label = @"Vita3K GXM scene";
    return commands;
}
void append(std::string &key, uint32_t value) {
    const char bytes[] = {static_cast<char>(value), static_cast<char>(value >> 8),
        static_cast<char>(value >> 16), static_cast<char>(value >> 24)};
    key.append(bytes, sizeof(bytes));
}
bool surface_texture_format_matches(SceGxmColorFormat color, SceGxmTextureFormat texture) {
    SceGxmTextureFormat mapped{};
    if (!gxm::convert_color_format_to_texture_format(color,mapped)
        || gxm::get_base_format(mapped)!=gxm::get_base_format(texture)) return false;
    // X8S8S8U8 decodes its two byte orders differently. A direct GPU view is
    // valid only for the matching color order; other views need a byte cast.
    return gxm::get_base_format(mapped)!=SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8
        || mapped==texture;
}
shader::metal::Program depth_only_program(uint32_t samples) {
    // Disabling the guest fragment program leaves raster depth/stencil active.
    // A small native function retains GXM's separate mask test, including each
    // MSAA sample, without running guest texture loads, DEPTHF, or buffer stores.
    shader::metal::Program program{};
    program.stage = shader::metal::Stage::Fragment;
    program.entry_point = "depth_only";
    program.writes_guest_memory = false;
    program.source = "#include <metal_stdlib>\nusing namespace metal;\nfragment void depth_only(float4 p [[position]], ";
    program.source += samples > 1 ? "texture2d_ms<float> mask" : "texture2d<float> mask";
    program.source += " [[texture(" + std::to_string(shader::metal::MASK_TEXTURE) + ")]]";
    if (samples > 1) program.source += ", uint sample [[sample_id]]";
    program.source += ") { if (all(mask.read(uint2(p.xy)";
    if (samples > 1) program.source += ", sample";
    program.source += ") < float4(0.5))) discard_fragment(); }\n";
    return program;
}
MTLPixelFormat color_format(SceGxmColorFormat format, id<MTLDevice> device) {
    switch (gxm::get_base_format(format)) {
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8:
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8: return MTLPixelFormatRGBA8Unorm;
    // GXM stores A8/R3/G3/B2 in 16 bits. Metal has no renderable format with
    // that channel precision, so retain logical RGBA in an expanded target.
    case SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2:
        require((uint32_t(format) & SCE_GXM_COLOR_SWIZZLE_MASK) == 0,
            "Metal: unsupported U8U3U3U2 color surface swizzle");
        return MTLPixelFormatRGBA8Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_S5S5U6:
        require(((uint32_t(format) & SCE_GXM_COLOR_SWIZZLE_MASK) >> 20) < 2,
            "Metal: unsupported S5S5U6 color surface swizzle");
        return MTLPixelFormatRGBA16Float;
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8: return MTLPixelFormatRGBA8Snorm;
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16: return MTLPixelFormatRGBA16Float;
    // Metal has no A2 + three unsigned 10-bit float attachment. Keep the
    // channels in an expanded float target, also usable by later texture reads.
    case SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10: return MTLPixelFormatRGBA16Float;
    // Two signed and two unsigned byte channels cannot share a native Metal
    // normalized attachment. Preserve their logical values in half floats.
    case SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8: return MTLPixelFormatRGBA16Float;
    case SCE_GXM_COLOR_BASE_FORMAT_F32F32: return MTLPixelFormatRG32Float;
    case SCE_GXM_COLOR_BASE_FORMAT_F32: return MTLPixelFormatR32Float;
    case SCE_GXM_COLOR_BASE_FORMAT_F16: return MTLPixelFormatR16Float;
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16: return MTLPixelFormatRG16Float;
    case SCE_GXM_COLOR_BASE_FORMAT_U8: return MTLPixelFormatR8Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_S8: return MTLPixelFormatR8Snorm;
    case SCE_GXM_COLOR_BASE_FORMAT_U16: return MTLPixelFormatR16Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_S16: return MTLPixelFormatR16Snorm;
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8: return MTLPixelFormatRG8Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8: return MTLPixelFormatRG8Snorm;
    case SCE_GXM_COLOR_BASE_FORMAT_U16U16: return MTLPixelFormatRG16Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_S16S16: return MTLPixelFormatRG16Snorm;
    case SCE_GXM_COLOR_BASE_FORMAT_F11F11F10: return MTLPixelFormatRG11B10Float;
    case SCE_GXM_COLOR_BASE_FORMAT_U5U6U5: return MTLPixelFormatB5G6R5Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_U4U4U4U4: return MTLPixelFormatABGR4Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9:
        // RGB9E5 is filter-only on Mac-family GPUs. Apple-family GPUs can
        // render to it; otherwise expand the attachment and pack on readback.
        return [device supportsFamily:MTLGPUFamilyApple2]
            ? MTLPixelFormatRGB9E5Float : MTLPixelFormatRGBA16Float;
    case SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10: return MTLPixelFormatBGR10A2Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_U1U5U5U5: {
        const uint32_t mode=(uint32_t(format)&SCE_GXM_COLOR_SWIZZLE_MASK)>>20;
        if (mode >= 4) throw std::runtime_error("Metal: unsupported 5:5:5:1 color surface swizzle");
        return mode < 2 ? MTLPixelFormatBGR5A1Unorm : MTLPixelFormatA1BGR5Unorm;
    }
    default: throw std::runtime_error("Metal: unsupported color surface format " + std::to_string(format));
    }
}
MTLVertexFormat attribute_format(SceGxmAttributeFormat f, uint32_t count, bool is_signed) {
    require(count >= 1 && count <= 4, "Metal: invalid vertex component count");
    static const MTLVertexFormat formats[][4] = {
        {MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4},
        {MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4},
        {MTLVertexFormatUShort, MTLVertexFormatUShort2, MTLVertexFormatUShort3, MTLVertexFormatUShort4},
        {MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4},
        {MTLVertexFormatUCharNormalized, MTLVertexFormatUChar2Normalized, MTLVertexFormatUChar3Normalized, MTLVertexFormatUChar4Normalized},
        {MTLVertexFormatCharNormalized, MTLVertexFormatChar2Normalized, MTLVertexFormatChar3Normalized, MTLVertexFormatChar4Normalized},
        {MTLVertexFormatUShortNormalized, MTLVertexFormatUShort2Normalized, MTLVertexFormatUShort3Normalized, MTLVertexFormatUShort4Normalized},
        {MTLVertexFormatShortNormalized, MTLVertexFormatShort2Normalized, MTLVertexFormatShort3Normalized, MTLVertexFormatShort4Normalized},
        {MTLVertexFormatHalf, MTLVertexFormatHalf2, MTLVertexFormatHalf3, MTLVertexFormatHalf4},
        {MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4},
        {MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3, MTLVertexFormatUInt4}
    };
    require(f <= SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED, "Metal: unsupported vertex format");
    if (f == SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED && is_signed) {
        static const MTLVertexFormat signed_formats[] = {
            MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3, MTLVertexFormatInt4
        };
        return signed_formats[count - 1];
    }
    return formats[f][count - 1];
}
MTLBlendOperation blend_op(SceGxmBlendFunc f) {
    switch (f) {
    case SCE_GXM_BLEND_FUNC_NONE:
    case SCE_GXM_BLEND_FUNC_ADD: return MTLBlendOperationAdd;
    case SCE_GXM_BLEND_FUNC_SUBTRACT: return MTLBlendOperationSubtract;
    case SCE_GXM_BLEND_FUNC_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
    case SCE_GXM_BLEND_FUNC_MIN: return MTLBlendOperationMin;
    case SCE_GXM_BLEND_FUNC_MAX: return MTLBlendOperationMax;
    }
    throw std::runtime_error("Metal: unknown blend operation");
}
MTLBlendFactor blend_factor(SceGxmBlendFactor f) {
    // Vulkan currently falls back to destination alpha for this GXM-only
    // saturated factor. Match that behavior until both backends emulate it.
    if (f == SCE_GXM_BLEND_FACTOR_DST_ALPHA_SATURATE) return MTLBlendFactorDestinationAlpha;
    static const MTLBlendFactor factors[] = {MTLBlendFactorZero, MTLBlendFactorOne, MTLBlendFactorSourceColor,
        MTLBlendFactorOneMinusSourceColor, MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha,
        MTLBlendFactorDestinationColor, MTLBlendFactorOneMinusDestinationColor, MTLBlendFactorDestinationAlpha,
        MTLBlendFactorOneMinusDestinationAlpha, MTLBlendFactorSourceAlphaSaturated};
    require(f < std::size(factors), "Metal: unknown blend factor");
    return factors[f];
}
MTLStencilOperation stencil_op(SceGxmStencilOp op) {
    switch (op) {
    case SCE_GXM_STENCIL_OP_KEEP: return MTLStencilOperationKeep;
    case SCE_GXM_STENCIL_OP_ZERO: return MTLStencilOperationZero;
    case SCE_GXM_STENCIL_OP_REPLACE: return MTLStencilOperationReplace;
    case SCE_GXM_STENCIL_OP_INCR: return MTLStencilOperationIncrementClamp;
    case SCE_GXM_STENCIL_OP_DECR: return MTLStencilOperationDecrementClamp;
    case SCE_GXM_STENCIL_OP_INVERT: return MTLStencilOperationInvert;
    case SCE_GXM_STENCIL_OP_INCR_WRAP: return MTLStencilOperationIncrementWrap;
    case SCE_GXM_STENCIL_OP_DECR_WRAP: return MTLStencilOperationDecrementWrap;
    }
    throw std::runtime_error("Metal: unknown stencil operation");
}
MTLStencilDescriptor *stencil_desc(const GxmStencilStateOp &ops, const GxmStencilStateValues &values) {
    MTLStencilDescriptor *result = [MTLStencilDescriptor new];
    result.stencilCompareFunction = static_cast<MTLCompareFunction>(static_cast<uint32_t>(ops.func) >> 25);
    result.stencilFailureOperation = stencil_op(ops.stencil_fail);
    result.depthFailureOperation = stencil_op(ops.depth_fail);
    result.depthStencilPassOperation = stencil_op(ops.depth_pass);
    result.readMask = values.compare_mask;
    result.writeMask = values.write_mask;
    return result;
}
// Disjoint scissors preserve blend/occlusion results when excluding an interior rectangle.
std::vector<MTLScissorRect> scissors(const GxmRecordState &r, uint32_t w, uint32_t h, float scale) {
    if (r.region_clip_mode == SCE_GXM_REGION_CLIP_ALL) return {};
    if (r.region_clip_mode == SCE_GXM_REGION_CLIP_NONE) return {{0, 0, w, h}};
    const auto clamp_edge = [](double v, uint32_t limit) { return static_cast<uint32_t>(std::clamp(v, 0.0, double(limit))); };
    const auto x0 = clamp_edge(std::floor(double(r.region_clip_min.x) * scale), w);
    const auto y0 = clamp_edge(std::floor(double(r.region_clip_min.y) * scale), h);
    const bool outside = r.region_clip_mode == SCE_GXM_REGION_CLIP_OUTSIDE;
    // Vulkan scales the tile-aligned scissor origin and extent separately,
    // truncating each. INSIDE must exclude exactly the area OUTSIDE includes.
    const auto x1 = std::max(x0, clamp_edge(double(x0)
        + std::floor((double(r.region_clip_max.x) - r.region_clip_min.x + 1) * scale), w));
    const auto y1 = std::max(y0, clamp_edge(double(y0)
        + std::floor((double(r.region_clip_max.y) - r.region_clip_min.y + 1) * scale), h));
    std::vector<MTLScissorRect> result;
    auto add = [&](uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        if (width && height) result.push_back({x, y, width, height});
    };
    if (outside) add(x0, y0, x1 - x0, y1 - y0);
    else {
        add(0, 0, w, y0); add(0, y1, w, h - y1);
        add(0, y0, x0, y1 - y0); add(x1, y0, w - x1, y1 - y0);
    }
    return result;
}
id<MTLTexture> make_texture(Device &device, MTLPixelFormat format, uint32_t w, uint32_t h, MTLTextureUsage usage, uint32_t samples = 1) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    desc.usage = usage;
    desc.storageMode = (samples > 1 || format == MTLPixelFormatDepth32Float_Stencil8) ? MTLStorageModePrivate : MTLStorageModeShared;
    desc.sampleCount = samples;
    if (samples > 1) desc.textureType = MTLTextureType2DMultisample;
    id<MTLTexture> result = [device.native_device() newTextureWithDescriptor:desc];
    require(result != nil, "Metal: texture allocation failed");
    return result;
}
}

// Match Plus: only discard beyond the far plane when both active faces fail
// the depth test there and discarding cannot suppress a stencil side effect.
static bool draw_drops_fragments_beyond_far_plane(const GxmRecordState &record) {
    const auto depth_fails = [](SceGxmDepthFunc func) {
        return func == SCE_GXM_DEPTH_FUNC_NEVER || func == SCE_GXM_DEPTH_FUNC_LESS
            || func == SCE_GXM_DEPTH_FUNC_EQUAL || func == SCE_GXM_DEPTH_FUNC_LESS_EQUAL;
    };
    const auto stencil_quiet = [](const GxmStencilStateOp &op) {
        return (op.func == SCE_GXM_STENCIL_FUNC_ALWAYS || op.stencil_fail == SCE_GXM_STENCIL_OP_KEEP)
            && op.depth_fail == SCE_GXM_STENCIL_OP_KEEP;
    };
    return depth_fails(record.front_depth_func) && stencil_quiet(record.front_stencil_state_op)
        && (record.two_sided == SCE_GXM_TWO_SIDED_DISABLED
            || (depth_fails(record.back_depth_func) && stencil_quiet(record.back_stencil_state_op)));
}

struct SurfaceWriteBounds {
    uint32_t x0 = UINT32_MAX, y0 = UINT32_MAX, x1 = 0, y1 = 0;
    bool empty() const { return x1 <= x0 || y1 <= y0; }
    void include(uint32_t left, uint32_t top, uint32_t right, uint32_t bottom) {
        if (right <= left || bottom <= top) return;
        x0 = std::min(x0, left); y0 = std::min(y0, top);
        x1 = std::max(x1, right); y1 = std::max(y1, bottom);
    }
};
struct HalfPixelStrips {
    uint32_t columns = 0, rows = 0;
    SurfaceWriteBounds column_span, row_span;
};
struct Surface {
    uint64_t revision = 0;
    uint64_t last_frame_rendered = 0;
    uint64_t binding_generation = 0;
    id<MTLTexture> color;
    // A float view of an independently written RGBA16Uint attachment. Keep
    // normal color for blending/display; raw register bits serve guest reads.
    id<MTLTexture> raw_color;
    bool raw_color_invalidated = false;
    id<MTLTexture> raw_multisample_color;
    bool raw_multisample_dirty = true;
    id<MTLTexture> multisample_color;
    bool multisample_dirty = true;
    float multisample_scale = 1;
    std::map<std::array<uint32_t, 3>, id<MTLTexture>> rgba8_casts;
    std::map<std::array<uint32_t, 4>, id<MTLTexture>> word_casts;
    std::map<std::array<uint32_t, 6>, id<MTLTexture>> word_rect_casts;
    std::map<std::array<uint32_t, 6>, id<MTLTexture>> halfword_rect_casts;
    std::map<std::array<uint32_t, 6>, id<MTLTexture>> byte_rect_casts;
    std::map<std::array<uint32_t, 6>, id<MTLTexture>> small_rect_casts;
    bool has_word_offset_view = false;
    std::map<std::array<uint32_t,4>,id<MTLTexture>> subrectangles;
    SceGxmColorSurface guest{};
    // Last CPU bytes imported or published. Compare against this baseline,
    // not a downsampled GPU image, to retain untouched high-resolution pixels.
    std::vector<uint8_t> cpu_snapshot;
    uint64_t cpu_hash = 0;
    size_t cpu_hash_size = 0;
    SurfaceWriteStamp cpu_writes;
    SurfaceWriteBounds scene_writes, written_tiles;
    // Scene-wide logical macroblock render area in native pixels, plus its
    // guest-space publication bounds. Keep the native union before rounding.
    SurfaceWriteBounds scene_render_area, scene_macroblock_bounds;
    uint32_t rendered_width = 0, rendered_height = 0;
    float render_scale = 1.f;
    uint32_t publication_samples_x = 1, publication_samples_y = 1;
};
static id<MTLTexture> raw_surface_color(const Surface &surface) {
    return surface.raw_color && !surface.raw_color_invalidated ? surface.raw_color : surface.color;
}
static bool write_surface_storage(Surface &surface, const SceGxmColorSurface &guest,
    std::span<const uint8_t> bytes, std::span<const SurfaceMemoryRange> ranges) {
    return write_surface_memory(surface.color,guest,bytes,ranges)
        && (!surface.raw_color || write_surface_memory(surface.raw_color,guest,bytes,ranges));
}
static void update_cpu_snapshot(Surface &surface, std::span<const uint8_t> bytes,
    std::span<const SurfaceMemoryRange> ranges) {
    surface.cpu_hash = XXH3_64bits(bytes.data(), bytes.size());
    surface.cpu_hash_size = bytes.size();
    if (surface.cpu_snapshot.size() != bytes.size()) return;
    for (const auto &range : ranges)
        std::memcpy(surface.cpu_snapshot.data()+range.offset, bytes.data()+range.offset, range.size);
}
static bool surface_cpu_bytes_changed(const Surface &entry, std::span<const uint8_t> bytes) {
    if (entry.cpu_writes.changed()) return true;
    if (bytes.empty()) return false;
    return entry.cpu_snapshot.size() == bytes.size()
        ? std::memcmp(entry.cpu_snapshot.data(), bytes.data(), bytes.size()) != 0
        : entry.cpu_hash_size == bytes.size() && XXH3_64bits(bytes.data(), bytes.size()) != entry.cpu_hash;
}
// Keep Plus' accumulated write ownership and render-target extent separate
// from the latest scene rectangle. Conversion must not publish unowned bytes.
enum class SurfacePublication { Unavailable, Published, CpuNewer };
static SurfacePublication read_surface_publication(const Surface &entry, const SceGxmColorSurface &surface,
    std::span<uint8_t> output, std::vector<SurfaceMemoryRange> &written,
    Device &device, std::unique_ptr<SurfaceCaster> &caster) {
    written.clear();
    // Plus suppresses small-surface writeback after a CPU write. Native guest
    // snapshots identify changed bytes even while this surface is still bound
    // (and therefore excluded from retire_cpu_overwritten_surfaces). Compare
    // before conversion replaces RAM, and never acknowledge these CPU bytes
    // in the GPU baseline: the next bind must still import them.
    const bool same_layout = entry.guest.width == surface.width && entry.guest.height == surface.height
        && entry.guest.strideInPixels == surface.strideInPixels && entry.guest.colorFormat == surface.colorFormat
        && entry.guest.surfaceType == surface.surfaceType;
    if (surface.width <= 512 && surface.height <= 512 && same_layout && surface_cpu_bytes_changed(entry, output))
        return SurfacePublication::CpuNewer;
    SurfaceWriteBounds bounds{0, 0, surface.width, surface.height};
    const auto intersect = [&](const SurfaceWriteBounds &limit) {
        const auto previous = bounds;
        bounds.x0 = std::max(bounds.x0, limit.x0); bounds.y0 = std::max(bounds.y0, limit.y0);
        bounds.x1 = std::min(bounds.x1, limit.x1); bounds.y1 = std::min(bounds.y1, limit.y1);
        return bounds.x0 != previous.x0 || bounds.y0 != previous.y0
            || bounds.x1 != previous.x1 || bounds.y1 != previous.y1;
    };
    const auto base = gxm::get_base_format(surface.colorFormat);
    const bool raw = entry.raw_color && !entry.raw_color_invalidated;
    // Expanded attachments need conversion before publishing guest words.
    // RGB9E5 joins this path on Mac-family GPUs without a renderable native
    // RGB9E5 attachment; Apple-family GPUs keep the packed attachment.
    const bool repack = base == SCE_GXM_COLOR_BASE_FORMAT_U8U8U8
        || base == SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10
        || (base == SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
            && entry.color.pixelFormat == MTLPixelFormatRGBA16Float);
    bool macroblock_clamped = false, ownership_clamped = false;
    if (!raw && !repack) {
        if (!entry.scene_render_area.empty()) macroblock_clamped = intersect(entry.scene_macroblock_bounds);
        ownership_clamped |= intersect(entry.written_tiles);
        if (entry.rendered_width && entry.rendered_height)
            ownership_clamped |= intersect({0, 0, entry.rendered_width, entry.rendered_height});
    }
    const bool small_linear = surface.surfaceType == SCE_GXM_COLOR_SURFACE_LINEAR
        && surface.width <= 512 && surface.height <= 512;
    const bool small_tiled = surface.surfaceType == SCE_GXM_COLOR_SURFACE_TILED
        && surface.width <= 512 && surface.height <= 512;
    const auto &tiles = entry.written_tiles;
    const bool covers_all = !tiles.empty() && tiles.x0 == 0 && tiles.y0 == 0
        && tiles.x1 >= surface.width && tiles.y1 >= surface.height;
    // Plus defers incomplete small tiled writebacks. Keep the existing native
    // small-linear conversion support while restricting it to the scene.
    if (small_tiled && !covers_all) return SurfacePublication::Published;
    if (small_linear && !covers_all) {
        intersect(entry.scene_writes);
        ownership_clamped = true;
    }
    if (bounds.empty()) return SurfacePublication::Published;
    if (!surface_memory_ranges(surface,
            {bounds.x0, bounds.y0, bounds.x1 - bounds.x0, bounds.y1 - bounds.y0}, written)) return SurfacePublication::Unavailable;
    for (const auto &range : written) {
        if (range.offset > output.size() || range.size > output.size() - range.offset) {
            written.clear();
            return SurfacePublication::Unavailable;
        }
    }
    id<MTLTexture> source = raw_surface_color(entry);
    if (entry.render_scale != 1.f) {
        SurfaceWriteBounds input{0, 0, uint32_t(source.width), uint32_t(source.height)};
        if (ownership_clamped) {
            const auto edge = [&](uint32_t value, NSUInteger limit) {
                return uint32_t(std::clamp(std::floor(value * double(entry.render_scale)), 0.0, double(limit)));
            };
            input = {edge(bounds.x0, source.width), edge(bounds.y0, source.height),
                edge(bounds.x1, source.width), edge(bounds.y1, source.height)};
        } else if (macroblock_clamped) {
            // The unclipped native macroblock interval is the blit source;
            // its truncated guest interval is the destination (Plus rt_clamped=false).
            const auto &area = entry.scene_render_area;
            input = {std::min<uint32_t>(area.x0 * entry.publication_samples_x, source.width),
                std::min<uint32_t>(area.y0 * entry.publication_samples_y, source.height),
                std::min<uint32_t>(area.x1 * entry.publication_samples_x, source.width),
                std::min<uint32_t>(area.y1 * entry.publication_samples_y, source.height)};
        }
        if (!caster) caster = std::make_unique<SurfaceCaster>(device);
        std::optional<PublicationClip> publication_clip;
        if (surface.clip_enabled && entry.publication_samples_x == 1 && entry.publication_samples_y == 1
            && source.textureType == MTLTextureType2D && source.sampleCount == 1) {
            const uint32_t x0 = std::min(uint32_t(surface.clip_x_min), surface.width);
            const uint32_t y0 = std::min(uint32_t(surface.clip_y_min), surface.height);
            const uint32_t x1 = std::min(uint32_t(surface.clip_x_max) + 1, surface.width);
            const uint32_t y1 = std::min(uint32_t(surface.clip_y_max) + 1, surface.height);
            const auto edge = [&](uint32_t coordinate, NSUInteger limit) {
                return uint32_t(std::clamp(std::floor(double(coordinate) * entry.render_scale), 0.0, double(limit)));
            };
            const uint32_t sx0 = edge(x0, source.width), sy0 = edge(y0, source.height);
            const uint32_t sx1 = edge(x1, source.width), sy1 = edge(y1, source.height);
            if (x0 < x1 && y0 < y1 && sx0 < sx1 && sy0 < sy1)
                publication_clip = {{x0, y0, x1 - x0, y1 - y0},
                    {sx0, sy0, sx1 - sx0, sy1 - sy0}};
        }
        source = caster->resample_publication(source, surface.width, surface.height,
            {input.x0,input.y0,input.x1-input.x0,input.y1-input.y0},
            {bounds.x0,bounds.y0,bounds.x1-bounds.x0,bounds.y1-bounds.y0},
            raw || base == SCE_GXM_COLOR_BASE_FORMAT_F32 || base == SCE_GXM_COLOR_BASE_FORMAT_F32F32,
            nil, publication_clip ? &*publication_clip : nullptr);
    }
    if (bounds.x0 == 0 && bounds.y0 == 0 && bounds.x1 == surface.width && bounds.y1 == surface.height)
        return read_surface_memory(source, surface, output) ? SurfacePublication::Published : SurfacePublication::Unavailable;
    // Convert into temporary guest-layout storage, then copy only the owned
    // bytes. Padding remains untouched for every layout and packed format.
    std::vector<uint8_t> converted(output.size());
    if (!read_surface_memory(source, surface, converted)) return SurfacePublication::Unavailable;
    for (const auto &range : written)
        std::memcpy(output.data() + range.offset, converted.data() + range.offset, range.size);
    return SurfacePublication::Published;
}
struct DepthSurface {
    uint64_t last_attached_frame = 0;
    std::map<std::array<uint32_t,5>,id<MTLTexture>> subrectangles;
    id<MTLTexture> texture;
    id<MTLTexture> mask;
    id<MTLTexture> sample_rate_copy, sample_rate_mask;
    // Expanded source grid whose target-based extent and guest mapping were
    // established at allocation. A new scene must not infer it from its color.
    uint32_t sample_grid_width = 0, sample_grid_height = 0;
    std::map<SceGxmTextureBaseFormat,id<MTLTexture>> snapshots;
    SceGxmDepthStencilSurface guest{};
    uint32_t width = 0, height = 0;
    float scale = 1;
    SceGxmMultisampleMode multisample = SCE_GXM_MULTISAMPLE_NONE;
    // Last observed/imported or published guest bytes. A no-store scene may
    // keep newer native contents while these bytes stay unchanged in RAM.
    std::vector<uint8_t> guest_depth_snapshot, guest_stencil_snapshot;
    float snapshot_background_depth = 0;
    uint32_t snapshot_background_stencil = 0;
    bool guest_snapshot_valid = false;
    bool depth_content_stored = true;
    Address last_scene_color_addr = 0;
    bool published = false;
};
struct MetalContext::Impl {
    struct PendingBatch {
        id<MTLCommandBuffer> commands;
        UploadBufferArena uploads;
    };
    struct VisibilityResult {
        id<MTLBuffer> buffer;
        uint32_t offset;
        Address address;
        uint32_t index;
        bool increment;
        uint64_t group;
    };
    MemState *mem = nullptr;
    id<MTLCommandBuffer> commands;
    id<MTLRenderCommandEncoder> encoder;
    id<MTLTexture> color;
    id<MTLTexture> depth;
    id<MTLTexture> transient_depth;
    id<MTLTexture> transient_color;
    id<MTLTexture> render_color;
    id<MTLTexture> raw_color;
    id<MTLTexture> raw_render_color;
    id<MTLTexture> raw_attachment;
    id<MTLTexture> raw_clip_snapshot;
    id<MTLTexture> color_clip_snapshot;
    bool color_clip_restore_pending = false;
    bool color_clip_samplewise = false;
    uint32_t samples = 1;
    float sample_scale = 1;
    float surface_downscale = 1;
    bool expanded_color = false, custom_samples = false;
    std::array<MTLSamplePosition,4> sample_positions{};
    std::pair<Address, Address> depth_key{};
    bool depth_is_sample_rate_copy = false;
    id<MTLTexture> mask;
    id<MTLTexture> transient_mask;
    bool mask_constant_valid = false;
    bool mask_constant_value = false;
    SceGxmColorSurface guest_color{};
    SceGxmDepthStencilSurface guest_depth{};
    std::optional<DepthMemoryLayout> depth_layout;
    uint32_t depth_scene_width = 0, depth_scene_height = 0;
    bool scene_active=false;
    uint32_t width = 0, height = 0;
    uint16_t macroblock_last_x = ~0u, macroblock_last_y = ~0u;
    uint16_t macroblock_visited = 0;
    bool macroblock_ignore = false;
    bool mask_pass = false;
    bool transient_color_initialized = false;
    bool depth_written = false;
    bool depth_scene_written = false;
    bool scene_depth_drawn = false;
    bool direct_guest_memory_used = false;
    // Watch restoration belongs to GPU retirement, including failed draws.
    bool guest_write_watches_pending = false;
    uint32_t pending_draws = 0;
    bool pending_color_writes = false;
    HalfPixelStrips half_pixel_strips;
    // A mid-scene finish may already have published the current color image.
    bool color_guest_current = false;
    bool color_guest_dirty = false;
    uint64_t color_binding_generation = 0;
    bool color_publication_discarded = false;
    size_t pending_upload_bytes = 0;
    size_t pending_uniform_bytes = 0;
    size_t pending_stream_bytes = 0;
    size_t pending_index_bytes = 0;
    UploadBufferArena uploads;
    std::deque<PendingBatch> pending_batches;
    Address visibility_address{};
    uint32_t visibility_stride = 0;
    bool visibility_enabled = false;
    uint32_t visibility_index = 0;
    bool visibility_increment = false;
    bool back_visibility_enabled = false;
    uint32_t back_visibility_index = 0;
    bool back_visibility_increment = false;
    id<MTLBuffer> pass_visibility_buffer;
    Address pass_visibility_address{};
    uint32_t pass_visibility_stride = 0;
    uint32_t pass_visibility_offset = 0;
    size_t visibility_buffer_capacity = 4096;
    bool pass_visibility_active = false;
    uint32_t pass_visibility_index = 0;
    bool pass_visibility_increment = false;
    std::vector<VisibilityResult> visibility_results;
    uint64_t visibility_epoch = 1;
    std::map<std::array<uint64_t, 3>, bool> published_set_results;
};

static bool owns_color_binding(const Surface &surface, const MetalContext &ctx) {
    return ctx.impl->color_binding_generation
        && surface.binding_generation == ctx.impl->color_binding_generation
        && surface.color == ctx.impl->color;
}

static void note_half_pixel_origin(MetalContext &ctx, const MTLViewport &viewport, const MTLScissorRect &scissor,
    uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
    const float scale = ctx.impl->sample_scale * ctx.impl->surface_downscale;
    if (ctx.impl->samples != 1 || std::abs(scale - 1.0f) < 1e-4f || scale <= 0) return;
    const auto format = gxm::get_base_format(ctx.impl->guest_color.colorFormat);
    const bool raw_words = format == SCE_GXM_COLOR_BASE_FORMAT_F32 || format == SCE_GXM_COLOR_BASE_FORMAT_F32F32;
    const auto lines_to_fill = [&](float edge, NSUInteger scissor_start, uint32_t limit) -> uint32_t {
        if (!std::isfinite(edge)) return 0;
        const float native_edge = edge / scale;
        const float native_pixel = std::floor(native_edge + 1e-4f);
        const float fraction = native_edge - native_pixel;
        if (std::abs(fraction) < 1e-3f || native_pixel != 0 || fraction > 0.5f + 1e-3f || scissor_start > 0)
            return 0;
        const float first = std::floor(edge - 0.5f + 1e-4f) + 1;
        if (first <= 0 || first >= limit || (raw_words && (first + 0.5f) / scale >= 1.0f - 1e-4f)) return 0;
        return uint32_t(first);
    };
    auto &pending = ctx.impl->half_pixel_strips;
    const auto columns = lines_to_fill(float(std::min(viewport.originX, viewport.originX + viewport.width)), scissor.x, x1);
    const auto rows = lines_to_fill(float(std::min(viewport.originY, viewport.originY + viewport.height)), scissor.y, y1);
    if (columns) {
        pending.columns = std::max(pending.columns, columns);
        pending.column_span.include(0, y0, columns, y1);
    }
    if (rows) {
        pending.rows = std::max(pending.rows, rows);
        pending.row_span.include(x0, 0, x1, rows);
    }
}

static void fill_half_pixel_strips(MetalContext &ctx) {
    const auto pending = ctx.impl->half_pixel_strips;
    ctx.impl->half_pixel_strips = {};
    if ((!pending.columns && !pending.rows) || !ctx.impl->commands || !ctx.impl->guest_color.data
        || !ctx.impl->color || ctx.impl->samples != 1) return;
    const auto fill = [&](id<MTLTexture> color) {
        if (!color) return;
        const uint32_t width = uint32_t(color.width), height = uint32_t(color.height);
        uint32_t left = 0, top = 0, right = width, bottom = height;
        const auto &surface = ctx.impl->guest_color;
        if (surface.clip_enabled) {
            // Keep Metal's existing guest color-clip ownership, even at a pass
            // restart before the deferred outside-color restoration runs.
            const auto edge = [&](uint32_t value, uint32_t limit) {
                return uint32_t(std::clamp(std::floor(double(value) * ctx.impl->sample_scale), 0.0, double(limit)));
            };
            left = edge(surface.clip_x_min, width); top = edge(surface.clip_y_min, height);
            right = edge(uint32_t(surface.clip_x_max) + 1, width);
            bottom = edge(uint32_t(surface.clip_y_max) + 1, height);
        }
        const uint32_t col_y0 = std::max(pending.column_span.y0, top);
        const uint32_t col_y1 = std::min(pending.column_span.y1, bottom);
        const bool fill_columns = pending.columns > left && pending.columns < right && col_y1 > col_y0;
        const uint32_t row_x0 = fill_columns ? left : std::max(pending.row_span.x0, left);
        const uint32_t row_x1 = std::min(pending.row_span.x1, right);
        const bool fill_rows = pending.rows > top && pending.rows < bottom && row_x1 > row_x0;
        if (!fill_columns && !fill_rows) return;
        // Copy exact texel bits via a one-column/one-row scratch texture. The row
        // snapshot follows the column writes, so their common corner is filled.
        const auto scratch = [&](uint32_t w, uint32_t h) {
            auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:color.pixelFormat width:w height:h mipmapped:NO];
            desc.storageMode = MTLStorageModePrivate;
            id<MTLTexture> texture = [ctx.impl->commands.device newTextureWithDescriptor:desc];
            require(texture != nil, "Metal: cannot allocate half-pixel strip snapshot");
            return texture;
        };
        id<MTLTexture> column = fill_columns ? scratch(1, col_y1 - col_y0) : nil;
        id<MTLTexture> row = fill_rows ? scratch(row_x1 - row_x0, 1) : nil;
        auto blit = [ctx.impl->commands blitCommandEncoder];
        require(blit != nil, "Metal: cannot fill half-pixel strips");
        if (fill_columns) {
            const auto size = MTLSizeMake(1, col_y1 - col_y0, 1);
            [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(pending.columns, col_y0, 0)
                sourceSize:size toTexture:column destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            for (uint32_t x = left; x < pending.columns; ++x)
                [blit copyFromTexture:column sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:size toTexture:color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(x, col_y0, 0)];
        }
        if (fill_rows) {
            const auto size = MTLSizeMake(row_x1 - row_x0, 1, 1);
            [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(row_x0, pending.rows, 0)
                sourceSize:size toTexture:row destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            for (uint32_t y = top; y < pending.rows; ++y)
                [blit copyFromTexture:row sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:size toTexture:color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(row_x0, y, 0)];
        }
        [blit endEncoding];
    };
    fill(ctx.impl->color);
    fill(ctx.impl->raw_color);
}

static id<MTLTexture> depth_alias_view(SurfaceCaster &caster, id<MTLTexture> source,
    uint32_t memory_width, uint32_t memory_height, SurfaceRect rect, id<MTLCommandBuffer> commands) {
    // A smaller view at the beginning of the same aspect ratio addresses the
    // whole depth image with 0..1 UVs. Plus scales it instead of cropping it.
    const bool full_view = rect.x == 0 && rect.y == 0
        && rect.width < memory_width && rect.height < memory_height
        && uint64_t(memory_width) * rect.height == uint64_t(memory_height) * rect.width;
    if (full_view) {
        const auto width = std::max<uint32_t>(1, uint64_t(source.width) * rect.width / memory_width);
        const auto height = std::max<uint32_t>(1, uint64_t(source.height) * rect.height / memory_height);
        return caster.scaled_snapshot(source, width, height, commands);
    }
    return commands ? caster.enqueue_subrectangle(source, memory_width, memory_height, rect, commands)
                    : caster.snapshot_subrectangle(source, memory_width, memory_height, rect);
}

static void end_pass(MetalContext &ctx) {
    if (ctx.impl->encoder) {
        [ctx.impl->encoder endEncoding];
        ctx.impl->encoder = nil;
    }
    fill_half_pixel_strips(ctx);
}

// Raw register halves follow the same guest sample grid as normal color, but
// are moved through integer views. Never resolve them by averaging the words.
static void expand_color_samples(MetalContext &ctx, SurfaceCaster &caster,
    id<MTLCommandBuffer> commands = nil) {
    caster.expand_multisample(ctx.impl->render_color,ctx.impl->color,ctx.impl->sample_scale,
        ctx.impl->guest_color.width,ctx.impl->guest_color.height,commands);
    if (ctx.impl->raw_render_color)
        caster.expand_multisample(ctx.impl->raw_render_color,ctx.impl->raw_color,ctx.impl->sample_scale,
            ctx.impl->guest_color.width,ctx.impl->guest_color.height,commands);
}

static void configure_raw_attachment(MTLRenderPipelineDescriptor *descriptor, id<MTLTexture> texture,
    MTLColorWriteMask mask) {
    if (!texture) return;
    descriptor.colorAttachments[1].pixelFormat=texture.pixelFormat;
    descriptor.colorAttachments[1].blendingEnabled=NO;
    descriptor.colorAttachments[1].writeMask=mask;
}

static void begin_pass(MetalContext &ctx, bool mask, bool clear_depth = false, bool clear_stencil = false) {
    end_pass(ctx);
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.renderTargetWidth=ctx.impl->width;
    pass.renderTargetHeight=ctx.impl->height;
    pass.colorAttachments[0].texture = mask ? ctx.impl->mask : ctx.impl->render_color;
    pass.colorAttachments[0].loadAction = !mask && !ctx.impl->guest_color.data
            && !ctx.impl->transient_color_initialized ? MTLLoadActionDontCare : MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (!mask && ctx.impl->raw_attachment) {
        pass.colorAttachments[1].texture = ctx.impl->raw_attachment;
        pass.colorAttachments[1].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[1].storeAction = MTLStoreActionStore;
    }
    if (!mask && !ctx.impl->guest_color.data) ctx.impl->transient_color_initialized = true;
    if (!mask && ctx.impl->samples > 1 && ctx.impl->color && !ctx.impl->expanded_color) {
        pass.colorAttachments[0].resolveTexture = ctx.impl->color;
        pass.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
    }
    if (ctx.impl->custom_samples) [pass setSamplePositions:ctx.impl->sample_positions.data() count:ctx.impl->samples];
    pass.depthAttachment.texture = pass.stencilAttachment.texture = ctx.impl->depth;
    pass.depthAttachment.loadAction = clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
    pass.stencilAttachment.loadAction = clear_stencil ? MTLLoadActionClear : MTLLoadActionLoad;
    pass.depthAttachment.clearDepth = ctx.record.depth_stencil_surface.background_depth;
    pass.stencilAttachment.clearStencil = ctx.record.depth_stencil_surface.stencil;
    pass.depthAttachment.storeAction = pass.stencilAttachment.storeAction = MTLStoreActionStore;
    if (ctx.impl->custom_samples) pass.depthAttachment.storeAction = MTLStoreActionCustomSampleDepthStore;
    ctx.impl->pass_visibility_buffer = nil;
    ctx.impl->pass_visibility_address = ctx.impl->visibility_address;
    ctx.impl->pass_visibility_stride = ctx.impl->visibility_stride;
    ctx.impl->pass_visibility_offset = 0;
    ctx.impl->pass_visibility_active = false;
    if (ctx.impl->visibility_address && ctx.impl->visibility_stride >= sizeof(uint32_t)) {
        // Each draw uses a fresh eight-byte result. Metal allows an offset only
        // once per encoder; separate passes must not reset earlier results.
        ctx.impl->pass_visibility_buffer = [ctx.impl->commands.device
            newBufferWithLength:ctx.impl->visibility_buffer_capacity options:MTLResourceStorageModeShared];
        require(ctx.impl->pass_visibility_buffer != nil, "Metal: visibility result buffer allocation failed");
        pass.visibilityResultBuffer = ctx.impl->pass_visibility_buffer;
    }
    ctx.impl->encoder = [ctx.impl->commands renderCommandEncoderWithDescriptor:pass];
    ctx.impl->mask_pass = mask;
    ctx.impl->depth_written |= clear_depth || clear_stencil;
    require(ctx.impl->encoder != nil, "Metal: cannot begin GXM render pass");
    [ctx.impl->encoder setDepthClipMode:MTLDepthClipModeClamp];
}

static void restore_color_outside_clip(MetalContext &ctx, SurfaceCaster *caster) {
    if (!ctx.impl->color_clip_snapshot || !ctx.impl->color_clip_restore_pending)
        return;
    const auto &surface = ctx.impl->guest_color;
    if (ctx.impl->color_clip_samplewise) {
        require(caster != nullptr, "Metal: samplewise color clip has no surface caster");
        caster->restore_clipped_multisample(ctx.impl->color_clip_snapshot,
            ctx.impl->render_color, surface, ctx.impl->commands,
            ctx.impl->custom_samples ? ctx.impl->sample_positions.data() : nullptr);
        if (ctx.impl->raw_clip_snapshot)
            caster->restore_clipped_multisample(ctx.impl->raw_clip_snapshot,
                ctx.impl->raw_render_color, surface, ctx.impl->commands,
                ctx.impl->custom_samples ? ctx.impl->sample_positions.data() : nullptr);
        ctx.impl->color_clip_restore_pending = false;
        return;
    }
    const uint32_t width = uint32_t(ctx.impl->render_color.width);
    const uint32_t height = uint32_t(ctx.impl->render_color.height);
    const double x_scale = double(ctx.impl->sample_scale)
        / (ctx.impl->expanded_color ? ctx.impl->samples / 2 : 1);
    const double y_scale = double(ctx.impl->sample_scale)
        / (ctx.impl->expanded_color ? 2 : 1);
    const auto edge = [](uint32_t coordinate, uint32_t limit, double scale) {
        return uint32_t(std::clamp(std::floor(double(coordinate) * scale), 0.0, double(limit)));
    };
    const uint32_t x0 = edge(surface.clip_x_min, width, x_scale);
    const uint32_t y0 = edge(surface.clip_y_min, height, y_scale);
    const uint32_t x1 = edge(uint32_t(surface.clip_x_max) + 1, width, x_scale);
    const uint32_t y1 = edge(uint32_t(surface.clip_y_max) + 1, height, y_scale);
    require(ctx.impl->commands != nil, "Metal: color clip restoration has no command buffer");
    auto blit = [ctx.impl->commands blitCommandEncoder];
    require(blit != nil, "Metal: cannot restore clipped color samples");
    const auto copy = [&](uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
        if (!w || !h) return;
        [blit copyFromTexture:ctx.impl->color_clip_snapshot sourceSlice:0 sourceLevel:0
            sourceOrigin:MTLOriginMake(x, y, 0) sourceSize:MTLSizeMake(w, h, 1)
            toTexture:ctx.impl->render_color destinationSlice:0 destinationLevel:0
            destinationOrigin:MTLOriginMake(x, y, 0)];
        if (ctx.impl->raw_clip_snapshot)
            [blit copyFromTexture:ctx.impl->raw_clip_snapshot sourceSlice:0 sourceLevel:0
                sourceOrigin:MTLOriginMake(x,y,0) sourceSize:MTLSizeMake(w,h,1)
                toTexture:ctx.impl->raw_render_color destinationSlice:0 destinationLevel:0
                destinationOrigin:MTLOriginMake(x,y,0)];
    };
    if (x1 <= x0 || y1 <= y0) copy(0, 0, width, height);
    else {
        copy(0, 0, width, y0);
        copy(0, y1, width, height - y1);
        copy(0, y0, x0, y1 - y0);
        copy(x1, y0, width - x1, y1 - y0);
    }
    [blit endEncoding];
    // The ordinary render pass resolved before the outside samples were
    // restored. Resolve them once more so guest publication sees the clip.
    if (ctx.impl->samples > 1 && !ctx.impl->expanded_color && ctx.impl->color) {
        auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = ctx.impl->render_color;
        pass.colorAttachments[0].resolveTexture = ctx.impl->color;
        pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
        auto encoder = [ctx.impl->commands renderCommandEncoderWithDescriptor:pass];
        require(encoder != nil, "Metal: cannot resolve clipped color samples");
        [encoder endEncoding];
    }
    ctx.impl->color_clip_restore_pending = false;
}
static bool needs_raw_color_resolve(const MetalContext &ctx) {
    return ctx.impl->commands && ctx.impl->raw_render_color && ctx.impl->raw_color
        && ctx.impl->color && ctx.impl->samples>1 && !ctx.impl->expanded_color;
}
static void resolve_raw_color_samples(MetalContext &ctx, SurfaceCaster &caster) {
    caster.resolve_raw_multisample(ctx.impl->raw_render_color,ctx.impl->color,
        ctx.impl->raw_color,ctx.impl->commands);
}
struct MetalState::Impl {
    static constexpr size_t STREAM_SNAPSHOT_RING_SIZE = 64u * 1024u * 1024u;
    static constexpr size_t STREAM_SNAPSHOT_MAX_BYTES = 16u * 1024u;
    struct StreamSnapshotHeader {
        uint64_t handle;
        uint32_t size;
        uint32_t reserved = 0;
    };
    std::mutex stream_snapshot_mutex;
    std::vector<uint8_t> stream_snapshot_ring;
    uint64_t stream_snapshot_cursor = 0;
    bool stream_snapshots_disabled = false;
    uint64_t frame_timestamp = 0;
    struct PipelineResult {
        id<MTLRenderPipelineState> pipeline = nil;
        std::string error;
    };
    std::atomic_bool async_compilation = false;
    std::map<std::string, std::future<PipelineResult>> compiling_pipelines;
    std::set<std::string> failed_shaders, failed_pipelines;
    // Cached variants are speculative. A warmup failure must not suppress a
    // later rebuild from the actual draw's GXP and native descriptor.
    std::set<std::string> warmup_failed_shaders, warmup_failed_pipelines;
    bool cache_enabled = false;
    bool cache_reported = false;
    bool sync_draws = std::getenv("VITA3K_METAL_SYNC_DRAWS") != nullptr;
    bool trace_batches = std::getenv("VITA3K_METAL_TRACE_BATCHES") != nullptr;
    std::filesystem::path trace_batches_trigger = std::getenv("VITA3K_METAL_TRACE_BATCH_TRIGGER")
        ? std::getenv("VITA3K_METAL_TRACE_BATCH_TRIGGER") : "";
    uint32_t traced_batches = 0;
    uint32_t traced_scene_ends = 0;
    uint32_t traced_draw_flushes = 0;
    uint32_t traced_mid_scene_flushes = 0;
    uint32_t traced_display_fallbacks = 0;
    uint32_t traced_transfers = 0;
    uint32_t traced_display_frames = 0;
    bool trace_display_pixel = std::getenv("VITA3K_METAL_TRACE_DISPLAY_PIXEL") != nullptr;
    uint32_t traced_display_pixels = 0;
    bool trace_textures = std::getenv("VITA3K_METAL_TRACE_TEXTURES") != nullptr;
    bool trace_finish_timing = std::getenv("VITA3K_METAL_TRACE_FINISH_TIMING") != nullptr;
    std::filesystem::path trace_depth_store_trigger = std::getenv("VITA3K_METAL_TRACE_DEPTH_STORE_TRIGGER")
        ? std::getenv("VITA3K_METAL_TRACE_DEPTH_STORE_TRIGGER") : "";
    uint32_t traced_depth_stores = 0;
    uint64_t timed_finishes = 0;
    uint64_t timed_submits = 0;
    uint64_t timed_pending_waits = 0;
    uint64_t timed_publications = 0;
    uint64_t timed_scene_ends = 0;
    uint64_t timed_deferred_scenes = 0;
    uint64_t timed_depth_blocked_scenes = 0;
    uint64_t timed_mapped_blocked_scenes = 0;
    uint64_t timed_mapped_draws = 0;
    uint64_t timed_mapped_bytes = 0;
    uint64_t timed_max_mapped_extent = 0;
    double submit_wait_ms = 0;
    double pending_wait_ms = 0;
    double publication_ms = 0;
    std::set<std::string> traced_textures;
    bool trace_draws = std::getenv("VITA3K_METAL_TRACE_DRAWS") != nullptr;
    std::set<std::string> traced_draws;
    uint32_t trace_target_width = std::getenv("VITA3K_METAL_TRACE_TARGET_WIDTH")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_TRACE_TARGET_WIDTH"), nullptr, 10)) : 0;
    uint32_t traced_target_draws = 0;
    std::string trace_pixel = std::getenv("VITA3K_METAL_TRACE_PIXEL") ? std::getenv("VITA3K_METAL_TRACE_PIXEL") : "";
    std::string trace_pixel_arm_shader = std::getenv("VITA3K_METAL_TRACE_PIXEL_ARM_SHADER")
        ? std::getenv("VITA3K_METAL_TRACE_PIXEL_ARM_SHADER") : "";
    bool trace_pixel_armed = trace_pixel_arm_shader.empty();
    uint32_t trace_pixel_target_address = std::getenv("VITA3K_METAL_TRACE_PIXEL_TARGET_ADDRESS")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_TRACE_PIXEL_TARGET_ADDRESS"), nullptr, 0)) : 0;
    uint32_t trace_pixel_target_width = std::getenv("VITA3K_METAL_TRACE_PIXEL_TARGET_WIDTH")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_TRACE_PIXEL_TARGET_WIDTH"), nullptr, 10)) : 0;
    std::filesystem::path trace_pixel_trigger = std::getenv("VITA3K_METAL_TRACE_PIXEL_TRIGGER")
        ? std::getenv("VITA3K_METAL_TRACE_PIXEL_TRIGGER") : "";
    uint32_t traced_pixel_draws = 0;
    float last_traced_pixel[4]{};
    bool has_traced_pixel = false;
    std::filesystem::path dump_pipeline_dir = std::getenv("VITA3K_METAL_DUMP_PIPELINE_DIR") ? std::getenv("VITA3K_METAL_DUMP_PIPELINE_DIR") : "";
    uint32_t dumped_pipelines = 0;
    std::filesystem::path dump_surface_dir = std::getenv("VITA3K_METAL_DUMP_SURFACE_DIR") ? std::getenv("VITA3K_METAL_DUMP_SURFACE_DIR") : "";
    std::set<std::string> dumped_surfaces;
    std::filesystem::path dump_draw_dir = std::getenv("VITA3K_METAL_DUMP_DRAW_DIR") ? std::getenv("VITA3K_METAL_DUMP_DRAW_DIR") : "";
    std::string dump_draw_shader = std::getenv("VITA3K_METAL_DUMP_DRAW_SHADER") ? std::getenv("VITA3K_METAL_DUMP_DRAW_SHADER") : "";
    uint32_t dump_draw_target_address = std::getenv("VITA3K_METAL_DUMP_DRAW_TARGET_ADDRESS")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_TARGET_ADDRESS"), nullptr, 0)) : 0;
    uint32_t dump_draw_target_width = std::getenv("VITA3K_METAL_DUMP_DRAW_TARGET_WIDTH")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_TARGET_WIDTH"), nullptr, 10)) : 0;
    std::string dump_draw_texture_format = std::getenv("VITA3K_METAL_DUMP_DRAW_TEXTURE_FORMAT") ? std::getenv("VITA3K_METAL_DUMP_DRAW_TEXTURE_FORMAT") : "";
    uint32_t dump_draw_texture_address = std::getenv("VITA3K_METAL_DUMP_DRAW_TEXTURE_ADDRESS")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_TEXTURE_ADDRESS"), nullptr, 0)) : 0;
    uint32_t dump_draw_texture_width = std::getenv("VITA3K_METAL_DUMP_DRAW_TEXTURE_WIDTH")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_TEXTURE_WIDTH"), nullptr, 10)) : 0;
    std::string dump_arm_shader = std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_SHADER") ? std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_SHADER") : "";
    uint32_t dump_arm_target_width = std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_TARGET_WIDTH")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_TARGET_WIDTH"), nullptr, 10)) : 0;
    uint32_t dump_arm_shader_skip = std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_SKIP")
        ? uint32_t(std::min<unsigned long>(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_SKIP"), nullptr, 10), 1000000)) : 0;
    uint32_t dump_arm_shader_matches = 0;
    std::string dump_arm_pixel_shader = std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_PIXEL_SHADER")
        ? std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_PIXEL_SHADER") : "";
    float dump_arm_pixel_min_rgb = std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_PIXEL_MIN_RGB")
        ? std::strtof(std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_PIXEL_MIN_RGB"), nullptr) : 0.0f;
    std::string dump_arm_pixel_rgb = std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_PIXEL_RGB")
        ? std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_PIXEL_RGB") : "";
    bool dump_pixel_armed = false;
    bool dump_armed = dump_arm_shader.empty();
    bool dump_attachments = std::getenv("VITA3K_METAL_DUMP_DRAW_ATTACHMENTS") != nullptr;
    std::string dump_vertex_shader = std::getenv("VITA3K_METAL_DUMP_VERTEX_SHADER") ? std::getenv("VITA3K_METAL_DUMP_VERTEX_SHADER") : "";
    std::filesystem::path dump_trigger_file = std::getenv("VITA3K_METAL_DUMP_DRAW_TRIGGER_FILE")
        ? std::getenv("VITA3K_METAL_DUMP_DRAW_TRIGGER_FILE") : "";
    uint32_t dump_stream_stride = std::getenv("VITA3K_METAL_DUMP_DRAW_STREAM_STRIDE")
        ? uint32_t(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_STREAM_STRIDE"), nullptr, 10)) : 0;
    bool draw_dumped = false;
    uint32_t dump_draw_saved = 0;
    uint32_t dump_draw_limit = std::getenv("VITA3K_METAL_DUMP_DRAW_LIMIT")
        ? uint32_t(std::clamp<unsigned long>(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_LIMIT"), nullptr, 10), 1, 16)) : 1;
    uint32_t dump_draw_matches = 0;
    uint32_t dump_draw_skip = std::getenv("VITA3K_METAL_DUMP_DRAW_SKIP")
        ? uint32_t(std::min<unsigned long>(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_SKIP"), nullptr, 10), 1000000)) : 0;
    std::unique_ptr<Device> device;
    std::unique_ptr<OverlayRenderer> overlay;
    std::string gpu_name;
    std::map<Address, Surface> surfaces;
    SurfaceWriteTracker surface_writes;
    // Never reset on cache clear: an old context must not match a reused entry.
    uint64_t next_color_binding_generation = 1;
    MappedGuestRegions mapped_memory;
    DirectGuestBufferCache direct_guest_buffers;
    struct RenderedImage {
        id<MTLTexture> texture, uploaded;
        std::vector<id<MTLTexture>> sources;
    };
    std::map<std::string, RenderedImage> rendered_images;
    std::map<std::string, RenderedImage> rendered_cubes;
    struct CubeAlias {
        id<MTLTexture> source = nil;
        id<MTLTexture> cube = nil;
    };
    std::map<uintptr_t, CubeAlias> cube_aliases;
    id<MTLTexture> invalid_cube_fallback = nil;
    std::map<std::pair<Address, Address>, DepthSurface> depth_surfaces;
    std::map<std::string, std::unique_ptr<CompiledProgram>> shaders;
    std::set<std::pair<Sha256Hash, Sha256Hash>> known_shader_pairs;
    std::map<std::string, id<MTLRenderPipelineState>> pipelines;
    std::map<std::array<uint32_t, 16>, id<MTLDepthStencilState>> depth_states;
    id<MTLDepthStencilState> macroblock_clear_depth_state;
    CAMetalLayer *layer;
    std::unique_ptr<ScreenRenderer> screen;
    std::unique_ptr<SurfaceCaster> caster;
    id<CAMetalDrawable> drawable;
    id<MTLCommandBuffer> screen_commands;
    std::mutex screenshot_mutex;
    std::condition_variable screenshot_ready;
    bool screenshot_requested = false;
    std::vector<uint32_t> screenshot_frame;
    uint32_t screenshot_width = 0, screenshot_height = 0;
};
static void clear_macroblock_depth(MetalState &state, MetalContext &ctx, MTLScissorRect tile) {
    static constexpr const char *source = R"(#include <metal_stdlib>
using namespace metal;
vertex float4 macroblock_clear_vertex(uint index [[vertex_id]],
    constant float &clear_depth [[buffer(0)]]) {
    const float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[index], clear_depth, 1);
}
fragment void macroblock_clear_fragment() {}
)";
    auto &device = *state.impl->device;
    std::string error;
    auto &vertex = state.impl->shaders["metal-macroblock-clear-vertex"];
    if (!vertex) {
        vertex = device.compile({source, "macroblock_clear_vertex", shader::metal::Stage::Vertex}, false, error);
        require(bool(vertex), "Metal macroblock clear vertex shader: " + error);
    }
    auto &fragment = state.impl->shaders["metal-macroblock-clear-fragment"];
    if (!fragment) {
        fragment = device.compile({source, "macroblock_clear_fragment", shader::metal::Stage::Fragment}, false, error);
        require(bool(fragment), "Metal macroblock clear fragment shader: " + error);
    }
    const std::string pipeline_key = "metal-macroblock-depth-clear-"
        + std::to_string(uint32_t(ctx.impl->render_color.pixelFormat)) + "-"
        + std::to_string(ctx.impl->samples) + "-" + std::to_string(bool(ctx.impl->raw_attachment));
    auto &pipeline = state.impl->pipelines[pipeline_key];
    if (!pipeline) {
        auto desc = [MTLRenderPipelineDescriptor new];
        desc.vertexFunction = vertex->function;
        desc.fragmentFunction = fragment->function;
        desc.rasterSampleCount = ctx.impl->samples;
        desc.depthAttachmentPixelFormat = desc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        desc.colorAttachments[0].pixelFormat = ctx.impl->render_color.pixelFormat;
        desc.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
        configure_raw_attachment(desc,ctx.impl->raw_attachment,MTLColorWriteMaskNone);
        pipeline = device.create_pipeline(desc, error);
        require(pipeline != nil, "Metal macroblock clear pipeline: " + error);
    }
    if (!state.impl->macroblock_clear_depth_state) {
        auto desc = [MTLDepthStencilDescriptor new];
        desc.depthCompareFunction = MTLCompareFunctionAlways;
        desc.depthWriteEnabled = YES;
        auto stencil = [MTLStencilDescriptor new];
        stencil.stencilCompareFunction = MTLCompareFunctionAlways;
        stencil.stencilFailureOperation = MTLStencilOperationReplace;
        stencil.depthFailureOperation = MTLStencilOperationReplace;
        stencil.depthStencilPassOperation = MTLStencilOperationReplace;
        stencil.readMask = stencil.writeMask = 0xff;
        desc.frontFaceStencil = desc.backFaceStencil = stencil;
        state.impl->macroblock_clear_depth_state = [device.native_device() newDepthStencilStateWithDescriptor:desc];
        require(state.impl->macroblock_clear_depth_state != nil, "Metal macroblock clear depth state failed");
    }
    begin_pass(ctx, false);
    auto encoder = ctx.impl->encoder;
    [encoder setRenderPipelineState:pipeline];
    [encoder setDepthStencilState:state.impl->macroblock_clear_depth_state];
    [encoder setStencilReferenceValue:ctx.record.depth_stencil_surface.stencil];
    [encoder setScissorRect:tile];
    const float depth = ctx.record.depth_stencil_surface.background_depth;
    [encoder setVertexBytes:&depth length:sizeof(depth) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    ctx.impl->encoder = nil;
    ctx.impl->pass_visibility_active = false;
    ctx.impl->depth_written = true;
    ctx.impl->depth_scene_written = true;
}
struct DisplaySurfaceRegion {
    id<MTLTexture> color = nil;
    uint32_t line = 0, width = 0, height = 0, available_rows = 0, destination_line = 0;
};
static std::optional<DisplaySurfaceRegion> find_display_surface_region(
    const std::map<Address, Surface> &surfaces, const DisplayFrameInfo &frame,
    float scale, bool allow_legacy_pitch) {
    const auto upper = surfaces.upper_bound(frame.base.address());
    auto match = [&](auto it) -> std::optional<DisplaySurfaceRegion> {
        id<MTLTexture> color = it->second.color;
        if (!color || (!frame.pitch && !allow_legacy_pitch)) return std::nullopt;
        const int64_t delta = int64_t(frame.base.address()) - it->first;
        const uint32_t pitch = frame.pitch ? frame.pitch : it->second.guest.strideInPixels;
        const uint64_t stride_bytes = uint64_t(it->second.guest.strideInPixels)
            * gxm::bits_per_pixel(gxm::get_base_format(it->second.guest.colorFormat)) / 8;
        const uint64_t pitch_bytes = uint64_t(pitch) * 4;
        if (!pitch_bytes || (frame.pitch && stride_bytes != pitch_bytes)
            || delta % int64_t(pitch_bytes) || (delta && !frame.pitch)) return std::nullopt;
        const int64_t row_delta = delta / int64_t(pitch_bytes);
        const uint64_t scaled_line = uint64_t(std::max<int64_t>(row_delta, 0) * scale);
        const uint64_t destination_line = uint64_t(std::max<int64_t>(-row_delta, 0) * scale);
        const uint32_t width = frame.image_size.x > 0 ? uint32_t(frame.image_size.x * scale) : uint32_t(color.width);
        const uint32_t height = frame.image_size.y > 0 ? uint32_t(frame.image_size.y * scale) : uint32_t(color.height);
        if (!width || !height || width > color.width || height > 16384
            || scaled_line >= color.height || destination_line >= height)
            return std::nullopt;
        return DisplaySurfaceRegion{color, uint32_t(scaled_line), width, height,
            std::min(uint32_t(height - destination_line), uint32_t(color.height - scaled_line)), uint32_t(destination_line)};
    };
    if (upper != surfaces.begin()) {
        if (auto region = match(std::prev(upper))) return region;
    }
    // Some games render a shorter image inside the display allocation, for
    // example 540 rows starting two rows into a 544-row framebuffer.
    if (upper != surfaces.end()) return match(upper);
    return std::nullopt;
}
static std::vector<uint32_t> display_border_pixels(const DisplayFrameInfo &frame,
    const DisplaySurfaceRegion &region, MemState *mem, uint32_t first_row, uint32_t rows) {
    std::vector<uint32_t> pixels(size_t(region.width) * rows, 0xff000000u);
    if (!rows || !mem || !frame.base || frame.image_size.x <= 0 || frame.image_size.y <= 0
        || frame.pitch < uint32_t(frame.image_size.x)) return pixels;
    const uint64_t end = uint64_t(frame.base.address()) + uint64_t(frame.pitch) * frame.image_size.y * 4;
    if (end > uint64_t(UINT32_MAX) - 4095
        || !is_valid_addr_range(*mem, frame.base.address(), Address(end))) return pixels;
    const auto *guest = static_cast<const uint32_t *>(frame.base.get(*mem));
    for (uint32_t y = 0; y < rows; ++y) {
        const size_t guest_y = uint64_t(first_row + y) * frame.image_size.y / region.height;
        for (uint32_t x = 0; x < region.width; ++x)
            pixels[size_t(y) * region.width + x] = guest[guest_y * frame.pitch
                + uint64_t(x) * frame.image_size.x / region.width];
    }
    return pixels;
}
MetalContext::MetalContext() : impl(std::make_unique<Impl>()) {
    vertex_stream_snapshots.fill(~0ull);
}
void freeze_program(SceGxmVertexProgram &program, const MemState &mem) {
    auto binding = std::make_shared<ProgramBinding>();
    const auto *gxp = program.program.get(mem);
    const auto *bytes = reinterpret_cast<const uint8_t *>(gxp);
    binding->gxp.assign(bytes, bytes + gxp->size);
    auto metadata = std::make_shared<VertexProgram>(*program.renderer_data);
    metadata->metal_binding.reset();
    binding->vertex_program = std::move(metadata);
    binding->streams = program.streams;
    binding->attributes = program.attributes;
    program.renderer_data->metal_binding = std::move(binding);
}
void freeze_program(SceGxmFragmentProgram &program, const MemState &mem) {
    auto binding = std::make_shared<ProgramBinding>();
    const auto *gxp = program.program.get(mem);
    const auto *bytes = reinterpret_cast<const uint8_t *>(gxp);
    binding->gxp.assign(bytes, bytes + gxp->size);
    auto metadata = std::make_shared<MetalFragmentProgram>(static_cast<const MetalFragmentProgram &>(*program.renderer_data));
    metadata->metal_binding.reset();
    binding->fragment_program = std::move(metadata);
    binding->is_maskupdate = program.is_maskupdate;
    program.renderer_data->metal_binding = std::move(binding);
}
MetalContext::~MetalContext() = default;
MetalState::MetalState() : impl(std::make_unique<Impl>()), texture_cache(*this) {
    current_backend = Backend::Metal;
    context = nullptr; res_multiplier = 1; should_display = false; disable_surface_sync = false;
    stretch_the_display_area = false; fullscreen_hd_res_pixel_perfect = false;
    features.direct_fragcolor = true;
    features.support_unknown_format = true;
    features.support_scaled_attribute_formats = false;
    features.use_mask_bit = true;
    features.use_texture_viewport = true;
}
MetalState::~MetalState() {
    try { cleanup(); } catch (const std::exception &error) { LOG_ERROR("Metal shutdown: {}", error.what()); }
}
bool MetalState::init() {
    std::string error;
    impl->device = Device::create(error);
    if (!impl->device) { LOG_ERROR("{}", error); return false; }
    texture_cache.support_dxt = impl->device->native_device().supportsBCTextureCompression;
    auto astc_probe = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatASTC_4x4_LDR
        width:4 height:4 mipmapped:NO];
    astc_probe.usage = MTLTextureUsageShaderRead;
    texture_cache.support_astc = [impl->device->native_device() newTextureWithDescriptor:astc_probe] != nil;
    impl->gpu_name = impl->device->native_device().name.UTF8String;
    shader_version = "metal" + std::to_string(shader::metal::SHADER_ABI_VERSION);
    if (frame) {
        auto handle = frame->handle();
        auto *mac = std::get_if<MacOSDisplayHandle>(&handle);
        if (!mac || !mac->view) return false;
        NSView *view = (__bridge NSView *)mac->view;
        if (!view.layer) view.wantsLayer = YES;
        CALayer *root = view.layer;
        if (!root) { LOG_ERROR("Metal: frame host has no backing layer"); return false; }
        CAMetalLayer *metal_layer = [root isKindOfClass:[CAMetalLayer class]] ? (CAMetalLayer *)root : nil;
        if (!metal_layer) {
            // Qt owns its QContainerLayer, even when the window requests a
            // MetalSurface. Replacing that layer is ignored by Qt; keep the
            // drawable layer as a child and let Qt resize its parent.
            for (CALayer *child in root.sublayers)
                if ([child isKindOfClass:[CAMetalLayer class]] && [child.name isEqualToString:@"Vita3K Metal"])
                    metal_layer = (CAMetalLayer *)child;
            if (!metal_layer) {
                metal_layer = [CAMetalLayer layer];
                metal_layer.name = @"Vita3K Metal";
                metal_layer.frame = root.bounds;
                metal_layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
                [root addSublayer:metal_layer];
            }
        }
        impl->layer = metal_layer;
        impl->layer.device = impl->device->native_device();
        impl->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        impl->layer.framebufferOnly = YES;
        init_overlay_font_dirs();
    }
    LOG_INFO("Native Metal 3 renderer: {}", impl->gpu_name);
    return true;
}
void MetalState::set_app(const char *title_id, const char *self_name) {
    // Workers may still be using the previous game's cache and device.
    impl->compiling_pipelines.clear();
    impl->failed_shaders.clear();
    impl->failed_pipelines.clear();
    impl->warmup_failed_shaders.clear();
    impl->warmup_failed_pipelines.clear();
    State::set_app(title_id, self_name);
    impl->known_shader_pairs.clear();
    shaders_cache_hashs.clear();
    pipelines_count_precompiled = 0;
    impl->device->configure_cache(impl->cache_enabled && !cache_path.empty()
        ? std::filesystem::path(shaders_path.string()) / "metal" : std::filesystem::path{});
    impl->cache_reported = false;
    if (frame) impl->overlay = std::make_unique<OverlayRenderer>(*impl->device, std::filesystem::path(static_assets.string()));
    if (!impl->device->cache_directory().empty())
        LOG_INFO("Metal persistent cache: {}", impl->device->cache_directory().string());
}
// The native backend commits directly; it never reserves queue slots with
// enqueue(). Uncommitted buffers have no GPU users and may be discarded.
// Wait also on terminal errors so completion handlers have returned before
// owners of no-copy guest memory or uploads are released.
static bool retire_commands_for_shutdown(id<MTLCommandBuffer> commands) {
    if (!commands || commands.status == MTLCommandBufferStatusNotEnqueued
        || commands.status == MTLCommandBufferStatusEnqueued)
        return false;
    [commands waitUntilCompleted];
    return commands.status == MTLCommandBufferStatusError;
}

void MetalState::cleanup() {
    assert(!render_thread);
    reset_vertex_stream_snapshots();
    // std::async futures join here, before the cache or its device is released.
    impl->compiling_pipelines.clear();
    impl->failed_shaders.clear();
    impl->failed_pipelines.clear();
    impl->warmup_failed_shaders.clear();
    impl->warmup_failed_pipelines.clear();
    if (context) finish_for_shutdown(*static_cast<MetalContext *>(context));
    if (impl->screen_commands) {
        // A failed swap may leave an already submitted buffer here. Never
        // resubmit it, or submit an incomplete presentation during teardown.
        const bool failed = retire_commands_for_shutdown(impl->screen_commands);
        impl->screen_commands = nil;
        impl->drawable = nil;
        if (failed) LOG_ERROR("Metal shutdown: presentation command buffer failed");
    }
    if (impl->device && !impl->cache_reported) {
        impl->device->flush_cache();
        const auto stats = impl->device->cache_stats();
        if (!impl->device->cache_directory().empty())
            LOG_INFO("Metal cache: shader hits {}, misses {}, writes {}; pipeline hits {}, misses {}; archive writes {}, rejected {}, I/O errors {}",
                stats.program_hits, stats.program_misses, stats.program_writes, stats.pipeline_hits,
                stats.pipeline_misses, stats.archive_writes, stats.rejected_files, stats.io_errors);
        impl->cache_reported = true;
    }
    impl->rendered_images.clear();
    impl->rendered_cubes.clear();
    impl->direct_guest_buffers.clear();
    impl->surfaces.clear(); impl->depth_surfaces.clear(); impl->pipelines.clear(); impl->depth_states.clear(); impl->shaders.clear();
}
void MetalState::late_init(const Config &cfg, std::string_view game_id, MemState &) {
    impl->cache_enabled = cfg.shader_cache;
    features.use_texture_viewport = !cfg.current_config.high_accuracy;
    texture_cache.backend = Backend::Metal;
    texture_cache.TextureCache::init(cfg.hashless_texture_cache, texture_folder(), game_id);
}
bool MetalState::map_memory(MemState &mem, Ptr<void> address, uint32_t size) {
    if (!size) return true;
    const uint64_t end = uint64_t(address.address()) + size;
    if (mem.use_page_table || end > uint64_t(UINT32_MAX) - 4095
        || !is_valid_addr_range(mem, address.address(), Address(end))) {
        LOG_WARN_ONCE("Metal: cannot extend uniform addresses across this GXM memory mapping");
        return false;
    }
    impl->mapped_memory.map(address.address(), size);
    return true;
}
void MetalState::unmap_memory(MemState &, Ptr<void> address) {
    // A mapped uniform may still be referenced by the active command buffer
    // or by a previously submitted internal batch. The guest can release its
    // storage as soon as sceGxmUnmapMemory returns, so retire those accesses
    // before removing the mapping, as Vulkan's device.waitIdle() does.
    // Unmap is a guest-visible memory-lifetime boundary. Publishing the
    // active surface also waits for every pending internal batch; suppressing
    // that publication leaves completed color writes out of guest memory.
    if (context) finish(*static_cast<MetalContext *>(context));
    impl->direct_guest_buffers.clear();
    impl->mapped_memory.unmap(address.address());
}
uint32_t MetalState::get_features_mask() {
    // The shared shader-hash index uses this value to reject entries from a
    // different global shader configuration. Draw-specific hints remain in
    // the complete Metal variant keys and the native cache is device-scoped.
    uint32_t mask = 0;
    unsigned bit = 0;
    for (bool enabled : {features.support_shader_interlock, features.support_texture_barrier,
             features.direct_fragcolor, features.spirv_shader, features.support_get_texture_sub_image,
             features.preserve_f16_nan_as_u16, features.support_unknown_format, features.support_rgb_attributes,
             features.use_mask_bit, true /* native uniform addresses */, features.support_scaled_attribute_formats,
             features.use_texture_viewport})
        mask |= uint32_t(enabled) << bit++;
    return mask;
}
int MetalState::get_supported_filters() {
    return int(Filter::NEAREST) | int(Filter::BILINEAR) | int(Filter::BICUBIC)
        | int(Filter::FXAA) | int(Filter::FSR)
        | ([MTLFXSpatialScalerDescriptor supportsDevice:impl->device->native_device()] ? int(Filter::METALFX_SPATIAL) : 0);
}
void MetalState::new_frame() {
    ++impl->frame_timestamp;
}
void MetalState::set_screen_filter(const std::string_view &filter) {
    if (!impl->screen) impl->screen = std::make_unique<ScreenRenderer>(*impl->device, std::filesystem::path(static_assets.string()));
    impl->screen->set_filter(filter);
}
void MetalState::set_anisotropic_filtering(int value) { texture_cache.anisotropic_filtering = std::clamp(value, 1, 16); }
std::string_view MetalState::get_gpu_name() { return impl->gpu_name; }
void MetalState::precompile_shader(const ShadersHash &hash) {
    @autoreleasepool {
        try {
            impl->known_shader_pairs.emplace(hash.frag, hash.vert);
            const auto remember_warmup_failure = [](std::set<std::string> &failures,
                                                     const std::string &key, const char *kind, const char *error) {
                if (failures.insert(key).second)
                    LOG_WARN("Metal: cached {} warmup {:016x} failed; draw will rebuild it: {}",
                        kind, XXH3_64bits(key.data(), key.size()), error);
            };
            const auto warm_shader = [&](const std::string &key, shader::metal::Stage stage, bool gamma) {
                if (render_abort.load(std::memory_order_relaxed) || impl->shaders.contains(key)
                    || impl->warmup_failed_shaders.contains(key)) return;
                try {
                    auto program = impl->device->load_cached_program(key);
                    if (!program || program->stage != stage) {
                        impl->warmup_failed_shaders.insert(key);
                        return;
                    }
                    std::string error;
                    auto compiled = impl->device->compile(*program, gamma, error);
                    require(bool(compiled), error.empty() ? "compiler returned no shader" : error);
                    impl->shaders.emplace(key, std::move(compiled));
                    ++shaders_count_compiled;
                } catch (const std::exception &error) {
                    remember_warmup_failure(impl->warmup_failed_shaders, key, "shader", error.what());
                }
            };
            // A GXM hash alone cannot reconstruct a draw-specific Metal key.
            // Warm only complete variants recorded by an earlier draw.
            for (const auto &[guest_hash, stage] : {std::pair{hex_string(hash.vert), shader::metal::Stage::Vertex},
                     std::pair{hex_string(hash.frag), shader::metal::Stage::Fragment}}) {
                if (render_abort.load(std::memory_order_relaxed)) break;
                for (const auto &variant : impl->device->cached_variants(guest_hash)) {
                    if (render_abort.load(std::memory_order_relaxed)) break;
                    warm_shader(variant.key, stage, variant.gamma_correction);
                }
            }
            if (!render_abort.load(std::memory_order_relaxed)) {
                for (auto &saved : impl->device->cached_pipeline_templates(hex_string(hash.frag), hex_string(hash.vert))) {
                    if (render_abort.load(std::memory_order_relaxed)) break;
                    if (impl->warmup_failed_pipelines.contains(saved.key)) continue;
                    if (const auto found = impl->pipelines.find(saved.key);
                        found != impl->pipelines.end() && found->second) continue;
                    try {
                        if (saved.fragment_key.starts_with("metal-depth-only"))
                            warm_shader(saved.fragment_key, shader::metal::Stage::Fragment, false);
                        if (render_abort.load(std::memory_order_relaxed)) break;
                        const auto vertex = impl->shaders.find(saved.vertex_key);
                        const auto fragment = impl->shaders.find(saved.fragment_key);
                        if (vertex == impl->shaders.end() || fragment == impl->shaders.end()
                            || !vertex->second || !fragment->second
                            || vertex->second->stage != shader::metal::Stage::Vertex
                            || fragment->second->stage != shader::metal::Stage::Fragment
                            || std::string_view(vertex->second->function.name.UTF8String ?: "") != saved.vertex_function
                            || std::string_view(fragment->second->function.name.UTF8String ?: "") != saved.fragment_function) continue;
                        saved.descriptor.vertexFunction = vertex->second->function;
                        saved.descriptor.fragmentFunction = fragment->second->function;
                        std::string error;
                        auto pipeline = impl->device->create_pipeline(saved.descriptor, error);
                        require(pipeline != nil, error.empty() ? "compiler returned no pipeline" : error);
                        impl->pipelines[saved.key] = pipeline;
                        ++pipelines_count_precompiled;
                    } catch (const std::exception &error) {
                        remember_warmup_failure(impl->warmup_failed_pipelines, saved.key, "pipeline", error.what());
                    }
                }
            }
        } catch (const std::exception &error) {
            // Match Plus: a cache loading failure is not a render-thread failure.
            LOG_WARN("Metal: cached program pair warmup skipped: {}", error.what());
        }
        ++programs_count_pre_compiled;
    }
}
void MetalState::preclose_action() {
    // Called on the UI thread before stop_render_thread joins the GXM worker.
    // GPU completion is handled on the worker and needs no wake-up here.
    // cleanup() runs after the join; ending encoders here races active draws.
}
void MetalState::set_async_compilation(bool enable) {
    impl->async_compilation.store(enable, std::memory_order_relaxed);
}
void MetalState::snapshot_vertex_streams(MemState &mem, CommandList &list) {
    if (!list.first) return;
    Command *from = list.first;
    // Earlier flushes already captured their own vertex inputs. Repeated
    // kicks must not replace those bytes with the guest's current contents.
    for (Command *cmd = list.first; cmd; cmd = cmd == list.last ? nullptr : cmd->next)
        if (cmd->opcode == CommandOpcode::MidSceneFlush)
            from = cmd == list.last ? nullptr : cmd->next;

    const std::lock_guard lock(impl->stream_snapshot_mutex);
    if (impl->stream_snapshots_disabled) return;
    for (Command *cmd = from; cmd; cmd = cmd == list.last ? nullptr : cmd->next) {
        if (cmd->opcode != CommandOpcode::SetState) continue;
        CommandHelper helper(cmd);
        if (helper.pop<GXMState>() != GXMState::VertexStream) continue;
        const auto stream = helper.pop<Ptr<const uint8_t>>();
        helper.pop<size_t>();
        const auto size = helper.pop<size_t>();
        uint64_t handle = ~0ull;
        const uint64_t end = uint64_t(stream.address()) + size;
        if (stream && size && size <= Impl::STREAM_SNAPSHOT_MAX_BYTES
            && end <= uint64_t(UINT32_MAX) - 4095
            && is_valid_addr_range(mem, stream.address(), Address(end))) {
            auto &ring = impl->stream_snapshot_ring;
            if (ring.empty()) ring.resize(Impl::STREAM_SNAPSHOT_RING_SIZE);
            const size_t slot = (sizeof(Impl::StreamSnapshotHeader) + size + 63) & ~size_t(63);
            handle = impl->stream_snapshot_cursor;
            size_t offset = size_t(handle % ring.size());
            if (offset + slot > ring.size()) {
                handle += ring.size() - offset;
                offset = 0;
            }
            impl->stream_snapshot_cursor = handle + slot;
            const Impl::StreamSnapshotHeader header{handle, uint32_t(size)};
            std::memcpy(ring.data() + offset, &header, sizeof(header));
            auto *destination = ring.data() + offset + sizeof(header);
            if (!mem.use_page_table) {
                std::memcpy(destination, stream.get(mem), size);
            } else {
                // Ptr::get resolves only its first page. Resolve each guest
                // page separately when external mappings are not contiguous.
                for (size_t copied = 0; copied < size;) {
                    const Address address = Address(uint64_t(stream.address()) + copied);
                    const size_t chunk = std::min(size - copied, size_t(4096 - (address & 4095)));
                    std::memcpy(destination + copied, Ptr<const uint8_t>(address).get(mem), chunk);
                    copied += chunk;
                }
            }
        }
        require(helper.push(handle), "Metal: vertex snapshot handle exceeds command storage");
    }
}
std::vector<uint8_t> MetalState::vertex_stream_snapshot(uint64_t handle, size_t size) {
    if (handle == ~0ull || !size || size > Impl::STREAM_SNAPSHOT_MAX_BYTES) return {};
    const std::lock_guard lock(impl->stream_snapshot_mutex);
    const auto &ring = impl->stream_snapshot_ring;
    if (impl->stream_snapshots_disabled || ring.empty() || impl->stream_snapshot_cursor < handle
        || impl->stream_snapshot_cursor - handle > ring.size()) return {};
    const size_t offset = size_t(handle % ring.size());
    if (offset > ring.size() - sizeof(Impl::StreamSnapshotHeader)) return {};
    Impl::StreamSnapshotHeader header;
    std::memcpy(&header, ring.data() + offset, sizeof(header));
    if (header.handle != handle || size > header.size || size > ring.size() - offset - sizeof(header)) return {};
    const auto *bytes = ring.data() + offset + sizeof(header);
    // Own the returned bytes: a guest-thread kick may wrap the ring as soon
    // as this lock is released, before the render thread uploads the stream.
    return {bytes, bytes + size};
}
void MetalState::disable_vertex_stream_snapshots() {
    const std::lock_guard lock(impl->stream_snapshot_mutex);
    impl->stream_snapshots_disabled = true;
    std::vector<uint8_t>().swap(impl->stream_snapshot_ring);
}
void MetalState::reset_vertex_stream_snapshots() {
    const std::lock_guard lock(impl->stream_snapshot_mutex);
    impl->stream_snapshots_disabled = false;
    std::vector<uint8_t>().swap(impl->stream_snapshot_ring);
    // Keep handles monotonic so an old command cannot match a new slot after
    // GXM termination/reinitialization on the same MetalState.
}
void MetalState::finish_for_shutdown(MetalContext &ctx) {
    assert(!render_thread);
    const auto discard_remaining = [&]() {
        // Avoid end_pass here: its half-pixel restoration can allocate and
        // encode more work. Discard unfinished work after closing its encoder.
        if (ctx.impl->encoder) {
            [ctx.impl->encoder endEncoding];
            ctx.impl->encoder = nil;
        }
        size_t failed = retire_commands_for_shutdown(ctx.impl->commands);
        for (const auto &batch : ctx.impl->pending_batches)
            failed += retire_commands_for_shutdown(batch.commands);
        // No uploads or guest-memory bindings are released before all known
        // submitted buffers have reached a terminal state, including errors.
        ctx.impl->commands = nil;
        ctx.impl->pending_batches.clear();
        ctx.impl->uploads.reset_after_completion();
        ctx.impl->visibility_results.clear();
        ctx.impl->pass_visibility_buffer = nil;
        ctx.impl->half_pixel_strips = {};
        ctx.impl->color_clip_restore_pending = false;
        ctx.impl->color_guest_dirty = false;
        ctx.impl->pending_color_writes = false;
        ctx.impl->pending_draws = 0;
        ctx.impl->scene_active = false;
        if (ctx.impl->guest_write_watches_pending && ctx.impl->mem) {
            impl->surface_writes.complete_gpu_writes(*ctx.impl->mem);
            ctx.impl->guest_write_watches_pending = false;
        }
        return failed;
    };
    const bool submitted_current = ctx.impl->commands
        && ctx.impl->commands.status != MTLCommandBufferStatusNotEnqueued;
    const bool failed_pending = std::any_of(ctx.impl->pending_batches.begin(), ctx.impl->pending_batches.end(),
        [](const auto &batch) { return batch.commands.status == MTLCommandBufferStatusError; });
    if (submitted_current || failed_pending) {
        // finish restores clipped color before submission. An interrupted
        // already-submitted buffer must never receive more encoding work.
        const size_t failed = discard_remaining();
        LOG_ERROR("Metal shutdown: retired interrupted submitted work, {} GPU errors", failed);
        return;
    }
    try {
        finish(ctx);
    } catch (const std::exception &error) {
        const size_t failed = discard_remaining();
        LOG_ERROR("Metal shutdown: finish failed ({}); retired remaining buffers, {} GPU errors",
            error.what(), failed);
    }
}

bool MetalState::finish(MetalContext &ctx, bool publish_color, bool wait_without_publication) {
    bool published_color = false;
    wait_without_publication |= ctx.impl->guest_write_watches_pending;
    end_pass(ctx);
    if (ctx.impl->color_clip_samplewise && ctx.impl->color_clip_restore_pending && !impl->caster)
        impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
    restore_color_outside_clip(ctx, impl->caster.get());
    if (needs_raw_color_resolve(ctx)) {
        if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
        resolve_raw_color_samples(ctx,*impl->caster);
    }
    const auto wait_pending_batch = [&](bool recycle_uploads) {
        auto batch = std::move(ctx.impl->pending_batches.front());
        ctx.impl->pending_batches.pop_front();
        std::chrono::steady_clock::time_point wait_started{};
        if (impl->trace_finish_timing) wait_started = std::chrono::steady_clock::now();
        [batch.commands waitUntilCompleted];
        if (impl->trace_finish_timing) {
            impl->pending_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wait_started).count();
            ++impl->timed_pending_waits;
        }
        require(batch.commands.status == MTLCommandBufferStatusCompleted,
            batch.commands.error.localizedDescription.UTF8String ?: "Metal internal batch failed");
        if (recycle_uploads) {
            batch.uploads.reset_after_completion();
            ctx.impl->uploads = std::move(batch.uploads);
        }
    };
    if (ctx.impl->commands) {
        // Internal draw limits have no guest-visible result. Keep at most two
        // completed-or-running batches ahead of the CPU, retaining their upload
        // slices until the corresponding GPU command buffer has finished.
        const bool deferred = !publish_color && !wait_without_publication && !ctx.impl->expanded_color
            && ctx.impl->visibility_results.empty();
        if (deferred) {
            require(ctx.impl->commands.status == MTLCommandBufferStatusNotEnqueued,
                "Metal internal batch was already submitted");
            [ctx.impl->commands commit];
            ctx.impl->pending_batches.push_back({ctx.impl->commands, std::move(ctx.impl->uploads)});
            ctx.impl->uploads = UploadBufferArena{};
        } else {
            std::string error;
            std::chrono::steady_clock::time_point submit_started{};
            if (impl->trace_finish_timing) submit_started = std::chrono::steady_clock::now();
            require(impl->device->submit_and_wait(ctx.impl->commands, error), error);
            if (impl->trace_finish_timing) {
                impl->submit_wait_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - submit_started).count();
                ++impl->timed_submits;
            }
            ctx.impl->uploads.reset_after_completion();
            while (!ctx.impl->pending_batches.empty()) wait_pending_batch(false);
        }
        ctx.impl->commands = nil;
        if (deferred && ctx.impl->pending_batches.size() >= 3)
            wait_pending_batch(true);
        // Face routing can switch the Metal query between front and back
        // several times. SET retains visibility across draws while the guest
        // query state is unchanged, so combine its Boolean segments first.
        std::map<std::array<uint64_t, 3>, std::pair<bool, size_t>> grouped_set_results;
        for (size_t i = 0; i < ctx.impl->visibility_results.size(); ++i) {
            const auto &result = ctx.impl->visibility_results[i];
            if (!result.group || result.increment) continue;
            uint64_t samples = 0;
            std::memcpy(&samples, static_cast<const uint8_t *>(result.buffer.contents) + result.offset, sizeof(samples));
            const std::array<uint64_t, 3> key{result.group, uint64_t(result.address), result.index};
            auto &combined = grouped_set_results[key];
            if (const auto prior = ctx.impl->published_set_results.find(key);
                prior != ctx.impl->published_set_results.end())
                combined.first |= prior->second;
            combined.first |= samples != 0;
            combined.second = i;
        }
        if (ctx.impl->mem) for (size_t i = 0; i < ctx.impl->visibility_results.size(); ++i) {
            const auto &result = ctx.impl->visibility_results[i];
            const uint64_t guest_address = uint64_t(result.address) + uint64_t(result.index) * sizeof(uint32_t);
            if (guest_address + sizeof(uint32_t) > uint64_t(UINT32_MAX) - 4095
                || !is_valid_addr_range(*ctx.impl->mem, Address(guest_address), Address(guest_address + sizeof(uint32_t)))) {
                LOG_WARN_ONCE("Metal: visibility result points outside guest memory");
                continue;
            }
            uint64_t samples = 0;
            std::memcpy(&samples, static_cast<const uint8_t *>(result.buffer.contents) + result.offset, sizeof(samples));
            if (result.group && !result.increment) {
                const std::array<uint64_t, 3> key{result.group, uint64_t(result.address), result.index};
                const auto &combined = grouped_set_results.at(key);
                if (combined.second != i) continue;
                samples = combined.first;
                ctx.impl->published_set_results[key] = combined.first;
            }
            auto *guest = Ptr<uint32_t>(Address(guest_address)).get(*ctx.impl->mem);
            *guest = result.increment ? *guest + uint32_t(samples) : uint32_t(samples != 0);
        }
        ctx.impl->visibility_results.clear();
        ctx.impl->pass_visibility_buffer = nil;
        if (ctx.impl->samples > 1 && ctx.impl->expanded_color && ctx.impl->color) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            expand_color_samples(ctx,*impl->caster);
        }
        const bool drew_color=ctx.impl->pending_color_writes;
        if (drew_color) {
            ctx.impl->color_guest_current = false;
            ctx.impl->color_guest_dirty = true;
        }
        if(impl->trace_batches && ctx.impl->pending_draws
            && (impl->trace_batches_trigger.empty() || std::filesystem::exists(impl->trace_batches_trigger))
            && impl->traced_batches++<64)
            LOG_INFO("Metal batch completed: draws={} input_bytes={} uniform_bytes={} stream_bytes={} index_bytes={}",
                ctx.impl->pending_draws,ctx.impl->pending_upload_bytes,ctx.impl->pending_uniform_bytes,
                ctx.impl->pending_stream_bytes,ctx.impl->pending_index_bytes);
        ctx.impl->pending_draws = 0;
        ctx.impl->pending_color_writes = false;
        ctx.impl->pending_upload_bytes = 0;
        ctx.impl->pending_uniform_bytes = 0;
        ctx.impl->pending_stream_bytes = 0;
        ctx.impl->pending_index_bytes = 0;
        ctx.impl->depth_scene_written |= ctx.impl->depth_written;
        if (ctx.impl->depth_written) {
            auto found = impl->depth_surfaces.find(ctx.impl->depth_key);
            if (found != impl->depth_surfaces.end() && found->second.texture==ctx.impl->depth) {
                found->second.snapshots.clear();
                found->second.subrectangles.clear();
            }
            ctx.impl->depth_written = false;
        }
    }
    if (publish_color || wait_without_publication)
        while (!ctx.impl->pending_batches.empty()) wait_pending_batch(false);
    // All guest-buffer GPU users are now retired. Restore watches before
    // publication, which can itself throw or perform guarded guest writes.
    if (ctx.impl->guest_write_watches_pending && ctx.impl->mem) {
        impl->surface_writes.complete_gpu_writes(*ctx.impl->mem);
        ctx.impl->guest_write_watches_pending = false;
    }
    // Batch-size flushes only retain upload resources; no guest-visible
    // notification has been signaled. Keep their color writes on the GPU and
    // publish the completed surface at the next CPU-visible boundary.
    const auto &surface = ctx.impl->guest_color;
    if (publish_color && surface.data && ctx.impl->color_binding_generation) {
        const auto cached = impl->surfaces.find(surface.data.address());
        if (cached == impl->surfaces.end() || !owns_color_binding(cached->second, ctx)) {
            // Plus drops post-sync requests when their cache generation was
            // destroyed. Native contexts can likewise outlive a retired or
            // rebound cache entry; their retained texture is not proof that
            // they still own these guest bytes. Never use a full-copy fallback.
            ctx.impl->color_guest_dirty = false;
            ctx.impl->color_guest_current = false;
            ctx.impl->color_publication_discarded = true;
            LOG_WARN_ONCE("Metal: discarding color publication from a retired surface binding");
        }
    }
    if (publish_color && ctx.impl->color_guest_dirty && !disable_surface_sync
        && ctx.impl->mem && surface.data && ctx.impl->color) {
        const size_t bytes=surface_memory_size(surface);
        const uint64_t end=uint64_t(surface.data.address())+bytes;
        if (bytes && end<=uint64_t(UINT32_MAX)-4095
            && is_valid_addr_range(*ctx.impl->mem,surface.data.address(),Address(end))) {
            const std::span<uint8_t> output{static_cast<uint8_t *>(surface.data.get(*ctx.impl->mem)),bytes};
            std::chrono::steady_clock::time_point publication_started{};
            if (impl->trace_finish_timing) publication_started = std::chrono::steady_clock::now();
            auto publication = SurfacePublication::Unavailable;
            std::vector<SurfaceMemoryRange> written;
            const auto cached = impl->surfaces.find(surface.data.address());
            if (cached != impl->surfaces.end() && owns_color_binding(cached->second, ctx)) {
                publication = impl->surface_writes.writeback(*ctx.impl->mem, [&] {
                    return read_surface_publication(cached->second, surface, output, written, *impl->device, impl->caster);
                });
            }
            if (impl->trace_finish_timing) {
                impl->publication_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - publication_started).count();
                ++impl->timed_publications;
            }
            if (publication == SurfacePublication::CpuNewer) {
                // This synchronization is handled by retaining the newer CPU
                // contents. Do not let EndScene's explicit-sync fallback or a
                // later finish publish the same obsolete image over them.
                published_color = true;
                ctx.impl->color_guest_dirty = false;
                ctx.impl->color_guest_current = false;
                ctx.impl->color_publication_discarded = true;
            } else if (publication == SurfacePublication::Unavailable)
                LOG_WARN_ONCE("Metal: automatic color surface sync unavailable for format={:#x}",
                    uint32_t(surface.colorFormat));
            else {
                published_color = true;
                ctx.impl->color_guest_current = true;
                ctx.impl->color_guest_dirty = false;
                update_cpu_snapshot(cached->second,output,written);
            }
        }
    }
    if (impl->trace_finish_timing && ++impl->timed_finishes == 256) {
        LOG_INFO("Metal finish timing: submits={} wait_ms={:.2f} pending_waits={} pending_ms={:.2f} publications={} copy_ms={:.2f}",
            impl->timed_submits, impl->submit_wait_ms,
            impl->timed_pending_waits, impl->pending_wait_ms,
            impl->timed_publications, impl->publication_ms);
        LOG_INFO("Metal scene timing: ends={} deferred={} depth_blocked={} mapped_blocked={} mapped_draws={} mapped_bytes={} max_mapped_extent={}",
            impl->timed_scene_ends, impl->timed_deferred_scenes,
            impl->timed_depth_blocked_scenes, impl->timed_mapped_blocked_scenes,
            impl->timed_mapped_draws, impl->timed_mapped_bytes,
            impl->timed_max_mapped_extent);
        LOG_INFO("Metal resource usage: allocated_bytes={} recommended_bytes={} direct_buffers={} color_surfaces={} depth_surfaces={} pipelines={} compiling={}",
            uint64_t(impl->device->native_device().currentAllocatedSize),
            uint64_t(impl->device->native_device().recommendedMaxWorkingSetSize),
            impl->direct_guest_buffers.size(), impl->surfaces.size(), impl->depth_surfaces.size(),
            impl->pipelines.size(), impl->compiling_pipelines.size());
        impl->timed_finishes = impl->timed_submits = impl->timed_pending_waits = impl->timed_publications = 0;
        impl->timed_scene_ends = impl->timed_deferred_scenes = impl->timed_depth_blocked_scenes = 0;
        impl->timed_mapped_blocked_scenes = impl->timed_mapped_draws = 0;
        impl->timed_mapped_bytes = impl->timed_max_mapped_extent = 0;
        impl->submit_wait_ms = impl->pending_wait_ms = impl->publication_ms = 0;
    }
    return published_color;
}
uint64_t MetalState::count_nonzero_color_samples(MetalContext &ctx) {
    finish(ctx);
    id<MTLTexture> source = ctx.impl->render_color;
    require(source && source.textureType == MTLTextureType2DMultisample
        && source.sampleCount >= 2 && source.sampleCount <= 4
        && (source.pixelFormat == MTLPixelFormatRGBA8Unorm
            || source.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB),
        "Metal: native MSAA sample diagnostic requires an RGBA8 color target");
    const uint32_t width = uint32_t(source.width * (source.sampleCount / 2));
    const uint32_t height = uint32_t(source.height * 2);
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
        width:width height:height mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    id<MTLTexture> packed = [impl->device->native_device() newTextureWithDescriptor:desc];
    require(packed != nil, "Metal: cannot allocate native MSAA sample diagnostic");
    if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
    impl->caster->expand_multisample(source, packed, 1.0f, width, height);
    std::vector<uint8_t> rgba(size_t(width) * height * 4);
    [packed getBytes:rgba.data() bytesPerRow:size_t(width) * 4
        fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0];
    uint64_t covered = 0;
    for (size_t pixel = 0; pixel < size_t(width) * height; ++pixel)
        covered += rgba[pixel * 4 + 3] != 0;
    return covered;
}
void MetalState::mid_scene_flush(MetalContext &ctx, bool wait_for_completion) {
    if (impl->trace_batches
        && (impl->trace_batches_trigger.empty() || std::filesystem::exists(impl->trace_batches_trigger))
        && impl->traced_mid_scene_flushes++ < 64)
        LOG_INFO("Metal mid-scene flush: wait={} pending_draws={} input_bytes={} commands={}",
            wait_for_completion, ctx.impl->pending_draws, ctx.impl->pending_upload_bytes, bool(ctx.impl->commands));
    if (wait_for_completion) {
        finish(ctx);
        return;
    }
    if (ctx.impl->encoder) {
        // End the pass to order its tracked resources before subsequent
        // encoders. Guest buffers are declared with useResource as well.
        // Apple GPUs cannot encode a within-pass barrier after Fragment;
        // no such barrier is needed across this encoder boundary.
        end_pass(ctx);
        ctx.impl->pass_visibility_active = false;
        if (ctx.impl->color_clip_samplewise && ctx.impl->color_clip_restore_pending && !impl->caster)
            impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
        restore_color_outside_clip(ctx, impl->caster.get());
        if (needs_raw_color_resolve(ctx)) {
            if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
            resolve_raw_color_samples(ctx,*impl->caster);
        }
    }
}
static std::pair<std::span<uint8_t>,std::span<uint8_t>> depth_memory_spans(MemState &mem,
    const SceGxmDepthStencilSurface &surface,const DepthMemoryLayout &layout) {
    auto span=[&](Ptr<void> pointer,size_t bytes) -> std::span<uint8_t> {
        if(!bytes) return {};
        const uint64_t end=uint64_t(pointer.address())+bytes;
        if (!pointer || end>uint64_t(UINT32_MAX)-4095 || !is_valid_addr_range(mem,pointer.address(),Address(end)))
            throw std::runtime_error(fmt::format(
                "Metal: depth/stencil allocation extends beyond mapped guest memory: address={:#x} bytes={} size={}x{} stride={} format={:#x} type={:#x}",
                pointer.address(),bytes,layout.width,layout.height,layout.stride,
                uint32_t(surface.get_format()),uint32_t(surface.get_type())));
        return {static_cast<uint8_t *>(pointer.get(mem)),bytes};
    };
    return {span(surface.depth_data,layout.depth_size),span(surface.stencil_data,layout.stencil_size)};
}
static std::vector<SurfaceMemoryRange> depth_scene_ranges(const MetalContext &ctx,bool stencil) {
    const auto &layout=*ctx.impl->depth_layout;
    require(ctx.impl->depth_scene_width<=layout.width && ctx.impl->depth_scene_height<=layout.height,
        "Metal: depth scene exceeds retained storage");
    const size_t size=stencil ? layout.stencil_size : layout.depth_size;
    if (!size) return {};
    if (ctx.impl->depth_scene_width==layout.width && ctx.impl->depth_scene_height==layout.height) return {{0,size}};
    const size_t bytes=stencil ? (layout.packed ? 4 : 1) : layout.depth_bytes;
    std::vector<SurfaceMemoryRange> result;
    for (uint32_t y=0;y<ctx.impl->depth_scene_height;++y) {
        if (!layout.tiled) {
            result.push_back({size_t(y)*layout.stride*bytes,size_t(ctx.impl->depth_scene_width)*bytes});
            continue;
        }
        for (uint32_t x=0;x<ctx.impl->depth_scene_width;) {
            const uint32_t count=std::min(32u,ctx.impl->depth_scene_width-x);
            const size_t offset=((size_t(y/32)*(layout.stride/32)+x/32)*1024+(y%32)*32)*bytes;
            result.push_back({offset,size_t(count)*bytes});x+=count;
        }
    }
    return result;
}
bool MetalState::end_scene(MetalContext &ctx, bool allow_deferred) {
    @autoreleasepool {
    if (impl->trace_batches && ctx.impl->scene_active
        && (impl->trace_batches_trigger.empty() || std::filesystem::exists(impl->trace_batches_trigger))
        && impl->traced_scene_ends++ < 64)
        LOG_INFO("Metal scene end: pending_draws={} input_bytes={} commands={} depth_written={} depth_scene_written={} force_store={}",
            ctx.impl->pending_draws, ctx.impl->pending_upload_bytes, bool(ctx.impl->commands),
            ctx.impl->depth_written, ctx.impl->depth_scene_written, bool(ctx.impl->guest_depth.force_store));
    DepthStoreReadback pending_depth_store;
    // A written depth surface is always published at scene end. Encode its
    // readback behind the draws in their command buffer, then consume the
    // shared buffer after finish() completes that same submission.
    if (ctx.impl->scene_active && ctx.impl->guest_depth.force_store && ctx.impl->depth_layout
        && ctx.impl->mem && ctx.impl->commands
        && (ctx.impl->depth_scene_written || ctx.impl->depth_written)) {
        const auto &layout=*ctx.impl->depth_layout;
        const auto [depth,stencil]=depth_memory_spans(*ctx.impl->mem,ctx.impl->guest_depth,layout);
        if(!depth.empty() || !stencil.empty()) {
            end_pass(ctx);
            if(!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->enqueue_depth_store(ctx.impl->depth,ctx.impl->guest_depth,layout,
                ctx.impl->sample_scale,depth.size(),stencil.size(),ctx.impl->mask,
                ctx.impl->commands,pending_depth_store),
                "Metal: cannot enqueue guest depth/stencil storage");
        }
    }
    auto cached_depth=impl->depth_surfaces.find(ctx.impl->depth_key);
    const auto published_depth_matches_guest=[&] {
        if (!ctx.impl->scene_active || !ctx.impl->guest_depth.force_store || !ctx.impl->depth_layout
            || !ctx.impl->mem || ctx.impl->depth_scene_written || ctx.impl->depth_written
            || cached_depth==impl->depth_surfaces.end() || cached_depth->second.texture!=ctx.impl->depth
            || !cached_depth->second.published) return false;
        const auto &layout=*ctx.impl->depth_layout;
        const auto [depth,stencil]=depth_memory_spans(*ctx.impl->mem,ctx.impl->guest_depth,layout);
        const auto &entry=cached_depth->second;
        return (!depth.empty() || !stencil.empty())
            && entry.guest_depth_snapshot.size()==depth.size()
            && entry.guest_stencil_snapshot.size()==stencil.size()
            && std::equal(entry.guest_depth_snapshot.begin(),entry.guest_depth_snapshot.end(),depth.begin())
            && std::equal(entry.guest_stencil_snapshot.begin(),entry.guest_stencil_snapshot.end(),stencil.begin());
    };
    bool reused_published_depth=published_depth_matches_guest();
    // With guest color synchronization disabled, EndScene has no CPU-visible
    // color result. A force_store depth attachment also needs no new guest
    // publication when its already published native image was not written.
    // Keep other depth and mapped-buffer results synchronous.
    const bool defer_scene = allow_deferred && ctx.impl->scene_active
        && (!ctx.impl->guest_depth.force_store || reused_published_depth)
        && !ctx.impl->direct_guest_memory_used;
    const bool submits_deferred = defer_scene && ctx.impl->commands
        && !ctx.impl->expanded_color && ctx.impl->visibility_results.empty();
    if (impl->trace_finish_timing && ctx.impl->scene_active) {
        ++impl->timed_scene_ends;
        impl->timed_deferred_scenes += submits_deferred;
        impl->timed_depth_blocked_scenes += allow_deferred && ctx.impl->guest_depth.force_store
            && !reused_published_depth;
        impl->timed_mapped_blocked_scenes += allow_deferred && ctx.impl->direct_guest_memory_used;
    }
    const bool published_color = finish(ctx, !defer_scene);
    // A synchronous completion can make a direct GXP store to the guest
    // depth range visible after the first comparison.
    if (reused_published_depth) reused_published_depth=published_depth_matches_guest();
    if (submits_deferred) ++deferred_scene_submits;
    if(!ctx.impl->scene_active) return ctx.impl->color_publication_discarded;
    bool stored_depth = false;
    if(ctx.impl->guest_depth.force_store && ctx.impl->depth_layout && ctx.impl->mem) {
        const auto &layout=*ctx.impl->depth_layout;
        const auto [depth,stencil]=depth_memory_spans(*ctx.impl->mem,ctx.impl->guest_depth,layout);
        if(!depth.empty() || !stencil.empty()) {
            if (!reused_published_depth) {
                if(!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                const bool partial=ctx.impl->depth_scene_width!=layout.width || ctx.impl->depth_scene_height!=layout.height;
                std::vector<uint8_t> depth_copy,stencil_copy;
                std::span<uint8_t> depth_output=depth,stencil_output=stencil;
                if (partial) {
                    depth_copy.assign(depth.begin(),depth.end());depth_output=depth_copy;
                    if (layout.packed && layout.stencil_size) stencil_output=depth_output;
                    else { stencil_copy.assign(stencil.begin(),stencil.end());stencil_output=stencil_copy; }
                }
                if (pending_depth_store.depth) {
                    impl->caster->finish_depth_store(pending_depth_store,ctx.impl->guest_depth,layout,depth_output,stencil_output);
                    ++inline_depth_stores;
                } else
                    require(impl->caster->store_depth_memory(ctx.impl->depth,ctx.impl->guest_depth,layout,
                        ctx.impl->sample_scale,depth_output,stencil_output,ctx.impl->mask),
                        "Metal: cannot publish guest depth/stencil storage");
                if (partial) {
                    for (const auto &range:depth_scene_ranges(ctx,false))
                        std::memcpy(depth.data()+range.offset,depth_output.data()+range.offset,range.size);
                    for (const auto &range:depth_scene_ranges(ctx,true))
                        std::memcpy(stencil.data()+range.offset,stencil_output.data()+range.offset,range.size);
                }
                stored_depth = true;
                if (!impl->trace_depth_store_trigger.empty()
                    && std::filesystem::exists(impl->trace_depth_store_trigger)
                    && impl->traced_depth_stores++ < 128)
                    LOG_INFO("Metal depth publication: format={} size={}x{} stride={} tiled={} packed={} depth_bytes={} depth_size={} stencil_size={} samples={} inline={} direct_packed_eligible={}",
                        uint32_t(ctx.impl->guest_depth.get_format()), layout.width, layout.height,
                        layout.stride, layout.tiled, layout.packed, layout.depth_bytes,
                        layout.depth_size, layout.stencil_size, ctx.impl->depth.sampleCount,
                        bool(pending_depth_store.depth), layout.packed && !layout.tiled
                            && layout.stride == layout.width && layout.depth_bytes == 4
                            && layout.depth_size && !layout.stencil_size
                            && ctx.impl->guest_depth.get_format() == SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24);
            }
            if (cached_depth!=impl->depth_surfaces.end() && cached_depth->second.texture==ctx.impl->depth) {
                auto &entry=cached_depth->second;
                if (stored_depth) {
                    const bool complete=ctx.impl->depth_scene_width==layout.width && ctx.impl->depth_scene_height==layout.height;
                    if (complete) {
                        entry.guest_depth_snapshot.assign(depth.begin(),depth.end());
                        entry.guest_stencil_snapshot.assign(stencil.begin(),stencil.end());
                    } else {
                        require(entry.guest_depth_snapshot.size()==depth.size() && entry.guest_stencil_snapshot.size()==stencil.size(),
                            "Metal: partial depth publication lost its storage baseline");
                        for (const auto &range:depth_scene_ranges(ctx,false))
                            std::memcpy(entry.guest_depth_snapshot.data()+range.offset,depth.data()+range.offset,range.size);
                        for (const auto &range:depth_scene_ranges(ctx,true))
                            std::memcpy(entry.guest_stencil_snapshot.data()+range.offset,stencil.data()+range.offset,range.size);
                    }
                    entry.snapshot_background_depth=ctx.impl->guest_depth.background_depth;
                    entry.snapshot_background_stencil=ctx.impl->guest_depth.stencil;
                    entry.guest_snapshot_valid=true;
                    entry.published=complete;
                }
            }
        }
    }
    if (cached_depth!=impl->depth_surfaces.end() && cached_depth->second.texture==ctx.impl->depth
        && !stored_depth && !reused_published_depth)
        cached_depth->second.published=false;
    if (cached_depth!=impl->depth_surfaces.end()
        && (cached_depth->second.texture==ctx.impl->depth || ctx.impl->depth_is_sample_rate_copy)
        && ctx.impl->scene_depth_drawn)
        cached_depth->second.depth_content_stored=ctx.impl->guest_depth.force_store;
    ctx.impl->scene_active=false;
    // A dropped obsolete publication is handled. Returning false here would
    // make the automatic scene-end caller sync a newer cache binding or retry
    // a GPU image whose publication was suppressed by newer CPU contents.
    if (ctx.impl->color_publication_discarded) return true;
    const auto &published = ctx.impl->guest_color;
    const auto &recorded = ctx.record.color_surface;
    // A previous mid-scene finish can have published the last color draw.
    // Accept it only while guest bytes still match that publication. CPU writes
    // between the flush and scene end retain the explicit sync behavior.
    bool earlier_color_publication = false;
    if (!published_color && ctx.impl->color_guest_current && ctx.impl->mem && published.data) {
        const auto found = impl->surfaces.find(published.data.address());
        const size_t bytes = surface_memory_size(published);
        const uint64_t end = uint64_t(published.data.address()) + bytes;
        if (found != impl->surfaces.end() && owns_color_binding(found->second, ctx)
            && bytes && found->second.cpu_snapshot.size() == bytes
            && end <= uint64_t(UINT32_MAX) - 4095
            && is_valid_addr_range(*ctx.impl->mem, published.data.address(), Address(end))) {
            const auto *guest = static_cast<const uint8_t *>(published.data.get(*ctx.impl->mem));
            earlier_color_publication = !surface_cpu_bytes_changed(found->second, {guest, bytes});
        }
    }
    // A depth/stencil store after finish can touch aliased color guest bytes.
    // Preserve the explicit color sync for that uncommon overlap.
    bool depth_overlaps_color = false;
    if (stored_depth && ctx.impl->depth_layout && published.data) {
        const uint64_t color_begin = published.data.address();
        const uint64_t color_end = color_begin + surface_memory_size(published);
        const auto &layout = *ctx.impl->depth_layout;
        const auto overlaps = [color_begin, color_end](Ptr<void> address, size_t bytes) {
            const uint64_t begin = address.address();
            return address && bytes && begin < color_end && begin + bytes > color_begin;
        };
        depth_overlaps_color = overlaps(ctx.impl->guest_depth.depth_data, layout.depth_size)
            || overlaps(ctx.impl->guest_depth.stencil_data, layout.stencil_size);
    }
    return (published_color || earlier_color_publication) && !depth_overlaps_color && published.data == recorded.data
        && published.width == recorded.width && published.height == recorded.height
        && published.strideInPixels == recorded.strideInPixels
        && published.colorFormat == recorded.colorFormat
        && published.surfaceType == recorded.surfaceType
        && published.disabled == recorded.disabled && published.downscale == recorded.downscale
        && published.gamma == recorded.gamma;
    }
}
void MetalState::set_visibility_buffer(MetalContext &ctx, Ptr<uint32_t> buffer, uint32_t stride) {
    if (ctx.impl->visibility_address != buffer.address() || ctx.impl->visibility_stride != stride)
        ++ctx.impl->visibility_epoch;
    if (ctx.impl->encoder && ctx.impl->pass_visibility_active
        && (ctx.impl->visibility_address != buffer.address() || ctx.impl->visibility_stride != stride)) {
        [ctx.impl->encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        ctx.impl->pass_visibility_active = false;
    }
    ctx.impl->visibility_address = buffer.address();
    ctx.impl->visibility_stride = stride;
}
void MetalState::set_visibility_index(MetalContext &ctx, bool enable, uint32_t index, bool increment) {
    if (ctx.impl->visibility_enabled != enable || ctx.impl->visibility_index != index
        || ctx.impl->visibility_increment != increment)
        ++ctx.impl->visibility_epoch;
    if (ctx.impl->encoder && ctx.impl->pass_visibility_active
        && (!enable || ctx.impl->visibility_index != index || ctx.impl->visibility_increment != increment)) {
        [ctx.impl->encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        ctx.impl->pass_visibility_active = false;
    }
    ctx.impl->visibility_enabled = enable;
    ctx.impl->visibility_index = index;
    ctx.impl->visibility_increment = increment;
}
void MetalState::set_back_visibility_index(MetalContext &ctx, bool enable, uint32_t index, bool increment) {
    if (ctx.impl->back_visibility_enabled != enable || ctx.impl->back_visibility_index != index
        || ctx.impl->back_visibility_increment != increment)
        ++ctx.impl->visibility_epoch;
    ctx.impl->back_visibility_enabled = enable;
    ctx.impl->back_visibility_index = index;
    ctx.impl->back_visibility_increment = increment;
}
void MetalState::retire_cpu_overwritten_surfaces(MemState &mem,
    std::span<const std::pair<uint64_t, uint64_t>> ranges) {
    const auto *active = context ? static_cast<const MetalContext *>(context) : nullptr;
    for (auto it = impl->surfaces.begin(); it != impl->surfaces.end();) {
        const auto &surface = it->second;
        const size_t bytes = surface.cpu_hash_size;
        const uint64_t surface_end = uint64_t(it->first) + bytes;
        // An active render target is owned by its current scene. Completed
        // surfaces may have been repurposed by CPU writes since publication.
        const bool rendering = active && active->impl->scene_active && surface.color == active->impl->color;
        const auto overlap = std::find_if(ranges.begin(), ranges.end(), [&](const auto &range) {
            return it->first < range.second && surface_end > range.first;
        });
        if (!rendering && bytes && overlap != ranges.end()
            && surface_end <= uint64_t(UINT32_MAX) - 4095
            && is_valid_addr_range(mem, it->first, Address(surface_end))
            && (surface.cpu_writes.changed() || XXH3_64bits(surface.guest.data.get(mem), bytes) != surface.cpu_hash)) {
            LOG_INFO("Metal: retiring CPU-overwritten color surface at {:#x} before access {:#x}", it->first, overlap->first);
            it = impl->surfaces.erase(it);
        } else ++it;
    }
}
bool MetalState::sync_surface(MemState &mem, const SceGxmColorSurface &surface) {
    if (context) finish(*static_cast<MetalContext *>(context));
    const auto found = impl->surfaces.find(surface.data.address());
    const size_t bytes = surface_memory_size(surface);
    const uint64_t end = uint64_t(surface.data.address()) + bytes;
    if (!surface.data || !bytes || found == impl->surfaces.end()
        || found->second.guest.colorFormat != surface.colorFormat
        || found->second.guest.width != surface.width || found->second.guest.height != surface.height
        || end > uint64_t(UINT32_MAX) - 4095
        || !is_valid_addr_range(mem, surface.data.address(), Address(end))) return false;
    const std::span<uint8_t> output{static_cast<uint8_t *>(surface.data.get(mem)), bytes};
    std::vector<SurfaceMemoryRange> written;
    const auto publication = impl->surface_writes.writeback(mem, [&] {
        return read_surface_publication(found->second, surface, output, written, *impl->device, impl->caster);
    });
    if (publication == SurfacePublication::Unavailable) return false;
    if (publication == SurfacePublication::CpuNewer) return true;
    if (found->second.guest.strideInPixels == surface.strideInPixels
        && found->second.guest.surfaceType == surface.surfaceType) {
        update_cpu_snapshot(found->second, output, written);
    } else found->second.cpu_snapshot.clear();
    return true;
}
int MetalState::sync_surfaces_for_cpu_read(MemState &mem, Address address, uint32_t size) {
    if (disable_surface_sync || !size) return 0;
    const uint64_t range_end = uint64_t(address) + size;
    if (!address || range_end > uint64_t(UINT32_MAX) - 4095
        || !is_valid_addr_range(mem, address, Address(range_end))) return CommandErrorSurfaceSyncFailed;
    // Finish producers without first publishing the active target: CPU readers
    // must exclude CPU-newer surfaces of every size, unlike normal scene sync.
    if (context) finish(*static_cast<MetalContext *>(context), false, true);
    std::vector<Surface *> candidates;
    for (auto &[base, entry] : impl->surfaces) {
        const size_t bytes = surface_memory_size(entry.guest);
        const uint64_t end = uint64_t(base) + bytes;
        // Match Plus's three-frame window using subtraction to avoid overflow.
        if (!bytes || base >= range_end || end <= address
            || impl->frame_timestamp < entry.last_frame_rendered
            || impl->frame_timestamp - entry.last_frame_rendered >= 3
            || end > uint64_t(UINT32_MAX) - 4095
            || !is_valid_addr_range(mem, base, Address(end))) continue;
        const std::span<const uint8_t> guest{static_cast<const uint8_t *>(entry.guest.data.get(mem)), bytes};
        if (!surface_cpu_bytes_changed(entry, guest)) candidates.push_back(&entry);
    }
    int synced = 0;
    for (auto *entry : candidates) {
        const std::span<uint8_t> guest{static_cast<uint8_t *>(entry->guest.data.get(mem)), surface_memory_size(entry->guest)};
        std::vector<SurfaceMemoryRange> written;
        const auto result = impl->surface_writes.writeback(mem, [&] {
            return read_surface_publication(*entry, entry->guest, guest, written, *impl->device, impl->caster);
        });
        if (result == SurfacePublication::Unavailable) return CommandErrorSurfaceSyncFailed;
        if (result == SurfacePublication::CpuNewer) continue;
        update_cpu_snapshot(*entry, guest, written);
        if (context) {
            auto &ctx = *static_cast<MetalContext *>(context);
            if (owns_color_binding(*entry, ctx)) {
                ctx.impl->color_guest_dirty = false;
                ctx.impl->color_guest_current = true;
            }
        }
        ++synced;
    }
    return synced;
}
namespace {
std::vector<SurfaceMemoryRange> surface_intersections(uint64_t address,size_t bytes,
    std::span<const std::pair<uint64_t,uint64_t>> ranges) {
    std::vector<SurfaceMemoryRange> result;
    for (const auto &[begin,end]:ranges) {
        const auto first=std::max(begin,address),last=std::min(end,address+bytes);
        if (first<last) result.push_back({size_t(first-address),size_t(last-first)});
    }
    return result;
}
struct DepthTransfer {
    DepthSurface *surface;
    DepthMemoryLayout layout;
    bool read_depth=false,read_stencil=false;
    std::vector<SurfaceMemoryRange> cpu_depth_changes,cpu_stencil_changes;
    std::vector<uint8_t> depth,stencil;
};
std::vector<SurfaceMemoryRange> changed_memory_ranges(std::span<const uint8_t> bytes,std::span<const uint8_t> baseline) {
    if (bytes.empty()) return {};
    if (bytes.size()!=baseline.size()) return {{0,bytes.size()}};
    std::vector<SurfaceMemoryRange> ranges;
    if (!std::memcmp(bytes.data(),baseline.data(),bytes.size())) return ranges;
    for (size_t at=0;at<bytes.size();) {
        if (bytes[at]==baseline[at]) { ++at;continue; }
        const size_t first=at++;
        while (at<bytes.size() && bytes[at]!=baseline[at]) ++at;
        ranges.push_back({first,at-first});
    }
    return ranges;
}
void depth_memory_changed(MetalState &state,DepthSurface &surface,DepthMemoryWrite written) {
    if (!written.depth && !written.stencil && !written.mask) return;
    surface.snapshots.clear();surface.subrectangles.clear();surface.published=false;
    if (written.depth) surface.depth_content_stored=true;
    if (state.context) {
        auto &ctx=*static_cast<MetalContext *>(state.context);
        if (ctx.impl->depth==surface.texture) ctx.impl->depth_scene_written=true;
        if (written.mask && surface.mask && ctx.impl->mask==surface.mask) ctx.impl->mask_constant_valid=false;
    }
}
bool prepare_depth_transfers(MetalState &state,MemState &mem,
    std::span<const std::pair<uint64_t,uint64_t>> reads,
    std::span<const std::pair<uint64_t,uint64_t>> writes,std::vector<DepthTransfer> &transfers) {
    const auto mapped=[&](Address address,size_t size) {
        return !size || (address && uint64_t(address)+size<=uint64_t(UINT32_MAX)-4095
            && is_valid_addr_range(mem,address,Address(uint64_t(address)+size)));
    };
    for (auto &[key,surface]:state.impl->depth_surfaces) {
        // First exclude unrelated entries even if their descriptor cannot be
        // decoded. This bound includes 4X sample rows and packed stencil bytes.
        const size_t bound=size_t(std::min<__uint128_t>(UINT32_MAX,
            __uint128_t(surface.guest.get_stride())*((uint64_t(surface.height)*2+31)&~uint64_t(31))*4));
        const auto touches=[&](Address address,size_t size,auto ranges) {
            return address && !surface_intersections(address,size,ranges).empty();
        };
        if (!touches(key.first,bound,reads) && !touches(key.first,bound,writes)
            && !touches(key.second,bound,reads) && !touches(key.second,bound,writes)) continue;
        const auto layout=depth_memory_layout(surface.guest,surface.width,surface.height,surface.multisample);
        if (!layout) return false;
        DepthTransfer transfer{&surface,*layout};
        transfer.read_depth=touches(key.first,layout->depth_size,reads);
        transfer.read_stencil=touches(key.second,layout->stencil_size,reads);
        if (!transfer.read_depth && !transfer.read_stencil
            && !touches(key.first,layout->depth_size,writes) && !touches(key.second,layout->stencil_size,writes)) continue;
        if (!surface.guest_snapshot_valid || surface.guest_depth_snapshot.size()!=layout->depth_size
            || surface.guest_stencil_snapshot.size()!=layout->stencil_size
            || !mapped(key.first,layout->depth_size) || !mapped(key.second,layout->stencil_size)) return false;
        const auto [depth,stencil]=depth_memory_spans(mem,surface.guest,*layout);
        if (!state.impl->caster) state.impl->caster=std::make_unique<SurfaceCaster>(*state.impl->device);
        if (!state.impl->caster->patch_depth_memory(surface.texture,surface.guest,*layout,surface.scale,
            depth,stencil,{},{},surface.mask)) return false;
        transfer.cpu_depth_changes=changed_memory_ranges(depth,surface.guest_depth_snapshot);
        transfer.cpu_stencil_changes=changed_memory_ranges(stencil,surface.guest_stencil_snapshot);
        transfers.push_back(std::move(transfer));
    }
    // CPU edits precede this transfer. Import only those bytes, retaining every
    // other native pixel/sample, before reading a GPU-produced source image.
    for (auto &transfer:transfers) {
        auto &surface=*transfer.surface;
        const auto [depth,stencil]=depth_memory_spans(mem,surface.guest,transfer.layout);
        if (!transfer.cpu_depth_changes.empty() || !transfer.cpu_stencil_changes.empty()) {
            DepthMemoryWrite written;
            require(state.impl->caster->patch_depth_memory(surface.texture,surface.guest,transfer.layout,surface.scale,
                depth,stencil,transfer.cpu_depth_changes,transfer.cpu_stencil_changes,surface.mask,nil,&written),
                "Metal: cannot import CPU depth edits before transfer");
            surface.guest_depth_snapshot.assign(depth.begin(),depth.end());
            surface.guest_stencil_snapshot.assign(stencil.begin(),stencil.end());
            depth_memory_changed(state,surface,written);
        }
        if (transfer.read_depth || transfer.read_stencil) {
            transfer.depth.assign(depth.begin(),depth.end());transfer.stencil.assign(stencil.begin(),stencil.end());
            if (!state.impl->caster->store_depth_memory(surface.texture,surface.guest,transfer.layout,surface.scale,
                transfer.depth,transfer.stencil,surface.mask)) return false;
            const auto restore_cpu=[](auto bytes,const auto &ranges,auto &output) {
                for (const auto &range:ranges)
                    std::memcpy(output.data()+range.offset,bytes.data()+range.offset,range.size);
            };
            // CPU writes are newer than the preceding GPU producer, including
            // bytes that a reduced native extent cannot represent separately.
            restore_cpu(depth,transfer.cpu_depth_changes,transfer.depth);
            if (transfer.layout.packed && transfer.layout.stencil_size)
                restore_cpu(stencil,transfer.cpu_stencil_changes,transfer.depth);
            else restore_cpu(stencil,transfer.cpu_stencil_changes,transfer.stencil);
            // Packed stencil aliases the entire depth allocation. The direct
            // readback can write depth only; expose one consistent byte image.
            if (transfer.layout.packed && transfer.layout.stencil_size) transfer.stencil=transfer.depth;
        }
    }
    return true;
}
void finish_depth_transfers(MetalState &state,MemState &mem,std::span<DepthTransfer> transfers,
    std::span<const std::pair<uint64_t,uint64_t>> writes) {
    for (auto &transfer:transfers) {
        auto &surface=*transfer.surface;
        const auto depth_changes=surface_intersections(surface.guest.depth_data.address(),transfer.layout.depth_size,writes);
        const auto stencil_changes=surface_intersections(surface.guest.stencil_data.address(),transfer.layout.stencil_size,writes);
        if (depth_changes.empty() && stencil_changes.empty()) continue;
        const auto [depth,stencil]=depth_memory_spans(mem,surface.guest,transfer.layout);
        DepthMemoryWrite written;
        require(state.impl->caster->patch_depth_memory(surface.texture,surface.guest,transfer.layout,surface.scale,
            depth,stencil,depth_changes,stencil_changes,surface.mask,nil,&written),
            "Metal: validated depth transfer destination could not be updated");
        const auto update=[](auto bytes,const auto &ranges,auto &baseline) {
            for (const auto &range:ranges)
                std::memcpy(baseline.data()+range.offset,bytes.data()+range.offset,range.size);
        };
        update(depth,depth_changes,surface.guest_depth_snapshot);
        update(stencil,stencil_changes,surface.guest_stencil_snapshot);
        depth_memory_changed(state,surface,written);
    }
}
}
bool MetalState::transfer_fill(MemState &mem, const SceGxmTransferImage &image, uint32_t color) {
    const uint32_t bits = gxm::get_bits_per_pixel(image.format);
    // The GXM fill value contains at most four bytes. Never borrow adjacent
    // stack bytes when a guest supplies RAW64/RAW128 or an invalid format.
    if (!bits || bits > 32 || bits % 8 || !image.width || !image.height
        || image.width > 16384 || image.height > 16384 || !image.address) return false;
    const size_t pixel_bytes = bits / 8;
    std::vector<std::pair<uint64_t,uint64_t>> rows;
    for (uint32_t y = 0; y < image.height; ++y) {
        const __int128 begin = __int128(image.address.address()) + (__int128(image.y) + y) * image.stride
            + __int128(image.x) * pixel_bytes;
        const __int128 end = begin + size_t(image.width) * pixel_bytes;
        if (begin <= 0 || end > uint64_t(UINT32_MAX) - 4095
            || !is_valid_addr_range(mem, Address(begin), Address(end))) return false;
        rows.emplace_back(uint64_t(begin), uint64_t(end));
    }
    std::sort(rows.begin(), rows.end());
    std::vector<std::pair<uint64_t,uint64_t>> writes;
    for (const auto &row : rows) {
        if (!writes.empty() && row.first <= writes.back().second)
            writes.back().second = std::max(writes.back().second,row.second);
        else writes.push_back(row);
    }
    if (context) finish(*static_cast<MetalContext *>(context));
    struct Update { Surface *surface; size_t bytes; std::vector<SurfaceMemoryRange> ranges; };
    // Retire stale aliases before saving pointers or modifying the destination;
    // otherwise a partial fill would hide earlier CPU writes outside its rows.
    retire_cpu_overwritten_surfaces(mem, writes);
    std::vector<Update> updates;
    for (auto &[address,surface] : impl->surfaces) {
        // Estimate unsupported representations too, so an overlapping write
        // cannot silently leave such a cached GPU surface unchanged.
        const size_t rows = surface.guest.surfaceType == SCE_GXM_COLOR_SURFACE_TILED
            ? (size_t(surface.guest.height)+31)&~size_t(31) : surface.guest.height;
        const size_t extent = rows * surface.guest.strideInPixels
            * ((gxm::bits_per_pixel(gxm::get_base_format(surface.guest.colorFormat))+7)/8);
        Update update{&surface, surface_memory_size(surface.guest), {}};
        for (const auto &[begin,end] : writes) {
            const uint64_t first = std::max<uint64_t>(begin,address), last = std::min<uint64_t>(end,uint64_t(address)+extent);
            if (first < last) update.ranges.push_back({size_t(first-address),size_t(last-first)});
        }
        if (update.ranges.empty()) continue;
        const uint64_t end = uint64_t(address)+update.bytes;
        if (!update.bytes || end > uint64_t(UINT32_MAX)-4095 || !is_valid_addr_range(mem,address,Address(end))
            || !write_surface_storage(surface,surface.guest,
                {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},{})) return false;
        updates.push_back(std::move(update));
    }
    std::vector<DepthTransfer> depth_transfers;
    if (!prepare_depth_transfers(*this,mem,{},writes,depth_transfers)) return false;
    std::array<uint8_t,4> bytes;
    std::memcpy(bytes.data(),&color,4);
    // Keep row origins: a packed 24-bit pattern must restart at every pixel,
    // even when the signed row stride is not divisible by three.
    for (uint32_t y = 0; y < image.height; ++y) {
        const auto address = Address(__int128(image.address.address()) + (__int128(image.y)+y)*image.stride
            + __int128(image.x)*pixel_bytes);
        auto *row = Ptr<uint8_t>(address).get(mem);
        for (uint32_t x = 0; x < image.width; ++x) std::memcpy(row+x*pixel_bytes,bytes.data(),pixel_bytes);
    }
    for (auto &update : updates) {
        auto &surface = *update.surface;
        require(write_surface_storage(surface,surface.guest,
            {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},update.ranges),
            "Metal: validated transfer destination could not be updated");
        update_cpu_snapshot(surface,
            {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},update.ranges);
        ++surface.revision;
        surface.rgba8_casts.clear(); surface.word_casts.clear(); surface.word_rect_casts.clear(); surface.halfword_rect_casts.clear(); surface.byte_rect_casts.clear(); surface.small_rect_casts.clear(); surface.subrectangles.clear();
        if (surface.multisample_color && !surface.multisample_dirty) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->patch_multisample(surface.multisample_color,surface.guest,surface.multisample_scale,
                {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},update.ranges),
                "Metal: validated transfer could not preserve multisample storage");
        }
        if (surface.raw_multisample_color && !surface.raw_multisample_dirty) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->patch_multisample(surface.raw_multisample_color,surface.guest,surface.multisample_scale,
                {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},update.ranges),
                "Metal: validated transfer could not preserve raw multisample storage");
        }
    }
    finish_depth_transfers(*this,mem,depth_transfers,writes);
    return true;
}
namespace {
using TransferRanges = std::vector<std::pair<uint64_t, uint64_t>>;
void merge_transfer_ranges(TransferRanges &ranges) {
    std::sort(ranges.begin(), ranges.end());
    size_t count = 0;
    for (const auto range : ranges) {
        if (count && range.first <= ranges[count-1].second)
            ranges[count-1].second = std::max(ranges[count-1].second, range.second);
        else ranges[count++] = range;
    }
    ranges.resize(count);
}
void append_transfer_range(TransferRanges &ranges, uint64_t address, size_t bytes) {
    if (!ranges.empty() && address == ranges.back().second) ranges.back().second += bytes;
    else ranges.emplace_back(address, address+bytes);
}
struct TransferLayout {
    const SceGxmTransferImage &image;
    SceGxmTransferType type;
    size_t bytes;
    // All arithmetic is signed and wide until the validated guest address is
    // produced. In particular RGB24 does not have a stride measured in pixels.
    __int128 address(uint32_t x, uint32_t y) const {
        const uint64_t px = uint64_t(image.x)+x, py = uint64_t(image.y)+y;
        __int128 offset;
        if (type == SCE_GXM_TRANSFER_LINEAR) offset = __int128(py)*image.stride+px*bytes;
        else if (type == SCE_GXM_TRANSFER_TILED)
            offset = (__int128(py/32)*image.stride*32)+((px/32)*1024+(py%32)*32+px%32)*bytes;
        else offset = uint64_t(texture::encode_morton(uint16_t(px),uint16_t(py),image.width,image.height))*bytes;
        return __int128(image.address.address())+offset;
    }
    bool validate(MemState &mem, TransferRanges &ranges) const {
        if (!image.address || !image.width || !image.height || image.width>16384 || image.height>16384) return false;
        if (type == SCE_GXM_TRANSFER_SWIZZLED) {
            if (!std::has_single_bit(image.width) || !std::has_single_bit(image.height)
                || uint64_t(image.x)+image.width>UINT16_MAX+1ull
                || uint64_t(image.y)+image.height>UINT16_MAX+1ull) return false;
            if (!image.x && !image.y) {
                const uint64_t begin=image.address.address(), end=begin+uint64_t(image.width)*image.height*bytes;
                if (end>uint64_t(UINT32_MAX)-4095 || !is_valid_addr_range(mem,Address(begin),Address(end))) return false;
                ranges.emplace_back(begin,end);
                return true;
            }
            // Match the software/Vulkan transfer's Morton address formula,
            // including the offset origin. Validate the actual source/dest
            // addresses before reading or modifying guest memory.
            const bool exact_ranges = uint64_t(image.width)*image.height <= 65536;
            uint64_t first=UINT64_MAX, last=0;
            if (exact_ranges) ranges.reserve(ranges.size()+size_t(image.width)*image.height);
            for (uint32_t y=0;y<image.height;++y) for (uint32_t x=0;x<image.width;++x) {
                const auto begin=address(x,y), end=begin+bytes;
                if (begin<=0 || end>uint64_t(UINT32_MAX)-4095) return false;
                first=std::min(first,uint64_t(begin)); last=std::max(last,uint64_t(end));
                if (exact_ranges) append_transfer_range(ranges,uint64_t(begin),bytes);
            }
            if (!exact_ranges) ranges.emplace_back(first,last);
            merge_transfer_ranges(ranges);
            for (const auto &[begin,end] : ranges)
                if (!is_valid_addr_range(mem,Address(begin),Address(end))) return false;
            return true;
        } else if (type == SCE_GXM_TRANSFER_TILED) {
            if (!image.stride || int64_t(image.stride) % int64_t(32*bytes)) return false;
        } else if (type != SCE_GXM_TRANSFER_LINEAR) return false;
        for (uint32_t y=0;y<image.height;++y) for (uint32_t x=0;x<image.width;) {
            const uint32_t count=type==SCE_GXM_TRANSFER_TILED
                ? std::min<uint32_t>(image.width-x,32-(uint64_t(image.x)+x)%32) : image.width;
            const auto begin = address(x,y), end = begin+count*bytes;
            if (begin<=0 || end>uint64_t(UINT32_MAX)-4095) return false;
            append_transfer_range(ranges,uint64_t(begin),count*bytes);
            x+=count;
        }
        merge_transfer_ranges(ranges);
        for (const auto &[begin,end] : ranges)
            if (!is_valid_addr_range(mem,Address(begin),Address(end))) return false;
        return true;
    }
};
size_t cached_surface_extent(const Surface &surface) {
    const size_t rows = surface.guest.surfaceType == SCE_GXM_COLOR_SURFACE_TILED
        ? (size_t(surface.guest.height)+31)&~size_t(31) : surface.guest.height;
    return rows*surface.guest.strideInPixels*((gxm::bits_per_pixel(gxm::get_base_format(surface.guest.colorFormat))+7)/8);
}
}
bool MetalState::transfer_copy(MemState &mem, const SceGxmTransferImage &source, const SceGxmTransferImage &destination,
    SceGxmTransferType source_type, SceGxmTransferType destination_type,
    SceGxmTransferColorKeyMode mode, uint32_t key, uint32_t mask) {
    return transfer_image(mem,source,destination,source_type,destination_type,mode,key,mask,false);
}
bool MetalState::transfer_downscale(MemState &mem, const SceGxmTransferImage &source, const SceGxmTransferImage &destination) {
    return transfer_image(mem,source,destination,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
        SCE_GXM_TRANSFER_COLORKEY_NONE,0,0,true);
}
bool MetalState::try_transfer_depth_gpu(MemState &mem, const SceGxmTransferImage &source,
    const SceGxmTransferImage &destination, SceGxmTransferType source_type, SceGxmTransferType destination_type,
    std::span<const std::pair<uint64_t,uint64_t>> reads,
    std::span<const std::pair<uint64_t,uint64_t>> writes) {
    // Guest depth words and separate stencil bytes may share a native image,
    // but only packed S8D24 transfers own both aspects of that image.
    const auto overlaps=[](uint64_t address,size_t size,auto ranges) {
        return size && std::any_of(ranges.begin(),ranges.end(),[&](const auto &range) {
            return address<range.second && range.first<address+size;
        });
    };
    struct Candidate { DepthSurface *entry; DepthMemoryLayout layout; bool stencil; };
    const auto find=[&](const SceGxmTransferImage &image) -> std::optional<Candidate> {
        std::optional<Candidate> result;
        const uint32_t bits=gxm::get_bits_per_pixel(image.format);
        for (auto &[key,entry]:impl->depth_surfaces) {
            if (key.first!=image.address.address() && key.second!=image.address.address()) continue;
            const auto layout=depth_memory_layout(entry.guest,entry.width,entry.height,entry.multisample);
            if (!layout) return std::nullopt;
            const bool depth=key.first==image.address.address() && layout->depth_size && bits==layout->depth_bytes*8;
            const bool stencil=key.second==image.address.address() && !layout->packed && layout->stencil_size && bits==8;
            if (!depth && !stencil) continue;
            if (result || (depth && stencil)) return std::nullopt;
            result=Candidate{&entry,*layout,stencil};
        }
        return result;
    };
    const auto source_entry=find(source),destination_entry=find(destination);
    if (!source_entry || !destination_entry || source_entry->stencil!=destination_entry->stencil) return false;
    auto *src=source_entry->entry,*dst=destination_entry->entry;
    const bool stencil=source_entry->stencil;
    if (!src || !dst || src==dst || !src->texture || !dst->texture
        || src->texture==dst->texture || src->texture.sampleCount!=dst->texture.sampleCount
        || (!stencil && src->guest.get_format()!=dst->guest.get_format()))
        return false;
    const auto *src_layout=&source_entry->layout,*dst_layout=&destination_entry->layout;
    const size_t destination_size=stencil ? dst_layout->stencil_size : dst_layout->depth_size;
    for (const auto &[first,last]:writes)
        if (first<destination.address.address() || last<first
            || last>uint64_t(destination.address.address())+destination_size) return false;
    const auto region=[stencil](const DepthSurface &entry,const DepthMemoryLayout &layout,
                          const SceGxmTransferImage &image,SceGxmTransferType type) -> std::optional<SurfaceRect> {
        const uint32_t bytes=stencil ? 1 : layout.depth_bytes;
        if (!bytes || image.stride<=0
            || uint64_t(image.stride)!=uint64_t(layout.stride)*bytes
            || type!=(layout.tiled ? SCE_GXM_TRANSFER_TILED : SCE_GXM_TRANSFER_LINEAR)
            || uint64_t(image.x)+image.width>layout.width || uint64_t(image.y)+image.height>layout.height)
            return std::nullopt;
        const uint32_t sx=entry.texture.sampleCount>1 ? uint32_t(entry.texture.sampleCount/2) : 1;
        const uint32_t sy=entry.texture.sampleCount>1 ? 2 : 1;
        if (image.x%sx || image.width%sx || image.y%sy || image.height%sy) return std::nullopt;
        const uint64_t left=uint64_t(image.x)*entry.texture.width;
        const uint64_t right=(uint64_t(image.x)+image.width)*entry.texture.width;
        const uint64_t top=uint64_t(image.y)*entry.texture.height;
        const uint64_t bottom=(uint64_t(image.y)+image.height)*entry.texture.height;
        // The copy must address whole native pixels and whole sample groups.
        // Fractional-resolution edges are eligible only when exactly aligned.
        if (left%layout.width || right%layout.width || top%layout.height || bottom%layout.height)
            return std::nullopt;
        return SurfaceRect{uint32_t(left/layout.width),uint32_t(top/layout.height),
            uint32_t((right-left)/layout.width),uint32_t((bottom-top)/layout.height)};
    };
    const auto src_rect=region(*src,*src_layout,source,source_type);
    const auto dst_rect=region(*dst,*dst_layout,destination,destination_type);
    if (!src_rect || !dst_rect || !src_rect->width || !src_rect->height
        || src_rect->width!=dst_rect->width || src_rect->height!=dst_rect->height)
        return false;
    // A second overlapping representation needs ownership/order resolution,
    // not a source chosen by address-map iteration order.
    const auto touches=[&](const DepthSurface &entry,const DepthMemoryLayout &layout,auto ranges) {
        return overlaps(entry.guest.depth_data.address(),layout.depth_size,ranges)
            || overlaps(entry.guest.stencil_data.address(),layout.stencil_size,ranges);
    };
    const auto aliased=[](const DepthSurface &entry,const DepthMemoryLayout &layout) {
        return !layout.packed && layout.depth_size && layout.stencil_size
            && uint64_t(entry.guest.depth_data.address())<uint64_t(entry.guest.stencil_data.address())+layout.stencil_size
            && uint64_t(entry.guest.stencil_data.address())<uint64_t(entry.guest.depth_data.address())+layout.depth_size;
    };
    if (touches(*src,*src_layout,writes) || touches(*dst,*dst_layout,reads)
        || aliased(*src,*src_layout) || aliased(*dst,*dst_layout)) return false;
    for (const auto &[address,entry]:impl->surfaces)
        if (overlaps(address,cached_surface_extent(entry),reads)
            || overlaps(address,cached_surface_extent(entry),writes)) return false;
    for (const auto &[key,entry]:impl->depth_surfaces) {
        if (&entry==src || &entry==dst) continue;
        const auto layout=depth_memory_layout(entry.guest,entry.width,entry.height,entry.multisample);
        if (!layout) return false;
        if (overlaps(key.first,layout->depth_size,reads) || overlaps(key.first,layout->depth_size,writes)
            || overlaps(key.second,layout->stencil_size,reads) || overlaps(key.second,layout->stencil_size,writes))
            return false;
    }
    const auto unchanged=[&](const DepthSurface &entry,const DepthMemoryLayout &layout) {
        const auto matches=[&](Address address,size_t size,const std::vector<uint8_t> &snapshot) {
            if (!size) return true;
            const uint64_t end=uint64_t(address)+size;
            return snapshot.size()==size && end<=uint64_t(UINT32_MAX)-4095
                && is_valid_addr_range(mem,address,Address(end))
                && std::memcmp(snapshot.data(),Ptr<uint8_t>(address).get(mem),size)==0;
        };
        return entry.guest_snapshot_valid
            && matches(entry.guest.depth_data.address(),layout.depth_size,entry.guest_depth_snapshot)
            && matches(entry.guest.stencil_data.address(),layout.stencil_size,entry.guest_stencil_snapshot);
    };
    if (!unchanged(*src,*src_layout) || !unchanged(*dst,*dst_layout)) return false;
    const auto masked=[](const DepthSurface &entry) {
        return entry.guest.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M
            || entry.guest.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8;
    };
    const auto valid_mask=[](const DepthSurface &entry) {
        return entry.mask && entry.mask.pixelFormat==MTLPixelFormatRGBA8Unorm
            && entry.mask.width==entry.texture.width && entry.mask.height==entry.texture.height
            && entry.mask.sampleCount==entry.texture.sampleCount;
    };
    const bool copy_mask=!stencil && masked(*src);
    if ((masked(*dst) && !valid_mask(*dst)) || (copy_mask && !valid_mask(*src))) return false;
    std::vector<uint8_t> depth_output=dst->guest_depth_snapshot,stencil_output=dst->guest_stencil_snapshot;
    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
    auto commands=scene_command_buffer(*impl->device);
    const auto copy=[&](id<MTLTexture> input,id<MTLTexture> output) {
        auto blit=[commands blitCommandEncoder];
        require(blit!=nil,"Metal: cannot encode depth transfer");
        [blit copyFromTexture:input sourceSlice:0 sourceLevel:0
            sourceOrigin:MTLOriginMake(src_rect->x,src_rect->y,0)
            sourceSize:MTLSizeMake(src_rect->width,src_rect->height,1)
            toTexture:output destinationSlice:0 destinationLevel:0
            destinationOrigin:MTLOriginMake(dst_rect->x,dst_rect->y,0)];
        [blit endEncoding];
    };
    if (src_layout->packed) copy(src->texture,dst->texture);
    else if (!impl->caster->copy_depth_stencil_region(src->texture,dst->texture,*src_rect,*dst_rect,stencil,commands))
        return false;
    if (copy_mask) copy(src->mask,dst->mask);
    DepthStoreReadback readback;
    // Retain native precision in the destination, but keep Metal's existing
    // guest-visible transfer write. Convert behind the copy in the same buffer.
    require(impl->caster->enqueue_depth_store(dst->texture,dst->guest,*dst_layout,dst->scale,
        dst_layout->depth_size,dst_layout->stencil_size,dst->mask,commands,readback),
        "Metal: cannot read back depth transfer");
    std::string error;
    require(impl->device->submit_and_wait(commands,error),error);
    impl->caster->finish_depth_store(readback,dst->guest,*dst_layout,depth_output,stencil_output);
    const auto &output=stencil ? stencil_output : depth_output;
    auto &baseline=stencil ? dst->guest_stencil_snapshot : dst->guest_depth_snapshot;
    auto *guest=static_cast<uint8_t *>(destination.address.get(mem));
    for (const auto &[first,last]:writes) {
        const size_t offset=first-destination.address.address(),size=last-first;
        std::memcpy(guest+offset,output.data()+offset,size);
        std::memcpy(baseline.data()+offset,output.data()+offset,size);
    }
    if (dst_layout->packed && dst_layout->stencil_size) dst->guest_stencil_snapshot=dst->guest_depth_snapshot;
    dst->snapshots.clear();dst->subrectangles.clear();
    if (!stencil) dst->depth_content_stored=true;
    // Only transferred bytes were published; untouched native pixels may still
    // differ from RAM, so this is not a complete surface publication.
    dst->published=false;
    if (context) {
        auto &ctx=*static_cast<MetalContext *>(context);
        if (ctx.impl->depth==dst->texture) ctx.impl->depth_scene_written=true;
        if (copy_mask && ctx.impl->mask==dst->mask) ctx.impl->mask_constant_valid=false;
    }
    return true;
}
bool MetalState::transfer_image(MemState &mem, const SceGxmTransferImage &source, const SceGxmTransferImage &destination,
    SceGxmTransferType source_type, SceGxmTransferType destination_type,
    SceGxmTransferColorKeyMode mode, uint32_t key, uint32_t mask, bool downscale) {
    const uint32_t bits=gxm::get_bits_per_pixel(source.format);
    if (downscale && bits!=8 && bits!=16 && bits!=24 && bits!=32) return false;
    const bool matching_dimensions = downscale
        ? source.width / 2 == destination.width && source.height / 2 == destination.height
        : source.width == destination.width && source.height == destination.height;
    if (source.format!=destination.format || !bits || bits%8 || bits>128
        || !matching_dimensions
        || (mode!=SCE_GXM_TRANSFER_COLORKEY_NONE && mode!=SCE_GXM_TRANSFER_COLORKEY_PASS && mode!=SCE_GXM_TRANSFER_COLORKEY_REJECT)
        || (mode!=SCE_GXM_TRANSFER_COLORKEY_NONE && bits!=32)) return false;
    const size_t bytes=bits/8;
    const TransferLayout src{source,source_type,bytes}, dst{destination,destination_type,bytes};
    TransferRanges reads, destinations;
    if (!src.validate(mem,reads) || !dst.validate(mem,destinations)) return false;
    const bool trace_transfer = impl->trace_batches && impl->traced_transfers++ < 512;
    if (trace_transfer)
        LOG_INFO("Metal transfer: src={:#x} {}x{} stride={} xy={},{} type={} dst={:#x} {}x{} stride={} xy={},{} type={} format={:#x} downscale={}",
            source.address.address(), source.width, source.height, source.stride, source.x, source.y, uint32_t(source_type),
            destination.address.address(), destination.width, destination.height, destination.stride, destination.x, destination.y,
            uint32_t(destination_type), uint32_t(source.format), downscale);
    if (context) finish(*static_cast<MetalContext *>(context));
    // A transfer can read a former render target without binding it as a
    // texture first. Apply the same CPU-authority check before GPU snapshots
    // replace source bytes or partial writes update destination cache hashes.
    retire_cpu_overwritten_surfaces(mem, reads);
    retire_cpu_overwritten_surfaces(mem, destinations);
    if (!downscale && (bits==8 || bits==16 || bits==32) && mode==SCE_GXM_TRANSFER_COLORKEY_NONE
        && try_transfer_depth_gpu(mem,source,destination,source_type,destination_type,reads,destinations))
        return true;
    struct Snapshot {
        uint64_t address;
        std::vector<uint8_t> data;
        std::vector<SurfaceMemoryRange> published;
        std::vector<SurfaceMemoryRange> claimed;
    };
    struct Update { Surface *surface; size_t bytes; };
    std::vector<Snapshot> snapshots;
    std::vector<Update> updates;
    const auto conflicts_with_snapshot = [&](const Snapshot &candidate) {
        for (const auto &range : candidate.claimed) {
            const uint64_t first_address = candidate.address + range.offset;
            const uint64_t last_address = first_address + range.size;
            for (const auto &other : snapshots) {
                for (const auto &other_range : other.claimed) {
                    const uint64_t other_first = other.address + other_range.offset;
                    const uint64_t first = std::max(first_address, other_first);
                    const uint64_t last = std::min(last_address, other_first + other_range.size);
                    if (first >= last) continue;
                    // Both views may be valid despite overlapping ownership.
                    // Equal bytes are independent of which draw wrote last;
                    // differing bytes still require write-order provenance.
                    for (const auto &read : surface_intersections(first, last - first, reads)) {
                        const uint64_t at = first + read.offset;
                        if (std::memcmp(candidate.data.data() + at - candidate.address,
                                other.data.data() + at - other.address, read.size) != 0)
                            return true;
                    }
                }
            }
        }
        return false;
    };
    // Preflight every overlapping cached representation before modifying RAM.
    for (auto &[address,surface] : impl->surfaces) {
        const size_t extent=cached_surface_extent(surface);
        const bool read=!surface_intersections(address,extent,reads).empty();
        const bool write=!surface_intersections(address,extent,destinations).empty();
        if (!read && !write) continue;
        if (trace_transfer)
            LOG_INFO("Metal transfer surface: address={:#x} {}x{} stride={} type={} format={:#x} native={}x{} read={} write={}",
                address, surface.guest.width, surface.guest.height, surface.guest.strideInPixels, uint32_t(surface.guest.surfaceType),
                uint32_t(surface.guest.colorFormat), surface.color.width, surface.color.height, read, write);
        const size_t size=surface_memory_size(surface.guest);
        const uint64_t end=uint64_t(address)+size;
        if (!size || end>uint64_t(UINT32_MAX)-4095 || !is_valid_addr_range(mem,address,Address(end))) return false;
        const auto *guest=static_cast<const uint8_t *>(surface.guest.data.get(mem));
        if (write) {
            if (!write_surface_storage(surface,surface.guest,{guest,size},{})) return false;
            updates.push_back({&surface,size});
        }
        if (read) {
            Snapshot snapshot{address,std::vector<uint8_t>(guest,guest+size),{}, {}};
            // Plus transfers read through perform_surface_sync, including its
            // write-ownership bounds and scaled blit. Preserve guest bytes
            // outside those bounds in this private source snapshot as well.
            // This is not a guest publication: leave the CPU baseline intact.
            const auto publication = read_surface_publication(surface,surface.guest,snapshot.data,
                snapshot.published,*impl->device,impl->caster);
            if (publication == SurfacePublication::Unavailable) return false;
            // A CPU-newer cached alias claims its entire source range until
            // the exact changed bytes are known. Other aliases may contribute
            // only when their published bytes do not compete with that range.
            snapshot.claimed = publication == SurfacePublication::CpuNewer
                ? surface_intersections(address,size,reads) : snapshot.published;
            if (conflicts_with_snapshot(snapshot)) return false;
            snapshots.push_back(std::move(snapshot));
        }
    }
    std::vector<DepthTransfer> depth_transfers;
    if (!prepare_depth_transfers(*this,mem,reads,destinations,depth_transfers)) return false;
    const auto add_depth_snapshot=[&](uint64_t address,std::vector<uint8_t> &data) {
        const std::array published{SurfaceMemoryRange{0,data.size()}};
        Snapshot snapshot{address,std::move(data),{published.begin(),published.end()},
            {published.begin(),published.end()}};
        if (conflicts_with_snapshot(snapshot)) return false;
        snapshots.push_back(std::move(snapshot));
        return true;
    };
    for (auto &transfer:depth_transfers) {
        if (transfer.read_depth && !add_depth_snapshot(transfer.surface->guest.depth_data.address(),transfer.depth)) return false;
        if (transfer.read_stencil && (!transfer.layout.packed || !transfer.read_depth)
            && !add_depth_snapshot(transfer.surface->guest.stencil_data.address(),transfer.stencil)) return false;
    }
    // Snapshot the complete source rectangle before any destination writes:
    // overlapping copies never consume bytes already replaced by this command.
    std::vector<uint8_t> pixels(size_t(source.width)*source.height*bytes);
    for (uint32_t y=0;y<source.height;++y) {
        auto *row=pixels.data()+size_t(y)*source.width*bytes;
        if (source_type==SCE_GXM_TRANSFER_LINEAR) {
            const uint64_t address=uint64_t(src.address(0,y));
            const size_t row_bytes=size_t(source.width)*bytes;
            std::memcpy(row,Ptr<uint8_t>(Address(address)).get(mem),row_bytes);
            for (const auto &snapshot : snapshots) for (const auto &range : snapshot.published) {
                const uint64_t begin=snapshot.address+range.offset;
                const uint64_t first=std::max(address,begin);
                const uint64_t last=std::min(address+row_bytes,begin+range.size);
                if (first<last) std::memcpy(row+first-address,snapshot.data.data()+first-snapshot.address,last-first);
            }
        } else for (uint32_t x=0;x<source.width;++x) {
            const uint64_t address=uint64_t(src.address(x,y));
            auto *pixel=row+size_t(x)*bytes;
            std::memcpy(pixel,Ptr<uint8_t>(Address(address)).get(mem),bytes);
            for (const auto &snapshot : snapshots) for (const auto &range : snapshot.published) {
                const uint64_t begin=snapshot.address+range.offset;
                const uint64_t first=std::max(address,begin), last=std::min(address+bytes,begin+range.size);
                if (first<last) std::memcpy(pixel+first-address,snapshot.data.data()+first-snapshot.address,last-first);
            }
        }
    }
    if (downscale) {
        std::vector<uint8_t> reduced(size_t(destination.width)*destination.height*bytes);
        AVPixelFormat pixel_format=AV_PIX_FMT_NONE;
        if (source.stride>0 && destination.stride>0) {
            if (source.format==SCE_GXM_TRANSFER_FORMAT_U5U6U5_BGR) pixel_format=AV_PIX_FMT_RGB565LE;
            if (source.format==SCE_GXM_TRANSFER_FORMAT_U8U8U8_BGR) pixel_format=AV_PIX_FMT_RGB24;
            if (source.format==SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR) pixel_format=AV_PIX_FMT_RGBA;
        }
        if (pixel_format!=AV_PIX_FMT_NONE) {
            SwsContext *scale=sws_getContext(source.width,source.height,pixel_format,
                destination.width,destination.height,pixel_format,SWS_AREA,nullptr,nullptr,nullptr);
            if (!scale) return false;
            const uint8_t *source_plane=pixels.data();
            uint8_t *destination_plane=reduced.data();
            const int source_stride=int(source.width*bytes), destination_stride=int(destination.width*bytes);
            const int rows=sws_scale(scale,&source_plane,&source_stride,0,source.height,
                &destination_plane,&destination_stride);
            sws_freeContext(scale);
            if (rows!=destination.height) return false;
        } else {
            // The software/Vulkan path uses its nearest fallback for every
            // other transfer format and for negative strides.
            for (uint32_t y=0;y<destination.height;++y) for (uint32_t x=0;x<destination.width;++x)
                std::memcpy(reduced.data()+(size_t(y)*destination.width+x)*bytes,
                    pixels.data()+((size_t(y)*2)*source.width+x*2)*bytes,bytes);
        }
        pixels=std::move(reduced);
    }
    TransferRanges writes;
    const auto passes = [&](const uint8_t *pixel) {
        if (mode==SCE_GXM_TRANSFER_COLORKEY_NONE) return true;
        uint32_t value; std::memcpy(&value,pixel,4);
        const bool equal=(value&mask)==key;
        return mode==SCE_GXM_TRANSFER_COLORKEY_PASS ? equal : !equal;
    };
    for (uint32_t y=0;y<destination.height;++y) {
        const auto *row=pixels.data()+size_t(y)*destination.width*bytes;
        if (destination_type==SCE_GXM_TRANSFER_LINEAR && mode==SCE_GXM_TRANSFER_COLORKEY_NONE) {
            const Address address=Address(dst.address(0,y));
            const size_t row_bytes=size_t(destination.width)*bytes;
            std::memcpy(Ptr<uint8_t>(address).get(mem),row,row_bytes);
            append_transfer_range(writes,address,row_bytes);
        } else for (uint32_t x=0;x<destination.width;++x) {
            const auto *pixel=row+size_t(x)*bytes;
            if (!passes(pixel)) continue;
            const Address address=Address(dst.address(x,y));
            std::memcpy(Ptr<uint8_t>(address).get(mem),pixel,bytes);
            append_transfer_range(writes,address,bytes);
        }
    }
    merge_transfer_ranges(writes);
    for (const auto &update : updates) {
        auto &surface=*update.surface;
        const auto ranges=surface_intersections(surface.guest.data.address(),update.bytes,writes);
        if (ranges.empty()) continue;
        require(write_surface_storage(surface,surface.guest,
            {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},ranges),
            "Metal: validated transfer copy destination could not be updated");
        update_cpu_snapshot(surface,
            {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},ranges);
        ++surface.revision;
        surface.rgba8_casts.clear(); surface.word_casts.clear(); surface.word_rect_casts.clear(); surface.halfword_rect_casts.clear(); surface.byte_rect_casts.clear(); surface.small_rect_casts.clear(); surface.subrectangles.clear();
        if (surface.multisample_color && !surface.multisample_dirty) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->patch_multisample(surface.multisample_color,surface.guest,surface.multisample_scale,
                {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},ranges),
                "Metal: validated transfer could not preserve multisample storage");
        }
        if (surface.raw_multisample_color && !surface.raw_multisample_dirty) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->patch_multisample(surface.raw_multisample_color,surface.guest,surface.multisample_scale,
                {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},ranges),
                "Metal: validated transfer could not preserve raw multisample storage");
        }
    }
    finish_depth_transfers(*this,mem,depth_transfers,writes);
    return true;
}
void MetalState::set_context(MetalContext &ctx, MemState &mem) {
    @autoreleasepool {
        if (context && context != &ctx) end_scene(*static_cast<MetalContext *>(context));
        end_scene(ctx);
        ctx.impl->published_set_results.clear();
        context = &ctx; ctx.impl->mem = &mem;
        ctx.impl->color_guest_current = false;
        ctx.impl->color_guest_dirty = false;
        ctx.impl->color_binding_generation = 0;
        ctx.impl->color_publication_discarded = false;
        ctx.impl->half_pixel_strips = {};
        ctx.impl->color_clip_snapshot = nil;
        ctx.impl->color_clip_restore_pending = false;
        ctx.impl->color_clip_samplewise = false;
        ctx.impl->depth_scene_written = false;
        ctx.impl->scene_depth_drawn = false;
        ctx.impl->depth_is_sample_rate_copy = false;
        ctx.impl->direct_guest_memory_used = false;
        auto *target = static_cast<MetalRenderTarget *>(ctx.current_render_target);
        require(target != nullptr, "Metal: scene without render target");
        const auto &surface = ctx.record.color_surface;
        if (impl->trace_textures && impl->traced_textures.size() < 4096) {
            const auto &ds = ctx.record.depth_stencil_surface;
            const auto detail = fmt::format("color={:#x} fmt={:#x} size={}x{} depth={:#x} stencil={:#x} depthfmt={:#x} load={} store={} depth_stride={} depth_type={:#x} multisample={} gamma={} downscale={} rt_native={}x{} scale={} custom_samples={} sample_locations={:#x} output_register_size={}",
                surface.data.address(), uint32_t(surface.colorFormat), surface.width, surface.height,
                ds.depth_data.address(), ds.stencil_data.address(), uint32_t(ds.get_format()), uint32_t(ds.force_load), uint32_t(ds.force_store),
                ds.get_stride(), uint32_t(ds.get_type()), uint32_t(target->multisample_mode), uint32_t(surface.gamma),
                uint32_t(surface.downscale), target->width, target->height, res_multiplier,
                target->custom_multisample_locations, target->multisample_locations, surface.outputRegisterSize);
            if (impl->traced_textures.insert(detail).second) LOG_INFO("Metal scene binding: {}", detail);
        }
        ctx.impl->guest_color = surface;
        const uint32_t samples=target->multisample_mode==SCE_GXM_MULTISAMPLE_NONE ? 1
            : target->multisample_mode==SCE_GXM_MULTISAMPLE_2X ? 2 : 4;
        require([impl->device->native_device() supportsTextureSampleCount:samples],"Metal: requested MSAA count is unavailable");
        ctx.impl->samples=samples; ctx.impl->sample_scale=res_multiplier;
        ctx.impl->expanded_color=samples>1 && surface.data && !surface.downscale;
        ctx.impl->custom_samples=samples>1 && target->custom_multisample_locations;
        for(uint32_t i=0;i<samples && ctx.impl->custom_samples;++i)
            ctx.impl->sample_positions[i]={float((target->multisample_locations>>(i*8))&15)/16,
                float((target->multisample_locations>>(i*8+4))&15)/16};
        const uint32_t color_width=surface.data ? uint32_t(surface.width*res_multiplier) : target->width;
        const uint32_t color_height=surface.data ? uint32_t(surface.height*res_multiplier) : target->height;
        // Plus scales guest viewport/scissor coordinates only when an explicit
        // downscale surface is at most half the base target on both axes.
        // Storage dimensions and native sample addressing retain their scale.
        ctx.impl->surface_downscale=surface.data && surface.downscale && color_width && color_height
                && uint64_t(target->width)>=uint64_t(color_width)*2
                && uint64_t(target->height)>=uint64_t(color_height)*2
            ? 0.5f : 1.f;
        require(!ctx.impl->expanded_color || (!(surface.width%(samples/2)) && !(surface.height%2)),"Metal: invalid expanded MSAA color extent");
        // Fractionally scaled expanded images may end inside a sample tile.
        // Keep the exact resolved extent and allocate one extra native pixel
        // for that partial tile; the conversion passes map its valid samples.
        const auto width=ctx.impl->expanded_color ? (color_width+samples/2-1)/(samples/2) : color_width;
        const auto height=ctx.impl->expanded_color ? (color_height+1)/2 : color_height;
        require(width && height, "Metal: empty render target");
        ctx.impl->width = width; ctx.impl->height = height;
        ctx.impl->raw_color = nil;
        ctx.impl->raw_render_color = nil;
        ctx.impl->raw_attachment = nil;
        ctx.impl->raw_clip_snapshot = nil;
        bool seed_raw_color = false;
        bool seed_raw_samples = false;
        if (surface.data) {
            auto &entry = impl->surfaces[surface.data.address()];
            const size_t watched_bytes = surface_memory_size(surface);
            const uint64_t watched_end = uint64_t(surface.data.address()) + watched_bytes;
            const auto pending_cpu_writes = watched_bytes && watched_end <= uint64_t(UINT32_MAX) - 4095
                    && is_valid_addr_range(mem, surface.data.address(), Address(watched_end))
                ? impl->surface_writes.capture(mem, surface.data.address(), watched_bytes) : SurfaceWriteStamp{};
            entry.scene_writes = {};
            entry.scene_render_area = {};
            entry.scene_macroblock_bounds = {};
            if (entry.guest.width != surface.width || entry.guest.height != surface.height
                || entry.guest.strideInPixels != surface.strideInPixels
                || entry.guest.colorFormat != surface.colorFormat || entry.guest.surfaceType != surface.surfaceType) {
                entry.written_tiles = {};
                entry.rendered_width = entry.rendered_height = 0;
            }
            ++entry.revision;
            entry.rgba8_casts.clear(); entry.word_casts.clear(); entry.word_rect_casts.clear(); entry.halfword_rect_casts.clear(); entry.byte_rect_casts.clear(); entry.small_rect_casts.clear(); entry.subrectangles.clear();
            auto format = color_format(surface.colorFormat, impl->device->native_device());
            if (surface.gamma) {
                // Vulkan uses an sRGB view only for RGBA8. For other render
                // formats it keeps the linear attachment and shader hint.
                if (format == MTLPixelFormatRGBA8Unorm) format = MTLPixelFormatRGBA8Unorm_sRGB;
                else LOG_WARN_ONCE("Metal: gamma correction requested for non-sRGB render format {}",uint32_t(format));
            }
            // A gamma interpretation change must preserve the stored bytes.
            if (entry.color && entry.color.width == color_width && entry.color.height == color_height
                && (entry.color.pixelFormat == MTLPixelFormatRGBA8Unorm || entry.color.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB)
                && (format == MTLPixelFormatRGBA8Unorm || format == MTLPixelFormatRGBA8Unorm_sRGB))
                entry.color = rgba8_gamma_view(entry.color, format == MTLPixelFormatRGBA8Unorm_sRGB);
            const bool new_color = !entry.color || entry.color.width != color_width || entry.color.height != color_height || entry.color.pixelFormat != format;
            if (new_color) {
                entry.has_word_offset_view = false;
                entry.written_tiles = {};
                entry.rendered_width = entry.rendered_height = 0;
                entry.raw_color = nil;
                entry.raw_color_invalidated = false;
                entry.raw_multisample_color = nil;
                entry.raw_multisample_dirty = true;
                entry.multisample_color = nil;
                entry.color = make_texture(*impl->device, format, color_width, color_height, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView | MTLTextureUsageShaderWrite);
                entry.cpu_snapshot.clear();
                if (gxm::get_base_format(surface.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16) {
                    std::vector<uint8_t> zero(size_t(color_width)*color_height*8);
                    [entry.color replaceRegion:MTLRegionMake2D(0,0,color_width,color_height)
                        mipmapLevel:0 withBytes:zero.data() bytesPerRow:size_t(color_width)*8];
                }
            }
            // Plus expands its target before recording the writeback extent.
            // Native MSAA stores a 1x2 (2x) or 2x2 (4x) guest sample grid.
            const uint32_t target_samples_x = ctx.impl->expanded_color ? samples / 2 : 1;
            const uint32_t target_samples_y = ctx.impl->expanded_color ? 2 : 1;
            entry.render_scale = res_multiplier;
            entry.publication_samples_x = target_samples_x;
            entry.publication_samples_y = target_samples_y;
            entry.rendered_width = std::max(entry.rendered_width,
                std::min<uint32_t>(surface.width, uint32_t(std::lround(target->width * target_samples_x / double(res_multiplier)))));
            entry.rendered_height = std::max(entry.rendered_height,
                std::min<uint32_t>(surface.height, uint32_t(std::lround(target->height * target_samples_y / double(res_multiplier)))));
            const bool rgb = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U8U8U8;
            const bool signed_rgb = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_S5S5U6;
            if (rgb && (new_color || gxm::get_base_format(entry.guest.colorFormat) != SCE_GXM_COLOR_BASE_FORMAT_U8U8U8)) {
                // RGB storage has implicit alpha one. Keep the backing alpha
                // fixed so destination-alpha blending and framebuffer fetch
                // observe the same value as texture sampling.
                std::vector<uint8_t> rgba(size_t(color_width)*color_height*4);
                if (!new_color) [entry.color getBytes:rgba.data() bytesPerRow:color_width*4
                    fromRegion:MTLRegionMake2D(0,0,color_width,color_height) mipmapLevel:0];
                for (size_t i=3;i<rgba.size();i+=4) rgba[i]=255;
                [entry.color replaceRegion:MTLRegionMake2D(0,0,color_width,color_height) mipmapLevel:0 withBytes:rgba.data() bytesPerRow:color_width*4];
                if (new_color && !disable_surface_sync) {
                    const size_t bytes=surface_memory_size(surface);
                    const uint64_t end=uint64_t(surface.data.address())+bytes;
                    if (bytes && end<=uint64_t(UINT32_MAX)-4095 && is_valid_addr_range(mem,surface.data.address(),Address(end))) {
                        const SurfaceMemoryRange all{0,bytes};
                        require(write_surface_storage(entry,surface,
                            {static_cast<const uint8_t *>(surface.data.get(mem)),bytes},{&all,1}),"Metal: RGB initial surface upload failed");
                    }
                }
            }
            if (signed_rgb && (new_color || gxm::get_base_format(entry.guest.colorFormat) != SCE_GXM_COLOR_BASE_FORMAT_S5S5U6)) {
                // No guest alpha channel: keep destination-alpha blending and
                // framebuffer reads at one even before the first CPU import.
                std::vector<uint16_t> rgba(size_t(color_width)*color_height*4);
                if (!new_color) [entry.color getBytes:rgba.data() bytesPerRow:color_width*8
                    fromRegion:MTLRegionMake2D(0,0,color_width,color_height) mipmapLevel:0];
                for (size_t i=3;i<rgba.size();i+=4) rgba[i]=0x3c00;
                [entry.color replaceRegion:MTLRegionMake2D(0,0,color_width,color_height)
                    mipmapLevel:0 withBytes:rgba.data() bytesPerRow:color_width*8];
            }
            // With synchronous surfaces guest memory is authoritative between scenes,
            // including CPU fills and transfers at an unchanged base address.
            if (!disable_surface_sync && res_multiplier > 0) {
                const size_t bytes=surface_memory_size(surface);
                const uint64_t end=uint64_t(surface.data.address())+bytes;
                if (bytes && end<=uint64_t(UINT32_MAX)-4095 && is_valid_addr_range(mem,surface.data.address(),Address(end))) {
                    const std::span<const uint8_t> cpu{static_cast<const uint8_t *>(surface.data.get(mem)),bytes};
                    const bool same_layout=entry.guest.width==surface.width && entry.guest.height==surface.height
                        && entry.guest.strideInPixels==surface.strideInPixels && entry.guest.colorFormat==surface.colorFormat
                        && entry.guest.surfaceType==surface.surfaceType;
                    std::vector<SurfaceMemoryRange> changes;
                    // A same-value store or write-restore still transfers
                    // ownership to the CPU. Import its full surface once.
                    if (entry.cpu_writes.changed() || !same_layout || entry.cpu_snapshot.size()!=bytes) changes.push_back({0,bytes});
                    else if (std::memcmp(cpu.data(), entry.cpu_snapshot.data(), bytes) != 0)
                        for (size_t at=0;at<bytes;) {
                            if (cpu[at]==entry.cpu_snapshot[at]) {++at;continue;}
                            const size_t first=at++;
                            while (at<bytes && cpu[at]!=entry.cpu_snapshot[at]) ++at;
                            changes.push_back({first,at-first});
                        }
                    if (!changes.empty()) {
                        require(write_surface_storage(entry,surface,cpu,changes),"Metal: CPU surface import failed");
                        // Patching only the edited bytes preserves every other
                        // sample. Reseeding from the resolved image would turn
                        // distinct edge samples into copies of their average.
                        if (samples>1 && !entry.multisample_dirty && entry.multisample_color
                            && entry.multisample_color.width==width && entry.multisample_color.height==height
                            && entry.multisample_color.sampleCount==samples) {
                            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                            require(impl->caster->patch_multisample(entry.multisample_color,surface,ctx.impl->sample_scale,cpu,changes),
                                "Metal: CPU surface import could not preserve multisample storage");
                        }
                        if (samples>1 && !entry.raw_multisample_dirty && entry.raw_multisample_color
                            && entry.raw_multisample_color.width==width && entry.raw_multisample_color.height==height
                            && entry.raw_multisample_color.sampleCount==samples) {
                            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                            require(impl->caster->patch_multisample(entry.raw_multisample_color,surface,ctx.impl->sample_scale,cpu,changes),
                                "Metal: CPU surface import could not preserve raw multisample storage");
                        }
                        entry.cpu_snapshot.assign(cpu.begin(),cpu.end());
                    }
                    // Capture happened before the import: a later write stays
                    // dirty instead of being acknowledged by this bind.
                    entry.cpu_writes = pending_cpu_writes;
                    // With a single native sample, an unchanged scene cannot
                    // alter the imported color image. Avoid a readback unless
                    // a draw writes color or guest bytes change meanwhile.
                    if (samples == 1 && entry.cpu_snapshot.size() == bytes)
                        ctx.impl->color_guest_current = true;
                }
            }
            if (new_color || samples==1) entry.multisample_dirty=true;
            if (new_color || samples==1) entry.raw_multisample_dirty=true;
            if (samples>1) {
                entry.multisample_scale=ctx.impl->sample_scale;
                if (entry.multisample_color && entry.multisample_color.width==width && entry.multisample_color.height==height
                    && entry.multisample_color.sampleCount==samples
                    && (entry.multisample_color.pixelFormat==MTLPixelFormatRGBA8Unorm || entry.multisample_color.pixelFormat==MTLPixelFormatRGBA8Unorm_sRGB)
                    && (format==MTLPixelFormatRGBA8Unorm || format==MTLPixelFormatRGBA8Unorm_sRGB))
                    entry.multisample_color=rgba8_gamma_view(entry.multisample_color,format==MTLPixelFormatRGBA8Unorm_sRGB);
                if (!entry.multisample_color || entry.multisample_color.width!=width || entry.multisample_color.height!=height
                    || entry.multisample_color.sampleCount!=samples || entry.multisample_color.pixelFormat!=format) {
                    entry.multisample_color=make_texture(*impl->device,format,width,height,
                        MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView,samples);
                    entry.multisample_dirty=true;
                }
                if (entry.multisample_dirty) {
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    impl->caster->seed_multisample(entry.color,entry.multisample_color,ctx.impl->sample_scale,ctx.impl->expanded_color,
                        ctx.impl->expanded_color ? surface.width : 0,ctx.impl->expanded_color ? surface.height : 0);
                    entry.multisample_dirty=false;
                }
            }
            ctx.impl->render_color=samples>1 ? entry.multisample_color : entry.color;
            if (gxm::get_base_format(surface.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16) {
                const bool new_raw_color=!entry.raw_color;
                if (new_raw_color) {
                    entry.raw_color=make_texture(*impl->device,format,color_width,color_height,
                        MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView|MTLTextureUsageShaderWrite);
                }
                if (new_raw_color || entry.raw_color_invalidated) {
                    seed_raw_color=true;
                    entry.raw_multisample_dirty=true;
                    entry.raw_color_invalidated=false;
                }
                ctx.impl->raw_color=entry.raw_color;
                ctx.impl->raw_render_color=entry.raw_color;
                if (samples>1) {
                    if (!entry.raw_multisample_color || entry.raw_multisample_color.width!=width
                        || entry.raw_multisample_color.height!=height || entry.raw_multisample_color.sampleCount!=samples) {
                        entry.raw_multisample_color=make_texture(*impl->device,format,width,height,
                            MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView,samples);
                        entry.raw_multisample_dirty=true;
                    }
                    ctx.impl->raw_render_color=entry.raw_multisample_color;
                    seed_raw_samples=entry.raw_multisample_dirty;
                }
                ctx.impl->raw_attachment=[ctx.impl->raw_render_color newTextureViewWithPixelFormat:MTLPixelFormatRGBA16Uint];
                require(ctx.impl->raw_attachment!=nil,"Metal: raw F16 attachment view failed");
            }
            entry.guest = surface;
            // Disabled surface synchronization does not acknowledge writes
            // to an existing image. New images/ranges get their own baseline.
            if (new_color || entry.cpu_writes.token != pending_cpu_writes.token)
                entry.cpu_writes = pending_cpu_writes;
            entry.last_frame_rendered = impl->frame_timestamp;
            const size_t guest_bytes = surface_memory_size(surface);
            const uint64_t guest_end = uint64_t(surface.data.address()) + guest_bytes;
            if (guest_bytes && guest_end <= uint64_t(UINT32_MAX) - 4095
                && is_valid_addr_range(mem, surface.data.address(), Address(guest_end))) {
                entry.cpu_hash = XXH3_64bits(surface.data.get(mem), guest_bytes);
                entry.cpu_hash_size = guest_bytes;
            } else entry.cpu_hash_size = 0;
            require(impl->next_color_binding_generation != 0, "Metal: color binding generation exhausted");
            entry.binding_generation = impl->next_color_binding_generation++;
            ctx.impl->color_binding_generation = entry.binding_generation;
            ctx.impl->color = entry.color;
        } else {
            // Vulkan binds the render target's transient RGBA8 color image
            // even when the guest provides no color surface. Metal also needs
            // a real attachment for fragment shaders using framebuffer fetch.
            if (!ctx.impl->transient_color || ctx.impl->transient_color.width!=width
                || ctx.impl->transient_color.height!=height || ctx.impl->transient_color.sampleCount!=samples)
                ctx.impl->transient_color=make_texture(*impl->device,MTLPixelFormatRGBA8Unorm,width,height,
                    MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead,samples);
            ctx.impl->render_color=ctx.impl->transient_color;
            ctx.impl->color=nil;
        }
        ctx.impl->transient_color_initialized=false;
        ctx.record.is_gamma_corrected = bool(surface.data && surface.gamma);
        const auto &ds = ctx.record.depth_stencil_surface;
        ctx.impl->guest_depth=ds;
        const uint32_t guest_depth_width=surface.data
            ? surface.width/(ctx.impl->expanded_color ? samples/2 : 1)
            : target->guest_width ? target->guest_width : uint32_t(std::lround(double(width)/res_multiplier));
        const uint32_t guest_depth_height=surface.data
            ? surface.height/(ctx.impl->expanded_color ? 2 : 1)
            : target->guest_height ? target->guest_height : uint32_t(std::lround(double(height)/res_multiplier));
        ctx.impl->depth_layout=res_multiplier>0
            ? depth_memory_layout(ds,guest_depth_width,guest_depth_height,target->multisample_mode):std::nullopt;
        ctx.impl->depth_scene_width=ctx.impl->depth_layout ? ctx.impl->depth_layout->width : 0;
        ctx.impl->depth_scene_height=ctx.impl->depth_layout ? ctx.impl->depth_layout->height : 0;
        if(!ctx.impl->depth_layout && !ds.disabled() && (ds.depth_data || ds.stencil_data) && (ds.force_load || ds.force_store))
            LOG_WARN_ONCE("Metal: guest depth memory encoding/scale is not implemented, format={:#x}",uint32_t(ds.get_format()));
        // Store flags control guest RAM publication, not GPU cache ownership.
        const bool backed_depth = !ds.disabled() && (ds.depth_data || ds.stencil_data);
        ctx.impl->depth_key = backed_depth ? std::make_pair(ds.depth_data.address(), ds.stencil_data.address()) : std::pair<Address,Address>{};
        id<MTLTexture> selected = backed_depth ? impl->depth_surfaces[ctx.impl->depth_key].texture : ctx.impl->transient_depth;
        bool new_depth = !selected || selected.width != width || selected.height != height || selected.sampleCount != samples;
        bool retained_larger_depth=false;
        bool resample_depth=false;
        if (backed_depth && selected) {
            const auto &entry = impl->depth_surfaces[ctx.impl->depth_key];
            const bool compatible=entry.guest.get_format()==ds.get_format() && entry.guest.get_type()==ds.get_type()
                && entry.guest.get_stride()==ds.get_stride() && entry.multisample==target->multisample_mode
                && selected.sampleCount==samples && entry.scale==ctx.impl->sample_scale;
            new_depth|=!compatible || entry.width!=guest_depth_width || entry.height!=guest_depth_height;
            const auto storage=compatible ? depth_memory_layout(ds,entry.width,entry.height,entry.multisample) : std::nullopt;
            const auto mapped=[&](auto pointer,size_t bytes) {
                return !bytes || (pointer && uint64_t(pointer.address())+bytes<=uint64_t(UINT32_MAX)-4095
                    && is_valid_addr_range(mem,pointer.address(),Address(uint64_t(pointer.address())+bytes)));
            };
            const bool storage_known=storage && entry.guest_snapshot_valid
                && (!storage->tiled || storage->stride%32==0)
                && entry.guest_depth_snapshot.size()==storage->depth_size
                && entry.guest_stencil_snapshot.size()==storage->stencil_size
                && mapped(ds.depth_data,storage->depth_size) && mapped(ds.stencil_data,storage->stencil_size);
            const bool needs_mask=ds.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M
                || ds.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8;
            const bool mask_known=!needs_mask || (entry.mask && entry.mask.width==selected.width
                && entry.mask.height==selected.height && entry.mask.sampleCount==samples
                && entry.mask.pixelFormat==MTLPixelFormatRGBA8Unorm);
            const uint32_t target_divisor=ctx.impl->surface_downscale==0.5f ? 2 : 1;
            // Plus returns a temporary scene view, leaving the full cached
            // image intact. Only use a source grid whose target/guest mapping
            // was recorded, and whose requested target matches this attachment.
            resample_depth=storage_known && mask_known && samples>1 && surface.data
                && surface.downscale && !ds.force_store && ctx.impl->depth_layout
                && target->width/target_divisor==width && target->height/target_divisor==height
                && ctx.impl->depth_layout->width==uint32_t(width/res_multiplier)*(samples/2)
                && ctx.impl->depth_layout->height==uint32_t(height/res_multiplier)*2
                && entry.sample_grid_width>=uint64_t(width)*(samples/2)
                && entry.sample_grid_height>=uint64_t(height)*2;
            if (resample_depth) {
                new_depth=false;
                retained_larger_depth=selected.width!=width || selected.height!=height
                    || entry.width!=guest_depth_width || entry.height!=guest_depth_height;
                ctx.impl->depth_layout=storage;
            }
            // Reuse a larger storage image only when its prefix has the same
            // native/guest coordinate map. Sample-rate resampling is separate.
            if (new_depth && compatible && ctx.impl->depth_layout && entry.guest_snapshot_valid
                && selected.width>=width && selected.height>=height
                && entry.width>=guest_depth_width && entry.height>=guest_depth_height
                && uint64_t(selected.width)*guest_depth_width==uint64_t(width)*entry.width
                && uint64_t(selected.height)*guest_depth_height==uint64_t(height)*entry.height
                && !(samples>1 && surface.downscale && !ds.force_store)) {
                // Absent-aspect background changes require a region history;
                // retain the existing exact-size path until that is available.
                if (storage_known
                    && (storage->depth_size || entry.snapshot_background_depth==ds.background_depth)
                    && (storage->stencil_size || (storage->packed && storage->depth_size)
                        || entry.snapshot_background_stencil==ds.stencil)) {
                    new_depth=false;retained_larger_depth=true;ctx.impl->depth_layout=storage;
                }
            }
        }
        bool discard_cached_depth=false;
        if (new_depth)
            selected = make_texture(*impl->device, MTLPixelFormatDepth32Float_Stencil8, width, height, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, samples);
        if (backed_depth) {
            auto &entry = impl->depth_surfaces[ctx.impl->depth_key];
            if (new_depth) {
                entry.snapshots.clear(); entry.subrectangles.clear(); entry.published=false;
                entry.mask=nil;
                entry.sample_rate_copy=nil;entry.sample_rate_mask=nil;
                entry.sample_grid_width=entry.sample_grid_height=0;
                // The expanded color attachment defines the sample grid even
                // when its extent differs from the render target's base extent.
                // Retain that grid for a later downscaled depth continuation.
                if (ctx.impl->expanded_color && ctx.impl->depth_layout
                    && ctx.impl->depth_layout->width==uint32_t(color_width/res_multiplier)
                    && ctx.impl->depth_layout->height==uint32_t(color_height/res_multiplier)) {
                    entry.sample_grid_width=color_width;entry.sample_grid_height=color_height;
                }
                entry.guest_snapshot_valid=false;
                entry.depth_content_stored=true;
            }
            // Plus keeps a no-store depth result only for a continuation with
            // the same color target. Stencil validity is independent of this.
            discard_cached_depth=!new_depth && ds.force_store && !entry.depth_content_stored
                && entry.last_scene_color_addr!=surface.data.address();
            entry.last_scene_color_addr=surface.data.address();
            entry.texture = selected;
            entry.guest = ds;
            if (!retained_larger_depth) {
                entry.width = guest_depth_width;
                entry.height = guest_depth_height;
            }
            entry.scale = ctx.impl->sample_scale;
            entry.multisample = target->multisample_mode;
            entry.last_attached_frame = impl->frame_timestamp;
        } else ctx.impl->transient_depth = selected;
        ctx.impl->depth = selected;
        ctx.impl->commands = scene_command_buffer(*impl->device);
        if (seed_raw_color) {
            // Follow prior submissions on the same queue. A CPU getBytes here
            // could race the previous scene's deferred color producer.
            auto blit=[ctx.impl->commands blitCommandEncoder];
            require(blit!=nil,"Metal: cannot seed raw F16 attachment");
            [blit copyFromTexture:ctx.impl->color sourceSlice:0 sourceLevel:0
                sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(color_width,color_height,1)
                toTexture:ctx.impl->raw_color destinationSlice:0 destinationLevel:0
                destinationOrigin:MTLOriginMake(0,0,0)];
            [blit endEncoding];
        }
        if (seed_raw_samples) {
            // The initial normal-to-raw copy, seed and scene draws must stay
            // ordered in this buffer; a separately submitted seed runs too soon.
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            impl->caster->seed_multisample(ctx.impl->raw_color,ctx.impl->raw_render_color,
                ctx.impl->sample_scale,ctx.impl->expanded_color,
                ctx.impl->expanded_color ? surface.width : 0,
                ctx.impl->expanded_color ? surface.height : 0,ctx.impl->commands);
            impl->surfaces.at(surface.data.address()).raw_multisample_dirty=false;
        }
        // Treat the descriptor's clip as a color-output boundary. Preserve
        // samples outside its inclusive rectangle before any scene draws;
        // restoring them leaves depth, stencil and visibility untouched.
        const bool clipped_color = surface.data && surface.clip_enabled
            && (surface.clip_x_min || surface.clip_y_min
                || uint32_t(surface.clip_x_max) + 1 < surface.width
                || uint32_t(surface.clip_y_max) + 1 < surface.height);
        const uint32_t x_group = samples / 2;
        const bool grouped_clip = !ctx.impl->expanded_color
            || (ctx.impl->sample_scale == 1.f
                && !(surface.clip_x_min % x_group)
                && !((uint32_t(surface.clip_x_max) + 1) % x_group)
                && !(surface.clip_y_min % 2)
                && !((uint32_t(surface.clip_y_max) + 1) % 2));
        if (clipped_color) {
            ctx.impl->color_clip_samplewise = !grouped_clip;
            ctx.impl->color_clip_snapshot = make_texture(*impl->device,
                ctx.impl->render_color.pixelFormat, uint32_t(ctx.impl->render_color.width),
                uint32_t(ctx.impl->render_color.height),
                MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, samples);
            auto blit = [ctx.impl->commands blitCommandEncoder];
            require(blit != nil, "Metal: cannot capture color clip samples");
            [blit copyFromTexture:ctx.impl->render_color sourceSlice:0 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(ctx.impl->render_color.width, ctx.impl->render_color.height, 1)
                toTexture:ctx.impl->color_clip_snapshot destinationSlice:0 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            if (ctx.impl->raw_render_color) {
                ctx.impl->raw_clip_snapshot=make_texture(*impl->device,ctx.impl->raw_render_color.pixelFormat,
                    width,height,MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView,samples);
                [blit copyFromTexture:ctx.impl->raw_render_color sourceSlice:0 sourceLevel:0
                    sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(width,height,1)
                    toTexture:ctx.impl->raw_clip_snapshot destinationSlice:0 destinationLevel:0
                    destinationOrigin:MTLOriginMake(0,0,0)];
            }
            [blit endEncoding];
        }
        const bool masked_depth=ds.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M
            || ds.get_format()==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8;
        auto &selected_mask=backed_depth && masked_depth
            ? impl->depth_surfaces.at(ctx.impl->depth_key).mask : ctx.impl->transient_mask;
        const uint32_t mask_width=backed_depth && masked_depth ? uint32_t(selected.width) : width;
        const uint32_t mask_height=backed_depth && masked_depth ? uint32_t(selected.height) : height;
        const bool new_mask=!selected_mask || selected_mask.width!=mask_width || selected_mask.height!=mask_height
            || selected_mask.sampleCount!=samples || selected_mask.pixelFormat!=MTLPixelFormatRGBA8Unorm;
        if (new_mask)
            selected_mask=make_texture(*impl->device,MTLPixelFormatRGBA8Unorm,mask_width,mask_height,
                MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead,samples);
        if (ctx.impl->mask!=selected_mask) {
            ctx.impl->mask=selected_mask;
            ctx.impl->mask_constant_valid=false;
        }
        bool loaded_mask=false;
        bool loaded_depth=false, retained_native_depth=false;
        bool clear_background_depth=false, clear_background_stencil=false;
        if(backed_depth && ctx.impl->depth_layout) {
            const auto &layout=*ctx.impl->depth_layout;
            auto &entry=impl->depth_surfaces[ctx.impl->depth_key];
            const auto mapped=[&](auto pointer,size_t bytes) {
                const uint64_t end=uint64_t(pointer.address())+bytes;
                return !bytes || (pointer && end<=uint64_t(UINT32_MAX)-4095
                    && is_valid_addr_range(mem,pointer.address(),Address(end)));
            };
            // No-load/no-store pointers need not be readable. Track a baseline
            // only if available; an explicit load retains its strict validation.
            if (ds.force_load || (mapped(ds.depth_data,layout.depth_size) && mapped(ds.stencil_data,layout.stencil_size))) {
                const auto [depth,stencil]=depth_memory_spans(mem,ds,layout);
                const auto unchanged=[](const std::vector<uint8_t> &saved,std::span<const uint8_t> current) {
                    return saved.size()==current.size() && std::equal(saved.begin(),saved.end(),current.begin());
                };
                // Unchanged RAM can describe either a published image or the
                // baseline beneath newer, intentionally un-published GPU data.
                const bool known_baseline=entry.guest_snapshot_valid && !new_depth
                    && entry.guest_depth_snapshot.size()==depth.size() && entry.guest_stencil_snapshot.size()==stencil.size();
                const bool reuse=known_baseline
                    && unchanged(entry.guest_depth_snapshot,depth) && unchanged(entry.guest_stencil_snapshot,stencil);
                if ((ds.force_load || retained_larger_depth || resample_depth) && (!depth.empty() || !stencil.empty())) {
                    // Background values supply only aspects absent from guest
                    // memory. Changing them must not reload another aspect's
                    // unchanged RAM over a newer GPU result.
                    clear_background_depth=known_baseline && !layout.depth_size
                        && entry.snapshot_background_depth!=ds.background_depth;
                    clear_background_stencil=known_baseline && !layout.stencil_size && !(layout.packed && layout.depth_size)
                        && entry.snapshot_background_stencil!=ds.stencil;
                    if (!reuse) {
                        if(!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                        if (known_baseline) {
                            // A newly allocated mask has no retained pixels to
                            // preserve. Initialize it before applying byte edits.
                            if (masked_depth && new_mask && !depth.empty()) {
                                const bool initialized_mask=impl->caster->load_mask_memory(selected_mask,ds,layout,
                                    ctx.impl->sample_scale,depth,ctx.impl->commands);
                                require(initialized_mask,"Metal: cannot initialize retained depth mask");
                                loaded_mask=ds.force_load;
                                ctx.impl->mask_constant_valid=false;
                            }
                            if (discard_cached_depth) {
                                // Clear invalid no-store depth before patching
                                // CPU bytes, not afterwards. Stencil survives.
                                impl->caster->clear_depth_region(selected,{0,0,width,height},ds.background_depth,ds.stencil,
                                    {true,false,false},ctx.impl->commands,ctx.impl->custom_samples ? ctx.impl->sample_positions.data() : nullptr);
                                ctx.impl->depth_written=true;
                                discard_cached_depth=false;clear_background_depth=false;
                                entry.snapshots.clear();entry.subrectangles.clear();entry.published=false;
                            }
                            const auto depth_changes=changed_memory_ranges(depth,entry.guest_depth_snapshot);
                            const auto stencil_changes=changed_memory_ranges(stencil,entry.guest_stencil_snapshot);
                            DepthMemoryWrite written;
                            require(impl->caster->patch_depth_memory(selected,ds,layout,ctx.impl->sample_scale,
                                depth,stencil,depth_changes,stencil_changes,selected_mask,ctx.impl->commands,&written),
                                "Metal: cannot patch retained guest depth/stencil storage");
                            ctx.impl->depth_written|=written.depth || written.stencil || written.mask;
                            depth_memory_changed(*this,entry,written);
                        } else {
                            require(impl->caster->load_depth_memory(selected,ds,layout,ctx.impl->sample_scale,depth,stencil,
                                ctx.impl->commands),
                                "Metal: cannot load guest depth/stencil storage");
                            ctx.impl->depth_written=true;
                            entry.snapshots.clear();entry.subrectangles.clear();
                            entry.depth_content_stored=true;
                            entry.published=false;
                            // A complete import replaces the stale cache.
                            discard_cached_depth=false;
                        }
                    }
                    retained_native_depth=ds.force_load && known_baseline;
                    loaded_depth=ds.force_load;
                }
                entry.guest_depth_snapshot.assign(depth.begin(),depth.end());
                entry.guest_stencil_snapshot.assign(stencil.begin(),stencil.end());
                if (!resample_depth) {
                    entry.snapshot_background_depth=ds.background_depth;
                    entry.snapshot_background_stencil=ds.stencil;
                }
                entry.guest_snapshot_valid=!depth.empty() || !stencil.empty();
            } else entry.guest_snapshot_valid=false;
        }
        ctx.impl->scene_active=true;
        ctx.impl->macroblock_last_x = ctx.impl->macroblock_last_y = ~0u;
        ctx.impl->macroblock_visited = 0;
        ctx.impl->macroblock_ignore = false;
        loaded_mask|=masked_depth && ds.depth_data && retained_native_depth && !new_mask;
        if(!loaded_mask && masked_depth && ds.force_load && ctx.impl->depth_layout && ds.depth_data) {
            const auto [depth,stencil]=depth_memory_spans(mem,ds,*ctx.impl->depth_layout);
            if(!depth.empty()) {
                if(!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                loaded_mask=impl->caster->load_mask_memory(ctx.impl->mask,ds,*ctx.impl->depth_layout,
                    ctx.impl->sample_scale,depth,ctx.impl->commands);
                require(loaded_mask,"Metal: cannot load guest depth mask storage");
                ctx.impl->mask_constant_valid=false;
            }
        }
        if (resample_depth) {
            auto &entry=impl->depth_surfaces.at(ctx.impl->depth_key);
            const auto allocate_view=[&](id<MTLTexture> __strong &texture,MTLPixelFormat format) {
                if (!texture || texture.width!=width || texture.height!=height || texture.sampleCount!=samples)
                    texture=make_texture(*impl->device,format,width,height,
                        MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView,samples);
            };
            allocate_view(entry.sample_rate_copy,MTLPixelFormatDepth32Float_Stencil8);
            if (masked_depth) allocate_view(entry.sample_rate_mask,MTLPixelFormatRGBA8Unorm);
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->resample_depth(selected,entry.sample_rate_copy,
                entry.sample_grid_width,entry.sample_grid_height,ctx.impl->depth_layout->width,ctx.impl->depth_layout->height,
                ctx.impl->commands,masked_depth ? selected_mask : nil,masked_depth ? entry.sample_rate_mask : nil,
                ctx.impl->custom_samples ? ctx.impl->sample_positions.data() : nullptr),
                "Metal: cannot reconstruct sample-rate depth view");
            selected=entry.sample_rate_copy;
            ctx.impl->depth=selected;
            ctx.impl->depth_is_sample_rate_copy=true;
            ctx.impl->depth_written=true;
            if (masked_depth) {
                ctx.impl->mask=entry.sample_rate_mask;
                ctx.impl->mask_constant_valid=false;
            }
        }
        const bool clear_mask = ds.disabled() || ds.mask;
        if(!loaded_mask && (!ctx.impl->mask_constant_valid || ctx.impl->mask_constant_value != clear_mask)) {
            const bool partial_mask=ctx.impl->mask.width!=width || ctx.impl->mask.height!=height;
            if (partial_mask) {
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                impl->caster->clear_depth_region(ctx.impl->mask,{0,0,width,height},float(clear_mask),0,
                    {false,false,true},ctx.impl->commands,ctx.impl->custom_samples ? ctx.impl->sample_positions.data() : nullptr);
                // The retained border need not have this constant value.
                ctx.impl->mask_constant_valid=false;
            } else {
                auto mask_pass=[MTLRenderPassDescriptor renderPassDescriptor];
                mask_pass.colorAttachments[0].texture=ctx.impl->mask;
                mask_pass.colorAttachments[0].loadAction=MTLLoadActionClear;
                // Disabled-surface initialization can request mask one.
                mask_pass.colorAttachments[0].clearColor=clear_mask
                    ? MTLClearColorMake(1,1,1,1) : MTLClearColorMake(0,0,0,0);
                mask_pass.colorAttachments[0].storeAction=MTLStoreActionStore;
                auto mask_clear=[ctx.impl->commands renderCommandEncoderWithDescriptor:mask_pass];
                require(mask_clear!=nil,"Metal: cannot initialize scene mask");
                [mask_clear endEncoding];
                ctx.impl->mask_constant_valid=true;
                ctx.impl->mask_constant_value=clear_mask;
            }
        }
        const bool clear_both=!loaded_depth && (new_depth || !ds.force_load);
        bool clear_depth=clear_both || discard_cached_depth || clear_background_depth;
        bool clear_stencil=clear_both || clear_background_stencil;
        if (retained_larger_depth && (clear_depth || clear_stencil)) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            impl->caster->clear_depth_region(selected,{0,0,width,height},ds.background_depth,ds.stencil,
                {clear_depth,clear_stencil,false},ctx.impl->commands,ctx.impl->custom_samples ? ctx.impl->sample_positions.data() : nullptr);
            ctx.impl->depth_written=true;
            clear_depth=false;clear_stencil=false;
        }
        begin_pass(ctx,false,clear_depth,clear_stencil);
    }
}
void set_uniform_buffer(MetalContext &ctx, const ShaderProgram &program, bool vertex, int block, uint32_t size, const uint8_t *data, Address address) {
    const auto offset = program.uniform_buffer_data_offsets.at(block);
    if (offset == uint32_t(-1)) return;
    const auto count = std::min<size_t>(size, program.uniform_buffer_sizes.at(block) * 4ull);
    require(data != nullptr && count, "Metal: empty uniform binding");
    ctx.uniforms[vertex ? 0 : 1].at(block) = {const_cast<uint8_t *>(data), count, address};
}
void set_viewport(MetalContext &ctx, float x, float y, float sx, float sy) {
    ctx.record.viewport_flip[0] = sx < 0 ? -1.0f : 1.0f;
    ctx.viewport = {x - std::abs(sx), y - std::abs(sy), 2 * std::abs(sx), 2 * std::abs(sy)};
}

struct MetalTextureCache::Impl {
    MetalState &state;
    std::array<id<MTLTexture>, TextureCacheSize> textures{};
    std::array<bool, TextureCacheSize> placeholders{};
    std::array<id<MTLTexture>, 2> invalid_source_placeholders{};
    id<MTLTexture> invalid_source_binding = nil;
    std::array<bool, TextureCacheSize> expanded_packed{};
    size_t current = 0;
    uint32_t width = 0, height = 0, mips = 1;
    bool cube = false;
    uint32_t gamma = 0;
    std::array<ImportedTextureView, TextureCacheSize> imported_views{};
    std::map<std::tuple<uint32_t,uint32_t,bool>,astcenc_context *> astc_contexts;
    SceGxmTexture source{};
    std::filesystem::path dump_dir = std::getenv("VITA3K_METAL_DUMP_TEXTURE_DIR") ? std::getenv("VITA3K_METAL_DUMP_TEXTURE_DIR") : "";
    uint32_t dump_limit = std::getenv("VITA3K_METAL_DUMP_TEXTURE_LIMIT")
        ? uint32_t(std::min<unsigned long>(std::strtoul(std::getenv("VITA3K_METAL_DUMP_TEXTURE_LIMIT"), nullptr, 10), 4096)) : 64;
    std::set<std::string> dumped_textures;
    explicit Impl(MetalState &state) : state(state) {}
    ~Impl() {
        for (const auto &[key,context]:astc_contexts) astcenc_context_free(context);
    }
};
static id<MTLTexture> sampled_placeholder(Device &device, bool cube) {
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:1 height:1 mipmapped:NO];
    desc.textureType = cube ? MTLTextureTypeCube : MTLTextureType2D;
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    id<MTLTexture> result = [device.native_device() newTextureWithDescriptor:desc];
    require(result != nil, "Metal: cannot allocate 1x1 fallback texture");
    const uint8_t magenta[4] = {255, 0, 255, 255};
    for (uint32_t side = 0; side < (cube ? 6u : 1u); ++side)
        [result replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:side
            withBytes:magenta bytesPerRow:4 bytesPerImage:4];
    return result;
}
MetalTextureCache::MetalTextureCache(MetalState &state) : impl(std::make_unique<Impl>(state)) {
    support_e5rgb9 = true;
    support_dxt_software_import = true;
    support_astc_software_import = true;
}
MetalTextureCache::~MetalTextureCache() = default;
void MetalTextureCache::cache_and_bind_image(const SceGxmTexture &texture, MemState &mem) {
    impl->invalid_source_binding = nil;
    const SceGxmTexture image = texture_image_descriptor(texture);
    const uint64_t address = uint64_t(image.data_addr) << 2;
    const size_t bytes = address ? texture_storage_size(image) : 0;
    const uint64_t end = address + bytes;
    // A queued bind can outlive its guest allocation. Keep this fallback
    // separate from cache entries so an invalid bind cannot reuse or replace
    // a valid image that a later draw still needs.
    if (!address || !bytes || end > uint64_t(UINT32_MAX) - 4095
        || !is_valid_addr_range(mem, Address(address), Address(end))) {
        LOG_WARN_ONCE("Metal: texture source at 0x{:08X} ({} bytes) is not allocated", address, bytes);
        const bool cube = image.texture_type() == SCE_GXM_TEXTURE_CUBE
            || image.texture_type() == SCE_GXM_TEXTURE_CUBE_ARBITRARY;
        auto &placeholder = impl->invalid_source_placeholders[cube ? 1 : 0];
        if (!placeholder) placeholder = sampled_placeholder(*impl->state.impl->device, cube);
        impl->invalid_source_binding = placeholder;
        return;
    }
    TextureCache::cache_and_bind_texture(image, mem);
}
id<MTLTexture> current_texture(const MetalTextureCache &cache) {
    return cache.impl->invalid_source_binding ? cache.impl->invalid_source_binding
        : cache.impl->textures[cache.impl->current];
}
static std::optional<SceGxmColorFormat> repacked_u2_color(SceGxmTextureFormat format) {
    const uint32_t mode = (uint32_t(format) & SCE_GXM_TEXTURE_SWIZZLE_MASK) >> 12;
    if (gxm::get_base_format(format) != SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10
        || !(mode & 2)) return std::nullopt;
    return SceGxmColorFormat(uint32_t(SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10)
        | ((mode & 3) << 20));
}
id<MTLTexture> current_texture_view(const MetalTextureCache &cache, SceGxmTextureFormat format) {
    if (cache.impl->invalid_source_binding) return cache.impl->invalid_source_binding;
    if (cache.impl->placeholders[cache.impl->current]) return current_texture(cache);
    const auto &view = cache.impl->imported_views[cache.impl->current];
    const auto color = view.active ? std::nullopt : repacked_u2_color(format);
    return sampling_view(current_texture(cache), format, color ? &*color : nullptr,
        view.active ? &view : nullptr);
}
void MetalTextureCache::select(size_t index, const SceGxmTexture &texture) {
    require(index < TextureCacheSize, "Metal: texture cache index out of bounds");
    impl->invalid_source_binding = nil;
    impl->current = index;
    impl->source = texture;
    impl->width = gxm::get_width(texture); impl->height = gxm::get_height(texture);
    impl->mips = renderer::texture::get_upload_mip(texture.true_mip_count(), impl->width, impl->height);
    impl->cube = texture.texture_type() == SCE_GXM_TEXTURE_CUBE || texture.texture_type() == SCE_GXM_TEXTURE_CUBE_ARBITRARY;
    impl->gamma = texture.gamma_mode;
    // Partial RG gamma cannot be represented by a compressed sRGB view.
    // Keep the shared CPU expansion for that guest-only mode.
    support_dxt = impl->state.impl->device->native_device().supportsBCTextureCompression
        && (importing_texture || texture.gamma_mode != 3);
    if (current_info && current_info->is_imported) {
        // An unchanged replacement skips import_configure_impl on reupload.
        // Keep its dimensions and color space instead of the guest descriptor's.
        impl->width = current_info->width;
        impl->height = current_info->height;
        impl->mips = current_info->mip_count;
        impl->gamma = current_info->is_srgb
            ? (current_info->format == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8 ? 3 : 1) : 0;
    }
}
void MetalTextureCache::configure_texture(const SceGxmTexture &) {
    impl->textures[impl->current] = nil;
    impl->placeholders[impl->current] = false;
    impl->expanded_packed[impl->current] = false;
    impl->imported_views[impl->current] = {};
    impl->width = gxm::get_width(impl->source);
    impl->height = gxm::get_height(impl->source);
    impl->mips = renderer::texture::get_upload_mip(impl->source.true_mip_count(), impl->width, impl->height);
    impl->gamma = impl->source.gamma_mode;
}
uint32_t MetalTextureCache::texture_size_to_protect(const SceGxmTexture &texture, uint32_t first_mip_size) const {
    const auto type = texture.texture_type();
    const bool cube = type == SCE_GXM_TEXTURE_CUBE || type == SCE_GXM_TEXTURE_CUBE_ARBITRARY;
    if (!cube && texture.true_mip_count() <= 1) return first_mip_size;
    const size_t bytes = texture_storage_size(texture);
    return bytes <= UINT32_MAX ? uint32_t(bytes) : first_mip_size;
}
uint64_t MetalTextureCache::additional_texture_hash(const SceGxmTexture &texture, const MemState &mem,
    bool storage_protected) const {
    const auto type = texture.texture_type();
    const bool cube = type == SCE_GXM_TEXTURE_CUBE || type == SCE_GXM_TEXTURE_CUBE_ARBITRARY;
    if (!texture.data_addr || (!cube && texture.true_mip_count() <= 1)) return 0;
    const size_t bytes = texture_storage_size(texture);
    const Address address = texture.data_addr << 2;
    const uint64_t end = uint64_t(address)+bytes;
    require(bytes && end <= uint64_t(UINT32_MAX)-4095 && is_valid_addr_range(mem,address,Address(end)),
        "Metal: texture mip/face storage extends beyond mapped guest memory");
    // Page protection spans the complete mip chain and every cube face.
    if (storage_protected && !import_textures && !export_textures) return 0;
    // The shared cache already hashes the contiguous first mip on each bind.
    // Check only the remaining storage here, including all other cube faces.
    // Replacement/export hashes omit padding, so retain the full check there.
    const size_t covered = import_textures || export_textures ? 0
        : std::min(bytes, size_t(gxm::texture_size_first_mip(texture))) & ~size_t(3);
    if (covered == bytes) return 0;
    auto tail = texture;
    tail.data_addr += covered / 4;
    return renderer::texture::hash_texture_data(tail,uint32_t(bytes-covered),mem);
}
void MetalTextureCache::import_configure_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height, bool srgb, uint16_t components, uint16_t mips, bool swap_rb) {
    impl->width = width; impl->height = height; impl->mips = mips;
    impl->imported_views[impl->current] = {format, components, true, swap_rb};
    // Replacement metadata requests sRGB for all stored color channels,
    // independently of the guest descriptor's partial-channel gamma mode.
    impl->gamma = srgb ? (format == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8 ? 3 : 1) : 0;
    impl->textures[impl->current] = nil;
    impl->placeholders[impl->current] = false;
    impl->expanded_packed[impl->current] = false;
}
void MetalTextureCache::upload_texture_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height,
    uint32_t mip, const void *pixels, int face, uint32_t stride) {
    const uint32_t slice = face > 0 ? face - 1 : 0;
    if (!mip && !slice) impl->placeholders[impl->current] = false;
    if (impl->placeholders[impl->current]) return;
    const auto placeholder = [&](const char *reason) {
        LOG_WARN_ONCE("Metal: substituting a 1x1 texture for invalid or unsupported image ({})", reason);
        impl->textures[impl->current] = sampled_placeholder(*impl->state.impl->device, impl->cube);
        impl->placeholders[impl->current] = true;
    };
    constexpr uint32_t max_dim = 16384;
    const uint32_t allowed_mips = std::bit_width(std::max(impl->width, impl->height));
    if (!impl->width || !impl->height || impl->width > max_dim || impl->height > max_dim
        || (impl->cube && impl->width != impl->height)
        || !impl->mips || impl->mips > allowed_mips) {
        placeholder("dimensions or mip count");
        return;
    }
    if (!width || !height || width > impl->width || height > impl->height
        || mip >= impl->mips || slice >= (impl->cube ? 6u : 1u)) {
        if (!mip && !slice) placeholder("upload extent or slice");
        return;
    }
    MTLPixelFormat native;
    uint32_t bytes;
    uint32_t block_bytes = 0;
    uint32_t block_width = 4, block_height = 4;
    switch (format) {
#define TEX(gxm, mtl, size) case SCE_GXM_TEXTURE_BASE_FORMAT_##gxm: native = MTLPixelFormat##mtl; bytes = size; break
    TEX(U8, R8Unorm, 1); TEX(S8, R8Snorm, 1);
    TEX(U8U8, RG8Unorm, 2); TEX(S8S8, RG8Snorm, 2);
    TEX(U8U8U8, RGBA8Unorm, 4); TEX(S8S8S8, RGBA8Snorm, 4);
    TEX(U8U8U8U8, RGBA8Unorm, 4); TEX(S8S8S8S8, RGBA8Snorm, 4);
    TEX(U16, R16Unorm, 2); TEX(S16, R16Snorm, 2); TEX(F16, R16Float, 2);
    TEX(U16U16, RG16Unorm, 4); TEX(S16S16, RG16Snorm, 4); TEX(F16F16, RG16Float, 4);
    TEX(U16U16U16U16, RGBA16Unorm, 8); TEX(S16S16S16S16, RGBA16Snorm, 8); TEX(F16F16F16F16, RGBA16Float, 8);
    TEX(U32, R32Uint, 4); TEX(S32, R32Sint, 4); TEX(F32, R32Float, 4); TEX(F32M, R32Float, 4);
    TEX(U32U32, RG32Uint, 8); TEX(F32F32, RG32Float, 8); TEX(F11F11F10, RG11B10Float, 4);
    TEX(U5U6U5, B5G6R5Unorm, 2);
    TEX(U4U4U4U4, ABGR4Unorm, 2);
    TEX(SE5M9M9M9, RGB9E5Float, 4);
    TEX(U2U10U10U10, BGR10A2Unorm, 4);
    case SCE_GXM_TEXTURE_BASE_FORMAT_UBC1: native=MTLPixelFormatBC1_RGBA; block_bytes=8; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_UBC2: native=MTLPixelFormatBC2_RGBA; block_bytes=16; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_UBC3: native=MTLPixelFormatBC3_RGBA; block_bytes=16; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_UBC4: native=MTLPixelFormatBC4_RUnorm; block_bytes=8; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_SBC4: native=MTLPixelFormatBC4_RSnorm; block_bytes=8; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_UBC5: native=MTLPixelFormatBC5_RGUnorm; block_bytes=16; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_SBC5: native=MTLPixelFormatBC5_RGSnorm; block_bytes=16; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_UBC6H: native=MTLPixelFormatBC6H_RGBUfloat; block_bytes=16; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_SBC6H: native=MTLPixelFormatBC6H_RGBFloat; block_bytes=16; bytes=0; break;
    case SCE_GXM_TEXTURE_BASE_FORMAT_UBC7: native=MTLPixelFormatBC7_RGBAUnorm; block_bytes=16; bytes=0; break;
#define ASTC_FMT(b_x, b_y) case SCE_GXM_TEXTURE_BASE_FORMAT_ASTC##b_x##x##b_y: \
        native=MTLPixelFormatASTC_##b_x##x##b_y##_LDR; block_bytes=16; \
        block_width=b_x; block_height=b_y; bytes=0; break;
#include "../texture/astc_formats.inc"
#undef ASTC_FMT
    case SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5:
        require(impl->source.swizzle_format < 8,"Metal: unsupported 5:5:5:1 texture channel mapping");
        native = impl->imported_views[impl->current].active ? MTLPixelFormatBGR5A1Unorm
            : (impl->source.swizzle_format & 2) ? MTLPixelFormatA1BGR5Unorm : MTLPixelFormatBGR5A1Unorm;
        bytes = 2;
        break;
#undef TEX
    default:
        placeholder("decoded format");
        return;
    }
    std::vector<uint8_t> decoded_astc;
    if (renderer::texture::is_astc_format(format) && !support_astc) {
        const auto [block_x,block_y]=gxm::get_block_size(format);
        const auto key=std::tuple{block_x,block_y,impl->gamma==1};
        auto &context=impl->astc_contexts[key];
        if (!context) {
            astcenc_config config{};
            const auto profile=impl->gamma==1 ? ASTCENC_PRF_LDR_SRGB : ASTCENC_PRF_LDR;
            require(astcenc_config_init(profile,block_x,block_y,1,ASTCENC_PRE_FASTEST,
                ASTCENC_FLG_DECOMPRESS_ONLY,&config)==ASTCENC_SUCCESS,
                "Metal: ASTC decoder configuration failed");
            require(astcenc_context_alloc(&config,1,&context,nullptr)==ASTCENC_SUCCESS,
                "Metal: ASTC decoder allocation failed");
        }
        const size_t blocks_x=(width+block_x-1)/block_x;
        const size_t blocks_y=(height+block_y-1)/block_y;
        const size_t source_blocks_x=(std::max(stride,width)+block_x-1)/block_x;
        std::vector<uint8_t> compressed(blocks_x*blocks_y*16);
        const auto *source=static_cast<const uint8_t *>(pixels);
        for (size_t row=0;row<blocks_y;++row)
            std::memcpy(compressed.data()+row*blocks_x*16,
                source+row*source_blocks_x*16,blocks_x*16);
        decoded_astc.resize(size_t(width)*height*4);
        void *slice=decoded_astc.data();
        astcenc_image image{width,height,1,ASTCENC_TYPE_U8,&slice};
        const astcenc_swizzle swizzle{ASTCENC_SWZ_R,ASTCENC_SWZ_G,ASTCENC_SWZ_B,ASTCENC_SWZ_A};
        const auto result=astcenc_decompress_image(context,compressed.data(),compressed.size(),
            &image,&swizzle,0);
        require(result==ASTCENC_SUCCESS,"Metal: ASTC decompression failed");
        pixels=decoded_astc.data();
        stride=width;
        native=MTLPixelFormatRGBA8Unorm;
        block_bytes=0;
        bytes=4;
    }
    if (!impl->dump_dir.empty() && !mip && format == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 && impl->dumped_textures.size() < impl->dump_limit) {
        auto name = fmt::format("{:08x}-{:08x}-{}x{}", impl->source.data_addr << 2, uint32_t(gxm::get_format(impl->source)), width, height);
        if (slice) name += fmt::format("-face{}",slice);
        if (impl->dumped_textures.insert(name).second) {
            std::filesystem::create_directories(impl->dump_dir);
            std::ofstream rgba(impl->dump_dir / (name + ".rgba"), std::ios::binary);
            std::ofstream preview(impl->dump_dir / (name + ".ppm"), std::ios::binary);
            preview << "P6\n" << width << " " << height << "\n255\n";
            const auto *source = static_cast<const uint8_t *>(pixels);
            const size_t pitch = size_t(std::max(stride,width))*4;
            for (uint32_t y=0;y<height;++y) {
                rgba.write(reinterpret_cast<const char *>(source+y*pitch),width*4);
                for (uint32_t x=0;x<width;++x)
                    preview.write(reinterpret_cast<const char *>(source+y*pitch+x*4),3);
            }
            require(bool(rgba) && bool(preview), "Metal: diagnostic texture write failed");
            LOG_INFO("Metal decoded texture saved: {}", (impl->dump_dir / name).string());
        }
    }
    std::vector<uint8_t> expanded_rgb;
    if (format == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8 || format == SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8) {
        const auto *source = static_cast<const uint8_t *>(pixels);
        const size_t source_pitch = size_t(std::max(stride, width)) * 3;
        expanded_rgb.resize(size_t(width) * height * 4);
        const uint8_t alpha = format == SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8 ? 127 : 255;
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                const auto *input = source + y * source_pitch + x * 3;
                auto *output = expanded_rgb.data() + (size_t(y) * width + x) * 4;
                std::memcpy(output, input, 3);
                output[3] = alpha;
            }
        pixels = expanded_rgb.data();
        stride = width;
    }
    std::vector<uint32_t> repacked_10;
    if (format == SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10
        && !impl->imported_views[impl->current].active && (impl->source.swizzle_format & 2)) {
        const uint32_t mode = impl->source.swizzle_format & 3;
        const auto *source = static_cast<const uint8_t *>(pixels);
        const size_t source_pitch = size_t(std::max(stride, width)) * 4;
        repacked_10.resize(size_t(width) * height);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                uint32_t guest;
                std::memcpy(&guest, source + y * source_pitch + x * 4, 4);
                repacked_10[size_t(y) * width + x] = packed_10_native_word(guest, mode);
            }
        pixels = repacked_10.data();
        stride = width;
    }
    std::vector<uint16_t> linear_pixels;
    if (impl->gamma == 1 && block_bytes) {
        if (native == MTLPixelFormatBC1_RGBA) native = MTLPixelFormatBC1_RGBA_sRGB;
        if (native == MTLPixelFormatBC2_RGBA) native = MTLPixelFormatBC2_RGBA_sRGB;
        if (native == MTLPixelFormatBC3_RGBA) native = MTLPixelFormatBC3_RGBA_sRGB;
        if (native == MTLPixelFormatBC7_RGBAUnorm) native = MTLPixelFormatBC7_RGBAUnorm_sRGB;
#define ASTC_FMT(b_x, b_y) if (native == MTLPixelFormatASTC_##b_x##x##b_y##_LDR) \
        native = MTLPixelFormatASTC_##b_x##x##b_y##_sRGB;
#include "../texture/astc_formats.inc"
#undef ASTC_FMT
    }
    if (impl->gamma == 1 && native == MTLPixelFormatRGBA8Unorm) {
        native = MTLPixelFormatRGBA8Unorm_sRGB;
    } else if (impl->gamma && (native == MTLPixelFormatR8Unorm || native == MTLPixelFormatRG8Unorm || native == MTLPixelFormatRGBA8Unorm)) {
        require(impl->gamma == 1 || impl->gamma == 3, "Metal: invalid texture gamma mode");
        // Metal has no RG-only sRGB RGBA view. Decode stored channels before
        // filtering and guest swizzling, using filterable half-float storage.
        // This also avoids depending on Apple-only R8/RG8 sRGB formats.
        static const auto tables = [] {
            std::array<std::array<uint16_t, 256>, 2> result{};
            std::array<float, 256> encoded{}, decoded{};
            for (size_t i = 0; i < 256; ++i) {
                const float value = float(i) / 255.0f;
                encoded[i] = value;
                decoded[i] = value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
            }
            float_to_half(encoded.data(), result[0].data(), 256);
            float_to_half(decoded.data(), result[1].data(), 256);
            return result;
        }();
        const uint32_t components = bytes;
        const uint32_t gamma_components = impl->gamma == 3 ? 2 : 1;
        const size_t source_pitch = size_t(std::max(stride, width)) * components;
        const auto *source = static_cast<const uint8_t *>(pixels);
        linear_pixels.resize(size_t(width) * height * components);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                for (uint32_t c = 0; c < components; ++c)
                    linear_pixels[(size_t(y) * width + x) * components + c] = tables[c < gamma_components][source[y * source_pitch + x * components + c]];
        native = components == 1 ? MTLPixelFormatR16Float : components == 2 ? MTLPixelFormatRG16Float : MTLPixelFormatRGBA16Float;
        bytes = components * 2;
        pixels = linear_pixels.data();
        stride = width;
    }
    // Each upload gets fresh storage; pending draws may still reference an older version.
    if (mip == 0 && slice == 0) {
        impl->expanded_packed[impl->current] = false;
        impl->textures[impl->current] = nil;
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native width:impl->width height:impl->height mipmapped:NO];
        desc.textureType = impl->cube ? MTLTextureTypeCube : MTLTextureType2D;
        desc.mipmapLevelCount = impl->mips;
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
        const bool packed = format == SCE_GXM_TEXTURE_BASE_FORMAT_U5U6U5
            || format == SCE_GXM_TEXTURE_BASE_FORMAT_U4U4U4U4
            || format == SCE_GXM_TEXTURE_BASE_FORMAT_U1U5U5U5;
        const bool force_packed_fallback = packed && std::getenv("VITA3K_METAL_FORCE_PACKED_FALLBACK");
        if (!force_packed_fallback)
            impl->textures[impl->current] = [impl->state.impl->device->native_device() newTextureWithDescriptor:desc];
        if (!impl->textures[impl->current] && packed) {
            // Mac-family GPUs can lack packed 16-bit texture formats. Expand
            // the native channel order to RGBA8; sampling_view still applies
            // the guest's swizzle to those channels.
            desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
            impl->textures[impl->current] = [impl->state.impl->device->native_device() newTextureWithDescriptor:desc];
            impl->expanded_packed[impl->current] = impl->textures[impl->current] != nil;
        }
        if (!impl->textures[impl->current]) {
            placeholder("native format or allocation");
            return;
        }
    }
    id<MTLTexture> texture = impl->textures[impl->current];
    require(texture != nil, "Metal: texture upload allocation failed");
    std::vector<uint8_t> expanded_packed;
    if (impl->expanded_packed[impl->current]) {
        expanded_packed.resize(size_t(width) * height * 4);
        const auto *source = static_cast<const uint8_t *>(pixels);
        const size_t source_pitch = size_t(std::max(stride, width)) * 2;
        const auto channel5 = [](uint32_t value) { return uint8_t((value << 3) | (value >> 2)); };
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                uint16_t word;
                std::memcpy(&word, source + y * source_pitch + x * 2, sizeof(word));
                auto *rgba = expanded_packed.data() + (size_t(y) * width + x) * 4;
                if (format == SCE_GXM_TEXTURE_BASE_FORMAT_U5U6U5) {
                    rgba[0] = channel5((word >> 11) & 31);
                    rgba[1] = uint8_t(((word >> 5) & 63) * 255 / 63);
                    rgba[2] = channel5(word & 31);
                    rgba[3] = 255;
                } else if (format == SCE_GXM_TEXTURE_BASE_FORMAT_U4U4U4U4) {
                    for (uint32_t c = 0; c < 4; ++c) {
                        const uint8_t nibble = (word >> (12 - c * 4)) & 15;
                        rgba[c] = (nibble << 4) | nibble;
                    }
                } else if (native == MTLPixelFormatA1BGR5Unorm) {
                    rgba[0] = channel5((word >> 11) & 31);
                    rgba[1] = channel5((word >> 6) & 31);
                    rgba[2] = channel5((word >> 1) & 31);
                    rgba[3] = word & 1 ? 255 : 0;
                } else {
                    rgba[0] = channel5((word >> 10) & 31);
                    rgba[1] = channel5((word >> 5) & 31);
                    rgba[2] = channel5(word & 31);
                    rgba[3] = word & 0x8000 ? 255 : 0;
                }
            }
        pixels = expanded_packed.data();
        bytes = 4;
        stride = width;
    }
    const auto pitch = block_bytes ? ((std::max(stride,width)+block_width-1)/block_width)*block_bytes : std::max(stride,width)*bytes;
    const auto rows = block_bytes ? (height+block_height-1)/block_height : height;
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:mip slice:slice withBytes:pixels bytesPerRow:pitch bytesPerImage:pitch * rows];
}

void MetalState::draw(MetalContext &ctx, MemState &mem, SceGxmPrimitiveType primitive, SceGxmIndexFormat index_format,
    const void *indices, uint32_t count, uint32_t instances) {
    // GXM commands run on a long-lived std::thread, outside Cocoa's event-loop
    // pool. Completed command buffers otherwise retain their draw resources
    // until thread exit. Persistent state owns its objects with strong ARC refs.
    @autoreleasepool {
        require(index_format == SCE_GXM_INDEX_FORMAT_U16 || index_format == SCE_GXM_INDEX_FORMAT_U32, "Metal: invalid index format");
        require(ctx.impl->mem == &mem && ctx.current_render_target && (indices || !count || !instances),
            "Metal: draw outside a scene or without indices");
        auto &record = ctx.record;
        const auto *vp = ctx.vertex_program.get();
        const auto *fp = ctx.fragment_program.get();
        require(vp && fp, "Metal: missing GXM shader program");
        const bool triangles = primitive == SCE_GXM_PRIMITIVE_TRIANGLES
            || primitive == SCE_GXM_PRIMITIVE_TRIANGLE_EDGES
            || primitive == SCE_GXM_PRIMITIVE_TRIANGLE_STRIP || primitive == SCE_GXM_PRIMITIVE_TRIANGLE_FAN;
        const bool wireframe = record.front_polygon_mode == SCE_GXM_POLYGON_MODE_LINE
            || record.front_polygon_mode == SCE_GXM_POLYGON_MODE_TRIANGLE_LINE;
        const uint32_t effective_line_width = triangles && record.two_sided == SCE_GXM_TWO_SIDED_ENABLED
            ? std::max({1u, record.line_width, record.back_line_width}) : std::max(1u, record.line_width);
        const bool wide_line = (primitive == SCE_GXM_PRIMITIVE_LINES || (triangles && wireframe))
            && float(effective_line_width) * res_multiplier > 1;
        const bool point_polygon = triangles && (record.front_polygon_mode == SCE_GXM_POLYGON_MODE_POINT
            || record.front_polygon_mode == SCE_GXM_POLYGON_MODE_POINT_10UV
            || record.front_polygon_mode == SCE_GXM_POLYGON_MODE_POINT_01UV
            || record.front_polygon_mode == SCE_GXM_POLYGON_MODE_TRIANGLE_POINT);
        const bool shader_point_size = point_polygon
            && (gxp::get_vertex_outputs(*vp->program()) & SCE_GXM_VERTEX_PROGRAM_OUTPUT_PSIZE);
        const auto empty_visibility_draw = [&](bool preserve_front, bool preserve_back) {
            if (!ctx.impl->visibility_address) return;
            const uint32_t entries = ctx.impl->visibility_stride / sizeof(uint32_t);
            if (!entries) return;
            for (bool front : {true, false}) {
                const bool enabled = front ? ctx.impl->visibility_enabled : ctx.impl->back_visibility_enabled;
                const bool increment = front ? ctx.impl->visibility_increment : ctx.impl->back_visibility_increment;
                const uint32_t index = front ? ctx.impl->visibility_index : ctx.impl->back_visibility_index;
                const bool preserve = front ? preserve_front : preserve_back;
                if (!enabled || increment || preserve) continue;
                const uint32_t query_index = index < entries ? index : 0;
                const uint64_t guest_address = uint64_t(ctx.impl->visibility_address)
                    + uint64_t(query_index) * sizeof(uint32_t);
                if (guest_address + sizeof(uint32_t) <= uint64_t(UINT32_MAX) - 4095
                    && is_valid_addr_range(mem, Address(guest_address), Address(guest_address + sizeof(uint32_t))))
                    *Ptr<uint32_t>(Address(guest_address)).get(mem) = 0;
            }
        };
        const auto preserve_active_set_query = [&](bool front) {
            const uint32_t entries = ctx.impl->visibility_stride / sizeof(uint32_t);
            const uint32_t index = front ? ctx.impl->visibility_index : ctx.impl->back_visibility_index;
            const uint32_t query_index = index < entries ? index : 0;
            const bool published = entries && ctx.impl->published_set_results.contains(
                {ctx.impl->visibility_epoch, uint64_t(ctx.impl->visibility_address), query_index});
            return published || (ctx.impl->pass_visibility_active
                && ctx.impl->pass_visibility_address == ctx.impl->visibility_address
                && ctx.impl->pass_visibility_index == query_index
                && !ctx.impl->pass_visibility_increment
                && !(front ? ctx.impl->visibility_increment : ctx.impl->back_visibility_increment));
        };
        if (!count || !instances || (primitive == SCE_GXM_PRIMITIVE_LINES && count < 2)
            || ((primitive == SCE_GXM_PRIMITIVE_TRIANGLE_FAN || point_polygon) && count < 3)) {
            empty_visibility_draw(preserve_active_set_query(true), preserve_active_set_query(false));
            return;
        }
        std::vector<uint32_t> point_capture_unique_indices;
        std::vector<uint32_t> point_capture_source_indices;
        uint32_t point_capture_max_index = 0;
        bool compact_point_capture = false;
        if (point_polygon || wide_line) {
            const uint32_t used_count = primitive == SCE_GXM_PRIMITIVE_TRIANGLES
                || primitive == SCE_GXM_PRIMITIVE_TRIANGLE_EDGES ? count / 3 * 3
                : primitive == SCE_GXM_PRIMITIVE_LINES ? count / 2 * 2 : count;
            point_capture_unique_indices.reserve(used_count);
            point_capture_source_indices.reserve(used_count);
            for (uint32_t i = 0; i < used_count; ++i) {
                const uint32_t index = index_format == SCE_GXM_INDEX_FORMAT_U16
                    ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i];
                point_capture_unique_indices.push_back(index);
                point_capture_source_indices.push_back(index);
                point_capture_max_index = std::max(point_capture_max_index, index);
            }
            std::sort(point_capture_unique_indices.begin(), point_capture_unique_indices.end());
            point_capture_unique_indices.erase(std::unique(point_capture_unique_indices.begin(), point_capture_unique_indices.end()),
                point_capture_unique_indices.end());
            const uint64_t indexed_span = uint64_t(point_capture_max_index) + 1;
            // Keep the single indexed draw for moderately sparse meshes. The
            // compact variant trades one draw per vertex for bounded storage.
            compact_point_capture = indexed_span > uint64_t(point_capture_unique_indices.size()) * 64
                || indexed_span > impl->device->native_device().maxBufferLength / sizeof(CapturedVertexOutputs);
        }
        const bool cull_front = triangles && record.cull_mode == SCE_GXM_CULL_CCW;
        const bool cull_back = triangles && record.cull_mode == SCE_GXM_CULL_CW;
        const bool two_sided = record.two_sided == SCE_GXM_TWO_SIDED_ENABLED;
        const bool front_fragment_disabled = record.front_side_fragment_program_mode == SCE_GXM_FRAGMENT_PROGRAM_DISABLED;
        const bool back_fragment_disabled = two_sided
            ? record.back_side_fragment_program_mode == SCE_GXM_FRAGMENT_PROGRAM_DISABLED : front_fragment_disabled;
        // Culled triangles cannot invoke the fragment stage. Select the
        // disabled-program path when every remaining face disables it.
        const bool fragment_disabled = !record.is_maskupdate
            && (cull_front || front_fragment_disabled) && (cull_back || back_fragment_disabled);
        // A depth-only program can still discard fragments or write guest
        // memory. Execute it, but never publish its undefined color output.
        // Mask updates write the separate mask attachment and are exempt.
        const bool fragment_color_disabled = fragment_disabled || (!record.is_maskupdate
            && (fp->program()->program_flags & SCE_GXM_PROGRAM_FLAG_OUTPUT_UNDEFINED));
        const bool fragment_resources = !record.is_maskupdate && !fragment_disabled;
        if (!impl->trace_pixel_armed && fragment_resources
            && impl->trace_pixel_arm_shader == hex_string(fp->fragment_program->hash)
            && (!impl->trace_pixel_target_width || ctx.impl->width == impl->trace_pixel_target_width))
            impl->trace_pixel_armed = true;
        if (!impl->dump_armed && fragment_resources && impl->dump_arm_shader == hex_string(fp->fragment_program->hash)
            && (!impl->dump_arm_target_width || ctx.impl->width == impl->dump_arm_target_width)
            && impl->dump_arm_shader_matches++ >= impl->dump_arm_shader_skip)
            impl->dump_armed = true;
        const bool capture_draw = !impl->dump_draw_dir.empty() && !impl->draw_dumped && impl->dump_armed
            && ((impl->dump_arm_pixel_shader.empty() && impl->dump_arm_pixel_rgb.empty()) || impl->dump_pixel_armed)
            && (impl->dump_draw_shader.empty() || impl->dump_draw_shader == hex_string(fp->fragment_program->hash))
            && (!impl->dump_draw_target_address || ctx.impl->guest_color.data.address() == impl->dump_draw_target_address)
            && (!impl->dump_draw_target_width || ctx.impl->width == impl->dump_draw_target_width)
            && (!impl->dump_draw_texture_address || (fp->fragment_program->textures_used[0]
                && (ctx.textures[0].data_addr << 2) == impl->dump_draw_texture_address))
            && (!impl->dump_draw_texture_width || (fp->fragment_program->textures_used[0]
                && gxm::get_width(ctx.textures[0]) == impl->dump_draw_texture_width))
            && (impl->dump_draw_texture_format.empty() || (fp->fragment_program->textures_used[0]
                && uint32_t(gxm::get_format(ctx.textures[0])) == std::strtoul(impl->dump_draw_texture_format.c_str(), nullptr, 0)))
            && (impl->dump_vertex_shader.empty() || impl->dump_vertex_shader == hex_string(vp->vertex_program->hash))
            && (!impl->dump_stream_stride || (!vp->streams.empty() && vp->streams[0].stride == impl->dump_stream_stride))
            && (impl->dump_trigger_file.empty() || std::filesystem::exists(impl->dump_trigger_file))
            && impl->dump_draw_matches++ >= impl->dump_draw_skip;
        const auto draw_capture_dir = impl->dump_draw_limit == 1 ? impl->dump_draw_dir
            : impl->dump_draw_dir / fmt::format("candidate-{:02}", impl->dump_draw_saved + 1);
        std::ofstream draw_metadata;
        std::ofstream texture_metadata;
        auto dump_bytes = [&](const std::string &name, const void *bytes, size_t size) {
            std::ofstream out(draw_capture_dir/name, std::ios::binary);
            out.write(static_cast<const char *>(bytes), size);
            require(bool(out), "Metal: cannot write draw diagnostic " + name);
        };
        // Optional attachment probes are separate from draw.txt so older replay
        // readers remain usable. Snapshot only completed single-sample images.
        auto dump_attachment = [&](const char *name, id<MTLTexture> source) {
            if (!capture_draw || !impl->dump_attachments || !source) return;
            std::filesystem::create_directories(draw_capture_dir);
            std::ofstream metadata(draw_capture_dir/"attachments.txt",std::ios::app);
            const size_t bytes=size_t(source.width)*source.height*16;
            if (source.sampleCount>1 || bytes>64*1024*1024) {
                metadata << "skipped " << name << ' ' << source.width << ' ' << source.height << '\n';
                return;
            }
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            auto snapshot=impl->caster->sampling_snapshot(source,0,0);
            if (!snapshot) {
                metadata << "unsupported " << name << ' ' << uint32_t(source.pixelFormat) << '\n';
                return;
            }
            std::vector<uint8_t> pixels(bytes);
            [snapshot getBytes:pixels.data() bytesPerRow:source.width*16
                fromRegion:MTLRegionMake2D(0,0,source.width,source.height) mipmapLevel:0];
            dump_bytes(std::string(name)+".rgba32f",pixels.data(),pixels.size());
            metadata << name << ' ' << source.width << ' ' << source.height << ' ' << uint32_t(source.pixelFormat) << '\n';
            require(bool(metadata),"Metal: cannot write attachment metadata");
        };
        auto dump_raw_attachment = [&](const char *name) {
            const auto source=ctx.impl->raw_color;
            if (!capture_draw || !impl->dump_attachments || !source) return;
            const size_t size=source.width*source.height*8;
            if (size>64*1024*1024) return;
            std::vector<uint8_t> bytes(size);
            [source getBytes:bytes.data() bytesPerRow:source.width*8
                fromRegion:MTLRegionMake2D(0,0,source.width,source.height) mipmapLevel:0];
            dump_bytes(std::string(name)+".u16",bytes.data(),bytes.size());
        };
        auto dump_depth_attachment = [&](const char *name) {
            id<MTLTexture> source = ctx.impl->depth;
            if (!capture_draw || !impl->dump_attachments || !source) return;
            std::filesystem::create_directories(draw_capture_dir);
            std::ofstream metadata(draw_capture_dir/"attachments.txt",std::ios::app);
            const size_t expanded_width=size_t(source.width)*(source.sampleCount>1 ? source.sampleCount/2 : 1);
            const size_t expanded_height=size_t(source.height)*(source.sampleCount>1 ? 2 : 1);
            const size_t bytes=expanded_width*expanded_height*sizeof(float);
            if (bytes>64*1024*1024) {
                metadata << "skipped " << name << ' ' << expanded_width << ' ' << expanded_height << '\n';
                return;
            }
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            auto snapshot=impl->caster->depth_snapshot(source,false,1,0,0,true);
            std::vector<uint8_t> pixels(bytes);
            [snapshot getBytes:pixels.data() bytesPerRow:expanded_width*sizeof(float)
                fromRegion:MTLRegionMake2D(0,0,expanded_width,expanded_height) mipmapLevel:0];
            dump_bytes(std::string(name)+".f32",pixels.data(),pixels.size());
            metadata << name << ' ' << expanded_width << ' ' << expanded_height
                << " samples " << source.sampleCount
                << " guest_format " << uint32_t(record.depth_stencil_surface.get_format())
                << " background " << record.depth_stencil_surface.background_depth
                << " force_load " << bool(record.depth_stencil_surface.force_load)
                << " force_store " << bool(record.depth_stencil_surface.force_store) << '\n';
            require(bool(metadata),"Metal: cannot write depth attachment metadata");
        };
        auto dump_stencil_attachment = [&](const char *name) {
            id<MTLTexture> source = ctx.impl->depth;
            if (!capture_draw || !impl->dump_attachments || !source) return;
            std::filesystem::create_directories(draw_capture_dir);
            std::ofstream metadata(draw_capture_dir/"attachments.txt",std::ios::app);
            const size_t expanded_width=size_t(source.width)*(source.sampleCount>1 ? source.sampleCount/2 : 1);
            const size_t expanded_height=size_t(source.height)*(source.sampleCount>1 ? 2 : 1);
            const size_t bytes=expanded_width*expanded_height;
            if (bytes>64*1024*1024) {
                metadata << "skipped " << name << ' ' << expanded_width << ' ' << expanded_height << '\n';
                return;
            }
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            auto snapshot=impl->caster->stencil_snapshot(source,false,1,0,0,true);
            std::vector<uint8_t> pixels(bytes);
            [snapshot getBytes:pixels.data() bytesPerRow:expanded_width
                fromRegion:MTLRegionMake2D(0,0,expanded_width,expanded_height) mipmapLevel:0];
            dump_bytes(std::string(name)+".u8",pixels.data(),pixels.size());
            metadata << name << ' ' << expanded_width << ' ' << expanded_height
                << " samples " << source.sampleCount << '\n';
            require(bool(metadata),"Metal: cannot write stencil attachment metadata");
        };
        const float raster_scale = res_multiplier * ctx.impl->surface_downscale;
        auto clip = scissors(record, ctx.impl->width, ctx.impl->height, raster_scale);
        // A zero-area scissor discards every fragment but still runs the
        // vertex stage. This preserves guest-memory stores in a GXP vertex
        // shader even when REGION_CLIP_ALL excludes the complete target.
        if (clip.empty()) clip.push_back(MTLScissorRect{0, 0, 0, 0});
        if (auto *target = ctx.current_render_target;
            target && target->has_macroblock_sync && clip[0].width && clip[0].height) {
            require(target->macroblock_width && target->macroblock_height,
                "Metal: macroblock target has zero-sized tiles");
            const bool single_tile = clip.size() == 1
                && clip[0].width <= target->macroblock_width
                && clip[0].height <= target->macroblock_height;
            ctx.impl->macroblock_ignore |= !single_tile;
            if (ctx.impl->macroblock_ignore) {
                // Match Vulkan's full-target fallback when a draw spans more
                // than one tile: every draw starts a new render pass.
                mid_scene_flush(ctx, false);
            } else {
                const uint16_t x = clip[0].x / target->macroblock_width;
                const uint16_t y = clip[0].y / target->macroblock_height;
                require(x < 4 && y < 4, "Metal: macroblock scissor exceeds four tiles");
                // Vulkan also intersects the scissor with this tile's render
                // area when its origin and size do not end at a tile edge.
                clip[0].width = std::min<uint32_t>(clip[0].width,
                    (x + 1) * target->macroblock_width - clip[0].x);
                clip[0].height = std::min<uint32_t>(clip[0].height,
                    (y + 1) * target->macroblock_height - clip[0].y);
                if (x != ctx.impl->macroblock_last_x || y != ctx.impl->macroblock_last_y) {
                    if (ctx.impl->macroblock_last_x != uint16_t(~0u))
                        mid_scene_flush(ctx, false);
                    const uint16_t bit = uint16_t(1u << (y * 4 + x));
                    const bool revisit = (ctx.impl->macroblock_visited & bit) != 0;
                    ctx.impl->macroblock_visited |= bit;
                    ctx.impl->macroblock_last_x = x;
                    ctx.impl->macroblock_last_y = y;
                    if (revisit && !record.depth_stencil_surface.force_load
                        && !record.depth_stencil_surface.disabled()) {
                        const uint32_t left = x * target->macroblock_width;
                        const uint32_t top = y * target->macroblock_height;
                        clear_macroblock_depth(*this, ctx, MTLScissorRect{left, top,
                            std::min<uint32_t>(target->macroblock_width, ctx.impl->width - left),
                            std::min<uint32_t>(target->macroblock_height, ctx.impl->height - top)});
                    }
                }
            }
        }
        if (capture_draw) finish(ctx); // Publish every producer before diagnostic texture reads.
        dump_attachment("color-before",ctx.impl->color);
        dump_raw_attachment("raw-color-before");
        dump_attachment("mask-before",ctx.impl->mask);
        dump_depth_attachment("depth-before");
        dump_stencil_attachment("stencil-before");
        ctx.shader_hints.metal_samples = ctx.impl->samples;
        ctx.shader_hints.metal_far_clip = draw_drops_fragments_beyond_far_plane(record);
        ctx.shader_hints.metal_missing_vertex_outputs = fragment_resources
            ? (uint32_t(gxp::get_fragment_inputs(*fp->program()))
                  & ~uint32_t(gxp::get_vertex_outputs(*vp->program())) & 0x3ffeu)
            : 0;
        ctx.shader_hints.metal_mip_sampling = true;
        ctx.shader_hints.metal_float_cube_filter = !impl->device->native_device().supports32BitFloatFiltering;
        ctx.shader_hints.metal_raw_color_attachment = !record.is_maskupdate && ctx.impl->raw_attachment;
        ctx.shader_hints.metal_raw_color_fetch = false;
        if (ctx.shader_hints.metal_raw_color_attachment) {
            const auto surface=impl->surfaces.find(ctx.impl->guest_color.data.address());
            const auto &blend=static_cast<const MetalFragmentProgram &>(*fp->fragment_program).blend;
            ctx.shader_hints.metal_raw_color_fetch=surface!=impl->surfaces.end()
                && !surface->second.raw_color_invalidated
                && blend.colorFunc==SCE_GXM_BLEND_FUNC_NONE && blend.alphaFunc==SCE_GXM_BLEND_FUNC_NONE;
        }
        ctx.shader_hints.attributes = &vp->attributes;
        ctx.shader_hints.color_format = record.color_surface.colorFormat;
        ctx.shader_hints.metal_red_alpha_shader_blend = false;
        ctx.shader_hints.metal_quantized_color_blend = false;
        if (!fragment_color_disabled && !record.is_maskupdate
            && gxm::is_red_alpha_color_format(record.color_surface.colorFormat)) {
            const auto blend = static_cast<const MetalFragmentProgram &>(*fp->fragment_program).blend;
            ctx.shader_hints.metal_red_alpha_shader_blend = blend.colorFunc != SCE_GXM_BLEND_FUNC_NONE
                || blend.alphaFunc != SCE_GXM_BLEND_FUNC_NONE;
            ctx.shader_hints.metal_red_alpha_blend = blend;
        }
        const auto color_base = gxm::get_base_format(record.color_surface.colorFormat);
        if (!fragment_color_disabled && !record.is_maskupdate
            && static_cast<const MetalFragmentProgram &>(*fp->fragment_program).blend.colorMask != SCE_GXM_COLOR_MASK_NONE
            && (color_base == SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2
                || color_base == SCE_GXM_COLOR_BASE_FORMAT_S5S5U6
                || color_base == SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8)) {
            ctx.shader_hints.metal_quantized_color_blend = true;
            ctx.shader_hints.metal_quantized_blend = static_cast<const MetalFragmentProgram &>(*fp->fragment_program).blend;
        }
        ctx.shader_hints.metal_output_register_size = record.color_surface.outputRegisterSize;
        ctx.shader_hints.metal_output_register_format = fp->fragment_program->output_register_format;
        for (uint32_t slot = 0; slot < SCE_GXM_MAX_TEXTURE_UNITS; ++slot) {
            ctx.shader_hints.fragment_textures[slot] = gxm::get_format(ctx.textures[slot]);
            ctx.shader_hints.vertex_textures[slot] = gxm::get_format(ctx.textures[slot + SCE_GXM_MAX_TEXTURE_UNITS]);
            ctx.shader_hints.metal_fragment_lod_bias[slot] = sampler_lod_bias(ctx.textures[slot]);
            ctx.shader_hints.metal_vertex_lod_bias[slot] = sampler_lod_bias(ctx.textures[slot + SCE_GXM_MAX_TEXTURE_UNITS]);
        }
        auto shader_key = [&](bool vertex) {
            if (!vertex && fragment_disabled) {
                std::string key = "metal-depth-only";
                append(key, ctx.impl->samples);
                return key;
            }
            std::string key = hex_string(vertex ? vp->vertex_program->hash : fp->fragment_program->hash);
            key.reserve(key.size() + 4 * (16 + 3 * vp->attributes.size()
                + 2 * SCE_GXM_MAX_TEXTURE_UNITS));
            append(key, vertex);
            if (vertex) {
                append(key, ctx.shader_hints.metal_missing_vertex_outputs);
                append(key, ctx.shader_hints.metal_far_clip);
            }
            append(key, ctx.shader_hints.metal_mip_sampling);
            append(key, ctx.shader_hints.metal_float_cube_filter);
            append(key, get_features_mask());
            append(key, ctx.shader_hints.color_format);
            if (!vertex) {
                append(key, ctx.shader_hints.metal_raw_color_attachment);
                append(key, ctx.shader_hints.metal_raw_color_fetch);
                append(key, ctx.shader_hints.metal_red_alpha_shader_blend);
                if (ctx.shader_hints.metal_red_alpha_shader_blend) {
                    const auto &blend = ctx.shader_hints.metal_red_alpha_blend;
                    append(key, blend.colorFunc); append(key, blend.alphaFunc);
                    append(key, blend.colorSrc); append(key, blend.colorDst);
                    append(key, blend.alphaSrc); append(key, blend.alphaDst);
                }
                append(key, ctx.shader_hints.metal_quantized_color_blend);
                if (ctx.shader_hints.metal_quantized_color_blend) {
                    const auto &blend = ctx.shader_hints.metal_quantized_blend;
                    append(key, blend.colorFunc); append(key, blend.alphaFunc);
                    append(key, blend.colorSrc); append(key, blend.colorDst);
                    append(key, blend.alphaSrc); append(key, blend.alphaDst);
                    append(key, blend.colorMask);
                }
            }
            append(key, ctx.shader_hints.metal_output_register_size);
            if (!vertex) append(key, ctx.shader_hints.metal_output_register_format);
            append(key, vertex ? 1u : ctx.impl->samples);
            append(key, !vertex && record.is_maskupdate);
            append(key, !vertex && !record.is_maskupdate && record.is_gamma_corrected);
            append(key, vp->attributes.size());
            for (auto &a : vp->attributes) {
                append(key, a.regIndex); append(key, a.format); append(key, a.componentCount);
            }
            for (uint32_t slot = 0; slot < SCE_GXM_MAX_TEXTURE_UNITS; ++slot) {
                append(key, vertex ? ctx.shader_hints.vertex_textures[slot] : ctx.shader_hints.fragment_textures[slot]);
                append(key, std::bit_cast<uint32_t>(vertex ? ctx.shader_hints.metal_vertex_lod_bias[slot] : ctx.shader_hints.metal_fragment_lod_bias[slot]));
            }
            return key;
        };
        const std::string vertex_key = shader_key(true);
        const std::string fragment_key = shader_key(false);
        std::string error;
        const auto remember_failure = [](std::set<std::string> &failures, const std::string &key,
                                          const char *kind, const std::string &message) {
            if (failures.insert(key).second)
                LOG_ERROR("Metal {} variant {:016x} rejected; skipping this variant until the next game: {}",
                    kind, XXH3_64bits(key.data(), key.size()), message.empty() ? "compiler returned no result" : message);
        };
        const auto compile_shader_once = [&](const std::string &key, auto &&build) -> CompiledProgram * {
            if (impl->failed_shaders.contains(key)) return nullptr;
            auto &cached = impl->shaders[key];
            if (!cached) {
                try {
                    cached = build();
                    require(bool(cached), error);
                } catch (const std::exception &failure) {
                    remember_failure(impl->failed_shaders, key, "shader", failure.what());
                    return nullptr;
                }
            }
            return cached.get();
        };
        const auto compile = [&](bool vertex) -> CompiledProgram * {
            const auto &key = vertex ? vertex_key : fragment_key;
            return compile_shader_once(key, [&]() {
                std::unique_ptr<CompiledProgram> compiled;
                const auto *gxp = vertex ? vp->program() : fp->program();
                // Keep GXM's vertex upload mode independent of the shader's
                // native uniform addresses.
                auto shader_features = features;
                shader_features.enable_memory_mapping = true;
                const bool gamma = !vertex && !record.is_maskupdate && record.is_gamma_corrected;
                if (!impl->warmup_failed_shaders.contains(key)) {
                    try {
                        if (auto msl = impl->device->load_cached_program(key)) {
                            const auto expected_stage = vertex ? shader::metal::Stage::Vertex : shader::metal::Stage::Fragment;
                            if (msl->stage == expected_stage)
                                compiled = impl->device->compile(*msl, gamma, error);
                            else
                                error = "cached shader stage does not match the draw";
                            if (!compiled) LOG_WARN("Metal: cached shader failed to compile; rebuilding: {}", error);
                        }
                    } catch (const std::exception &failure) {
                        LOG_WARN("Metal: cached shader could not be loaded; rebuilding: {}", failure.what());
                    }
                }
                if (!compiled) {
                    auto msl = !vertex && fragment_disabled ? depth_only_program(ctx.impl->samples)
                        : shader::metal::convert_gxp(*gxp, hex_string(vertex ? vp->vertex_program->hash : fp->fragment_program->hash), shader_features, ctx.shader_hints, !vertex && record.is_maskupdate);
                    compiled = impl->device->compile(msl, gamma, error);
                    require(bool(compiled), error);
                    try {
                        impl->device->store_cached_program(key, msl,
                            hex_string(vertex ? vp->vertex_program->hash : fp->fragment_program->hash), gamma);
                    } catch (const std::exception &failure) {
                        LOG_WARN("Metal: compiled shader could not be cached: {}", failure.what());
                    }
                }
                ++shaders_count_compiled;
                return compiled;
            });
        };
        auto *vertex_shader = compile(true);
        if (!vertex_shader) return;
        auto *fragment_shader = compile(false);
        if (!fragment_shader) return;
        auto &vs = *vertex_shader;
        auto &fs = *fragment_shader;
        // An indexed shader with external stores can execute for every source
        // occurrence. The regular capture deduplicates indices and would drop
        // those stores; use one compact draw per source occurrence instead.
        const bool capture_per_occurrence = (point_polygon || wide_line) && vs.writes_guest_memory;
        if (capture_per_occurrence) compact_point_capture = true;
        const bool routed_point_polygon = point_polygon;
        const bool routed_wide_line = wide_line;
        const bool routed_geometry = routed_point_polygon || routed_wide_line;
        const bool visibility_faces_differ = ctx.impl->visibility_enabled != ctx.impl->back_visibility_enabled
            || (ctx.impl->visibility_enabled && ctx.impl->back_visibility_enabled
                && (ctx.impl->visibility_index != ctx.impl->back_visibility_index
                    || ctx.impl->visibility_increment != ctx.impl->back_visibility_increment));
        const bool many_face_queries = ctx.impl->visibility_address && triangles && !cull_front && !cull_back
            && visibility_faces_differ;
        const uint64_t max_query_capacity = impl->device->native_device().maxBufferLength;
        uint64_t query_capacity = 4096;
        if (many_face_queries) {
            const uint64_t bytes_per_index = uint64_t(instances) * std::max<size_t>(1, clip.size()) * 16;
            require(max_query_capacity >= 4096 && bytes_per_index
                    && count <= (max_query_capacity - 8) / bytes_per_index,
                "Metal: face visibility query buffer exceeds native maximum length");
            query_capacity = std::max<uint64_t>(4096, uint64_t(count) * bytes_per_index + 8);
        }
        ctx.impl->visibility_buffer_capacity = size_t(query_capacity);
        if (impl->cache_enabled && impl->known_shader_pairs.emplace(fp->fragment_program->hash, vp->vertex_program->hash).second) {
            shaders_cache_hashs.push_back({fp->fragment_program->hash, vp->vertex_program->hash});
            renderer::save_shaders_cache_hashs(*this, shaders_cache_hashs);
        }
        const bool synchronous = impl->sync_draws || vs.writes_guest_memory || fs.writes_guest_memory;
        if (vs.writes_guest_memory || fs.writes_guest_memory)
            disable_vertex_stream_snapshots();
        if (!disable_surface_sync && ctx.impl->guest_color.data
            && (ctx.impl->pending_color_writes || ctx.impl->color_guest_dirty)) {
            // Plus sync_surface_for_gpu_read makes a recent color producer
            // visible before binding an aliased uniform. Metal's uniform GPU
            // addresses point at guest RAM, so the active producer must finish
            // and publish before this draw reads it. Other scenes have already
            // published at EndScene when surface synchronization is enabled.
            const uint64_t color_begin = ctx.impl->guest_color.data.address();
            const uint64_t color_end = color_begin + surface_memory_size(ctx.impl->guest_color);
            const ShaderProgram *uniform_programs[] = {vp->vertex_program.get(), fp->fragment_program.get()};
            bool reads_color = false;
            for (uint32_t stage = 0; stage < 2 && !reads_color; ++stage) {
                if (stage == 1 && !fragment_resources) continue;
                const auto &program = *uniform_programs[stage];
                for (uint32_t block = 0; block < program.buffer_count; ++block) {
                    if (!program.uniform_buffer_sizes.at(block)) continue;
                    const auto &binding = ctx.uniforms[stage].at(block);
                    if (!binding.data || !binding.size) continue;
                    // Preserve Plus's interior-alias threshold. Small uniforms
                    // inside a former surface allocation are often unrelated.
                    const uint64_t begin = binding.address;
                    const uint64_t read_size = std::min<uint64_t>(binding.size,
                        uint64_t(program.uniform_buffer_sizes.at(block)) * 4);
                    reads_color = begin == color_begin
                        || (read_size >= 16 * 1024 && begin >= color_begin
                            && begin + read_size <= color_end);
                    if (reads_color) break;
                }
            }
            if (reads_color)
                require(finish(ctx), "Metal: cannot publish color surface for a uniform reader");
        }
        size_t captured_texture_bytes = 0;
        // Plus keeps CPU-dirty render images eligible for texture reads during
        // the first two frames. Older images are rejected by the reader, not
        // destroyed: an active attachment and a later rebind still own them.
        // Freeze each decision within this draw so all alias paths agree.
        std::map<Address, bool> texture_surface_eligibility;
        const auto texture_surface_usable = [&](const Surface &surface) {
            const Address address = surface.guest.data.address();
            if (const auto known = texture_surface_eligibility.find(address);
                known != texture_surface_eligibility.end()) return known->second;
            bool usable = true;
            const bool recent = impl->frame_timestamp >= surface.last_frame_rendered
                && impl->frame_timestamp - surface.last_frame_rendered < 2;
            if (!recent) {
                usable = !surface.cpu_writes.changed();
                const size_t bytes = surface_memory_size(surface.guest);
                const uint64_t end = uint64_t(address) + bytes;
                if (usable && bytes && end <= uint64_t(UINT32_MAX) - 4095
                    && is_valid_addr_range(mem, address, Address(end)))
                    usable = !surface_cpu_bytes_changed(surface,
                        {static_cast<const uint8_t *>(surface.guest.data.get(mem)), bytes});
            }
            texture_surface_eligibility.emplace(address, usable);
            return usable;
        };
        const auto find_texture_surface = [&](Address address) {
            const auto found = impl->surfaces.find(address);
            return found != impl->surfaces.end() && texture_surface_usable(found->second)
                ? found : impl->surfaces.end();
        };
        const auto find_subrectangle = [&](const SceGxmTexture &texture) {
            return find_color_subrectangle(impl->surfaces, texture, texture_surface_usable);
        };
        const auto find_word_subrectangle = [&](const SceGxmTexture &texture) {
            auto found=impl->surfaces.upper_bound(texture.data_addr<<2);
            while (found!=impl->surfaces.begin()) {
                --found;
                if (surface_texture_layout_overlap(found->second.guest,texture))
                    return texture_surface_usable(found->second)
                        && surface_word_subrectangle(found->second.guest,texture)
                        ? found : impl->surfaces.end();
            }
            return impl->surfaces.end();
        };
        const auto find_halfword_subrectangle = [&](const SceGxmTexture &texture) {
            auto found=impl->surfaces.upper_bound(texture.data_addr<<2);
            while (found!=impl->surfaces.begin()) {
                --found;
                if (surface_texture_layout_overlap(found->second.guest,texture))
                    return texture_surface_usable(found->second)
                        && surface_halfword_subrectangle(found->second.guest,texture)
                        ? found : impl->surfaces.end();
            }
            return impl->surfaces.end();
        };
        const auto find_byte_subrectangle = [&](const SceGxmTexture &texture) {
            auto found=impl->surfaces.upper_bound(texture.data_addr<<2);
            while (found!=impl->surfaces.begin()) {
                --found;
                if (surface_texture_layout_overlap(found->second.guest,texture))
                    return texture_surface_usable(found->second)
                        && surface_byte_subrectangle(found->second.guest,texture)
                        ? found : impl->surfaces.end();
            }
            return impl->surfaces.end();
        };
        const auto find_rgba8_byte_subrectangle = [&](const SceGxmTexture &texture) {
            auto found=impl->surfaces.upper_bound(texture.data_addr<<2);
            while (found!=impl->surfaces.begin()) {
                --found;
                if (surface_texture_layout_overlap(found->second.guest,texture))
                    return texture_surface_usable(found->second)
                        && surface_rgba8_byte_subrectangle(found->second.guest,texture)
                        ? found : impl->surfaces.end();
            }
            return impl->surfaces.end();
        };
        const auto find_wide_small_subrectangle = [&](const SceGxmTexture &texture) {
            auto found=impl->surfaces.upper_bound(texture.data_addr<<2);
            while (found!=impl->surfaces.begin()) {
                --found;
                if (surface_texture_layout_overlap(found->second.guest,texture))
                    return texture_surface_usable(found->second)
                        && surface_wide_small_subrectangle(found->second.guest,texture)
                        ? found : impl->surfaces.end();
            }
            return impl->surfaces.end();
        };
        const auto find_32_small_subrectangle = [&](const SceGxmTexture &texture) {
            auto found=impl->surfaces.upper_bound(texture.data_addr<<2);
            while (found!=impl->surfaces.begin()) {
                --found;
                if (surface_texture_layout_overlap(found->second.guest,texture))
                    return texture_surface_usable(found->second)
                        && (gxm::get_base_format(found->second.guest.colorFormat)
                                != SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
                            || found->second.color.pixelFormat==MTLPixelFormatRGB9E5Float)
                        && surface_32_small_subrectangle(found->second.guest,texture)
                        ? found : impl->surfaces.end();
            }
            return impl->surfaces.end();
        };
        const auto find_direct_surface = [&](const SceGxmTexture &texture) {
            const auto found = find_texture_surface(texture.data_addr << 2);
            if (found == impl->surfaces.end()) return found;
            // An equal base address alone does not establish ownership of a
            // texture view. Plus checks the row pitch and tiling first.
            return surface_texture_layout_overlap(found->second.guest, texture)
                || surface_subrectangle(found->second.guest, texture)
                ? found : impl->surfaces.end();
        };
        const auto rg32_linear_alias = [&](const Surface &surface, const SceGxmTexture &texture,
            Address texture_address) {
            const auto type=texture.texture_type();
            const auto &guest=surface.guest;
            const uint64_t pitch=type==SCE_GXM_TEXTURE_LINEAR_STRIDED
                ? gxm::get_stride_in_bytes(texture)
                : uint64_t(align(gxm::get_width(texture),8))*4;
            return surface.color.pixelFormat==MTLPixelFormatRG32Float
                && surface_word_target_supported(gxm::get_base_format(gxm::get_format(texture)))
                && (gxm::get_base_format(gxm::get_format(texture))!=SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10
                    || texture.swizzle_format<8)
                && (gxm::get_base_format(gxm::get_format(texture))!=SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8
                    || texture.swizzle_format<2)
                && (texture_address==surface.guest.data.address()
                    || uint64_t(texture_address)==uint64_t(surface.guest.data.address())+4)
                && guest.surfaceType==SCE_GXM_COLOR_SURFACE_LINEAR
                && (type==SCE_GXM_TEXTURE_LINEAR || type==SCE_GXM_TEXTURE_LINEAR_STRIDED)
                && guest.strideInPixels==guest.width
                && gxm::get_width(texture)==guest.width*2
                && gxm::get_height(texture)==guest.height
                && pitch==uint64_t(guest.strideInPixels)*8 && res_multiplier>0;
        };
        const auto rgba16_linear_alias = [&](const Surface &surface, const SceGxmTexture &texture,
            Address texture_address) {
            const auto type=texture.texture_type();
            const auto &guest=surface.guest;
            const uint64_t pitch=type==SCE_GXM_TEXTURE_LINEAR_STRIDED
                ? gxm::get_stride_in_bytes(texture)
                : uint64_t(align(gxm::get_width(texture),8))*4;
            return guest.surfaceType==SCE_GXM_COLOR_SURFACE_LINEAR
                && surface_word_target_supported(gxm::get_base_format(gxm::get_format(texture)))
                && (gxm::get_base_format(gxm::get_format(texture))!=SCE_GXM_TEXTURE_BASE_FORMAT_U2U10U10U10
                    || texture.swizzle_format<8)
                && (gxm::get_base_format(gxm::get_format(texture))!=SCE_GXM_TEXTURE_BASE_FORMAT_X8S8S8U8
                    || texture.swizzle_format<2)
                && gxm::get_base_format(guest.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16
                && surface.color.pixelFormat==MTLPixelFormatRGBA16Float
                && (texture_address==guest.data.address()
                    || uint64_t(texture_address)==uint64_t(guest.data.address())+4)
                && (type==SCE_GXM_TEXTURE_LINEAR || type==SCE_GXM_TEXTURE_LINEAR_STRIDED)
                && guest.strideInPixels==guest.width && gxm::get_width(texture)==guest.width*2
                && gxm::get_height(texture)==guest.height
                && pitch==uint64_t(guest.strideInPixels)*8 && res_multiplier>0;
        };
        // Discover separate word views before creating any cast for this draw.
        // Once a high-word view has been used, Plus duplicates each view's word
        // across both columns instead of interleaving different words. Persist
        // that choice until the backing color texture is recreated.
        for (uint32_t index = 0; index < SCE_GXM_MAX_TEXTURE_UNITS * 2; ++index) {
            const bool vertex = index >= SCE_GXM_MAX_TEXTURE_UNITS;
            if (!vertex && !fragment_resources) continue;
            const auto &program = vertex ? static_cast<const ShaderProgram &>(*vp->vertex_program)
                                         : static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index % SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            if ((vertex ? vs.cube_texture_mask : fs.cube_texture_mask)
                & (1u << (index % SCE_GXM_MAX_TEXTURE_UNITS))) continue;
            const auto &texture = ctx.textures[index];
            if (texture.texture_type() != SCE_GXM_TEXTURE_LINEAR_STRIDED
                && texture::get_upload_mip(texture.true_mip_count(), gxm::get_width(texture),
                    gxm::get_height(texture)) > 1) continue;
            const auto base = gxm::get_base_format(gxm::get_format(texture));
            const Address address = texture.data_addr << 2;
            if (address < 4 || !surface_word_target_supported(base)) continue;
            const auto source = find_texture_surface(address - 4);
            if (source != impl->surfaces.end()
                && (rg32_linear_alias(source->second, texture, address)
                    || rgba16_linear_alias(source->second, texture, address))
                && !source->second.has_word_offset_view) {
                source->second.has_word_offset_view = true;
                source->second.rgba8_casts.clear();
                source->second.word_casts.clear();
                source->second.word_rect_casts.clear();
            }
            const auto owner=find_word_subrectangle(texture);
            if (owner!=impl->surfaces.end()) {
                const auto rect=*surface_word_subrectangle(owner->second.guest,texture);
                if ((rect.x&1u) && !owner->second.has_word_offset_view) {
                    owner->second.has_word_offset_view=true;
                    owner->second.rgba8_casts.clear();
                    owner->second.word_casts.clear();
                    owner->second.word_rect_casts.clear();
                }
            }
        }
        // A mip chain or cube can span several independently rendered surfaces.
        // Publish incompatible aliases and preserve native pixels for matching levels.
        std::array<id<MTLTexture>, SCE_GXM_MAX_TEXTURE_UNITS*2> prepared_images{};
        std::array<std::optional<std::pair<float,float>>, SCE_GXM_MAX_TEXTURE_UNITS*2> surface_viewports{};
        std::array<shader::metal::TextureMipInfo,SCE_GXM_MAX_TEXTURE_UNITS*2> texture_mip_info{};
        std::array<bool,SCE_GXM_MAX_TEXTURE_UNITS*2> narrow_cast_coords{};
        const auto rendered_cube=[&](const SceGxmTexture &texture,bool &raw)->id<MTLTexture> {
            const auto type=texture.texture_type();
            // Retain the existing full mip-chain assembler. This direct path
            // handles six complete base faces without reading padding/mips.
            if ((type!=SCE_GXM_TEXTURE_CUBE && type!=SCE_GXM_TEXTURE_CUBE_ARBITRARY)
                || texture.true_mip_count()>1) return nil;
            const uint32_t width=gxm::get_width(texture),height=gxm::get_height(texture);
            const auto format=gxm::get_format(texture);
            const auto base=gxm::get_base_format(format);
            SceGxmColorBaseFormat color_base;
            if (!texture.data_addr || !width || width!=height
                || !renderer::texture::convert_base_texture_format_to_base_color_format(base,color_base)) return nil;
            const uint32_t bits=gxm::bits_per_pixel(base);
            if (!bits || bits%8) return nil;
            const uint64_t layout_width=std::bit_ceil(width),layout_height=std::bit_ceil(height);
            uint64_t chain=0;
            for (uint64_t w=layout_width,h=layout_height;w && h;w>>=1,h>>=1) chain+=w*h*bits/8;
            const uint64_t alignment=((width>=32 && height>=32 && bits<=8)
                || (width>=16 && height>=16 && (bits==16 || bits==32))
                || (width>=8 && height>=8 && bits==64)) ? 2048 : 4;
            const std::array<uint64_t,2> steps{(layout_width*layout_height*bits/8+3)&~uint64_t(3),
                (chain+alignment-1)&~(alignment-1)};
            const uint64_t address=uint64_t(texture.data_addr)<<2;
            const auto selection = find_color_cube_faces(impl->surfaces, address, steps,
                [&](const Surface &surface) {
                    return surface.guest.width == width && surface.guest.height == height
                        && gxm::get_base_format(surface.guest.colorFormat) == color_base && surface.color;
                }, texture_surface_usable);
            if (!selection) return nil;
            const auto &[selected_step, faces] = *selection;
            const auto &first=*faces[0];
            for (const auto *face:faces)
                if (face->guest.colorFormat!=first.guest.colorFormat
                    || face->color.pixelFormat!=first.color.pixelFormat
                    || face->color.width!=first.color.width || face->color.height!=first.color.height
                    || face->color.textureType!=MTLTextureType2D || face->color.sampleCount!=1
                    || face->color.width!=face->color.height) return nil;
            const bool raw_cast=surface_raw_cast_required(first.guest.colorFormat,base);
            const bool cast=raw_cast || !surface_texture_format_matches(first.guest.colorFormat,format);
            if (cast && !surface_format_cast_enqueueable(first.guest.colorFormat,base,texture.swizzle_format)) return nil;
            const bool rgba8=first.color.pixelFormat==MTLPixelFormatRGBA8Unorm
                || first.color.pixelFormat==MTLPixelFormatRGBA8Unorm_sRGB;
            if (!raw_cast && texture.gamma_mode && !rgba8) return nil;
            raw=raw_cast;
            const auto descriptor=texture_image_descriptor(texture);
            std::string key(reinterpret_cast<const char *>(&descriptor),sizeof(descriptor));
            const auto append_wide=[&](uint64_t value) { append(key,uint32_t(value));append(key,uint32_t(value>>32)); };
            append_wide(selected_step);append(key,raw_cast);append(key,uint32_t(first.guest.colorFormat));
            for (const auto *face:faces) {
                append_wide(reinterpret_cast<uintptr_t>((__bridge void *)face->color));
                append_wide(reinterpret_cast<uintptr_t>((__bridge void *)raw_surface_color(*face)));
                append_wide(face->revision);
            }
            if (const auto cached=impl->rendered_cubes.find(key);cached!=impl->rendered_cubes.end()) {
                if (capture_draw || impl->sync_draws) finish(ctx);
                return cached->second.texture;
            }
            const bool queued=!capture_draw && !impl->sync_draws;
            if (queued) mid_scene_flush(ctx,false);
            else finish(ctx);
            if (queued && !ctx.impl->commands) ctx.impl->commands=scene_command_buffer(*impl->device);
            auto commands=queued ? ctx.impl->commands : scene_command_buffer(*impl->device);
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            if (queued && ctx.impl->expanded_color && std::any_of(faces.begin(),faces.end(),[&](const Surface *face) {
                return face->color==ctx.impl->color;
            })) expand_color_samples(ctx,*impl->caster,commands);
            std::array<id<MTLTexture>,6> images;
            const SceGxmColorFormat *rendered_format=&first.guest.colorFormat;
            const auto repacked_color=cast ? repacked_u2_color(format) : std::nullopt;
            if (cast) rendered_format=repacked_color ? &*repacked_color : nullptr;
            const bool gamma=!raw_cast && (texture.gamma_mode || first.color.pixelFormat==MTLPixelFormatRGBA8Unorm_sRGB);
            for (uint32_t face=0;face<6;++face) {
                images[face]=cast ? impl->caster->surface_format_cast(raw_surface_color(*faces[face]),
                    faces[face]->guest.colorFormat,base,texture.swizzle_format,commands,raw_cast) : faces[face]->color;
                require(images[face]!=nil,"Metal: rendered cube face cast failed");
                if (gamma) images[face]=impl->caster->rgba8_surface_sampling(images[face],
                    rendered_format ? *rendered_format : SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,texture.gamma_mode,commands);
            }
            auto image=impl->caster->rendered_cube(images,commands);
            if (!raw_cast) image=sampling_view(image,format,gamma ? nullptr : rendered_format);
            if (!queued) {
                std::string error;
                require(impl->device->submit_and_wait(commands,error),error);
            }
            if (impl->rendered_cubes.size()>=4) impl->rendered_cubes.erase(impl->rendered_cubes.begin());
            auto &cached=impl->rendered_cubes[key];
            cached.texture=image;
            for (const auto *face:faces) {
                cached.sources.push_back(face->color);
                cached.sources.push_back(raw_surface_color(*face));
            }
            return image;
        };
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            const uint32_t slot=index%SCE_GXM_MAX_TEXTURE_UNITS;
            if (!vertex && !fragment_resources) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program):static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[slot]) continue;
            const auto &texture=ctx.textures[index];
            const bool cube=((vertex ? vs.cube_texture_mask : fs.cube_texture_mask)&(1u<<slot))!=0;
            const auto texture_type=texture.texture_type();
            if (cube) {
                bool raw=false;
                prepared_images[index]=rendered_cube(texture,raw);
                if (prepared_images[index]) { texture_mip_info[index].control[3]=raw;continue; }
            }
            const bool cube_storage=texture_type==SCE_GXM_TEXTURE_SWIZZLED
                || texture_type==SCE_GXM_TEXTURE_SWIZZLED_ARBITRARY
                || texture_type==SCE_GXM_TEXTURE_CUBE
                || texture_type==SCE_GXM_TEXTURE_CUBE_ARBITRARY;
            if (cube && !cube_storage) {
                // A game may bind its small linear fallback texture to a cube
                // sampler. Decode the one real 2D image, then repeat it on all
                // faces instead of reading six nonexistent guest faces.
                const Address address=texture.data_addr<<2;
                const size_t bytes=texture_storage_size(texture);
                const uint64_t end=uint64_t(address)+bytes;
                if (!address || end>uint64_t(UINT32_MAX)-4095
                    || !is_valid_addr_range(mem,address,Address(end))) {
                    if (!impl->invalid_cube_fallback)
                        impl->invalid_cube_fallback=sampled_placeholder(*impl->device,true);
                    prepared_images[index]=impl->invalid_cube_fallback;
                    continue;
                }
                bool raw_cast_requested=false;
                for (auto &[base,surface]:impl->surfaces) {
                    const size_t size=surface_memory_size(surface.guest);
                    if (size && uint64_t(base)<end && uint64_t(base)+size>address && texture_surface_usable(surface)) {
                        raw_cast_requested|=surface_raw_cast_required(surface.guest.colorFormat,
                            gxm::get_base_format(gxm::get_format(texture)));
                        require(sync_surface(mem,surface.guest),"Metal: cannot publish cube fallback alias");
                    }
                }
                texture_cache.cache_and_bind_image(texture,mem);
                id<MTLTexture> source=current_texture(texture_cache);
                require(source && source.textureType==MTLTextureType2D
                    && source.width==source.height,
                    "Metal: cube fallback requires a square 2D source");
                const uintptr_t key=reinterpret_cast<uintptr_t>((__bridge void *)source);
                auto found=impl->cube_aliases.find(key);
                if (found==impl->cube_aliases.end()) {
                    finish(ctx);
                    auto desc=[MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:source.pixelFormat
                        size:source.width mipmapped:NO];
                    desc.mipmapLevelCount=source.mipmapLevelCount;
                    desc.storageMode=MTLStorageModePrivate;
                    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
                    id<MTLTexture> replica=[impl->device->native_device() newTextureWithDescriptor:desc];
                    require(replica!=nil,"Metal: cube fallback allocation failed");
                    auto commands=[impl->device->command_queue() commandBuffer];
                    auto blit=[commands blitCommandEncoder];
                    for (uint32_t mip=0;mip<source.mipmapLevelCount;++mip) {
                        const uint32_t width=std::max(1u,uint32_t(source.width)>>mip);
                        for (uint32_t face=0;face<6;++face)
                            [blit copyFromTexture:source sourceSlice:0 sourceLevel:mip
                                sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(width,width,1)
                                toTexture:replica destinationSlice:face destinationLevel:mip
                                destinationOrigin:MTLOriginMake(0,0,0)];
                    }
                    [blit endEncoding];
                    std::string error;
                    require(impl->device->submit_and_wait(commands,error),error);
                    if (impl->cube_aliases.size()>=8) impl->cube_aliases.erase(impl->cube_aliases.begin());
                    found=impl->cube_aliases.emplace(key,Impl::CubeAlias{source,replica}).first;
                    LOG_INFO("Metal: repeated 2D fallback texture over six cube faces at {:#x}",address);
                }
                if (raw_cast_requested && raw_texture_snapshot_supported(found->second.cube)) {
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    prepared_images[index]=impl->caster->raw_texture_snapshot(found->second.cube);
                    require(prepared_images[index]!=nil,"Metal: repeated cube raw surface carrier failed");
                    texture_mip_info[index].control[3]=1;
                } else prepared_images[index]=sampling_view(found->second.cube,gxm::get_format(texture),nullptr);
                continue;
            }
            const Address texture_address=texture.data_addr<<2;
            const auto texture_base=gxm::get_base_format(gxm::get_format(texture));
            bool incompatible_alias=false;
            auto source=find_texture_surface(texture_address);
            const auto word_owner=find_word_subrectangle(texture);
            if (word_owner!=impl->surfaces.end()) source=word_owner;
            const auto halfword_owner=find_halfword_subrectangle(texture);
            if (halfword_owner!=impl->surfaces.end()) source=halfword_owner;
            const auto byte_owner=find_byte_subrectangle(texture);
            if (byte_owner!=impl->surfaces.end()) source=byte_owner;
            const auto rgba8_byte_owner=find_rgba8_byte_subrectangle(texture);
            if (rgba8_byte_owner!=impl->surfaces.end()) source=rgba8_byte_owner;
            const auto wide_small_owner=find_wide_small_subrectangle(texture);
            if (wide_small_owner!=impl->surfaces.end()) source=wide_small_owner;
            const auto small32_owner=find_32_small_subrectangle(texture);
            if (small32_owner!=impl->surfaces.end()) source=small32_owner;
            // Persona 4 describes a 1024x1024 texture over a 960x544 target.
            // Sample its rendered prefix directly, preserving guest texel
            // coordinates without reading the unallocated trailing rows.
            if (!cube && features.use_texture_viewport && source!=impl->surfaces.end()) {
                surface_viewports[index]=surface_texture_viewport(source->second.guest,texture);
                if (surface_viewports[index]) continue;
            }
            if (source!=impl->surfaces.end() && surface_memory_size(source->second.guest)) {
                const bool direct_word_alias=rgba16_linear_alias(source->second,texture,texture_address)
                    || rg32_linear_alias(source->second,texture,texture_address)
                    || surface_word_subrectangle(source->second.guest,texture).has_value();
                const bool direct_halfword_alias=surface_halfword_subrectangle(source->second.guest,texture).has_value();
                const bool direct_byte_alias=surface_byte_subrectangle(source->second.guest,texture).has_value();
                const bool direct_rgba8_byte_alias=surface_rgba8_byte_subrectangle(source->second.guest,texture).has_value();
                const bool direct_wide_small_alias=surface_wide_small_subrectangle(source->second.guest,texture).has_value();
                const bool direct_32_small_alias=(gxm::get_base_format(source->second.guest.colorFormat)
                        != SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
                    || source->second.color.pixelFormat==MTLPixelFormatRGB9E5Float)
                    && surface_32_small_subrectangle(source->second.guest,texture).has_value();
                // A native surface can only be sampled directly when both
                // its byte layout and format match the guest texture view.
                incompatible_alias=!direct_word_alias && !direct_halfword_alias
                    && !direct_byte_alias && !direct_rgba8_byte_alias
                    && !direct_wide_small_alias && !direct_32_small_alias
                    && !surface_subrectangle(source->second.guest,texture);
            }
            if (surface_word_target_supported(texture_base)) {
                if (source==impl->surfaces.end() && texture_address>=4)
                    source=find_texture_surface(texture_address-4);
                incompatible_alias|=source!=impl->surfaces.end()
                    && ((source->second.color.pixelFormat==MTLPixelFormatRG32Float
                            && !rg32_linear_alias(source->second,texture,texture_address)
                            && !surface_word_subrectangle(source->second.guest,texture))
                        || (source->second.color.pixelFormat==MTLPixelFormatRGBA16Float
                            && gxm::get_base_format(source->second.guest.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16
                            && !rgba16_linear_alias(source->second,texture,texture_address)
                            && !surface_word_subrectangle(source->second.guest,texture)));
            }
            const bool single_mip = texture.texture_type()==SCE_GXM_TEXTURE_LINEAR_STRIDED
                || texture::get_upload_mip(texture.true_mip_count(),gxm::get_width(texture),gxm::get_height(texture))<=1;
            bool direct_precise_surface = false;
            if (!features.use_texture_viewport && !cube && single_mip) {
                // Plus crops a compatible GPU owner even without texture
                // viewports. Keep the same age/layout selection here: a RAM
                // upload would instead observe a recent CPU-dirty surface's
                // older guest bytes. The queued crop/feedback passes below
                // handle both inactive producers and the current attachment.
                const auto owner = find_subrectangle(texture);
                direct_precise_surface = owner != impl->surfaces.end()
                    && (surface_texture_format_matches(owner->second.guest.colorFormat, gxm::get_format(texture))
                        || surface_format_cast_enqueueable(owner->second.guest.colorFormat,
                            texture_base, texture.swizzle_format));
                direct_precise_surface |= word_owner!=impl->surfaces.end();
                direct_precise_surface |= halfword_owner!=impl->surfaces.end();
                direct_precise_surface |= byte_owner!=impl->surfaces.end();
                direct_precise_surface |= rgba8_byte_owner!=impl->surfaces.end();
                direct_precise_surface |= wide_small_owner!=impl->surfaces.end();
                direct_precise_surface |= small32_owner!=impl->surfaces.end();
            }
            if (!cube && single_mip
                && ((!incompatible_alias && features.use_texture_viewport) || direct_precise_surface)) continue;
            const auto upload=cube?cube_texture_descriptor(texture):texture;
            const Address address=texture.data_addr<<2;
            const size_t storage=texture_storage_size(upload), face_stride=cube?storage/6:storage;
            const uint64_t end=uint64_t(address)+storage;
            if (!address || end>uint64_t(UINT32_MAX)-4095
                || !is_valid_addr_range(mem,address,Address(end))) {
                texture_cache.cache_and_bind_image(upload,mem);
                prepared_images[index]=current_texture_view(texture_cache,gxm::get_format(texture));
                continue;
            }
            std::vector<Surface *> overlaps;
            for (auto &[base,surface]:impl->surfaces) {
                const size_t bytes=surface_memory_size(surface.guest);
                if (bytes && uint64_t(base)<end && uint64_t(base)+bytes>address && texture_surface_usable(surface)) overlaps.push_back(&surface);
            }
            // Without a color producer, retain the ordinary 2D path, including
            // its depth-surface aliases. Cubes still need their six-face upload.
            if (overlaps.empty() && !cube) continue;
            if (overlaps.empty()) {
                texture_cache.cache_and_bind_image(upload,mem);
                prepared_images[index]=current_texture_view(texture_cache,gxm::get_format(texture));
                continue;
            }
            const bool raw_cast_requested=std::any_of(overlaps.begin(),overlaps.end(),[&](const Surface *surface) {
                return surface_raw_cast_required(surface->guest.colorFormat,texture_base);
            });
            if (!features.use_texture_viewport) {
                // Precise mode samples the guest texture grid. Publish each
                // overlapping GPU producer before decoding its guest layout.
                for (auto *surface : overlaps)
                    require(sync_surface(mem, surface->guest), "Metal: cannot publish precise texture alias");
                texture_cache.cache_and_bind_image(upload, mem);
                prepared_images[index] = current_texture_view(texture_cache, gxm::get_format(texture));
                if (raw_cast_requested && raw_texture_snapshot_supported(current_texture(texture_cache))) {
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    prepared_images[index]=impl->caster->raw_texture_snapshot(current_texture(texture_cache));
                    require(prepared_images[index]!=nil,"Metal: precise raw surface carrier failed");
                    texture_mip_info[index].control[3]=1;
                }
                continue;
            }
            const uint32_t width=gxm::get_width(upload), height=gxm::get_height(upload);
            const uint32_t mips=texture::get_upload_mip(upload.true_mip_count(),width,height);
            auto &mip_info=texture_mip_info[index];
            for (uint32_t mip=0;mip<mips;++mip) mip_info.sizes[mip]={std::max(1u,width>>mip),std::max(1u,height>>mip)};
            const auto base_format=gxm::get_base_format(gxm::get_format(upload));
            const uint32_t bits=gxm::bits_per_pixel(base_format);
            const auto [block_width,block_height]=gxm::get_block_size(base_format);
            const auto type=upload.texture_type();
            const size_t align_width=std::max(block_width,type==SCE_GXM_TEXTURE_LINEAR?8u:type==SCE_GXM_TEXTURE_TILED?32u:1u);
            const size_t align_height=std::max(block_height,type==SCE_GXM_TEXTURE_TILED?32u:1u);
            struct RenderedFace { Surface *surface; uint32_t face,mip; bool cast; };
            std::vector<RenderedFace> rendered;
            const float scale=res_multiplier;
            for (auto *surface:overlaps) {
                bool matched=false;
                const bool same_format=surface_texture_format_matches(surface->guest.colorFormat,gxm::get_format(upload));
                const bool can_cast=surface_format_cast_supported(surface->guest.colorFormat,base_format,upload.swizzle_format);
                if ((same_format || can_cast) && !gxm::is_block_compressed_format(base_format)
                    && gxm::bits_per_pixel(gxm::get_base_format(surface->guest.colorFormat))==bits) {
                    size_t offset=0;
                    for (uint32_t mip=0;mip<mips;++mip) {
                        const uint32_t w=std::max(1u,width>>mip),h=std::max(1u,height>>mip);
                        for (uint32_t face=0;face<(cube?6u:1u);++face) {
                            auto level=upload;
                            level.mip_count=15;
                            level.data_addr=(uint64_t(address)+face*face_stride+offset)>>2;
                            if (cube) level.type=(type==SCE_GXM_TEXTURE_CUBE?SCE_GXM_TEXTURE_SWIZZLED:SCE_GXM_TEXTURE_SWIZZLED_ARBITRARY)>>29;
                            if (level.texture_type()==SCE_GXM_TEXTURE_SWIZZLED) {
                                level.width_base2=std::bit_width(w)-1;level.height_base2=std::bit_width(h)-1;
                            } else { level.width=w-1;level.height=h-1; }
                            const auto rect=surface_subrectangle(surface->guest,level);
                            if (!rect || rect->x || rect->y || rect->width!=surface->guest.width || rect->height!=surface->guest.height) continue;
                            rendered.push_back({surface,face,mip,!same_format});matched=true;
                            if (!cube) mip_info.sizes[mip]={uint32_t(surface->color.width),uint32_t(surface->color.height)};
                            require(surface->color.width==std::max(1u,uint32_t(double(w)*scale))
                                && surface->color.height==std::max(1u,uint32_t(double(h)*scale)),
                                "Metal: incompatible rendered texture resolution");
                        }
                        const size_t layout_w=std::bit_ceil(uint32_t(width))>>mip, layout_h=std::bit_ceil(uint32_t(height))>>mip;
                        offset+=((layout_w+align_width-1)/align_width)*align_width
                            *((layout_h+align_height-1)/align_height)*align_height*bits/8;
                    }
                }
                // For a different memory layout/format, reinterpret the freshly
                // published guest bytes through the ordinary upload decoder.
                if (!matched) require(sync_surface(mem,surface->guest),"Metal: cannot publish texture memory alias");
            }
            texture_cache.cache_and_bind_image(upload,mem);
            auto base_image=current_texture(texture_cache);
            auto uploaded=current_texture_view(texture_cache,gxm::get_format(texture));
            // A replacement image can use a different byte width. It has no
            // guest word representation; retain its existing sampled path.
            const bool raw_cast=raw_cast_requested && raw_texture_snapshot_supported(base_image);
            mip_info.control[3]=raw_cast;
            if (rendered.empty()) {
                if (raw_cast) {
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    uploaded=impl->caster->raw_texture_snapshot(base_image);
                    require(uploaded!=nil,"Metal: uploaded raw surface carrier failed");
                }
                prepared_images[index]=uploaded;continue;
            }
            if (!cube) for (uint32_t mip=0;mip<mips;++mip)
                mip_info.control[0]|=mip_info.sizes[mip]!=std::array<uint32_t,2>{
                    std::max(1u,uint32_t(double(uploaded.width)*scale)>>mip),
                    std::max(1u,uint32_t(double(uploaded.height)*scale)>>mip)};

            const auto image_descriptor=texture_image_descriptor(upload);
            std::string image_key(reinterpret_cast<const char *>(&image_descriptor),sizeof(image_descriptor));
            const auto append_wide=[&](uint64_t value) { append(image_key,uint32_t(value));append(image_key,uint32_t(value>>32)); };
            append_wide(reinterpret_cast<uintptr_t>((__bridge void *)base_image));
            append(image_key,std::bit_cast<uint32_t>(scale));
            append(image_key,raw_cast);
            for (const auto &face:rendered) {
                append_wide(reinterpret_cast<uintptr_t>((__bridge void *)face.surface->color));
                append_wide(reinterpret_cast<uintptr_t>((__bridge void *)raw_surface_color(*face.surface)));
                append_wide(face.surface->revision);append(image_key,face.face);append(image_key,face.mip);
            }
            if (auto found=impl->rendered_images.find(image_key);found!=impl->rendered_images.end()) {
                prepared_images[index]=found->second.texture;continue;
            }
            const bool queued_snapshot=!capture_draw && !impl->sync_draws
                && std::all_of(rendered.begin(),rendered.end(),[&](const RenderedFace &face) {
                    return (!face.cast && !raw_cast) || surface_format_cast_enqueueable(face.surface->guest.colorFormat,
                        base_format,upload.swizzle_format);
                });
            id<MTLCommandBuffer> snapshot_commands=nil;
            if (queued_snapshot) {
                mid_scene_flush(ctx,false);
                if (!ctx.impl->commands) ctx.impl->commands=scene_command_buffer(*impl->device);
                snapshot_commands=ctx.impl->commands;
                if (ctx.impl->expanded_color && std::any_of(rendered.begin(),rendered.end(),[&](const RenderedFace &face) {
                    return face.surface->color==ctx.impl->color;
                })) {
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    expand_color_samples(ctx,*impl->caster,snapshot_commands);
                }
            } else finish(ctx);
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            if (raw_cast) {
                uploaded=impl->caster->raw_texture_snapshot(base_image,snapshot_commands);
                require(uploaded!=nil,"Metal: mip/cube raw surface carrier failed");
            }
            std::vector<CubeSurface> sources;
            for (const auto &face:rendered) {
                auto native=face.cast || raw_cast ? raw_surface_color(*face.surface) : face.surface->color;
                const SceGxmColorFormat *format=&face.surface->guest.colorFormat;
                std::optional<SceGxmColorFormat> repacked_color;
                if (face.cast || raw_cast) {
                    native=impl->caster->surface_format_cast(native,*format,base_format,upload.swizzle_format,
                        snapshot_commands,raw_cast);
                    require(native!=nil,"Metal: incompatible rendered texture format cast");
                    repacked_color=repacked_u2_color(gxm::get_format(texture));
                    format=repacked_color ? &*repacked_color : nullptr;
                }
                if (!raw_cast && (native.pixelFormat==MTLPixelFormatRGBA8Unorm_sRGB || texture.gamma_mode)) {
                    native=impl->caster->rgba8_surface_sampling(native,
                        format?*format:SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,texture.gamma_mode,
                        snapshot_commands);format=nullptr;
                }
                sources.push_back({raw_cast ? native : sampling_view(native,gxm::get_format(texture),format),face.face,face.mip});
            }
            prepared_images[index]=impl->caster->texture_snapshot(uploaded,sources,scale,snapshot_commands);
            if (impl->rendered_images.size()>=4) impl->rendered_images.erase(impl->rendered_images.begin());
            auto &cached=impl->rendered_images[image_key];
            cached.texture=prepared_images[index];cached.uploaded=base_image;
            for (const auto &face:rendered) cached.sources.push_back(face.surface->color);
            if (impl->trace_textures) LOG_INFO("Metal rendered texture assembled: address={:#x} faces_mips={} scale={}",address,rendered.size(),scale);
        }
        bool inline_color_expanded=false;
        std::array<bool,SCE_GXM_MAX_TEXTURE_UNITS*2> inline_crops{};
        std::array<id<MTLTexture>,SCE_GXM_MAX_TEXTURE_UNITS*2> inline_gamma_images{};
        std::array<id<MTLTexture>,SCE_GXM_MAX_TEXTURE_UNITS*2> inline_rg32_images{};
        std::array<id<MTLTexture>,SCE_GXM_MAX_TEXTURE_UNITS*2> inline_cast_images{};
        const auto prepare_inline_commands = [&] {
            mid_scene_flush(ctx,false);
            if (!ctx.impl->commands)
                ctx.impl->commands=scene_command_buffer(*impl->device);
        };
        const auto prepare_inline_color = [&] {
            prepare_inline_commands();
            if (ctx.impl->expanded_color && !inline_color_expanded) {
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                expand_color_samples(ctx,*impl->caster,ctx.impl->commands);
                inline_color_expanded=true;
            }
        };
        // Prepare separate images before opening/binding this draw's encoder.
        // This also handles feedback from an offset inside the active target.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if (!vertex && !fragment_resources) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program):static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            if (prepared_images[index]) continue;
            const auto &texture=ctx.textures[index]; auto found=find_subrectangle(texture);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second; const auto rect=*surface_subrectangle(entry.guest,texture);
            const bool cropped=rect.x || rect.y || rect.width!=entry.guest.width || rect.height!=entry.guest.height;
            const bool same_format=surface_texture_format_matches(entry.guest.colorFormat,gxm::get_format(texture));
            const auto texture_base=gxm::get_base_format(gxm::get_format(texture));
            const bool gpu_cast=!same_format && !texture.gamma_mode
                && surface_format_cast_enqueueable(entry.guest.colorFormat,texture_base,texture.swizzle_format);
            const bool raw_cast=surface_raw_cast_required(entry.guest.colorFormat,texture_base);
            if (surface_texture_needs_native_resolution(entry.guest.colorFormat, texture,
                    res_multiplier, features.use_texture_viewport)
                && (same_format || surface_format_cast_enqueueable(entry.guest.colorFormat,
                    texture_base, texture.swizzle_format))) {
                // Restore the guest texel grid for small point-filtered data
                // textures at fractional resolution, as in Plus' cast path.
                // Do not publish through RAM: recent GPU owners can coexist
                // with newer CPU bytes that this texture reader must preserve.
                if (capture_draw) {
                    finish(ctx, false, true);
                    inline_color_expanded = false;
                } else if (entry.color == ctx.impl->color) prepare_inline_color();
                else prepare_inline_commands();
                if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                auto source = same_format ? entry.color : raw_surface_color(entry);
                const uint32_t x = std::min<uint32_t>(uint32_t(rect.x * double(res_multiplier)), source.width - 1);
                const uint32_t y = std::min<uint32_t>(uint32_t(rect.y * double(res_multiplier)), source.height - 1);
                const uint32_t w = std::min<uint32_t>(std::max(1u, uint32_t(rect.width * double(res_multiplier))), source.width - x);
                const uint32_t h = std::min<uint32_t>(std::max(1u, uint32_t(rect.height * double(res_multiplier))), source.height - y);
                const auto commands = capture_draw ? nil : ctx.impl->commands;
                auto native = impl->caster->resample_publication(source, rect.width, rect.height,
                    {x, y, w, h}, {0, 0, rect.width, rect.height}, !same_format, commands);
                std::optional<SceGxmColorFormat> format = entry.guest.colorFormat;
                if (!same_format) {
                    native = impl->caster->surface_format_cast(native, entry.guest.colorFormat,
                        texture_base, texture.swizzle_format, commands);
                    require(native != nil, "Metal: native-resolution texture reinterpretation failed");
                    format = repacked_u2_color(gxm::get_format(texture));
                }
                if (native.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB || texture.gamma_mode) {
                    native = impl->caster->rgba8_surface_sampling(native,
                        format ? *format : SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR, texture.gamma_mode, commands);
                    format.reset();
                }
                prepared_images[index] = sampling_view(native, gxm::get_format(texture), format ? &*format : nullptr);
                continue;
            }
            if (!cropped) {
                if (gpu_cast && !raw_cast && entry.color!=ctx.impl->color && !capture_draw) {
                    prepare_inline_commands();
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    inline_cast_images[index]=impl->caster->surface_format_cast(raw_surface_color(entry),
                        entry.guest.colorFormat,texture_base,texture.swizzle_format,ctx.impl->commands);
                    require(inline_cast_images[index]!=nil,"Metal: queued inactive color format cast failed");
                }
                continue;
            }
            auto &image=entry.subrectangles[{rect.x,rect.y,rect.width,rect.height}];
            // A compatible crop, including gamma sampling, can be copied and
            // decoded after the producer in the active command buffer.
            inline_crops[index]=(same_format || gpu_cast) && !capture_draw;
            if (!image) {
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                if (inline_crops[index]) {
                    if (entry.color==ctx.impl->color) prepare_inline_color();
                    else prepare_inline_commands();
                    image=impl->caster->enqueue_subrectangle(raw_surface_color(entry),entry.guest.width,entry.guest.height,
                        rect,ctx.impl->commands);
                } else {
                    finish(ctx);
                    inline_color_expanded=false;
                    image=impl->caster->color_subrectangle(raw_surface_color(entry),entry.guest,rect);
                }
            }
            if (inline_crops[index] && gpu_cast && !raw_cast) {
                if (entry.color==ctx.impl->color) prepare_inline_color();
                else prepare_inline_commands();
                inline_cast_images[index]=impl->caster->surface_format_cast(image,
                    entry.guest.colorFormat,texture_base,texture.swizzle_format,ctx.impl->commands);
                require(inline_cast_images[index]!=nil,"Metal: inline color crop format cast failed");
            }
            if (inline_crops[index] && same_format && !raw_cast
                && (image.pixelFormat==MTLPixelFormatRGBA8Unorm_sRGB || texture.gamma_mode)) {
                prepare_inline_color();
                inline_gamma_images[index]=impl->caster->rgba8_surface_sampling(image,
                    entry.guest.colorFormat,texture.gamma_mode,ctx.impl->commands);
            }
        }
        id<MTLTexture> color_feedback = nil;
        id<MTLTexture> raw_color_feedback = nil;
        bool inline_feedback=false;
        if (ctx.impl->color && !record.is_maskupdate
            && find_texture_surface(ctx.impl->guest_color.data.address()) != impl->surfaces.end()) {
            bool needs_feedback=false;
            inline_feedback=!capture_draw;
            for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
                const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
                if (!vertex && !fragment_resources) continue;
                const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program):static_cast<const ShaderProgram &>(*fp->fragment_program);
                if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
                if (prepared_images[index]) continue;
                const auto &texture=ctx.textures[index];
                const Address address=texture.data_addr<<2;
                const auto base=gxm::get_base_format(gxm::get_format(texture));
                const bool word_alias=uint64_t(address)==uint64_t(ctx.impl->guest_color.data.address())+4
                    && ((ctx.impl->color.pixelFormat==MTLPixelFormatRG32Float
                            && surface_word_target_supported(base))
                        || (ctx.impl->color.pixelFormat==MTLPixelFormatRGBA16Float
                            && surface_word_target_supported(base)
                            && gxm::get_base_format(ctx.impl->guest_color.colorFormat)==SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16));
                const auto word_region=find_word_subrectangle(texture);
                const bool region_alias=word_region!=impl->surfaces.end()
                    && word_region->first==ctx.impl->guest_color.data.address();
                const auto halfword_region=find_halfword_subrectangle(texture);
                const bool halfword_alias=halfword_region!=impl->surfaces.end()
                    && halfword_region->first==ctx.impl->guest_color.data.address();
                const auto byte_region=find_byte_subrectangle(texture);
                const bool byte_alias=byte_region!=impl->surfaces.end()
                    && byte_region->first==ctx.impl->guest_color.data.address();
                const auto rgba8_byte_region=find_rgba8_byte_subrectangle(texture);
                const bool rgba8_byte_alias=rgba8_byte_region!=impl->surfaces.end()
                    && rgba8_byte_region->first==ctx.impl->guest_color.data.address();
                const auto wide_small_region=find_wide_small_subrectangle(texture);
                const bool wide_small_alias=wide_small_region!=impl->surfaces.end()
                    && wide_small_region->first==ctx.impl->guest_color.data.address();
                const auto small32_region=find_32_small_subrectangle(texture);
                const bool small32_alias=small32_region!=impl->surfaces.end()
                    && small32_region->first==ctx.impl->guest_color.data.address();
                if (address==ctx.impl->guest_color.data.address() || word_alias || region_alias
                    || halfword_alias || byte_alias || rgba8_byte_alias || wide_small_alias
                    || small32_alias) {
                    needs_feedback=true;
                    const auto active=impl->surfaces.find(ctx.impl->guest_color.data.address());
                    const bool packed_rg32=ctx.impl->color.pixelFormat==MTLPixelFormatRG32Float
                        && (base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 || base==SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8)
                        && active!=impl->surfaces.end() && active->second.color==ctx.impl->color
                        && rg32_linear_alias(active->second,texture,address)
                        && !texture.gamma_mode && impl->dump_surface_dir.empty();
                    const bool same_format=!word_alias
                        && surface_texture_format_matches(ctx.impl->guest_color.colorFormat,gxm::get_format(texture));
                    const bool gpu_cast=!word_alias && !texture.gamma_mode
                        && active!=impl->surfaces.end() && active->second.color==ctx.impl->color
                        && surface_subrectangle(active->second.guest,texture).has_value()
                        && surface_format_cast_enqueueable(ctx.impl->guest_color.colorFormat,base,texture.swizzle_format);
                    inline_feedback &= packed_rg32 || same_format || gpu_cast;
                }
            }
            if (needs_feedback) {
                if (inline_feedback) prepare_inline_color();
                else { finish(ctx); inline_color_expanded=false; }
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                color_feedback=impl->caster->color_snapshot(ctx.impl->color,
                    inline_feedback ? ctx.impl->commands : nil);
                if (const auto active=impl->surfaces.find(ctx.impl->guest_color.data.address());
                    active!=impl->surfaces.end() && active->second.raw_color && !active->second.raw_color_invalidated)
                    raw_color_feedback=impl->caster->color_snapshot(active->second.raw_color,
                        inline_feedback ? ctx.impl->commands : nil);
                if (inline_feedback) for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
                    const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
                    if (!vertex && !fragment_resources) continue;
                    const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program):static_cast<const ShaderProgram &>(*fp->fragment_program);
                    if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS] || prepared_images[index]) continue;
                    const auto &texture=ctx.textures[index];
                    const Address address=texture.data_addr<<2;
                    const auto base=gxm::get_base_format(gxm::get_format(texture));
                    if (color_feedback.pixelFormat==MTLPixelFormatRG32Float
                        && (base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 || base==SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8)
                        && (address==ctx.impl->guest_color.data.address()
                            || uint64_t(address)==uint64_t(ctx.impl->guest_color.data.address())+4)) {
                        const auto active=impl->surfaces.find(ctx.impl->guest_color.data.address());
                        require(active!=impl->surfaces.end() && rg32_linear_alias(active->second,texture,address),
                            "Metal: active RG32 alias lost its validated row layout");
                        const bool signed_normalized=base==SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8;
                        const bool separate_word = res_multiplier != 1 && active->second.has_word_offset_view;
                        inline_rg32_images[index]=impl->caster->rgba8_from_rg32(color_feedback,
                            (active->second.guest.colorFormat & SCE_GXM_COLOR_SWIZZLE_MASK)==SCE_GXM_COLOR_SWIZZLE2_RG,
                            (address-ctx.impl->guest_color.data.address())/4,signed_normalized,
                            res_multiplier<1 && !separate_word && vertex ? active->second.guest.width : 0,
                            res_multiplier<1 && !separate_word && vertex ? active->second.guest.height : 0,
                            ctx.impl->commands, separate_word);
                        continue;
                    }
                    const bool same_format=surface_texture_format_matches(
                        ctx.impl->guest_color.colorFormat,gxm::get_format(texture));
                    if (address==ctx.impl->guest_color.data.address() && !same_format
                        && !surface_raw_cast_required(ctx.impl->guest_color.colorFormat,base)
                        && surface_format_cast_enqueueable(ctx.impl->guest_color.colorFormat,base,texture.swizzle_format)) {
                        inline_cast_images[index]=impl->caster->surface_format_cast(raw_color_feedback ? raw_color_feedback : color_feedback,
                            ctx.impl->guest_color.colorFormat,base,texture.swizzle_format,ctx.impl->commands);
                        require(inline_cast_images[index]!=nil,"Metal: inline color feedback format cast failed");
                        continue;
                    }
                    if ((texture.data_addr<<2)!=ctx.impl->guest_color.data.address()
                        || surface_raw_cast_required(ctx.impl->guest_color.colorFormat,base)
                        || (color_feedback.pixelFormat!=MTLPixelFormatRGBA8Unorm_sRGB && !texture.gamma_mode)) continue;
                    inline_gamma_images[index]=impl->caster->rgba8_surface_sampling(color_feedback,
                        ctx.impl->guest_color.colorFormat,texture.gamma_mode,ctx.impl->commands);
                }
            }
        }
        // Plus reinterprets cropped 64-bit stores by byte offset. Reconstruct
        // the complete word grid before cropping so the requested row pitch
        // and high-word phase keep their guest-memory meaning.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if ((!vertex && !fragment_resources) || prepared_images[index]) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program)
                : static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            const auto &texture=ctx.textures[index];
            auto found=find_word_subrectangle(texture);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second;
            const Address address=texture.data_addr<<2;
            if (rg32_linear_alias(entry,texture,address)
                || rgba16_linear_alias(entry,texture,address)) continue;
            const auto rect=*surface_word_subrectangle(entry.guest,texture);
            const auto base=gxm::get_base_format(gxm::get_format(texture));
            auto &crop=entry.word_rect_casts[{rect.x,rect.y,rect.width,rect.height,
                uint32_t(base),uint32_t(texture.swizzle_format)}];
            if (!crop) {
                const bool active=entry.color==ctx.impl->color && !record.is_maskupdate;
                if (active) require(color_feedback!=nil,"Metal: missing cropped word feedback snapshot");
                auto source=active ? (raw_color_feedback ? raw_color_feedback : color_feedback)
                    : raw_surface_color(entry);
                prepare_inline_commands();
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                id<MTLTexture> full=nil;
                if (source.pixelFormat==MTLPixelFormatRG32Float)
                    full=impl->caster->word_texture_from_rg32(source,base,texture.swizzle_format,
                        (entry.guest.colorFormat&SCE_GXM_COLOR_SWIZZLE_MASK)==SCE_GXM_COLOR_SWIZZLE2_RG,
                        0,0,0,ctx.impl->commands,false);
                else
                    full=impl->caster->word_texture_from_rgba16(source,entry.guest.colorFormat,
                        base,texture.swizzle_format,0,0,0,ctx.impl->commands,false);
                crop=impl->caster->enqueue_subrectangle(full,entry.guest.width*2,
                    entry.guest.height,rect,ctx.impl->commands);
            }
            auto native=crop;
            if (texture.gamma_mode && (base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                || base==SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8)) {
                prepare_inline_commands();
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                native=impl->caster->rgba8_surface_sampling(native,
                    SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,texture.gamma_mode,ctx.impl->commands);
            }
            prepared_images[index]=sampling_view(native,gxm::get_format(texture),nullptr);
        }
        // Split RGBA8 storage into guest-ordered 16-bit texels before a
        // partial crop. The cast stays on the GPU, including active feedback.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if ((!vertex && !fragment_resources) || prepared_images[index]) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program)
                : static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            const auto &texture=ctx.textures[index];
            auto found=find_halfword_subrectangle(texture);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second;
            const auto rect=*surface_halfword_subrectangle(entry.guest,texture);
            const auto base=gxm::get_base_format(gxm::get_format(texture));
            auto &crop=entry.halfword_rect_casts[{rect.x,rect.y,rect.width,rect.height,
                uint32_t(base),uint32_t(texture.swizzle_format)}];
            if (!crop) {
                const bool active=entry.color==ctx.impl->color && !record.is_maskupdate;
                if (active) require(color_feedback!=nil,"Metal: missing cropped halfword feedback snapshot");
                auto source=active ? (raw_color_feedback ? raw_color_feedback : color_feedback)
                    : raw_surface_color(entry);
                prepare_inline_commands();
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                auto full=impl->caster->halfword_texture_from_rgba8(source,entry.guest.colorFormat,
                    base,texture.swizzle_format,ctx.impl->commands);
                crop=impl->caster->enqueue_subrectangle(full,entry.guest.width*2,
                    entry.guest.height,rect,ctx.impl->commands);
            }
            prepared_images[index]=sampling_view(crop,gxm::get_format(texture),nullptr);
            narrow_cast_coords[index]=!vertex && rect.x==0 && rect.y==0
                && uint64_t(rect.width)==uint64_t(entry.guest.width)*2
                && rect.height==entry.guest.height;
        }
        // A 16-bit color texel can be sampled as two consecutive guest bytes.
        // Recover the byte order before cropping the linear view.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if ((!vertex && !fragment_resources) || prepared_images[index]) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program)
                : static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            const auto &texture=ctx.textures[index];
            auto found=find_byte_subrectangle(texture);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second;
            const auto rect=*surface_byte_subrectangle(entry.guest,texture);
            const auto base=gxm::get_base_format(gxm::get_format(texture));
            auto &crop=entry.byte_rect_casts[{rect.x,rect.y,rect.width,rect.height,
                uint32_t(base),uint32_t(texture.swizzle_format)}];
            if (!crop) {
                const bool active=entry.color==ctx.impl->color && !record.is_maskupdate;
                if (active) require(color_feedback!=nil,"Metal: missing cropped byte feedback snapshot");
                auto source=active ? (raw_color_feedback ? raw_color_feedback : color_feedback)
                    : raw_surface_color(entry);
                prepare_inline_commands();
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                auto full=impl->caster->byte_texture_from_16bit_surface(source,entry.guest.colorFormat,
                    base,ctx.impl->commands);
                crop=impl->caster->enqueue_subrectangle(full,entry.guest.width*2,
                    entry.guest.height,rect,ctx.impl->commands);
            }
            prepared_images[index]=sampling_view(crop,gxm::get_format(texture),nullptr);
            narrow_cast_coords[index]=!vertex && rect.x==0 && rect.y==0
                && uint64_t(rect.width)==uint64_t(entry.guest.width)*2
                && rect.height==entry.guest.height;
        }
        // A four-byte RGBA8 color texel may also be addressed as four
        // individual guest bytes, including a rectangle within the surface.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if ((!vertex && !fragment_resources) || prepared_images[index]) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program)
                : static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            const auto &texture=ctx.textures[index];
            auto found=find_rgba8_byte_subrectangle(texture);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second;
            const auto rect=*surface_rgba8_byte_subrectangle(entry.guest,texture);
            const auto base=gxm::get_base_format(gxm::get_format(texture));
            auto &crop=entry.byte_rect_casts[{rect.x,rect.y,rect.width,rect.height,
                uint32_t(base),uint32_t(texture.swizzle_format)}];
            if (!crop) {
                const bool active=entry.color==ctx.impl->color && !record.is_maskupdate;
                if (active) require(color_feedback!=nil,"Metal: missing RGBA8 byte feedback snapshot");
                auto source=active ? (raw_color_feedback ? raw_color_feedback : color_feedback)
                    : raw_surface_color(entry);
                prepare_inline_commands();
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                auto full=impl->caster->byte_texture_from_rgba8(source,entry.guest.colorFormat,
                    base,ctx.impl->commands);
                crop=impl->caster->enqueue_subrectangle(full,entry.guest.width*4,
                    entry.guest.height,rect,ctx.impl->commands);
            }
            prepared_images[index]=sampling_view(crop,gxm::get_format(texture),nullptr);
        }
        // Split 64-bit F16x4/RG32 storage into one- or two-byte guest texels.
        // Cropping after reconstruction keeps offsets in guest byte units.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if ((!vertex && !fragment_resources) || prepared_images[index]) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program)
                : static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            const auto &texture=ctx.textures[index];
            auto found=find_wide_small_subrectangle(texture);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second;
            const auto rect=*surface_wide_small_subrectangle(entry.guest,texture);
            const auto base=gxm::get_base_format(gxm::get_format(texture));
            const uint32_t bytes=gxm::bits_per_pixel(base)/8;
            auto &crop=entry.small_rect_casts[{rect.x,rect.y,rect.width,rect.height,
                uint32_t(base),uint32_t(texture.swizzle_format)}];
            if (!crop) {
                const bool active=entry.color==ctx.impl->color && !record.is_maskupdate;
                if (active) require(color_feedback!=nil,"Metal: missing small-word feedback snapshot");
                auto source=active ? (raw_color_feedback ? raw_color_feedback : color_feedback)
                    : raw_surface_color(entry);
                prepare_inline_commands();
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                auto full=impl->caster->small_texture_from_wide_surface(source,
                    entry.guest.colorFormat,base,texture.swizzle_format,ctx.impl->commands);
                crop=impl->caster->enqueue_subrectangle(full,entry.guest.width*(8/bytes),
                    entry.guest.height,rect,ctx.impl->commands);
            }
            prepared_images[index]=sampling_view(crop,gxm::get_format(texture),nullptr);
        }
        // Other native 32-bit stores (F32 and two 16-bit components) can
        // expose their raw guest bytes through the same narrow GPU caster.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if ((!vertex && !fragment_resources) || prepared_images[index]) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program)
                : static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            const auto &texture=ctx.textures[index];
            auto found=find_32_small_subrectangle(texture);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second;
            const auto rect=*surface_32_small_subrectangle(entry.guest,texture);
            const auto base=gxm::get_base_format(gxm::get_format(texture));
            const uint32_t bytes=gxm::bits_per_pixel(base)/8;
            auto &crop=entry.small_rect_casts[{rect.x,rect.y,rect.width,rect.height,
                uint32_t(base),uint32_t(texture.swizzle_format)}];
            if (!crop) {
                const bool active=entry.color==ctx.impl->color && !record.is_maskupdate;
                if (active) require(color_feedback!=nil,"Metal: missing 32-bit small feedback snapshot");
                auto source=active ? (raw_color_feedback ? raw_color_feedback : color_feedback)
                    : raw_surface_color(entry);
                prepare_inline_commands();
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                auto full=impl->caster->small_texture_from_wide_surface(source,
                    entry.guest.colorFormat,base,texture.swizzle_format,ctx.impl->commands);
                crop=impl->caster->enqueue_subrectangle(full,entry.guest.width*(4/bytes),
                    entry.guest.height,rect,ctx.impl->commands);
            }
            prepared_images[index]=sampling_view(crop,gxm::get_format(texture),nullptr);
            narrow_cast_coords[index]=!vertex && bytes==2 && rect.x==0 && rect.y==0
                && uint64_t(rect.width)==uint64_t(entry.guest.width)*2
                && rect.height==entry.guest.height;
        }
        // Carry direct 64-bit float aliases through normalized halfwords. A
        // bit-exact copy alone is insufficient: sampling an RG32Float view
        // can canonicalize NaN payloads before the guest sees its two words.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if ((!vertex && !fragment_resources) || prepared_images[index]) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program):static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            const auto &texture=ctx.textures[index];
            auto found=find_subrectangle(texture);
            if (found==impl->surfaces.end() && surface_viewports[index])
                found=find_texture_surface(texture.data_addr<<2);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second;
            const auto base=gxm::get_base_format(gxm::get_format(texture));
            if (!surface_raw_cast_required(entry.guest.colorFormat,base)
                || !surface_format_cast_enqueueable(entry.guest.colorFormat,base,texture.swizzle_format)) continue;
            const auto rect=surface_subrectangle(entry.guest,texture);
            const bool cropped=rect && (rect->x || rect->y || rect->width!=entry.guest.width || rect->height!=entry.guest.height);
            auto source=raw_surface_color(entry);
            if (cropped) {
                const auto crop=entry.subrectangles.find({rect->x,rect->y,rect->width,rect->height});
                require(crop!=entry.subrectangles.end() && crop->second!=nil,"Metal: raw surface crop was not prepared");
                source=crop->second;
            } else if (entry.color==ctx.impl->color) {
                require(color_feedback!=nil,"Metal: raw surface feedback was not prepared");
                source=raw_color_feedback ? raw_color_feedback : color_feedback;
            }
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            if (!capture_draw) prepare_inline_commands();
            else finish(ctx);
            inline_cast_images[index]=impl->caster->surface_format_cast(source,entry.guest.colorFormat,
                base,texture.swizzle_format,capture_draw ? nil : ctx.impl->commands,true);
            require(inline_cast_images[index]!=nil,"Metal: raw surface carrier creation failed");
            inline_gamma_images[index]=nil;
            texture_mip_info[index].control[3]=1;
        }
        // Publish an active depth producer before a draw samples it. A
        // depth alias and its crops can follow the render pass in the same
        // command buffer without waiting on the CPU, including MSAA samples.
        if (ctx.impl->depth_written && (ctx.impl->depth_key.first || ctx.impl->depth_key.second)) {
            auto found=impl->depth_surfaces.find(ctx.impl->depth_key);
            std::set<SceGxmTextureBaseFormat> sampled_bases;
            std::vector<std::pair<SceGxmTextureBaseFormat,SurfaceRect>> sampled_aliases;
            for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
                const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
                if(!vertex && !fragment_resources) continue;
                const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program):static_cast<const ShaderProgram &>(*fp->fragment_program);
                if(prepared_images[index] || !program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]
                    || found==impl->depth_surfaces.end() || found->second.texture!=ctx.impl->depth) continue;
                const auto base=gxm::get_base_format(gxm::get_format(ctx.textures[index]));
                if (base!=SCE_GXM_TEXTURE_BASE_FORMAT_U8 && base!=SCE_GXM_TEXTURE_BASE_FORMAT_S8
                    && base!=SCE_GXM_TEXTURE_BASE_FORMAT_X8U24 && base!=SCE_GXM_TEXTURE_BASE_FORMAT_F32
                    && base!=SCE_GXM_TEXTURE_BASE_FORMAT_F32M && base!=SCE_GXM_TEXTURE_BASE_FORMAT_U16
                    && base!=SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8) continue;
                if (base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                    && impl->frame_timestamp-found->second.last_attached_frame>2) continue;
                const auto rect=depth_subrectangle(found->second.guest,found->second.width,found->second.height,
                    found->second.multisample,ctx.textures[index]);
                if (!rect) continue;
                sampled_bases.insert(base);
                sampled_aliases.emplace_back(base,*rect);
            }
            if (!sampled_bases.empty()) {
                // An address registered for future visibility queries does not
                // require a CPU wait. Only results from submitted queries do.
                // Normal color-surface publication also happens at finish(),
                // after this command buffer's dependent snapshot and draw.
                const bool inline_snapshot=found->second.texture==ctx.impl->depth
                    && ctx.impl->visibility_results.empty()
                    && !impl->sync_draws && !capture_draw;
                if (!inline_snapshot) finish(ctx);
                else {
                    mid_scene_flush(ctx,false);
                    ctx.impl->depth_scene_written=true;
                    ctx.impl->depth_written=false;
                    found->second.snapshots.clear();
                    found->second.subrectangles.clear();
                    if (!ctx.impl->commands)
                        ctx.impl->commands=scene_command_buffer(*impl->device);
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    const uint32_t guest_width=found->second.width
                        *(found->second.multisample==SCE_GXM_MULTISAMPLE_4X ? 2 : 1);
                    const uint32_t guest_height=found->second.height
                        *(found->second.multisample==SCE_GXM_MULTISAMPLE_NONE ? 1 : 2);
                    for (const auto base:sampled_bases) {
                        found->second.snapshots[base] = base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                            ? impl->caster->packed_depth_snapshot(found->second.texture,false,ctx.impl->commands)
                            : base==SCE_GXM_TEXTURE_BASE_FORMAT_U8
                            || base==SCE_GXM_TEXTURE_BASE_FORMAT_S8
                            ? impl->caster->stencil_snapshot(found->second.texture,
                                base==SCE_GXM_TEXTURE_BASE_FORMAT_S8,res_multiplier,
                                guest_width,guest_height,false,ctx.impl->commands)
                            : impl->caster->depth_snapshot(found->second.texture,
                                base==SCE_GXM_TEXTURE_BASE_FORMAT_U16,res_multiplier,
                                guest_width,guest_height,false,ctx.impl->commands);
                        ++inline_depth_snapshots;
                    }
                    for (const auto &[base,rect]:sampled_aliases) {
                        if (!rect.x && !rect.y && rect.width==guest_width
                            && rect.height==guest_height) continue;
                        auto &crop=found->second.subrectangles[
                            {uint32_t(base),rect.x,rect.y,rect.width,rect.height}];
                        if (crop) continue;
                        crop=depth_alias_view(*impl->caster,found->second.snapshots.at(base),
                            guest_width,guest_height,rect,ctx.impl->commands);
                        ++inline_depth_crops;
                    }
                }
            }
        }
        // Prepare depth aliases before opening the draw encoder. After an
        // active producer has been finished above, snapshots and crops can
        // follow on this command buffer without another CPU wait. A registered
        // visibility address alone has no pending result to publish.
        if (!impl->depth_surfaces.empty() && ctx.impl->visibility_results.empty()
            && !impl->sync_draws && !capture_draw) {
            for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
                const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
                if (!vertex && !fragment_resources) continue;
                const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->vertex_program)
                    : static_cast<const ShaderProgram &>(*fp->fragment_program);
                if (prepared_images[index] || !program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
                const auto &texture=ctx.textures[index];
                const auto base=gxm::get_base_format(gxm::get_format(texture));
                if (base!=SCE_GXM_TEXTURE_BASE_FORMAT_U8 && base!=SCE_GXM_TEXTURE_BASE_FORMAT_S8
                    && base!=SCE_GXM_TEXTURE_BASE_FORMAT_X8U24 && base!=SCE_GXM_TEXTURE_BASE_FORMAT_F32
                    && base!=SCE_GXM_TEXTURE_BASE_FORMAT_F32M && base!=SCE_GXM_TEXTURE_BASE_FORMAT_U16
                    && base!=SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8) continue;
                if (find_direct_surface(texture)!=impl->surfaces.end()
                    || find_subrectangle(texture)!=impl->surfaces.end()) continue;
                for (auto &[key,entry]:impl->depth_surfaces) {
                    // Color textures frequently reuse old depth allocations.
                    // Match Plus's two-frame window only for packed color aliases.
                    if (base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                        && impl->frame_timestamp-entry.last_attached_frame>2) continue;
                    const auto rect=depth_subrectangle(entry.guest,entry.width,entry.height,
                        entry.multisample,texture);
                    if (!rect) continue;
                    auto &snapshot=entry.snapshots[base];
                    const uint32_t guest_width=entry.width
                        *(entry.multisample==SCE_GXM_MULTISAMPLE_4X ? 2 : 1);
                    const uint32_t guest_height=entry.height
                        *(entry.multisample==SCE_GXM_MULTISAMPLE_NONE ? 1 : 2);
                    const bool cropped=rect->x || rect->y || rect->width!=guest_width
                        || rect->height!=guest_height;
                    const std::array<uint32_t,5> crop_key{
                        uint32_t(base),rect->x,rect->y,rect->width,rect->height};
                    const auto existing_crop=cropped ? entry.subrectangles.find(crop_key)
                        : entry.subrectangles.end();
                    if (snapshot && (!cropped || existing_crop!=entry.subrectangles.end())) break;
                    mid_scene_flush(ctx,false);
                    if (!ctx.impl->commands)
                        ctx.impl->commands=scene_command_buffer(*impl->device);
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    if (!snapshot) {
                        snapshot=base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                            ? impl->caster->packed_depth_snapshot(entry.texture,false,ctx.impl->commands)
                            : base==SCE_GXM_TEXTURE_BASE_FORMAT_U8
                            || base==SCE_GXM_TEXTURE_BASE_FORMAT_S8
                            ? impl->caster->stencil_snapshot(entry.texture,
                                base==SCE_GXM_TEXTURE_BASE_FORMAT_S8,res_multiplier,
                                guest_width,guest_height,false,ctx.impl->commands)
                            : impl->caster->depth_snapshot(entry.texture,
                                base==SCE_GXM_TEXTURE_BASE_FORMAT_U16,res_multiplier,
                                guest_width,guest_height,false,ctx.impl->commands);
                        ++inline_depth_snapshots;
                    }
                    if (cropped) {
                        auto &crop=entry.subrectangles[crop_key];
                        crop=depth_alias_view(*impl->caster,snapshot,guest_width,guest_height,
                            *rect,ctx.impl->commands);
                        ++inline_depth_crops;
                    }
                    break;
                }
            }
        }
        if (!ctx.impl->encoder && ctx.impl->samples>1 && ctx.impl->guest_color.data) {
            auto &surface=impl->surfaces.at(ctx.impl->guest_color.data.address());
            if (surface.multisample_dirty) {
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                impl->caster->seed_multisample(surface.color,ctx.impl->render_color,ctx.impl->sample_scale,ctx.impl->expanded_color,
                    ctx.impl->expanded_color ? surface.guest.width : 0,
                    ctx.impl->expanded_color ? surface.guest.height : 0);
                surface.multisample_dirty=false;
            }
            if (ctx.impl->raw_render_color && surface.raw_multisample_dirty) {
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                if (!ctx.impl->commands) ctx.impl->commands=scene_command_buffer(*impl->device);
                impl->caster->seed_multisample(surface.raw_color,ctx.impl->raw_render_color,
                    ctx.impl->sample_scale,ctx.impl->expanded_color,
                    ctx.impl->expanded_color ? surface.guest.width : 0,
                    ctx.impl->expanded_color ? surface.guest.height : 0,ctx.impl->commands);
                surface.raw_multisample_dirty=false;
            }
        }
        if (!ctx.impl->encoder) {
            if (!ctx.impl->commands)
                ctx.impl->commands = scene_command_buffer(*impl->device);
            begin_pass(ctx, record.is_maskupdate);
        }
        if (ctx.impl->mask_pass != record.is_maskupdate) begin_pass(ctx, record.is_maskupdate);
        if ((ctx.impl->visibility_enabled || ctx.impl->back_visibility_enabled) && ctx.impl->visibility_address
            && (ctx.impl->pass_visibility_address != ctx.impl->visibility_address
                || ctx.impl->pass_visibility_stride != ctx.impl->visibility_stride))
            begin_pass(ctx, record.is_maskupdate);
        if (many_face_queries && ctx.impl->pass_visibility_buffer
            && uint64_t(ctx.impl->pass_visibility_offset) + query_capacity > ctx.impl->pass_visibility_buffer.length)
            begin_pass(ctx, record.is_maskupdate);
        if (capture_draw) {
            std::filesystem::create_directories(draw_capture_dir);
            draw_metadata.open(draw_capture_dir/"draw.txt");
            texture_metadata.open(draw_capture_dir/"textures.txt");
            texture_metadata << "version 3\n";
            draw_metadata.precision(17);
            draw_metadata << "version 2\nscale " << res_multiplier << "\nsize " << ctx.impl->width << ' ' << ctx.impl->height
                          << "\nfragment " << hex_string(fp->fragment_program->hash)
                          << "\nvertex " << hex_string(vp->vertex_program->hash)
                          << "\noutput_register_size " << record.color_surface.outputRegisterSize << '\n';
            if (ctx.shader_hints.metal_raw_color_attachment)
                draw_metadata << "raw_color_attachment 1\n";
            std::ofstream attachment_info(draw_capture_dir/"attachments.txt",std::ios::app);
            attachment_info << "target " << ctx.impl->guest_color.data.address() << " recorded " << record.color_surface.data.address()
                << " same_native " << (ctx.impl->color == ctx.impl->render_color) << '\n';
            attachment_info << "mask_state valid " << ctx.impl->mask_constant_valid
                << " constant " << ctx.impl->mask_constant_value
                << " descriptor " << record.depth_stencil_surface.mask
                << " depth_disabled " << record.depth_stencil_surface.disabled()
                << " writing " << record.writing_mask << '\n';
            auto shader_features = features;
            shader_features.enable_memory_mapping = true;
            auto msl = shader::metal::convert_gxp(*vp->program(), hex_string(vp->vertex_program->hash), shader_features, ctx.shader_hints, false);
            dump_bytes("vertex.metal", msl.source.data(), msl.source.size());
            const auto *vertex_gxp = vp->program(), *fragment_gxp = fp->program();
            dump_bytes("vertex.gxp", vertex_gxp, vertex_gxp->size);
            dump_bytes("fragment.gxp", fragment_gxp, fragment_gxp->size);
            auto fragment_msl = fragment_disabled ? depth_only_program(ctx.impl->samples)
                : shader::metal::convert_gxp(*fragment_gxp, hex_string(fp->fragment_program->hash),
                    shader_features, ctx.shader_hints, record.is_maskupdate);
            dump_bytes("fragment.metal", fragment_msl.source.data(), fragment_msl.source.size());
            draw_metadata << "entry " << msl.entry_point << '\n';
            for (uint32_t slot=0;slot<SCE_GXM_MAX_TEXTURE_UNITS;++slot)
                if (vp->vertex_program->textures_used[slot]) draw_metadata << "vertex_texture " << slot << '\n';
        }
        struct VertexAttributeLayout {
            MTLVertexFormat format = MTLVertexFormatInvalid;
            NSUInteger offset = 0;
            NSUInteger buffer_index = 0;
            size_t source_offset = 0;
            size_t byte_size = 0;
        };
        struct VertexStreamLayout {
            NSUInteger stride = 0;
            MTLVertexStepFunction step_function = MTLVertexStepFunctionPerVertex;
            NSUInteger step_rate = 0;
        };
        std::array<VertexAttributeLayout, 31> vertex_attributes{};
        std::array<VertexStreamLayout, SCE_GXM_MAX_VERTEX_STREAMS> vertex_stream_layouts{};
        std::string key = vertex_key + fragment_key;
        key.reserve(key.size() + 4 * (4 * vp->attributes.size()
            + 2 * vp->streams.size() + 12));
        uint32_t streams_used = 0;
        std::array<uint32_t, SCE_GXM_MAX_VERTEX_STREAMS> stream_extents{};
        std::array<bool, SCE_GXM_MAX_VERTEX_STREAMS> repack_stream{};
        for (const auto &a : vp->attributes) {
            auto it = vp->vertex_program->attribute_infos.find(a.regIndex);
            if (it == vp->vertex_program->attribute_infos.end()) continue;
            const auto &info = it->second;
            const auto shape = vertex_attribute_shape(a, info);
            require(shape.has_value(), "Metal: invalid vertex attribute shape");
            const auto components = shape->components;
            const auto f = shape->format;
            const auto element_size = shape->component_size;
            require(a.streamIndex < vp->streams.size() && a.streamIndex < SCE_GXM_MAX_VERTEX_STREAMS, "Metal: invalid vertex stream");
            stream_extents[a.streamIndex] = std::max(stream_extents[a.streamIndex], uint32_t(a.offset + components * element_size));
            for (uint32_t element = 0; element < (components + 3) / 4; ++element) {
                const auto location = info.location + element;
                require(location < 31, "Metal: attribute location exceeds hardware limit");
                const auto n = std::min(components - 4 * element, 4u);
                auto &attribute = vertex_attributes[location];
                attribute.format = attribute_format(f, n, !info.regformat && info.is_signed);
                attribute.offset = a.offset + 4 * element * element_size;
                attribute.buffer_index = shader::metal::VERTEX_STREAM_BUFFER_BASE + a.streamIndex;
                attribute.source_offset = attribute.offset;
                attribute.byte_size = n * element_size;
                const size_t stride = vp->streams[a.streamIndex].stride;
                // Whole-stride padding cannot preserve attributes that cross
                // a record boundary. Metal also needs each fetched attribute
                // to fit the native record and its offset to be aligned.
                repack_stream[a.streamIndex] = repack_stream[a.streamIndex]
                    || attribute.offset % 4 != 0
                    || (stride && attribute.offset + attribute.byte_size > stride);
                append(key, location); append(key, attribute.format);
                append(key, attribute.offset); append(key, a.streamIndex);
            }
            streams_used |= 1u << a.streamIndex;
        }
        require(vp->streams.size() <= SCE_GXM_MAX_VERTEX_STREAMS, "Metal: too many vertex streams");
        for (uint32_t stream = 0; stream < vp->streams.size(); ++stream) if (streams_used & (1u << stream)) {
            auto &binding = vertex_stream_layouts[stream];
            const auto stride = vp->streams[stream].stride;
            binding.stride = ((stride ? stride : stream_extents[stream]) + 3) & ~size_t(3);
            binding.step_function = !stride ? MTLVertexStepFunctionConstant
                : gxm::is_stream_instancing(static_cast<SceGxmIndexSource>(vp->streams[stream].indexSource))
                    ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
            binding.step_rate = stride ? 1 : 0;
            if (repack_stream[stream]) {
                size_t offset = 0;
                for (auto &attribute : vertex_attributes) {
                    if (attribute.format == MTLVertexFormatInvalid
                        || attribute.buffer_index != shader::metal::VERTEX_STREAM_BUFFER_BASE + stream)
                        continue;
                    attribute.offset = offset;
                    offset = (offset + attribute.byte_size + 3) & ~size_t(3);
                }
                binding.stride = offset;
            }
            // Include the layout policy so old cached descriptors cannot be
            // reused when a relocated attribute happens to keep the same stride.
            append(key, repack_stream[stream] ? 1 : 0);
            append(key, binding.stride); append(key, binding.step_function);
            if (capture_draw) draw_metadata << "layout " << shader::metal::VERTEX_STREAM_BUFFER_BASE+stream << ' '
                << binding.stride << ' ' << binding.step_function << ' ' << binding.step_rate << '\n';
        }
        if (capture_draw) {
            for (uint32_t location = 0; location < vertex_attributes.size(); ++location) {
                const auto &attribute = vertex_attributes[location];
                if (attribute.format != MTLVertexFormatInvalid)
                    draw_metadata << "attribute " << location << ' ' << attribute.format
                        << ' ' << attribute.offset << ' ' << attribute.buffer_index << '\n';
            }
        }
        SceGxmBlendInfo mask_blend{};
        mask_blend.colorMask = SCE_GXM_COLOR_MASK_ALL;
        auto blend = fragment_color_disabled ? SceGxmBlendInfo{}
            : record.is_maskupdate ? mask_blend : static_cast<const MetalFragmentProgram &>(*fp->fragment_program).blend;
        // Metal enables blending for the whole attachment. An independently
        // disabled channel group must still replace its destination. Normalize
        // before constructing either pipeline key so cached descriptors agree.
        if (blend.colorFunc == SCE_GXM_BLEND_FUNC_NONE) {
            blend.colorSrc = SCE_GXM_BLEND_FACTOR_ONE;
            blend.colorDst = SCE_GXM_BLEND_FACTOR_ZERO;
        }
        if (blend.alphaFunc == SCE_GXM_BLEND_FUNC_NONE) {
            blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
            blend.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
        }
        id<MTLTexture> attachment = record.is_maskupdate ? ctx.impl->mask : ctx.impl->render_color;
        const auto target_base=gxm::get_base_format(record.color_surface.colorFormat);
        const bool rgb_target=!record.is_maskupdate && (target_base==SCE_GXM_COLOR_BASE_FORMAT_U8U8U8
            || target_base==SCE_GXM_COLOR_BASE_FORMAT_S5S5U6);
        const bool alpha_target = !record.is_maskupdate
            && gxm::is_alpha_only_color_format(record.color_surface.colorFormat);
        const bool green_target = !record.is_maskupdate
            && gxm::is_green_only_color_format(record.color_surface.colorFormat);
        const bool red_alpha_target = !record.is_maskupdate
            && gxm::is_red_alpha_color_format(record.color_surface.colorFormat);
        const bool quantized_color_target = !fragment_disabled && !record.is_maskupdate
            && blend.colorMask != SCE_GXM_COLOR_MASK_NONE
            && (target_base == SCE_GXM_COLOR_BASE_FORMAT_U8U3U3U2
                || target_base == SCE_GXM_COLOR_BASE_FORMAT_S5S5U6
                || target_base == SCE_GXM_COLOR_BASE_FORMAT_U8S8S8U8);
        append(key, attachment ? attachment.pixelFormat : MTLPixelFormatInvalid);
        append(key, !record.is_maskupdate && bool(ctx.impl->raw_attachment));
        append(key, rgb_target);
        append(key, alpha_target);
        append(key, green_target);
        append(key, red_alpha_target);
        append(key, ctx.impl->samples);
        append(key, blend.colorMask); append(key, blend.colorFunc); append(key, blend.alphaFunc);
        append(key, blend.colorSrc); append(key, blend.colorDst); append(key, blend.alphaSrc); append(key, blend.alphaDst);
        const auto make_vertex_layout = [&] {
            MTLVertexDescriptor *layout = [MTLVertexDescriptor vertexDescriptor];
            for (uint32_t location = 0; location < vertex_attributes.size(); ++location) {
                const auto &attribute = vertex_attributes[location];
                if (attribute.format == MTLVertexFormatInvalid) continue;
                layout.attributes[location].format = attribute.format;
                layout.attributes[location].offset = attribute.offset;
                layout.attributes[location].bufferIndex = attribute.buffer_index;
            }
            for (uint32_t stream = 0; stream < vp->streams.size(); ++stream) if (streams_used & (1u << stream)) {
                auto *binding = layout.layouts[shader::metal::VERTEX_STREAM_BUFFER_BASE + stream];
                binding.stride = vertex_stream_layouts[stream].stride;
                binding.stepFunction = vertex_stream_layouts[stream].step_function;
                binding.stepRate = vertex_stream_layouts[stream].step_rate;
            }
            return layout;
        };
        const auto compile_pipeline_once = [&](const std::string &pipeline_key, MTLRenderPipelineDescriptor *descriptor) -> id<MTLRenderPipelineState> {
            if (impl->failed_pipelines.contains(pipeline_key)) return nil;
            try {
                auto result = impl->device->create_pipeline(descriptor, error);
                if (result) return result;
            } catch (const std::exception &failure) {
                error = failure.what();
            }
            remember_failure(impl->failed_pipelines, pipeline_key, "pipeline", error);
            return nil;
        };
        if (impl->failed_pipelines.contains(key)) return;
        auto &pipeline = impl->pipelines[key];
        if (!pipeline) {
            if (auto pending = impl->compiling_pipelines.find(key); pending != impl->compiling_pipelines.end()) {
                if (impl->async_compilation.load(std::memory_order_relaxed)
                    && pending->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                    return;
                // Remove the future before get(): an exception must not leave
                // an invalid future that is polled again on every draw.
                auto future = std::move(pending->second);
                impl->compiling_pipelines.erase(pending);
                Impl::PipelineResult result;
                try {
                    result = future.get();
                } catch (const std::exception &failure) {
                    result.error = failure.what();
                }
                if (!result.pipeline) {
                    remember_failure(impl->failed_pipelines, key, "pipeline", result.error);
                    return;
                }
                pipeline = result.pipeline;
            }
        }
        if (!pipeline) {
            MTLVertexDescriptor *layout = make_vertex_layout();
            MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
            desc.vertexFunction = vs.function; desc.fragmentFunction = fs.function; desc.vertexDescriptor = layout;
            desc.rasterSampleCount = ctx.impl->samples;
            desc.depthAttachmentPixelFormat = desc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
            auto *color = desc.colorAttachments[0];
            color.pixelFormat = attachment ? attachment.pixelFormat : MTLPixelFormatInvalid;
            color.writeMask = MTLColorWriteMaskNone;
            if (blend.colorMask & SCE_GXM_COLOR_MASK_R) color.writeMask |= MTLColorWriteMaskRed;
            if (blend.colorMask & SCE_GXM_COLOR_MASK_G) color.writeMask |= MTLColorWriteMaskGreen;
            if (blend.colorMask & SCE_GXM_COLOR_MASK_B) color.writeMask |= MTLColorWriteMaskBlue;
            if (!rgb_target && (blend.colorMask & SCE_GXM_COLOR_MASK_A)) color.writeMask |= MTLColorWriteMaskAlpha;
            color.blendingEnabled = blend.colorFunc != SCE_GXM_BLEND_FUNC_NONE || blend.alphaFunc != SCE_GXM_BLEND_FUNC_NONE;
            color.rgbBlendOperation = blend_op(blend.colorFunc); color.alphaBlendOperation = blend_op(blend.alphaFunc);
            color.sourceRGBBlendFactor = blend_factor(blend.colorSrc); color.destinationRGBBlendFactor = blend_factor(blend.colorDst);
            color.sourceAlphaBlendFactor = blend_factor(blend.alphaSrc); color.destinationAlphaBlendFactor = blend_factor(blend.alphaDst);
            if (alpha_target) {
                color.writeMask = (blend.colorMask & SCE_GXM_COLOR_MASK_A) ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;
                color.blendingEnabled = blend.alphaFunc != SCE_GXM_BLEND_FUNC_NONE;
                color.rgbBlendOperation = blend_op(blend.alphaFunc);
                const auto alpha_factor = [](SceGxmBlendFactor factor) {
                    // Native R holds destination alpha; source alpha remains A.
                    switch (factor) {
                    case SCE_GXM_BLEND_FACTOR_DST_ALPHA: return MTLBlendFactorDestinationColor;
                    case SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationColor;
                    case SCE_GXM_BLEND_FACTOR_DST_ALPHA_SATURATE: return MTLBlendFactorDestinationColor;
                    case SCE_GXM_BLEND_FACTOR_SRC_ALPHA_SATURATE: return MTLBlendFactorOne;
                    default: return blend_factor(factor);
                    }
                };
                color.sourceRGBBlendFactor = alpha_factor(blend.alphaSrc);
                color.destinationRGBBlendFactor = alpha_factor(blend.alphaDst);
            }
            if (green_target)
                color.writeMask = (blend.colorMask & SCE_GXM_COLOR_MASK_G) ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;
            if (red_alpha_target) {
                if (ctx.shader_hints.metal_red_alpha_shader_blend) color.blendingEnabled = NO;
                color.writeMask = MTLColorWriteMaskNone;
                if (blend.colorMask & SCE_GXM_COLOR_MASK_R)
                    color.writeMask |= MTLColorWriteMaskRed;
                if (blend.colorMask & SCE_GXM_COLOR_MASK_A)
                    color.writeMask |= MTLColorWriteMaskGreen;
            }
            if (quantized_color_target) {
                color.blendingEnabled = NO;
                color.writeMask = MTLColorWriteMaskAll;
            }
            configure_raw_attachment(desc,record.is_maskupdate ? nil : ctx.impl->raw_attachment,color.writeMask);
            if (!impl->dump_pipeline_dir.empty() && impl->dumped_pipelines < 4096) {
                // Metal can abort inside descriptor validation before returning
                // NSError. Close every diagnostic file BEFORE calling it.
                try {
                    const auto dir=impl->dump_pipeline_dir/std::to_string(impl->dumped_pipelines++);
                    std::filesystem::create_directories(dir);
                    auto write_gxp=[&](const char *name, const SceGxmProgram *gxp) {
                        std::ofstream file(dir/name,std::ios::binary);
                        file.write(reinterpret_cast<const char *>(gxp),gxp->size);
                        if (!file) throw std::runtime_error("Cannot save pipeline GXP");
                    };
                    write_gxp("vertex.gxp",vp->program()); write_gxp("fragment.gxp",fp->program());
                    std::ofstream meta(dir/"pipeline.txt");
                    meta<<"vertex "<<hex_string(vp->vertex_program->hash)<<"\nfragment "<<hex_string(fp->fragment_program->hash)
                        <<"\ncolor "<<uint32_t(record.color_surface.colorFormat)<<"\nsamples "<<desc.rasterSampleCount
                        <<"\nnative_color "<<color.pixelFormat<<"\nnative_depth "<<desc.depthAttachmentPixelFormat
                        <<"\nnative_stencil "<<desc.stencilAttachmentPixelFormat<<'\n';
                    for (const auto &a:vp->attributes)
                        meta<<"gxm_attribute "<<unsigned(a.streamIndex)<<' '<<a.offset<<' '<<unsigned(a.format)<<' '<<unsigned(a.componentCount)<<' '<<unsigned(a.regIndex)<<'\n';
                    for (size_t i=0;i<vp->streams.size();++i)
                        meta<<"gxm_stream "<<i<<' '<<vp->streams[i].stride<<' '<<unsigned(vp->streams[i].indexSource)<<'\n';
                    for (uint32_t i=0;i<31;++i) if(layout.attributes[i].format!=MTLVertexFormatInvalid)
                        meta<<"attribute "<<i<<' '<<layout.attributes[i].format<<' '<<layout.attributes[i].offset<<' '<<layout.attributes[i].bufferIndex<<'\n';
                    for (uint32_t i=0;i<31;++i) if(layout.layouts[i].stride)
                        meta<<"layout "<<i<<' '<<layout.layouts[i].stride<<' '<<layout.layouts[i].stepFunction<<' '<<layout.layouts[i].stepRate<<'\n';
                    for (MTLVertexAttribute *a in vs.function.vertexAttributes)
                        meta<<"shader_input "<<a.attributeIndex<<' '<<a.attributeType<<' '<<bool(a.active)<<' '<<(a.name.UTF8String?:"")<<'\n';
                    meta.flush();
                    if(!meta) throw std::runtime_error("Cannot save pipeline descriptor");
                    LOG_INFO("Metal pipeline diagnostic saved before validation: {}",dir.string());
                } catch(const std::exception &e) { LOG_WARN("Metal pipeline diagnostic: {}",e.what()); }
            }
            // Like Vulkan, defer newly seen pipelines for larger draws. Small
            // quads stay synchronous so menus and simple overlays appear at once.
            if (impl->async_compilation.load(std::memory_order_relaxed)
                && !routed_geometry && (count > 6 || instances > 1) && impl->compiling_pipelines.size() < 2) {
                MTLRenderPipelineDescriptor *owned_descriptor = [desc copy];
                Device *device = impl->device.get();
                const std::string fragment_hash=hex_string(fp->fragment_program->hash);
                const std::string vertex_hash=hex_string(vp->vertex_program->hash);
                impl->compiling_pipelines.emplace(key, std::async(std::launch::async,
                    [device, owned_descriptor, fragment_hash, vertex_hash, key, vertex_key, fragment_key]() -> Impl::PipelineResult {
                        @autoreleasepool {
                            Impl::PipelineResult result;
                            result.pipeline = device->create_pipeline(owned_descriptor, result.error);
                            if (result.pipeline) {
                                try {
                                    device->store_cached_pipeline_template(fragment_hash,vertex_hash,
                                        key,vertex_key,fragment_key,owned_descriptor);
                                } catch (const std::exception &failure) {
                                    LOG_WARN("Metal: compiled pipeline could not be cached: {}", failure.what());
                                }
                            }
                            return result;
                        }
                    }));
                return;
            }
            pipeline = compile_pipeline_once(key, desc);
            if (!pipeline) return;
            try {
                impl->device->store_cached_pipeline_template(hex_string(fp->fragment_program->hash),
                    hex_string(vp->vertex_program->hash),key,vertex_key,fragment_key,desc);
            } catch (const std::exception &failure) {
                LOG_WARN("Metal: compiled pipeline could not be cached: {}", failure.what());
            }
        }
        id<MTLRenderCommandEncoder> encoder = ctx.impl->encoder;
        [encoder setRenderPipelineState:pipeline];
        const bool depth_enabled = !record.is_maskupdate && !record.depth_stencil_surface.disabled();
        const bool split_face_depth = depth_enabled && triangles && two_sided && !cull_front && !cull_back
            && (record.front_depth_func != record.back_depth_func
                || record.front_depth_write_mode != record.back_depth_write_mode
                || record.depth_bias_unit != record.back_depth_bias_unit
                || record.depth_bias_slope != record.back_depth_bias_slope);
        const bool split_face_visibility = triangles && !cull_front && !cull_back
            && ctx.impl->visibility_address && visibility_faces_differ;
        const bool split_face_draw = split_face_depth || (split_face_visibility && !routed_geometry);
        // The two face draws must not execute a memory-writing GXP vertex
        // shader twice. Capture its outputs once, then split only the replay.
        const bool capture_split_faces = split_face_draw && vs.writes_guest_memory && !routed_geometry;
        // Reject unavailable capture/replay variants before retaining upload slices.
        id<MTLRenderPipelineState> point_capture_pipeline = nil;
        id<MTLRenderPipelineState> point_replay_pipeline = nil;
        if (routed_geometry || capture_split_faces) {
            const bool compact_capture = compact_point_capture || capture_split_faces;
            const std::string capture_suffix = compact_capture ? "-polygon-capture-compact" : "-polygon-capture";
            const std::string capture_key = vertex_key + capture_suffix;
            auto *capture_shader = compile_shader_once(capture_key, [&]() {
                auto hints = ctx.shader_hints;
                hints.metal_capture_vertex_outputs = true;
                hints.metal_capture_vertex_outputs_compact = compact_capture;
                auto shader_features = features;
                shader_features.enable_memory_mapping = true;
                auto source = shader::metal::convert_gxp(*vp->program(),
                    hex_string(vp->vertex_program->hash), shader_features, hints);
                return impl->device->compile(source, false, error);
            });
            if (!capture_shader) return;
            const std::string replay_shader_key = "metal-point-replay-vertex";
            auto *replay_shader = compile_shader_once(replay_shader_key, [&]() {
                return impl->device->compile(shader::metal::point_replay_vertex_program(), false, error);
            });
            if (!replay_shader) return;
            auto &cached_capture = impl->pipelines[key + capture_suffix];
            if (!cached_capture) {
                auto desc = [MTLRenderPipelineDescriptor new];
                desc.vertexFunction = capture_shader->function;
                desc.vertexDescriptor = make_vertex_layout();
                desc.rasterSampleCount = ctx.impl->samples;
                desc.depthAttachmentPixelFormat = desc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
                desc.colorAttachments[0].pixelFormat = attachment.pixelFormat;
                desc.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
                configure_raw_attachment(desc,record.is_maskupdate ? nil : ctx.impl->raw_attachment,MTLColorWriteMaskNone);
                cached_capture = compile_pipeline_once(key + capture_suffix, desc);
                if (!cached_capture) return;
            }
            point_capture_pipeline = cached_capture;
            auto &cached_replay = impl->pipelines[key + "-polygon-replay"];
            if (!cached_replay) {
                auto desc = [MTLRenderPipelineDescriptor new];
                desc.vertexFunction = replay_shader->function;
                desc.fragmentFunction = fs.function;
                desc.rasterSampleCount = ctx.impl->samples;
                desc.depthAttachmentPixelFormat = desc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
                auto *color = desc.colorAttachments[0];
                color.pixelFormat = attachment.pixelFormat;
                color.writeMask = MTLColorWriteMaskNone;
                if (blend.colorMask & SCE_GXM_COLOR_MASK_R) color.writeMask |= MTLColorWriteMaskRed;
                if (blend.colorMask & SCE_GXM_COLOR_MASK_G) color.writeMask |= MTLColorWriteMaskGreen;
                if (blend.colorMask & SCE_GXM_COLOR_MASK_B) color.writeMask |= MTLColorWriteMaskBlue;
                if (!rgb_target && (blend.colorMask & SCE_GXM_COLOR_MASK_A)) color.writeMask |= MTLColorWriteMaskAlpha;
                color.blendingEnabled = blend.colorFunc != SCE_GXM_BLEND_FUNC_NONE || blend.alphaFunc != SCE_GXM_BLEND_FUNC_NONE;
                color.rgbBlendOperation = blend_op(blend.colorFunc); color.alphaBlendOperation = blend_op(blend.alphaFunc);
                color.sourceRGBBlendFactor = blend_factor(blend.colorSrc); color.destinationRGBBlendFactor = blend_factor(blend.colorDst);
                color.sourceAlphaBlendFactor = blend_factor(blend.alphaSrc); color.destinationAlphaBlendFactor = blend_factor(blend.alphaDst);
                if (alpha_target) {
                    color.writeMask = (blend.colorMask & SCE_GXM_COLOR_MASK_A) ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;
                    color.blendingEnabled = blend.alphaFunc != SCE_GXM_BLEND_FUNC_NONE;
                    color.rgbBlendOperation = blend_op(blend.alphaFunc);
                    const auto alpha_factor = [](SceGxmBlendFactor factor) {
                        switch (factor) {
                        case SCE_GXM_BLEND_FACTOR_DST_ALPHA: return MTLBlendFactorDestinationColor;
                        case SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationColor;
                        case SCE_GXM_BLEND_FACTOR_DST_ALPHA_SATURATE: return MTLBlendFactorDestinationColor;
                        case SCE_GXM_BLEND_FACTOR_SRC_ALPHA_SATURATE: return MTLBlendFactorOne;
                        default: return blend_factor(factor);
                        }
                    };
                    color.sourceRGBBlendFactor = alpha_factor(blend.alphaSrc);
                    color.destinationRGBBlendFactor = alpha_factor(blend.alphaDst);
                }
                if (green_target)
                    color.writeMask = (blend.colorMask & SCE_GXM_COLOR_MASK_G) ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;
                if (red_alpha_target) {
                    if (ctx.shader_hints.metal_red_alpha_shader_blend) color.blendingEnabled = NO;
                    color.writeMask = MTLColorWriteMaskNone;
                    if (blend.colorMask & SCE_GXM_COLOR_MASK_R)
                        color.writeMask |= MTLColorWriteMaskRed;
                    if (blend.colorMask & SCE_GXM_COLOR_MASK_A)
                        color.writeMask |= MTLColorWriteMaskGreen;
                }
                if (quantized_color_target) {
                    color.blendingEnabled = NO;
                    color.writeMask = MTLColorWriteMaskAll;
                }
                configure_raw_attachment(desc,record.is_maskupdate ? nil : ctx.impl->raw_attachment,color.writeMask);
                cached_replay = compile_pipeline_once(key + "-polygon-replay", desc);
                if (!cached_replay) return;
            }
            point_replay_pipeline = cached_replay;
        }
        const bool back_depth = two_sided && cull_front;
        // Raster depth derivatives shrink with internal upscaling. Keep the
        // guest's slope bias constant in guest pixels, including replay draws.
        const float front_depth_slope = float(record.depth_bias_slope) * res_multiplier;
        const float back_depth_slope = float(record.back_depth_bias_slope) * res_multiplier;
        const auto depth_func = back_depth ? record.back_depth_func : record.front_depth_func;
        const auto depth_write_mode = back_depth ? record.back_depth_write_mode : record.front_depth_write_mode;
        using DepthKey = std::array<uint32_t, 16>;
        const auto stencil_fields = [](const GxmStencilStateOp &ops, const GxmStencilStateValues &values) {
            return std::array<uint32_t, 6>{uint32_t(ops.func) >> 25,
                uint32_t(stencil_op(ops.stencil_fail)), uint32_t(stencil_op(ops.depth_fail)),
                uint32_t(stencil_op(ops.depth_pass)), values.compare_mask, values.write_mask};
        };
        DepthKey depth_key{};
        depth_key[0] = depth_enabled ? uint32_t(depth_func) >> 22 : MTLCompareFunctionAlways;
        depth_key[1] = depth_enabled && depth_write_mode == SCE_GXM_DEPTH_WRITE_ENABLED;
        ctx.impl->depth_written |= depth_key[1]
            || (split_face_depth && record.back_depth_write_mode == SCE_GXM_DEPTH_WRITE_ENABLED);
        if (depth_enabled) {
            const auto front = stencil_fields(record.front_stencil_state_op, record.front_stencil_state_values);
            const auto back = record.two_sided == SCE_GXM_TWO_SIDED_DISABLED ? front
                : stencil_fields(record.back_stencil_state_op, record.back_stencil_state_values);
            std::copy(front.begin(), front.end(), depth_key.begin() + 2);
            std::copy(back.begin(), back.end(), depth_key.begin() + 8);
            depth_key[14] = depth_key[15] = 1;
            const auto writes_stencil=[](const std::array<uint32_t, 6> &face) {
                return face[5] && (face[1] != MTLStencilOperationKeep
                    || face[2] != MTLStencilOperationKeep || face[3] != MTLStencilOperationKeep);
            };
            ctx.impl->depth_written |= writes_stencil(front) || writes_stencil(back);
        }
        if (capture_draw && impl->dump_attachments) {
            std::ofstream metadata(draw_capture_dir/"attachments.txt",std::ios::app);
            metadata << "draw_state depth_key";
            for (const uint32_t value : depth_key) metadata << ' ' << value;
            metadata << " front_ref " << unsigned(record.front_stencil_state_values.ref)
                << " back_ref " << unsigned(record.back_stencil_state_values.ref)
                << " cull " << uint32_t(record.cull_mode)
                << " two_sided " << uint32_t(record.two_sided)
                << " front_disabled " << front_fragment_disabled
                << " back_disabled " << back_fragment_disabled
                << " clip_count " << clip.size();
            for (const auto &rect : clip)
                metadata << " clip " << rect.x << ',' << rect.y << ',' << rect.width << ',' << rect.height;
            metadata << '\n';
            require(bool(metadata),"Metal: cannot write draw-state metadata");
        }
        const auto native_depth_state = [&](const DepthKey &key) -> id<MTLDepthStencilState> {
            auto [found, inserted] = impl->depth_states.try_emplace(key);
            if (inserted) {
                auto descriptor = [MTLDepthStencilDescriptor new];
                descriptor.depthCompareFunction = static_cast<MTLCompareFunction>(key[0]);
                descriptor.depthWriteEnabled = key[1];
                const auto make_stencil = [&](size_t offset) {
                    auto stencil = [MTLStencilDescriptor new];
                    stencil.stencilCompareFunction = static_cast<MTLCompareFunction>(key[offset]);
                    stencil.stencilFailureOperation = static_cast<MTLStencilOperation>(key[offset + 1]);
                    stencil.depthFailureOperation = static_cast<MTLStencilOperation>(key[offset + 2]);
                    stencil.depthStencilPassOperation = static_cast<MTLStencilOperation>(key[offset + 3]);
                    stencil.readMask = key[offset + 4];
                    stencil.writeMask = key[offset + 5];
                    return stencil;
                };
                if (key[14]) descriptor.frontFaceStencil = make_stencil(2);
                if (key[15]) descriptor.backFaceStencil = make_stencil(8);
                found->second = [impl->device->native_device() newDepthStencilStateWithDescriptor:descriptor];
            }
            require(found->second != nil, "Metal: depth-stencil state allocation failed");
            return found->second;
        };
        id<MTLDepthStencilState> front_depth_state = native_depth_state(depth_key);
        id<MTLDepthStencilState> back_depth_state = nil;
        id<MTLDepthStencilState> point_front_depth_state = front_depth_state;
        id<MTLDepthStencilState> point_back_depth_state = front_depth_state;
        if (routed_geometry && two_sided && depth_enabled) {
            auto point_front = depth_key;
            point_front[0] = uint32_t(record.front_depth_func) >> 22;
            point_front[1] = record.front_depth_write_mode == SCE_GXM_DEPTH_WRITE_ENABLED;
            point_front_depth_state = native_depth_state(point_front);
            auto point_back = depth_key;
            point_back[0] = uint32_t(record.back_depth_func) >> 22;
            point_back[1] = record.back_depth_write_mode == SCE_GXM_DEPTH_WRITE_ENABLED;
            std::copy(depth_key.begin() + 8, depth_key.begin() + 14, point_back.begin() + 2);
            point_back_depth_state = native_depth_state(point_back);
        }
        if (split_face_depth) {
            auto back = depth_key;
            back[0] = uint32_t(record.back_depth_func) >> 22;
            back[1] = record.back_depth_write_mode == SCE_GXM_DEPTH_WRITE_ENABLED;
            back_depth_state = native_depth_state(back);
        }
        [encoder setDepthStencilState:front_depth_state];
        [encoder setStencilFrontReferenceValue:record.front_stencil_state_values.ref backReferenceValue:record.two_sided == SCE_GXM_TWO_SIDED_DISABLED ? record.front_stencil_state_values.ref : record.back_stencil_state_values.ref];
        [encoder setDepthBias:back_depth ? record.back_depth_bias_unit : record.depth_bias_unit
            slopeScale:back_depth ? back_depth_slope : front_depth_slope clamp:0];
        // Use the same transformed front face as GL/Vulkan. Swapping both
        // winding and cull mode preserves which triangles disappear, but
        // reverses front_facing and the per-face fragment/stencil state.
        [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
        [encoder setCullMode:record.cull_mode == SCE_GXM_CULL_NONE ? MTLCullModeNone : (record.cull_mode == SCE_GXM_CULL_CW ? MTLCullModeBack : MTLCullModeFront)];
        // Polygon rasterization mode affects triangles, not native point/line
        // primitives. Sly has a vertex program with a point-size output; point
        // draws using it must not fail because a point polygon mode is set.
        if (triangles && record.front_polygon_mode != SCE_GXM_POLYGON_MODE_TRIANGLE_FILL && !wireframe && !point_polygon)
            throw std::runtime_error(fmt::format("Metal: triangle polygon mode {:#x} requires conversion (primitive={:#x}, vertex={}, fragment={})",
                uint32_t(record.front_polygon_mode), uint32_t(primitive),
                hex_string(vp->vertex_program->hash), hex_string(fp->fragment_program->hash)));
        [encoder setTriangleFillMode:wireframe && !routed_wide_line ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
        const auto &v = ctx.viewport;
        MTLViewport viewport = record.viewport_flat ? MTLViewport{0, 0, double(ctx.impl->width), double(ctx.impl->height), 0, 1}
            : MTLViewport{v[0] * raster_scale, v[1] * raster_scale, v[2] * raster_scale, v[3] * raster_scale, 0, 1};
        [encoder setViewport:viewport];
        if (impl->trace_draws && impl->traced_draws.size() < 4096) {
            const auto detail = fmt::format("vertex={} fragment={} color={:#x} fmt={:#x} size={}x{} primitive={:#x} count={} instances={} mask={} viewport={},{},{},{} clips={} cull={} depth_enabled={} depth_func={} depth_write={} blend={},{},{},{},{},{} color_mask={:#x}",
                hex_string(vp->vertex_program->hash), hex_string(fp->fragment_program->hash),
                uint32_t(record.color_surface.data.address()), uint32_t(record.color_surface.colorFormat),
                ctx.impl->width, ctx.impl->height, uint32_t(primitive), count, instances, record.is_maskupdate,
                viewport.originX, viewport.originY, viewport.width, viewport.height, clip.size(),
                uint32_t(record.cull_mode), depth_enabled, uint32_t(depth_func), uint32_t(depth_write_mode),
                uint32_t(blend.colorFunc), uint32_t(blend.alphaFunc), uint32_t(blend.colorSrc), uint32_t(blend.colorDst),
                uint32_t(blend.alphaSrc), uint32_t(blend.alphaDst), uint32_t(blend.colorMask));
            if (impl->traced_draws.insert(detail).second) LOG_INFO("Metal draw binding: {}", detail);
        }
        if (impl->trace_target_width == ctx.impl->width && impl->traced_target_draws < 10000) {
            const uint32_t sequence = impl->traced_target_draws++;
            const bool has_source = fragment_resources && fp->fragment_program->textures_used[0];
            const Address source = has_source ? ctx.textures[0].data_addr << 2 : 0;
            uint64_t source_hash = 0;
            bool source_hash_valid = false;
            if (source) {
                const size_t bytes = texture_storage_size(ctx.textures[0]);
                const uint64_t end = uint64_t(source) + bytes;
                if (bytes && bytes <= 1024 * 1024 && end <= uint64_t(UINT32_MAX) - 4095
                    && is_valid_addr_range(mem, source, Address(end))) {
                    source_hash = XXH3_64bits(Ptr<uint8_t>(source).get(mem), bytes);
                    source_hash_valid = true;
                }
            }
            LOG_INFO("Metal target-width draw: seq={} target={:#x} size={}x{} fs={} source0={:#x} source_hash_valid={} source_hash={:#x}",
                sequence, ctx.impl->guest_color.data.address(), ctx.impl->width, ctx.impl->height,
                hex_string(fp->fragment_program->hash), source, source_hash_valid, source_hash);
        }
        if (capture_draw) draw_metadata << "viewport " << viewport.originX << ' ' << viewport.originY << ' '
            << viewport.width << ' ' << viewport.height << ' ' << viewport.znear << ' ' << viewport.zfar << '\n'
            << "depth_clamp 1\nfar_clip " << ctx.shader_hints.metal_far_clip << '\n';
        // MSL constant structures round their size to the largest member alignment.
        // Keep padding local to Metal so shared GL/Vulkan uniform layouts are untouched.
        shader::RenderVertUniformBlockExtended vertex_info{};
        shader::RenderFragUniformBlockExtended fragment_info{};
        auto &vert = vertex_info.base_block;
        vert.viewport_flip = {record.viewport_flat ? 1.0f : record.viewport_flip[0], record.viewport_flat ? -1.0f : -record.viewport_flip[1], 1, 1};
        vert.viewport_flag = record.viewport_flat ? 0 : 1;
        vert.screen_width = ctx.impl->expanded_color
            ? float(ctx.impl->guest_color.width/(ctx.impl->samples/2)) : ctx.impl->width/res_multiplier;
        vert.screen_height = ctx.impl->expanded_color
            ? float(ctx.impl->guest_color.height/2) : ctx.impl->height/res_multiplier;
        vert.z_offset = record.z_offset; vert.z_scale = record.z_scale;
        vert.point_size = float(std::max(1u, record.line_width)) * res_multiplier;
        auto &frag = fragment_info.base_block;
        // Match Plus' fragment coordinate contract for downscale without
        // MSAA. Expanded MSAA still uses Metal's per-sample coordinate mapping.
        const float fragment_scale = ctx.impl->samples==1 && ctx.impl->guest_color.data
                && ctx.impl->guest_color.downscale ? res_multiplier*0.5f : res_multiplier;
        frag.res_multiplier = fragment_scale; frag.res_multiplier_y = fragment_scale;
        frag.frag_coord_samples = ctx.impl->expanded_color ? ctx.impl->samples : 1;
        frag.writing_mask = record.writing_mask;
        frag.front_disabled = record.front_side_fragment_program_mode == SCE_GXM_FRAGMENT_PROGRAM_DISABLED;
        frag.back_disabled = record.two_sided == SCE_GXM_TWO_SIDED_DISABLED ? frag.front_disabled : record.back_side_fragment_program_mode == SCE_GXM_FRAGMENT_PROGRAM_DISABLED;
        std::vector<GuestBufferRange> guest_ranges;
        std::array<std::array<GuestBufferRange, SCE_GXM_REAL_MAX_UNIFORM_BUFFER>, 2> uniform_ranges{};
        const ShaderProgram *programs[] = {vp->vertex_program.get(), fp->fragment_program.get()};
        const SceGxmProgram *gxps[] = {vp->program(), fp->program()};
        for (uint32_t stage = 0; stage < 2; ++stage) {
            if (stage == 1 && !fragment_resources) continue;
            const auto &program = *programs[stage];
            for (uint32_t block = 0; block < program.buffer_count; ++block) {
                if (!program.uniform_buffer_sizes.at(block)) continue;
                const auto &binding = ctx.uniforms[stage].at(block);
                require(binding.data && binding.size, "Metal: shader uses an unbound uniform buffer");
                auto &range = uniform_ranges[stage][block];
                range = {binding.data, binding.size, false};
                bool mapped_uniform = false;
                // GXP LDR/STR can address before or after its declared uniform
                // registers. Retain the complete containing GXM mapping, while
                // the shader's base still points at the bound guest address.
                if (memory_backed_uniform_slot(gxps[stage]->buffer_flags, block)) {
                    const auto mapped = impl->mapped_memory.range(binding.address, binding.size);
                    range = {binding.data - (binding.address - mapped.address), mapped.size, mapped.mapped};
                    mapped_uniform = mapped.mapped;
                }
                if (!mapped_uniform && (program.dynamic_uniform_buffers & (1u << block))) {
                    // Plus copies at most 16 KiB beyond a dynamically indexed
                    // uniform. Metal snapshots whole host pages; include only
                    // guest pages that remain allocated and stop at the next
                    // color surface, as Plus does for its DoubleBuffer path.
                    constexpr uint64_t slack_window = 16 * 1024;
                    uint64_t limit = std::min<uint64_t>(uint64_t(binding.address) + slack_window, UINT32_MAX);
                    const auto next_surface = impl->surfaces.upper_bound(binding.address);
                    if (next_surface != impl->surfaces.end())
                        limit = std::min<uint64_t>(limit, next_surface->first);
                    if (next_surface != impl->surfaces.begin()) {
                        const auto previous = std::prev(next_surface);
                        if (uint64_t(previous->first) + surface_memory_size(previous->second.guest) > binding.address)
                            limit = binding.address;
                    }
                    uint64_t valid_end = uint64_t(binding.address) + binding.size;
                    while (valid_end < limit) {
                        const uint64_t candidate = std::min<uint64_t>(limit, ((valid_end / 4096) + 1) * 4096);
                        if (!is_valid_addr_range(mem, binding.address, Address(candidate)))
                            break;
                        valid_end = candidate;
                    }
                    range.size = std::max(range.size, size_t(valid_end - binding.address));
                }
                // GXM-mapped memory is live GPU-visible memory, as in Vulkan.
                // Snapshotting its full extent on every draw copies entire
                // mappings even if LDR touches only a few words.
                guest_ranges.push_back(range);
            }
        }
        ctx.impl->direct_guest_memory_used |= std::any_of(guest_ranges.begin(), guest_ranges.end(),
            [](const GuestBufferRange &range) { return range.direct; });
        if (impl->trace_finish_timing) {
            bool mapped_draw = false;
            for (const auto &range : guest_ranges) if (range.direct) {
                mapped_draw = true;
                impl->timed_mapped_bytes += range.size;
                impl->timed_max_mapped_extent = std::max<uint64_t>(impl->timed_max_mapped_extent, range.size);
            }
            impl->timed_mapped_draws += mapped_draw;
        }
        const bool shader_guest_stores = vs.writes_guest_memory || fs.writes_guest_memory;
        if (shader_guest_stores) {
            ctx.impl->guest_write_watches_pending = true;
            const uintptr_t arena = reinterpret_cast<uintptr_t>(mem.memory.get());
            for (const auto &range : guest_ranges) {
                const uintptr_t begin = reinterpret_cast<uintptr_t>(range.data);
                if (begin >= arena && begin - arena <= UINT32_MAX
                    && range.size <= uint64_t(UINT32_MAX) - (begin - arena))
                    impl->surface_writes.prepare_gpu_writes(mem, Address(begin - arena), range.size);
            }
            // Discard resources previously created while their CPU pages were
            // ReadOnly. Commands already using them retain their own references.
            impl->direct_guest_buffers.clear();
        }
        GuestBufferBindings native_buffers(*impl->device, guest_ranges, mem.host_page_size, !synchronous,
            &ctx.impl->uploads, &impl->direct_guest_buffers);
        const size_t uniform_upload_bytes=native_buffers.allocated_bytes();
        size_t draw_upload_bytes=uniform_upload_bytes;
        native_buffers.make_resident(encoder);
        vertex_info.set_buffer_count(vp->vertex_program->buffer_count);
        fragment_info.set_buffer_count(fragment_resources ? fp->fragment_program->buffer_count : 0);
        vertex_info.set_texture_count(vp->vertex_program->texture_count);
        fragment_info.set_texture_count(fragment_resources ? fp->fragment_program->texture_count : 0);
        for (uint32_t slot=0;slot<SCE_GXM_MAX_TEXTURE_UNITS;++slot) {
            vertex_info.set_viewport_ratio(slot,{1,1});
            fragment_info.set_viewport_ratio(slot,{1,1});
        }
        for (uint32_t stage = 0; stage < 2; ++stage) {
            if (stage == 1 && !fragment_resources) continue;
            const auto &program = *programs[stage];
            for (uint32_t block = 0; block < program.buffer_count; ++block) {
                if (!program.uniform_buffer_sizes.at(block)) continue;
                const auto &binding = ctx.uniforms[stage].at(block);
                const uint64_t address = native_buffers.address({binding.data, binding.size});
                if (stage == 0) vertex_info.set_buffer_address(block, address);
                else fragment_info.set_buffer_address(block, address);
                if (capture_draw) {
                    // Record the prefix as well as the suffix. The base offset
                    // lets a replay preserve both negative and positive LDRs.
                    const auto &range = uniform_ranges[stage][block];
                    const size_t base_offset = binding.data - range.data;
                    if (stage == 0) {
                        dump_bytes(fmt::format("uniform-{}.bin",block),range.data,range.size);
                        draw_metadata << "uniform " << block << ' ' << range.size << ' ' << base_offset << '\n';
                    } else {
                        dump_bytes(fmt::format("fragment-uniform-{}.bin",block),range.data,range.size);
                        texture_metadata << "fragment_uniform " << block << ' ' << range.size << ' ' << base_offset << '\n';
                    }
                }
            }
        }
        if (capture_draw) {
            dump_bytes("vertex-info.bin",&vertex_info.base_block,sizeof(vertex_info.base_block));
            draw_metadata << "buffer_count " << vertex_info.buffer_count << '\n';
        }
        [encoder setFragmentTexture:record.is_maskupdate ? nil : ctx.impl->mask atIndex:shader::metal::MASK_TEXTURE];
        std::array<id<MTLTexture>, SCE_GXM_MAX_TEXTURE_UNITS> replay_fragment_textures{};
        std::array<id<MTLSamplerState>, SCE_GXM_MAX_TEXTURE_UNITS> replay_fragment_samplers{};
        size_t vertex_elements = 0;
        for (uint32_t stream = 0; stream < vp->streams.size(); ++stream) if (streams_used & (1u << stream)) {
            const auto &data = record.vertex_streams[stream];
            require(data.data && data.size, "Metal: empty vertex stream");
            const auto stride = vp->streams[stream].stride;
            const auto aligned = align(stride, 4);
            const auto snapshot = vertex_stream_snapshot(ctx.vertex_stream_snapshots[stream], data.size);
            const uint8_t *bytes = snapshot.empty() ? data.data.get(mem) : snapshot.data();
            size_t elements = stride && stride != aligned ? (data.size + stride - 1) / stride : 0;
            const auto &layout = vertex_stream_layouts[stream];
            if (repack_stream[stream]) {
                if (layout.step_function == MTLVertexStepFunctionConstant) {
                    elements = 1;
                } else if (layout.step_function == MTLVertexStepFunctionPerInstance) {
                    elements = instances;
                } else {
                    if (!vertex_elements) {
                        for (uint32_t i = 0; i < count; ++i) {
                            const uint32_t index = index_format == SCE_GXM_INDEX_FORMAT_U16
                                ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i];
                            vertex_elements = std::max(vertex_elements, size_t(index) + 1);
                        }
                    }
                    elements = vertex_elements;
                }
                require(elements && layout.stride && elements <= std::numeric_limits<size_t>::max() / layout.stride,
                    "Metal: invalid repacked vertex stream extent");
                const uint64_t last_record = uint64_t(elements - 1) * stride;
                for (const auto &attribute : vertex_attributes) {
                    if (attribute.format != MTLVertexFormatInvalid
                        && attribute.buffer_index == shader::metal::VERTEX_STREAM_BUFFER_BASE + stream)
                        require(last_record + attribute.source_offset + attribute.byte_size <= data.size,
                            "Metal: vertex stream does not cover the shader attribute");
                }
            }
            const size_t length = repack_stream[stream] ? elements * layout.stride
                : elements ? elements * aligned : data.size;
            const auto buffer = ctx.impl->uploads.allocate(*impl->device, length);
            auto *upload = static_cast<uint8_t *>(buffer.buffer.contents) + buffer.offset;
            if (repack_stream[stream]) {
                std::memset(upload, 0, length);
                for (const auto &attribute : vertex_attributes) {
                    if (attribute.format == MTLVertexFormatInvalid
                        || attribute.buffer_index != shader::metal::VERTEX_STREAM_BUFFER_BASE + stream)
                        continue;
                    for (size_t i = 0; i < elements; ++i)
                        std::memcpy(upload + i * layout.stride + attribute.offset,
                            bytes + i * stride + attribute.source_offset, attribute.byte_size);
                }
            } else if (elements) {
                // The old temporary vector zero-filled padding before copying
                // to the upload arena. Preserve those bytes while packing
                // directly into the arena's retained GPU-visible slice.
                std::memset(upload, 0, length);
                for (size_t i = 0; i < elements; ++i)
                    std::memcpy(upload + i * aligned, bytes + i * stride,
                        std::min<size_t>(stride, data.size - i * stride));
            } else std::memcpy(upload, bytes, length);
            draw_upload_bytes += length;
            [encoder setVertexBuffer:buffer.buffer offset:buffer.offset atIndex:shader::metal::VERTEX_STREAM_BUFFER_BASE + stream];
            if (capture_draw) {
                dump_bytes(fmt::format("stream-{}.bin",shader::metal::VERTEX_STREAM_BUFFER_BASE+stream),upload,length);
                draw_metadata << "stream " << shader::metal::VERTEX_STREAM_BUFFER_BASE+stream << ' ' << length << '\n';
            }
        }
        for (uint32_t index = 0; index < SCE_GXM_MAX_TEXTURE_UNITS * 2; ++index) {
            const bool vertex = index >= SCE_GXM_MAX_TEXTURE_UNITS;
            if (!fragment_resources && !vertex) continue;
            const uint32_t slot = index % SCE_GXM_MAX_TEXTURE_UNITS;
            const auto &program = vertex ? static_cast<const ShaderProgram &>(*vp->vertex_program) : static_cast<const ShaderProgram &>(*fp->fragment_program);
            if (!program.textures_used[slot]) continue;
            const auto &texture = ctx.textures[index];
            const Address texture_address = texture.data_addr << 2;
            const auto texture_base = gxm::get_base_format(gxm::get_format(texture));
            const bool prepared = prepared_images[index] != nil;
            auto surface = prepared ? impl->surfaces.end() : find_direct_surface(texture);
            const auto subrectangle_surface=prepared ? impl->surfaces.end() : find_subrectangle(texture);
            if (subrectangle_surface!=impl->surfaces.end()) surface=subrectangle_surface;
            // A 4-byte texture may begin at the second word of an 8-byte
            // RG32 or F16x4 color target.
            if (!prepared && surface == impl->surfaces.end() && texture_address >= 4
                && surface_word_target_supported(texture_base)) {
                auto previous = find_texture_surface(texture_address - 4);
                if (previous != impl->surfaces.end()
                    && ((rg32_linear_alias(previous->second,texture,texture_address))
                        || rgba16_linear_alias(previous->second,texture,texture_address)))
                    surface = previous;
            }
            if (impl->trace_textures && impl->traced_textures.size() < 4096) {
                const auto &ds = record.depth_stencil_surface;
                const auto detail = fmt::format("shader={} stage={} slot={} tex={:#x} fmt={:#x} size={}x{} cached_color={} colorfmt={:#x} depth={:#x} depthfmt={:#x} load={} store={} type={:#x} min={} mag={} stride={}",
                    hex_string(program.hash), vertex ? "vertex" : "fragment", slot, texture.data_addr << 2,
                    uint32_t(gxm::get_format(texture)), gxm::get_width(texture), gxm::get_height(texture),
                    surface != impl->surfaces.end(), surface != impl->surfaces.end() ? uint32_t(surface->second.guest.colorFormat) : 0,
                    ds.depth_data.address(), uint32_t(ds.get_format()), uint32_t(ds.force_load), uint32_t(ds.force_store),
                    uint32_t(texture.texture_type()), uint32_t(texture.min_filter), uint32_t(texture.mag_filter),
                    texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED ? gxm::get_stride_in_bytes(texture) : 0);
                if (impl->traced_textures.insert(detail).second) LOG_INFO("Metal texture binding: {}", detail);
            }
            id<MTLTexture> native = prepared_images[index];
            bool uploaded_view = false;
            if (surface != impl->surfaces.end()) {
                native=surface->second.color;
                if (const auto rect=surface_subrectangle(surface->second.guest,texture)) {
                    auto crop=surface->second.subrectangles.find({rect->x,rect->y,rect->width,rect->height});
                    if (crop!=surface->second.subrectangles.end()) native=crop->second;
                }
            }
            else if (!native) {
                const auto base = gxm::get_base_format(gxm::get_format(texture));
                if (base == SCE_GXM_TEXTURE_BASE_FORMAT_U8 || base == SCE_GXM_TEXTURE_BASE_FORMAT_S8
                    || base == SCE_GXM_TEXTURE_BASE_FORMAT_X8U24 || base == SCE_GXM_TEXTURE_BASE_FORMAT_F32
                    || base == SCE_GXM_TEXTURE_BASE_FORMAT_F32M
                    || base == SCE_GXM_TEXTURE_BASE_FORMAT_U16
                    || base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8) {
                    for (auto &[key, entry] : impl->depth_surfaces) {
                        if (base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                            && impl->frame_timestamp-entry.last_attached_frame>2) continue;
                        const auto rect=depth_subrectangle(entry.guest,entry.width,entry.height,entry.multisample,texture);
                        if (!rect) continue;
                        auto &snapshot=entry.snapshots[base];
                        const uint32_t memory_width=entry.width*(entry.multisample==SCE_GXM_MULTISAMPLE_4X ? 2 : 1);
                        const uint32_t memory_height=entry.height*(entry.multisample!=SCE_GXM_MULTISAMPLE_NONE ? 2 : 1);
                        const bool cropped=rect->x || rect->y || rect->width!=memory_width || rect->height!=memory_height;
                        id<MTLCommandBuffer> snapshot_commands=nil;
                        if (!snapshot && cropped) {
                            snapshot_commands=[impl->device->command_queue() commandBuffer];
                            snapshot_commands.label=@"Vita3K surface depth sampling";
                        }
                        if (!snapshot) {
                            if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                            const uint32_t guest_width=entry.width*(entry.multisample==SCE_GXM_MULTISAMPLE_4X ? 2 : 1);
                            const uint32_t guest_height=entry.height*(entry.multisample==SCE_GXM_MULTISAMPLE_NONE ? 1 : 2);
                            snapshot = base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                                ? impl->caster->packed_depth_snapshot(entry.texture,false,snapshot_commands)
                                : base == SCE_GXM_TEXTURE_BASE_FORMAT_U8 || base == SCE_GXM_TEXTURE_BASE_FORMAT_S8
                                ? impl->caster->stencil_snapshot(entry.texture, base == SCE_GXM_TEXTURE_BASE_FORMAT_S8,
                                    res_multiplier,guest_width,guest_height,false,snapshot_commands)
                                : impl->caster->depth_snapshot(entry.texture, base == SCE_GXM_TEXTURE_BASE_FORMAT_U16,
                                    res_multiplier,guest_width,guest_height,false,snapshot_commands);
                        }
                        // Publish all stored depth samples in the expanded guest sampling grid.
                        native = snapshot;
                        if (cropped) {
                            auto &crop=entry.subrectangles[{uint32_t(base),rect->x,rect->y,rect->width,rect->height}];
                            if (!crop) crop=depth_alias_view(*impl->caster,snapshot,memory_width,memory_height,
                                *rect,snapshot_commands);
                            native=crop;
                        }
                        if (snapshot_commands) {
                            std::string error;
                            require(impl->device->submit_and_wait(snapshot_commands,error),error);
                        }
                        if (base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8)
                            native=rgba8_gamma_view(native,texture.gamma_mode);
                        if (impl->trace_textures && impl->traced_textures.size() < 4096) {
                            const auto detail = fmt::format("tex={:#x} fmt={:#x} guest={}x{} native={}x{} multisample={}",
                                texture_address, uint32_t(gxm::get_format(texture)), gxm::get_width(texture), gxm::get_height(texture),
                                native.width, native.height, uint32_t(entry.multisample));
                            if (impl->traced_textures.insert(detail).second) LOG_INFO("Metal depth texture binding: {}", detail);
                        }
                        break;
                    }
                }
                if (!native) {
                    auto upload = texture;
                    const bool cube = ((vertex ? vs.cube_texture_mask : fs.cube_texture_mask) & (1u<<slot)) != 0;
                    if (cube) {
                        upload = cube_texture_descriptor(texture);
                        if (upload.texture_type()!=texture.texture_type())
                            LOG_INFO_ONCE("Metal: cube sampler resolves six swizzled faces at {:#x}, mip_count={}",
                                texture_address,uint32_t(texture.mip_count));
                    }
                    texture_cache.cache_and_bind_image(upload, mem);
                    native = current_texture_view(texture_cache, gxm::get_format(texture));
                    uploaded_view = true;
                }
            }
            require(native != nil, "Metal: missing sampled texture");
            if (native == ctx.impl->color && !record.is_maskupdate) {
                require(color_feedback != nil, "Metal: missing color feedback snapshot");
                native = color_feedback;
            }
            const SceGxmColorFormat *rendered_format = surface != impl->surfaces.end() ? &surface->second.guest.colorFormat : nullptr;
            if (inline_cast_images[index]) {
                native=inline_cast_images[index];
                rendered_format=nullptr; // The queued cast already reconstructed guest memory order.
            }
            if (inline_gamma_images[index]) {
                native=inline_gamma_images[index];
                rendered_format=nullptr; // The queued conversion already restored guest channel order.
            }
            std::optional<SceGxmColorFormat> repacked_color;
            if (surface != impl->surfaces.end() && (native.pixelFormat == MTLPixelFormatRG32Float || inline_rg32_images[index])
                && (texture_base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                    || texture_base == SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8)) {
                auto &entry = surface->second;
                const uint32_t word_offset = (texture_address - surface->first)/4;
                const bool signed_normalized = texture_base == SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8;
                const bool separate_word = res_multiplier != 1 && entry.has_word_offset_view;
                require(rg32_linear_alias(entry,texture,texture_address),
                    "Metal: RG32/RGBA8 alias requires equal guest row size and positive resolution scale");
                const bool guest_grid = vertex && res_multiplier < 1 && !separate_word;
                auto &cast = entry.rgba8_casts[{word_offset, uint32_t(signed_normalized), uint32_t(guest_grid)}];
                if (inline_rg32_images[index]) cast=inline_rg32_images[index];
                else if (!cast) {
                    if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                    cast = impl->caster->rgba8_from_rg32(native,
                        (entry.guest.colorFormat & SCE_GXM_COLOR_SWIZZLE_MASK) == SCE_GXM_COLOR_SWIZZLE2_RG,
                        word_offset, signed_normalized,
                        res_multiplier<1 && !separate_word && vertex ? entry.guest.width : 0,
                        res_multiplier<1 && !separate_word && vertex ? entry.guest.height : 0, nil, separate_word);
                }
                native = cast;
                // Fragment casts use Plus' screen-relative correction before
                // viewport mapping. Separate views already select their word.
                if (!vertex && !entry.has_word_offset_view && res_multiplier != 1) {
                    const auto &target = static_cast<const MetalRenderTarget &>(*ctx.current_render_target);
                    texture_mip_info[index].cast_coords = {fragment_scale,
                        1.f / target.width, 1.f / target.height,
                        float(word_offset)};
                }
                const std::pair<float,float> offset{separate_word || !vertex ? 0.f
                    : packed_alias_x_offset(res_multiplier,uint32_t(native.width)),0};
                if (vertex) vertex_info.set_viewport_offset(slot,offset);
                else fragment_info.set_viewport_offset(slot,offset);
                if (!impl->dump_surface_dir.empty() && impl->dumped_surfaces.size() < 32) {
                    const auto name = fmt::format("{:08x}-offset{}-snorm{}-{}x{}", surface->first, word_offset, signed_normalized, native.width, native.height);
                    if (impl->dumped_surfaces.insert(name).second) {
                        std::filesystem::create_directories(impl->dump_surface_dir);
                        std::vector<uint8_t> bytes(native.width*native.height*4);
                        [native getBytes:bytes.data() bytesPerRow:native.width*4
                            fromRegion:MTLRegionMake2D(0,0,native.width,native.height) mipmapLevel:0];
                        std::ofstream raw(impl->dump_surface_dir/(name+".rgba"),std::ios::binary);
                        raw.write(reinterpret_cast<const char *>(bytes.data()),bytes.size());
                        std::ofstream preview(impl->dump_surface_dir/(name+".ppm"),std::ios::binary);
                        preview << "P6\n" << native.width << " " << native.height << "\n255\n";
                        for (size_t pixel=0;pixel<bytes.size()/4;++pixel)
                            preview.write(reinterpret_cast<const char *>(bytes.data()+pixel*4),3);
                        require(bool(raw) && bool(preview),"Metal: cannot write surface diagnostic");
                        LOG_INFO("Metal surface alias saved: {}",(impl->dump_surface_dir/name).string());
                    }
                }
                rendered_format = nullptr; // The cast has reconstructed guest memory channel order.
            }
            if (surface != impl->surfaces.end() && native.pixelFormat==MTLPixelFormatRG32Float
                && texture_base!=SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                && texture_base!=SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8
                && rg32_linear_alias(surface->second,texture,texture_address)) {
                auto &entry=surface->second;
                const uint32_t word_offset=(texture_address-surface->first)/4;
                const bool separate_word=res_multiplier!=1 && entry.has_word_offset_view;
                const bool guest_grid=vertex && res_multiplier<1 && !separate_word;
                auto &cast=entry.word_casts[{word_offset,uint32_t(texture_base),uint32_t(guest_grid),
                    uint32_t(texture.swizzle_format)}];
                if (!cast) {
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    cast=impl->caster->word_texture_from_rg32(native,texture_base,
                        texture.swizzle_format,
                        (entry.guest.colorFormat & SCE_GXM_COLOR_SWIZZLE_MASK)==SCE_GXM_COLOR_SWIZZLE2_RG,
                        word_offset,guest_grid ? entry.guest.width : 0,
                        guest_grid ? entry.guest.height : 0,nil,separate_word);
                }
                native=cast;
                if (!vertex && !entry.has_word_offset_view && res_multiplier!=1) {
                    const auto &target=static_cast<const MetalRenderTarget &>(*ctx.current_render_target);
                    texture_mip_info[index].cast_coords={fragment_scale,
                        1.f/target.width,1.f/target.height,float(word_offset)};
                }
                const std::pair<float,float> offset{separate_word || !vertex ? 0.f
                    : packed_alias_x_offset(res_multiplier,uint32_t(native.width)),0};
                if (vertex) vertex_info.set_viewport_offset(slot,offset);
                else fragment_info.set_viewport_offset(slot,offset);
                rendered_format=nullptr;
            }
            if (surface != impl->surfaces.end()
                && rgba16_linear_alias(surface->second,texture,texture_address)) {
                auto &entry=surface->second;
                const uint32_t word_offset=(texture_address-surface->first)/4;
                const bool separate_word=res_multiplier!=1 && entry.has_word_offset_view;
                const bool guest_grid=vertex && res_multiplier<1 && !separate_word;
                auto &cast=entry.word_casts[{word_offset,uint32_t(texture_base),uint32_t(guest_grid),
                    uint32_t(texture.swizzle_format)}];
                if (!cast) {
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    if (native==entry.color) native=raw_surface_color(entry);
                    else if (native==color_feedback && raw_color_feedback) native=raw_color_feedback;
                    cast=impl->caster->word_texture_from_rgba16(native,entry.guest.colorFormat,
                        texture_base,texture.swizzle_format,word_offset,
                        guest_grid ? entry.guest.width : 0,guest_grid ? entry.guest.height : 0,
                        nil,separate_word);
                }
                native=cast;
                if (!vertex && !entry.has_word_offset_view && res_multiplier!=1) {
                    const auto &target=static_cast<const MetalRenderTarget &>(*ctx.current_render_target);
                    texture_mip_info[index].cast_coords={fragment_scale,
                        1.f/target.width,1.f/target.height,float(word_offset)};
                }
                const std::pair<float,float> offset{separate_word || !vertex ? 0.f
                    : packed_alias_x_offset(res_multiplier,uint32_t(native.width)),0};
                if (vertex) vertex_info.set_viewport_offset(slot,offset);
                else fragment_info.set_viewport_offset(slot,offset);
                rendered_format=nullptr;
                if (texture.gamma_mode && (texture_base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                    || texture_base==SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8)) {
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    native=impl->caster->rgba8_surface_sampling(native,
                        SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,texture.gamma_mode);
                }
            }
            bool cast_memory=false;
            if (rendered_format) {
                const bool same_format=surface_texture_format_matches(*rendered_format,gxm::get_format(texture));
                if (!same_format && surface_format_cast_supported(*rendered_format,texture_base,texture.swizzle_format)) {
                    const auto &entry = surface->second;
                    require(surface_subrectangle(surface->second.guest,texture).has_value(),
                        fmt::format("Metal: format alias has incompatible extent, stride or memory layout: color={:#x} fmt={:#x} size={}x{} stride={} type={} texture={:#x} fmt={:#x} size={}x{} stride={} type={:#x}",
                            uint32_t(entry.guest.data.address()),uint32_t(entry.guest.colorFormat),entry.guest.width,entry.guest.height,entry.guest.strideInPixels,uint32_t(entry.guest.surfaceType),
                            texture_address,uint32_t(gxm::get_format(texture)),gxm::get_width(texture),gxm::get_height(texture),texture.texture_type()==SCE_GXM_TEXTURE_LINEAR_STRIDED?gxm::get_stride_in_bytes(texture):0,uint32_t(texture.texture_type())));
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    if (native==entry.color) native=raw_surface_color(entry);
                    else if (native==color_feedback && raw_color_feedback) native=raw_color_feedback;
                    native=impl->caster->surface_format_cast(native,*rendered_format,texture_base,texture.swizzle_format);
                    require(native!=nil,"Metal: cannot reinterpret sampled surface format");
                    repacked_color=repacked_u2_color(gxm::get_format(texture));
                    rendered_format=repacked_color ? &*repacked_color : nullptr;
                    cast_memory=true;
                }
            }
            if ((rendered_format || cast_memory) && (native.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB || texture.gamma_mode)) {
                if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                // Decode guest memory channels before applying the texture's
                // swizzle, including color formats that relocate alpha.
                native = impl->caster->rgba8_surface_sampling(native,
                    rendered_format ? *rendered_format : SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,
                    texture.gamma_mode);
                rendered_format = nullptr;
            }
            if (!prepared && !uploaded_view && !texture_mip_info[index].control[3]) native = sampling_view(native, gxm::get_format(texture),
                rendered_format);
            if (capture_draw) {
                const char *stage=vertex ? "vertex" : "fragment";
                if (const auto producer=find_texture_surface(texture_address);producer!=impl->surfaces.end()) {
                    const auto &guest=producer->second.guest;
                    texture_metadata << "producer " << stage << ' ' << slot << ' ' << guest.data.address() << ' '
                        << guest.width << ' ' << guest.height << ' ' << guest.strideInPixels << ' '
                        << uint32_t(guest.colorFormat) << ' ' << uint32_t(guest.surfaceType) << ' '
                        << prepared << ' ' << uploaded_view << '\n';
                    dump_attachment(fmt::format("{}-producer-{}",stage,slot).c_str(),producer->second.color);
                }
                if (std::getenv("VITA3K_METAL_DUMP_RAW_TEXTURE") && !prepared
                    && surface == impl->surfaces.end() && slot == 0) {
                    const size_t bytes = texture_storage_size(texture);
                    const uint64_t end = uint64_t(texture_address) + bytes;
                    if (bytes && bytes <= 1024 * 1024 && end <= uint64_t(UINT32_MAX) - 4095
                        && is_valid_addr_range(mem, texture_address, Address(end))) {
                        const auto name = fmt::format("{}-guest-texture-{}.bin", stage, slot);
                        dump_bytes(name, Ptr<uint8_t>(texture_address).get(mem), bytes);
                        texture_metadata << "guest_texture " << stage << ' ' << slot << ' '
                            << texture_address << ' ' << bytes << ' ' << name << '\n';
                    }
                }
                const bool is_cube=native.textureType==MTLTextureTypeCube;
                const uint32_t faces=is_cube ? 6 : 1;
                const uint32_t levels=uint32_t(native.mipmapLevelCount);
                if (!is_cube && (native.pixelFormat == MTLPixelFormatA1BGR5Unorm
                    || native.pixelFormat == MTLPixelFormatBGR5A1Unorm)) {
                    std::vector<uint16_t> packed(native.width * native.height);
                    [native getBytes:packed.data() bytesPerRow:native.width * sizeof(uint16_t)
                        fromRegion:MTLRegionMake2D(0,0,native.width,native.height) mipmapLevel:0];
                    dump_bytes(fmt::format("{}-texture-{}-packed16.bin",stage,slot),packed.data(),packed.size()*sizeof(uint16_t));
                }
                size_t total_bytes=0;
                for (uint32_t mip=0;mip<levels;++mip)
                    total_bytes+=size_t(std::max(NSUInteger(1),native.width>>mip))*std::max(NSUInteger(1),native.height>>mip)*16*faces;
                texture_metadata << "sampled_image " << stage << ' ' << slot << ' ' << (is_cube ? "cube" : "2d") << ' '
                    << native.width << ' ' << native.height << ' ' << levels << ' ' << uint32_t(native.pixelFormat) << '\n';
                texture_metadata << "native_sampler " << stage << ' ' << slot << ' '
                    << std::clamp(texture_cache.anisotropic_filtering,1,16) << '\n';
                if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                // Budget the entire image, including every cube face and mip,
                // before writing any subresource. Never label a partial image complete.
                bool complete=total_bytes<=64*1024*1024 && captured_texture_bytes+total_bytes<=256*1024*1024;
                const char *reason=complete ? "unsupported_subresource" : "capture_budget";
                uint32_t written=0;
                for (uint32_t face=0;complete && face<faces;++face) for (uint32_t mip=0;complete && mip<levels;++mip) {
                    auto snapshot=impl->caster->sampling_snapshot(native,mip,face);
                    if (!snapshot) { complete=false;break; }
                    const auto name=is_cube ? fmt::format("{}-texture-{}-face-{}-mip-{}.rgba32f",stage,slot,face,mip)
                        : mip ? fmt::format("{}-texture-{}-mip-{}.rgba32f",stage,slot,mip)
                        : fmt::format("{}-texture-{}.rgba32f",stage,slot);
                    const size_t bytes=size_t(snapshot.width)*snapshot.height*16;
                    std::vector<uint8_t> pixels(bytes);
                    [snapshot getBytes:pixels.data() bytesPerRow:snapshot.width*16
                        fromRegion:MTLRegionMake2D(0,0,snapshot.width,snapshot.height) mipmapLevel:0];
                    dump_bytes(name,pixels.data(),pixels.size());captured_texture_bytes+=bytes;++written;
                    if (!is_cube && mip==0) texture_metadata << "sampled_texture " << stage << ' ' << slot << ' '
                        << snapshot.width << ' ' << snapshot.height << ' ' << name << '\n';
                    texture_metadata << "sampled_subresource " << stage << ' ' << slot << ' ' << face << ' ' << mip << ' '
                        << snapshot.width << ' ' << snapshot.height << ' ' << name << '\n';
                }
                if (complete) texture_metadata << "sampled_image_complete " << stage << ' ' << slot << ' ' << written << ' ' << total_bytes << '\n';
                else {
                    texture_metadata << "sampled_texture_skipped " << stage << ' ' << slot << ' '
                        << native.width << ' ' << native.height << ' ' << uint32_t(native.pixelFormat) << '\n';
                    texture_metadata << "sampled_image_skipped " << stage << ' ' << slot << ' ' << reason << '\n';
                }
                dump_bytes(fmt::format("{}-texture-{}.gxm",vertex ? "vertex" : "fragment",slot),&texture,sizeof(texture));
            }
            auto &mip_info=texture_mip_info[index];
            if (narrow_cast_coords[index] && fragment_scale!=1.f) {
                const auto &target=static_cast<const MetalRenderTarget &>(*ctx.current_render_target);
                // A complete two-texel view of each stored pixel has the
                // same screen-relative word phase as the Plus cast sampler.
                mip_info.cast_coords={fragment_scale,1.f/target.width,1.f/target.height,0.f};
            }
            if (surface_viewports[index]) {
                if (vertex) vertex_info.set_viewport_ratio(slot,*surface_viewports[index]);
                else fragment_info.set_viewport_ratio(slot,*surface_viewports[index]);
            }
            // The strided descriptor stores its row pitch in the min/mip/LOD
            // fields. Match the hardware sampler when sampling an aliased
            // surface through the reconstructed guest grid.
            mip_info.control[1]=sampler_metadata_flags(texture);
            mip_info.control[2]=effective_sampler_anisotropy(texture,texture_cache.anisotropic_filtering);
            const bool raw_word_carrier = mip_info.control[3] != 0;
            const bool software_float_filter = !impl->device->native_device().supports32BitFloatFiltering
                && (native.textureType == MTLTextureType2D || native.textureType == MTLTextureTypeCube)
                && (native.pixelFormat == MTLPixelFormatR32Float
                    || native.pixelFormat == MTLPixelFormatRG32Float
                    || native.pixelFormat == MTLPixelFormatRGBA32Float);
            auto sampling_texture = texture;
            if (software_float_filter) {
                // Apple7/8 can read 32-bit float textures but cannot filter
                // them. The native-mip shader path applies the guest's
                // min/mag/mip filters in float precision for 2D and cubes.
                // Keep a point sampler for LOD queries and Metal validation.
                mip_info.control[0] = 1;
                mip_info.control[2] = 1;
                for (uint32_t mip=0;mip<native.mipmapLevelCount && mip<mip_info.sizes.size();++mip)
                    if (!mip_info.sizes[mip][0] || !mip_info.sizes[mip][1])
                        mip_info.sizes[mip] = {std::max(1u,uint32_t(native.width>>mip)),
                            std::max(1u,uint32_t(native.height>>mip))};
                sampling_texture.mag_filter = SCE_GXM_TEXTURE_FILTER_POINT;
                if (texture.texture_type() != SCE_GXM_TEXTURE_LINEAR_STRIDED)
                    sampling_texture.min_filter = SCE_GXM_TEXTURE_FILTER_POINT;
                sampling_texture.mip_filter = 0;
            }
            if (raw_word_carrier) {
                // The UNORM16 channels hold integer word halves. Interpolating
                // them before bit reconstruction invents unrelated bit patterns.
                // Plus forces nearest sampling for this raw cast as well.
                sampling_texture.mag_filter = SCE_GXM_TEXTURE_FILTER_POINT;
                if (texture.texture_type() != SCE_GXM_TEXTURE_LINEAR_STRIDED)
                    sampling_texture.min_filter = SCE_GXM_TEXTURE_FILTER_POINT;
                sampling_texture.mip_filter = 0;
                mip_info.control[1] = sampler_metadata_flags(sampling_texture);
                mip_info.control[2] = 1;
            }
            auto sampling = make_sampler(*impl->device, sampling_texture,
                software_float_filter || raw_word_carrier ? 1 : texture_cache.anisotropic_filtering);
            if (vertex) { [encoder setVertexTexture:native atIndex:slot]; [encoder setVertexSamplerState:sampling atIndex:slot]; }
            else {
                [encoder setFragmentTexture:native atIndex:slot];
                [encoder setFragmentSamplerState:sampling atIndex:slot];
                replay_fragment_textures[slot] = native;
                replay_fragment_samplers[slot] = sampling;
            }
        }
        std::vector<uint8_t> vertex_bytes(align(vertex_info.get_size(), 16));
        std::vector<uint8_t> fragment_bytes(align(fragment_info.get_size(), 16));
        vertex_info.copy_to(vertex_bytes.data());
        fragment_info.copy_to(fragment_bytes.data());
        if (capture_draw) {
            dump_bytes("vertex-render-info.bin",vertex_bytes.data(),vertex_bytes.size());
            dump_bytes("fragment-render-info.bin",fragment_bytes.data(),fragment_bytes.size());
        }
        [encoder setVertexBytes:texture_mip_info.data()+SCE_GXM_MAX_TEXTURE_UNITS length:sizeof(shader::metal::TextureMipInfos) atIndex:shader::metal::TEXTURE_INFO_BUFFER];
        [encoder setFragmentBytes:texture_mip_info.data() length:sizeof(shader::metal::TextureMipInfos) atIndex:shader::metal::TEXTURE_INFO_BUFFER];
        if (capture_draw) {
            dump_bytes("vertex-texture-info.bin",texture_mip_info.data()+SCE_GXM_MAX_TEXTURE_UNITS,sizeof(shader::metal::TextureMipInfos));
            dump_bytes("fragment-texture-info.bin",texture_mip_info.data(),sizeof(shader::metal::TextureMipInfos));
        }
        [encoder setVertexBytes:vertex_bytes.data() length:vertex_bytes.size() atIndex:shader::metal::RENDER_INFO_BUFFER];
        [encoder setFragmentBytes:fragment_bytes.data() length:fragment_bytes.size() atIndex:shader::metal::RENDER_INFO_BUFFER];
        MTLPrimitiveType type;
        std::vector<uint32_t> converted_indices;
        std::vector<uint32_t> converted_ordinals;
        std::vector<uint32_t> split_capture_source_indices;
        size_t native_count = count;
        auto index_size = index_format == SCE_GXM_INDEX_FORMAT_U16 ? 2u : 4u;
        if (capture_split_faces) {
            const uint32_t used_count = primitive == SCE_GXM_PRIMITIVE_TRIANGLES
                || primitive == SCE_GXM_PRIMITIVE_TRIANGLE_EDGES ? count / 3 * 3 : count;
            split_capture_source_indices.reserve(used_count);
            for (uint32_t i = 0; i < used_count; ++i)
                split_capture_source_indices.push_back(index_size == 2
                    ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i]);
        }
        if (point_polygon) {
            type = MTLPrimitiveTypePoint;
            if (primitive == SCE_GXM_PRIMITIVE_TRIANGLES || primitive == SCE_GXM_PRIMITIVE_TRIANGLE_EDGES) {
                // Incomplete trailing vertices do not form a triangle.
                native_count = (count / 3) * 3;
            } else {
                native_count = size_t(count - 2) * 3;
                converted_indices.reserve(native_count);
                if (capture_per_occurrence) converted_ordinals.reserve(native_count);
                const auto at = [&](uint32_t i) -> uint32_t {
                    return index_size == 2 ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i];
                };
                for (uint32_t i = 2; i < count; ++i) {
                    if (primitive == SCE_GXM_PRIMITIVE_TRIANGLE_FAN) {
                        converted_indices.push_back(at(0));
                        converted_indices.push_back(at(i - 1));
                        if (capture_per_occurrence) {
                            converted_ordinals.push_back(0);
                            converted_ordinals.push_back(i - 1);
                        }
                    } else if (i & 1) {
                        converted_indices.push_back(at(i - 1));
                        converted_indices.push_back(at(i - 2));
                        if (capture_per_occurrence) {
                            converted_ordinals.push_back(i - 1);
                            converted_ordinals.push_back(i - 2);
                        }
                    } else {
                        converted_indices.push_back(at(i - 2));
                        converted_indices.push_back(at(i - 1));
                        if (capture_per_occurrence) {
                            converted_ordinals.push_back(i - 2);
                            converted_ordinals.push_back(i - 1);
                        }
                    }
                    converted_indices.push_back(at(i));
                    if (capture_per_occurrence) converted_ordinals.push_back(i);
                }
                indices = converted_indices.data();
                index_size = 4;
            }
        } else if (split_face_draw && !routed_geometry && primitive == SCE_GXM_PRIMITIVE_TRIANGLE_STRIP) {
            // Each split draw starts a new primitive. Preserve strip parity by
            // converting odd triangles with their first two indices swapped.
            type = MTLPrimitiveTypeTriangle;
            native_count = count < 3 ? 0 : size_t(count - 2) * 3;
            converted_indices.reserve(native_count);
            if (capture_split_faces) converted_ordinals.reserve(native_count);
            const auto at = [&](uint32_t i) -> uint32_t {
                return index_size == 2 ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i];
            };
            for (uint32_t i = 2; i < count; ++i) {
                if (i & 1) {
                    converted_indices.push_back(at(i - 1));
                    converted_indices.push_back(at(i - 2));
                    if (capture_split_faces) {
                        converted_ordinals.push_back(i - 1);
                        converted_ordinals.push_back(i - 2);
                    }
                } else {
                    converted_indices.push_back(at(i - 2));
                    converted_indices.push_back(at(i - 1));
                    if (capture_split_faces) {
                        converted_ordinals.push_back(i - 2);
                        converted_ordinals.push_back(i - 1);
                    }
                }
                converted_indices.push_back(at(i));
                if (capture_split_faces) converted_ordinals.push_back(i);
            }
            indices = converted_indices.data();
            index_size = 4;
        } else switch (primitive) {
        case SCE_GXM_PRIMITIVE_TRIANGLES:
            type = MTLPrimitiveTypeTriangle;
            if (routed_wide_line) native_count = count / 3 * 3;
            break;
        // Vulkan currently applies the same triangle-list fallback for this
        // GXM topology. Keep culling and fragment selection consistent with it.
        case SCE_GXM_PRIMITIVE_TRIANGLE_EDGES:
            type = MTLPrimitiveTypeTriangle;
            if (routed_wide_line) native_count = count / 3 * 3;
            break;
        case SCE_GXM_PRIMITIVE_TRIANGLE_STRIP: type = MTLPrimitiveTypeTriangleStrip; break;
        case SCE_GXM_PRIMITIVE_LINES: type = MTLPrimitiveTypeLine; native_count = count / 2 * 2; break;
        case SCE_GXM_PRIMITIVE_POINTS: type = MTLPrimitiveTypePoint; break;
        case SCE_GXM_PRIMITIVE_TRIANGLE_FAN: {
            // Metal has no fan topology. Keep the original anchor and winding,
            // including full-width guest indices, in an equivalent triangle list.
            type = MTLPrimitiveTypeTriangle;
            native_count = size_t(count - 2) * 3;
            converted_indices.reserve(native_count);
            if (capture_per_occurrence || capture_split_faces) converted_ordinals.reserve(native_count);
            const auto at = [&](uint32_t i) -> uint32_t {
                return index_size == 2 ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i];
            };
            const auto anchor = at(0);
            for (uint32_t i = 2; i < count; ++i) {
                converted_indices.push_back(anchor);
                converted_indices.push_back(at(i - 1));
                converted_indices.push_back(at(i));
                if (capture_per_occurrence || capture_split_faces) {
                    converted_ordinals.push_back(0);
                    converted_ordinals.push_back(i - 1);
                    converted_ordinals.push_back(i);
                }
            }
            indices = converted_indices.data();
            index_size = 4;
            break;
        }
        default: throw std::runtime_error("Metal: primitive conversion required");
        }
        if (split_face_draw && type == MTLPrimitiveTypeTriangle)
            native_count = (native_count / 3) * 3;
        if (!native_count) {
            empty_visibility_draw(preserve_active_set_query(true), preserve_active_set_query(false));
            return;
        }
        const auto index_buffer = ctx.impl->uploads.allocate(*impl->device, native_count * index_size);
        std::memcpy(static_cast<uint8_t *>(index_buffer.buffer.contents) + index_buffer.offset, indices, native_count * index_size);
        std::vector<RoutedPointPolygon> routed_polygons;
        std::vector<RoutedWideLine> routed_lines;
        std::vector<id<MTLBuffer>> split_captured_instances;
        id<MTLBuffer> split_replay_indices = nil;
        if (routed_geometry) {
            std::vector<uint32_t> guest_indices(native_count);
            for (size_t i = 0; i < native_count; ++i) {
                const uint32_t index = index_size == 2
                    ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i];
                guest_indices[i] = index;
            }
            const auto &capture_vertex_indices = capture_per_occurrence
                ? point_capture_source_indices : point_capture_unique_indices;
            const auto capture_indices = ctx.impl->uploads.allocate(*impl->device,
                capture_vertex_indices.size() * sizeof(uint32_t));
            std::memcpy(static_cast<uint8_t *>(capture_indices.buffer.contents) + capture_indices.offset,
                capture_vertex_indices.data(), capture_vertex_indices.size() * sizeof(uint32_t));
            constexpr size_t vertex_bytes = sizeof(CapturedVertexOutputs);
            const size_t captured_vertices = compact_point_capture ? capture_vertex_indices.size()
                : size_t(point_capture_max_index) + 1;
            require(captured_vertices <= impl->device->native_device().maxBufferLength / vertex_bytes,
                "Metal: point polygon capture exceeds maximum native buffer length");
            std::vector<id<MTLBuffer>> captured_instances;
            captured_instances.reserve(instances);
            DepthKey capture_depth{};
            capture_depth[0] = MTLCompareFunctionAlways;
            [encoder setRenderPipelineState:point_capture_pipeline];
            [encoder setDepthStencilState:native_depth_state(capture_depth)];
            [encoder setCullMode:MTLCullModeNone];
            [encoder setTriangleFillMode:MTLTriangleFillModeFill];
            [encoder setScissorRect:MTLScissorRect{0, 0, ctx.impl->width, ctx.impl->height}];
            const bool preserve_prior_front_set_query = preserve_active_set_query(true);
            const bool preserve_prior_back_set_query = preserve_active_set_query(false);
            [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
            ctx.impl->pass_visibility_active = false;
            for (uint32_t instance = 0; instance < instances; ++instance) {
                id<MTLBuffer> captured = [impl->device->native_device()
                    newBufferWithLength:captured_vertices * vertex_bytes options:MTLResourceStorageModeShared];
                require(captured != nil, "Metal: point polygon capture buffer allocation failed");
                captured_instances.push_back(captured);
                if (compact_point_capture) {
                    for (size_t ordinal = 0; ordinal < capture_vertex_indices.size(); ++ordinal) {
                        [encoder setVertexBuffer:captured offset:ordinal * vertex_bytes
                            atIndex:shader::metal::VERTEX_OUTPUT_CAPTURE_BUFFER];
                        [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:1
                            indexType:MTLIndexTypeUInt32 indexBuffer:capture_indices.buffer
                            indexBufferOffset:capture_indices.offset + ordinal * sizeof(uint32_t)
                            instanceCount:1 baseVertex:0 baseInstance:instance];
                    }
                } else {
                    [encoder setVertexBuffer:captured offset:0 atIndex:shader::metal::VERTEX_OUTPUT_CAPTURE_BUFFER];
                    [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:capture_vertex_indices.size()
                        indexType:MTLIndexTypeUInt32
                        indexBuffer:capture_indices.buffer indexBufferOffset:capture_indices.offset
                        instanceCount:1 baseVertex:0 baseInstance:instance];
                }
            }
            if (capture_per_occurrence) {
                // Converted corners retain the original source ordinal next
                // to the guest index in the conversion above. Unconverted
                // topologies already have one corner per source ordinal.
                if (!converted_ordinals.empty()) {
                    require(converted_ordinals.size() == native_count,
                        "Metal: point polygon occurrence remapping size mismatch");
                    guest_indices = std::move(converted_ordinals);
                } else {
                    for (uint32_t i = 0; i < native_count; ++i)
                        guest_indices[i] = i;
                }
            } else if (compact_point_capture) {
                for (uint32_t &index : guest_indices) {
                    const auto found = std::lower_bound(point_capture_unique_indices.begin(),
                        point_capture_unique_indices.end(), index);
                    require(found != point_capture_unique_indices.end() && *found == index,
                        "Metal: point polygon capture index remapping failed");
                    index = uint32_t(found - point_capture_unique_indices.begin());
                }
            }
            const bool replay_depth_written = ctx.impl->depth_written;
            // CPU clipping needs the completed vertex outputs. Keep the guest
            // render target, uniforms and sampled resources alive across the
            // pass boundary; begin_pass loads their existing contents.
            finish(ctx);
            for (id<MTLBuffer> captured : captured_instances) {
                const auto *captured_data = static_cast<const CapturedVertexOutputs *>(captured.contents);
                if (routed_point_polygon) {
                    auto polygons = route_point_polygons(PolygonTopology::List, guest_indices,
                        {captured_data, captured_vertices}, cull_front, cull_back);
                    routed_polygons.insert(routed_polygons.end(),
                        std::make_move_iterator(polygons.begin()), std::make_move_iterator(polygons.end()));
                } else {
                    const auto topology = type == MTLPrimitiveTypeTriangleStrip
                        ? PolygonTopology::Strip : PolygonTopology::List;
                    auto lines = route_wide_lines(triangles, topology, guest_indices,
                        {captured_data, captured_vertices}, cull_front, cull_back,
                        float(viewport.width), float(viewport.height),
                        float(std::max(1u, record.line_width)) * res_multiplier,
                        float(std::max(1u, record.back_line_width)) * res_multiplier);
                    routed_lines.insert(routed_lines.end(),
                        std::make_move_iterator(lines.begin()), std::make_move_iterator(lines.end()));
                }
            }
            if (routed_polygons.empty() && routed_lines.empty()) {
                empty_visibility_draw(preserve_prior_front_set_query, preserve_prior_back_set_query);
                return;
            }
            ctx.impl->commands = scene_command_buffer(*impl->device);
            begin_pass(ctx, record.is_maskupdate);
            encoder = ctx.impl->encoder;
            ctx.impl->depth_written |= replay_depth_written;
            [encoder setRenderPipelineState:point_replay_pipeline];
            [encoder setDepthStencilState:front_depth_state];
            [encoder setStencilFrontReferenceValue:record.front_stencil_state_values.ref
                backReferenceValue:record.front_stencil_state_values.ref];
            [encoder setDepthBias:back_depth ? record.back_depth_bias_unit : record.depth_bias_unit
                slopeScale:back_depth ? back_depth_slope : front_depth_slope clamp:0];
            [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [encoder setCullMode:MTLCullModeNone];
            [encoder setTriangleFillMode:MTLTriangleFillModeFill];
            [encoder setViewport:viewport];
            native_buffers.make_resident(encoder);
            [encoder setFragmentTexture:record.is_maskupdate ? nil : ctx.impl->mask
                atIndex:shader::metal::MASK_TEXTURE];
            for (uint32_t slot = 0; slot < SCE_GXM_MAX_TEXTURE_UNITS; ++slot) {
                if (replay_fragment_textures[slot])
                    [encoder setFragmentTexture:replay_fragment_textures[slot] atIndex:slot];
                if (replay_fragment_samplers[slot])
                    [encoder setFragmentSamplerState:replay_fragment_samplers[slot] atIndex:slot];
            }
            [encoder setFragmentBytes:texture_mip_info.data() length:sizeof(shader::metal::TextureMipInfos)
                atIndex:shader::metal::TEXTURE_INFO_BUFFER];
        } else if (capture_split_faces) {
            require(type == MTLPrimitiveTypeTriangle && !split_capture_source_indices.empty(),
                "Metal: face-routed capture requires complete triangles");
            const auto capture_indices = ctx.impl->uploads.allocate(*impl->device,
                split_capture_source_indices.size() * sizeof(uint32_t));
            std::memcpy(static_cast<uint8_t *>(capture_indices.buffer.contents) + capture_indices.offset,
                split_capture_source_indices.data(), split_capture_source_indices.size() * sizeof(uint32_t));
            constexpr size_t vertex_bytes = sizeof(CapturedVertexOutputs);
            require(split_capture_source_indices.size() <= impl->device->native_device().maxBufferLength / vertex_bytes,
                "Metal: face-routed capture exceeds maximum native buffer length");
            split_captured_instances.reserve(instances);
            DepthKey capture_depth{};
            capture_depth[0] = MTLCompareFunctionAlways;
            [encoder setRenderPipelineState:point_capture_pipeline];
            [encoder setDepthStencilState:native_depth_state(capture_depth)];
            [encoder setCullMode:MTLCullModeNone];
            [encoder setTriangleFillMode:MTLTriangleFillModeFill];
            [encoder setScissorRect:MTLScissorRect{0, 0, ctx.impl->width, ctx.impl->height}];
            [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
            ctx.impl->pass_visibility_active = false;
            for (uint32_t instance = 0; instance < instances; ++instance) {
                id<MTLBuffer> captured = [impl->device->native_device()
                    newBufferWithLength:split_capture_source_indices.size() * vertex_bytes
                    options:MTLResourceStorageModeShared];
                require(captured != nil, "Metal: face-routed capture buffer allocation failed");
                split_captured_instances.push_back(captured);
                for (size_t ordinal = 0; ordinal < split_capture_source_indices.size(); ++ordinal) {
                    [encoder setVertexBuffer:captured offset:ordinal * vertex_bytes
                        atIndex:shader::metal::VERTEX_OUTPUT_CAPTURE_BUFFER];
                    [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:1
                        indexType:MTLIndexTypeUInt32 indexBuffer:capture_indices.buffer
                        indexBufferOffset:capture_indices.offset + ordinal * sizeof(uint32_t)
                        instanceCount:1 baseVertex:0 baseInstance:instance];
                }
            }
            if (converted_ordinals.empty()) {
                converted_ordinals.resize(native_count);
                std::iota(converted_ordinals.begin(), converted_ordinals.end(), 0);
            }
            require(converted_ordinals.size() == native_count,
                "Metal: face-routed occurrence remapping size mismatch");
            const bool replay_depth_written = ctx.impl->depth_written;
            // A pass boundary makes all capture writes visible to the replay.
            finish(ctx);
            split_replay_indices = [impl->device->native_device()
                newBufferWithBytes:converted_ordinals.data() length:converted_ordinals.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
            require(split_replay_indices != nil, "Metal: face-routed replay index allocation failed");
            ctx.impl->commands = scene_command_buffer(*impl->device);
            begin_pass(ctx, record.is_maskupdate);
            encoder = ctx.impl->encoder;
            ctx.impl->depth_written |= replay_depth_written;
            [encoder setRenderPipelineState:point_replay_pipeline];
            [encoder setDepthStencilState:front_depth_state];
            [encoder setStencilFrontReferenceValue:record.front_stencil_state_values.ref
                backReferenceValue:record.back_stencil_state_values.ref];
            [encoder setDepthBias:back_depth ? record.back_depth_bias_unit : record.depth_bias_unit
                slopeScale:back_depth ? back_depth_slope : front_depth_slope clamp:0];
            [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
            [encoder setTriangleFillMode:wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
            [encoder setViewport:viewport];
            native_buffers.make_resident(encoder);
            [encoder setFragmentTexture:record.is_maskupdate ? nil : ctx.impl->mask
                atIndex:shader::metal::MASK_TEXTURE];
            for (uint32_t slot = 0; slot < SCE_GXM_MAX_TEXTURE_UNITS; ++slot) {
                if (replay_fragment_textures[slot])
                    [encoder setFragmentTexture:replay_fragment_textures[slot] atIndex:slot];
                if (replay_fragment_samplers[slot])
                    [encoder setFragmentSamplerState:replay_fragment_samplers[slot] atIndex:slot];
            }
            [encoder setFragmentBytes:fragment_bytes.data() length:fragment_bytes.size()
                atIndex:shader::metal::RENDER_INFO_BUFFER];
            [encoder setFragmentBytes:texture_mip_info.data() length:sizeof(shader::metal::TextureMipInfos)
                atIndex:shader::metal::TEXTURE_INFO_BUFFER];
        }
        if (capture_draw) {
            dump_bytes("indices.bin",indices,native_count*index_size);
            draw_metadata << "draw " << type << ' ' << index_size << ' ' << native_count << ' ' << instances << '\n';
            require(bool(draw_metadata) && bool(texture_metadata),"Metal: cannot write draw metadata");
            impl->draw_dumped = ++impl->dump_draw_saved >= impl->dump_draw_limit;
            LOG_INFO("Metal vertex draw saved: {}",draw_capture_dir.string());
        }
        const uint32_t visibility_entries = ctx.impl->visibility_stride / sizeof(uint32_t);
        const uint64_t visibility_group = ctx.impl->visibility_address && visibility_faces_differ
            ? ctx.impl->visibility_epoch : 0;
        if (visibility_group && ctx.impl->pass_visibility_active) {
            [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
            ctx.impl->pass_visibility_active = false;
        }
        const auto select_visibility = [&](bool front) {
            const bool enabled = front ? ctx.impl->visibility_enabled : ctx.impl->back_visibility_enabled;
            const uint32_t index = front ? ctx.impl->visibility_index : ctx.impl->back_visibility_index;
            const bool increment = front ? ctx.impl->visibility_increment : ctx.impl->back_visibility_increment;
            if (enabled && ctx.impl->pass_visibility_buffer && visibility_entries) {
                uint32_t query_index = index;
                if (query_index >= visibility_entries) {
                    LOG_WARN_ONCE("Metal: visibility index {} exceeds buffer entry count {}", query_index, visibility_entries);
                    query_index = 0;
                }
                if (!ctx.impl->pass_visibility_active || ctx.impl->pass_visibility_index != query_index
                    || ctx.impl->pass_visibility_increment != increment) {
                    require(ctx.impl->pass_visibility_offset + sizeof(uint64_t) <= ctx.impl->pass_visibility_buffer.length,
                        "Metal: visibility query capacity exhausted before batch submit");
                    const uint32_t offset = ctx.impl->pass_visibility_offset;
                    ctx.impl->pass_visibility_offset += sizeof(uint64_t);
                    [encoder setVisibilityResultMode:increment
                        ? MTLVisibilityResultModeCounting : MTLVisibilityResultModeBoolean offset:offset];
                    ctx.impl->visibility_results.push_back({ctx.impl->pass_visibility_buffer, offset,
                        ctx.impl->visibility_address, query_index, increment, visibility_group});
                    ctx.impl->pass_visibility_active = true;
                    ctx.impl->pass_visibility_index = query_index;
                    ctx.impl->pass_visibility_increment = increment;
                }
            } else if (ctx.impl->pass_visibility_active) {
                [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
                ctx.impl->pass_visibility_active = false;
            }
        };
        if (!routed_geometry && !split_face_draw)
            select_visibility(!cull_front);
        for (const auto &rect : clip) {
            [encoder setScissorRect:rect];
            const auto metal_index_type = index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
            if (routed_point_polygon) {
                for (auto &polygon : routed_polygons) {
                    const bool front = polygon.face == PolygonFace::Front;
                    select_visibility(front);
                    [encoder setDepthBias:(front || !two_sided) ? record.depth_bias_unit : record.back_depth_bias_unit
                        slopeScale:(front || !two_sided) ? front_depth_slope : back_depth_slope clamp:0];
                    // The capture shader supplies the front GXM width only when
                    // the guest vertex shader did not write its own point size.
                    if (two_sided && !front && !shader_point_size) {
                        const float back_size = float(std::max(1u, record.back_line_width)) * res_multiplier;
                        for (auto &point : polygon.points) point[14][0] = back_size;
                    }
                    fragment_info.base_block.point_replay_face = front ? 1.0f : -1.0f;
                    fragment_info.copy_to(fragment_bytes.data());
                    [encoder setFragmentBytes:fragment_bytes.data() length:fragment_bytes.size()
                        atIndex:shader::metal::RENDER_INFO_BUFFER];
                    [encoder setDepthStencilState:front ? point_front_depth_state : point_back_depth_state];
                    const uint8_t stencil_ref = (front || !two_sided) ? record.front_stencil_state_values.ref
                        : record.back_stencil_state_values.ref;
                    [encoder setStencilFrontReferenceValue:stencil_ref backReferenceValue:stencil_ref];
                    const size_t bytes = polygon.points.size() * sizeof(CapturedVertexOutputs);
                    if (bytes <= 4096) {
                        [encoder setVertexBytes:polygon.points.data() length:bytes
                            atIndex:shader::metal::VERTEX_STREAM_BUFFER_BASE];
                    } else {
                        id<MTLBuffer> points = [impl->device->native_device() newBufferWithBytes:polygon.points.data()
                            length:bytes options:MTLResourceStorageModeShared];
                        require(points != nil, "Metal: point polygon replay buffer allocation failed");
                        [encoder setVertexBuffer:points offset:0 atIndex:shader::metal::VERTEX_STREAM_BUFFER_BASE];
                    }
                    [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:polygon.points.size()];
                }
            } else if (routed_wide_line) {
                for (const auto &line : routed_lines) {
                    const bool front = line.face != PolygonFace::Back;
                    select_visibility(front);
                    [encoder setDepthBias:(front || !two_sided) ? record.depth_bias_unit : record.back_depth_bias_unit
                        slopeScale:(front || !two_sided) ? front_depth_slope : back_depth_slope clamp:0];
                    fragment_info.base_block.point_replay_face = front ? 1.0f : -1.0f;
                    fragment_info.copy_to(fragment_bytes.data());
                    [encoder setFragmentBytes:fragment_bytes.data() length:fragment_bytes.size()
                        atIndex:shader::metal::RENDER_INFO_BUFFER];
                    [encoder setDepthStencilState:front ? point_front_depth_state : point_back_depth_state];
                    const uint8_t stencil_ref = (front || !two_sided) ? record.front_stencil_state_values.ref
                        : record.back_stencil_state_values.ref;
                    [encoder setStencilFrontReferenceValue:stencil_ref backReferenceValue:stencil_ref];
                    [encoder setVertexBytes:line.quad.data() length:sizeof(line.quad)
                        atIndex:shader::metal::VERTEX_STREAM_BUFFER_BASE];
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
                }
            } else if (split_face_draw) {
                // Keep guest instance and primitive order: depth and stencil
                // writes from an earlier triangle affect every later triangle.
                for (uint32_t instance = 0; instance < instances; ++instance) {
                    if (capture_split_faces)
                        [encoder setVertexBuffer:split_captured_instances[instance] offset:0
                            atIndex:shader::metal::VERTEX_STREAM_BUFFER_BASE];
                    for (size_t first = 0; first < native_count; first += 3) {
                        const auto offset = capture_split_faces ? first * sizeof(uint32_t)
                            : index_buffer.offset + first * index_size;
                        id<MTLBuffer> draw_indices = capture_split_faces ? split_replay_indices : index_buffer.buffer;
                        const auto draw_index_type = capture_split_faces ? MTLIndexTypeUInt32 : metal_index_type;
                        select_visibility(true);
                        [encoder setDepthStencilState:front_depth_state];
                        [encoder setDepthBias:record.depth_bias_unit slopeScale:front_depth_slope clamp:0];
                        [encoder setCullMode:MTLCullModeBack];
                        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:3 indexType:draw_index_type
                            indexBuffer:draw_indices indexBufferOffset:offset instanceCount:1 baseVertex:0
                            baseInstance:capture_split_faces ? 0 : instance];
                        select_visibility(false);
                        [encoder setDepthStencilState:back_depth_state];
                        [encoder setDepthBias:record.back_depth_bias_unit slopeScale:back_depth_slope clamp:0];
                        [encoder setCullMode:MTLCullModeFront];
                        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:3 indexType:draw_index_type
                            indexBuffer:draw_indices indexBufferOffset:offset instanceCount:1 baseVertex:0
                            baseInstance:capture_split_faces ? 0 : instance];
                    }
                }
            } else {
                [encoder drawIndexedPrimitives:type indexCount:native_count indexType:metal_index_type indexBuffer:index_buffer.buffer indexBufferOffset:index_buffer.offset instanceCount:instances];
            }
        }
        // Read-only shaders own snapshots retained by the command buffer. Guest
        // memory writers still finish before CPU processing can observe/reuse it.
        // Limit retained resources even when a guest emits an unusually long scene.
        ++ctx.impl->pending_draws;
        // Plus retains the union of logical render-pass areas across internal
        // submissions. Native Metal opens a full attachment pass eagerly for
        // depth loads, so record the macroblock area only after a guest draw
        // was encoded, when its tile selection is known. Mask/depth-only draws
        // also open a logical pass; actual color ownership is intersected later.
        if (auto *target = static_cast<MetalRenderTarget *>(ctx.current_render_target);
            target && target->has_macroblock_sync && res_multiplier != 1.f && ctx.impl->guest_color.data) {
            const auto found = impl->surfaces.find(ctx.impl->guest_color.data.address());
            if (found != impl->surfaces.end() && found->second.color == ctx.impl->color) {
                uint32_t left = 0, top = 0;
                uint32_t right = std::min(target->width, ctx.impl->width);
                uint32_t bottom = std::min(target->height, ctx.impl->height);
                const bool known_tile = ctx.impl->macroblock_last_x != uint16_t(~0u)
                    && ctx.impl->macroblock_last_y != uint16_t(~0u);
                if (!ctx.impl->macroblock_ignore && known_tile) {
                    left = uint32_t(ctx.impl->macroblock_last_x) * target->macroblock_width;
                    top = uint32_t(ctx.impl->macroblock_last_y) * target->macroblock_height;
                    right = std::min<uint32_t>(left + target->macroblock_width, ctx.impl->width);
                    bottom = std::min<uint32_t>(top + target->macroblock_height, ctx.impl->height);
                }
                // Before a first nonempty tile selection there is no guest
                // macroblock area. An empty-scissor draw must not mark it full.
                if (ctx.impl->macroblock_ignore || known_tile) {
                    auto &surface = found->second;
                    surface.scene_render_area.include(left, top, right, bottom);
                    const auto &area = surface.scene_render_area;
                    if (!area.empty()) {
                        const double sx = (ctx.impl->expanded_color ? ctx.impl->samples / 2 : 1) / double(res_multiplier);
                        const double sy = (ctx.impl->expanded_color ? 2 : 1) / double(res_multiplier);
                        // Plus truncates both edges of positive rendered areas;
                        // draw ownership instead rounds its upper edge outward.
                        const auto edge = [](uint32_t value, double scale, uint32_t limit) {
                            return uint32_t(std::min(double(limit), std::floor(value * scale)));
                        };
                        surface.scene_macroblock_bounds = {
                            edge(area.x0, sx, surface.guest.width), edge(area.y0, sy, surface.guest.height),
                            edge(area.x1, sx, surface.guest.width), edge(area.y1, sy, surface.guest.height)};
                    }
                }
            }
        }
        // Track only submitted, potentially depth-writing guest draws for
        // Plus' no-store validity rule. Imports, clears, stencil and mask writes
        // still use depth_scene_written for publication/snapshot ordering.
        ctx.impl->scene_depth_drawn |= depth_enabled
            && ((record.front_depth_write_mode==SCE_GXM_DEPTH_WRITE_ENABLED && (!two_sided || !cull_front))
                || (two_sided && !cull_back && record.back_depth_write_mode==SCE_GXM_DEPTH_WRITE_ENABLED))
            && std::any_of(clip.begin(),clip.end(),[](const MTLScissorRect &rect) { return rect.width && rect.height; });
        if (record.is_maskupdate) ctx.impl->mask_constant_valid=false;
        ctx.impl->depth_scene_written |= record.is_maskupdate;
        const auto writable_color = alpha_target ? SCE_GXM_COLOR_MASK_A
            : green_target ? SCE_GXM_COLOR_MASK_G
            : red_alpha_target ? (SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_A)
            : rgb_target ? (SCE_GXM_COLOR_MASK_R | SCE_GXM_COLOR_MASK_G | SCE_GXM_COLOR_MASK_B)
            : SCE_GXM_COLOR_MASK_ALL;
        const bool wrote_color = !record.is_maskupdate && !fragment_disabled
            && (blend.colorMask & writable_color);
        ctx.impl->pending_color_writes |= wrote_color;
        if (wrote_color && ctx.impl->guest_color.data) {
            if (const auto found = impl->surfaces.find(ctx.impl->guest_color.data.address()); found != impl->surfaces.end()) {
                auto &surface = found->second;
                if (blend.colorFunc!=SCE_GXM_BLEND_FUNC_NONE || blend.alphaFunc!=SCE_GXM_BLEND_FUNC_NONE)
                    surface.raw_color_invalidated=true;
                // Resolved expanded MSAA stores samples in a 1x2/2x2 grid.
                const double scale_x = (ctx.impl->expanded_color ? ctx.impl->samples / 2 : 1) / double(res_multiplier);
                const double scale_y = (ctx.impl->expanded_color ? 2 : 1) / double(res_multiplier);
                const auto edge = [](double value, uint32_t limit, bool upper) {
                    return uint32_t(std::clamp(upper ? std::ceil(value) : std::floor(value), 0.0, double(limit)));
                };
                for (const auto &rect : clip) {
                    const double left = std::max(double(rect.x), std::floor(std::min(viewport.originX, viewport.originX + viewport.width)));
                    const double top = std::max(double(rect.y), std::floor(std::min(viewport.originY, viewport.originY + viewport.height)));
                    const double right = std::min(double(rect.x + rect.width), std::ceil(std::max(viewport.originX, viewport.originX + viewport.width)));
                    const double bottom = std::min(double(rect.y + rect.height), std::ceil(std::max(viewport.originY, viewport.originY + viewport.height)));
                    if (right <= left || bottom <= top) continue;
                    note_half_pixel_origin(ctx, viewport, rect, uint32_t(left), uint32_t(top), uint32_t(right), uint32_t(bottom));
                    const auto x0 = edge(left * scale_x, surface.guest.width, false);
                    const auto y0 = edge(top * scale_y, surface.guest.height, false);
                    const auto x1 = edge(right * scale_x, surface.guest.width, true);
                    const auto y1 = edge(bottom * scale_y, surface.guest.height, true);
                    surface.scene_writes.include(x0, y0, x1, y1);
                    surface.written_tiles.include(x0 & ~31u, y0 & ~31u,
                        std::min<uint32_t>((x1 + 31) & ~31u, surface.guest.width),
                        std::min<uint32_t>((y1 + 31) & ~31u, surface.guest.height));
                }
            }
        }
        ctx.impl->color_clip_restore_pending |= wrote_color && bool(ctx.impl->color_clip_snapshot);
        const size_t index_upload_bytes=native_count*index_size;
        ctx.impl->pending_upload_bytes+=draw_upload_bytes+index_upload_bytes;
        if (impl->trace_batches) {
            ctx.impl->pending_uniform_bytes+=uniform_upload_bytes;
            ctx.impl->pending_stream_bytes+=draw_upload_bytes-uniform_upload_bytes;
            ctx.impl->pending_index_bytes+=index_upload_bytes;
        }
        if (!record.is_maskupdate && ctx.impl->guest_color.data) {
            auto surface=impl->surfaces.find(ctx.impl->guest_color.data.address());
            if (surface!=impl->surfaces.end()) { ++surface->second.revision; surface->second.rgba8_casts.clear(); surface->second.word_casts.clear(); surface->second.word_rect_casts.clear(); surface->second.halfword_rect_casts.clear(); surface->second.byte_rect_casts.clear(); surface->second.small_rect_casts.clear(); surface->second.subrectangles.clear(); }
        }
        // Keep small draws together to avoid a GPU completion wait every 64
        // draws; the upload-byte cap still bounds retained resources.
        if(synchronous || ctx.impl->pending_draws>=256 || ctx.impl->pending_upload_bytes>=32*1024*1024) {
            if (impl->trace_batches
                && (impl->trace_batches_trigger.empty() || std::filesystem::exists(impl->trace_batches_trigger))
                && impl->traced_draw_flushes++ < 96)
                LOG_INFO("Metal draw flush: synchronous={} vertex_store={} fragment_store={} draws={} input_bytes={}",
                    synchronous, vs.writes_guest_memory, fs.writes_guest_memory,
                    ctx.impl->pending_draws, ctx.impl->pending_upload_bytes);
            finish(ctx, synchronous, shader_guest_stores);
        }
        if (capture_draw && impl->dump_attachments) {
            finish(ctx);
            dump_attachment("color-after",ctx.impl->color);
            dump_raw_attachment("raw-color-after");
            dump_attachment("mask-after",ctx.impl->mask);
            dump_depth_attachment("depth-after");
            dump_stencil_attachment("stencil-after");
        }
        if (!impl->trace_pixel.empty() && impl->trace_pixel_armed && impl->traced_pixel_draws < 1000
            && (!impl->trace_pixel_target_address
                || ctx.impl->guest_color.data.address() == impl->trace_pixel_target_address)
            && (!impl->trace_pixel_target_width || ctx.impl->width == impl->trace_pixel_target_width)
            && (impl->trace_pixel_trigger.empty() || std::filesystem::exists(impl->trace_pixel_trigger))) {
            unsigned x = 0, y = 0;
            if (std::sscanf(impl->trace_pixel.c_str(), "%u,%u", &x, &y) == 2
                && ctx.impl->color && x < ctx.impl->color.width && y < ctx.impl->color.height) {
                finish(ctx);
                if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                auto snapshot = impl->caster->sampling_snapshot(ctx.impl->color, 0, 0);
                if (snapshot) {
                    float pixel[4]{};
                    [snapshot getBytes:pixel bytesPerRow:sizeof(pixel)
                        fromRegion:MTLRegionMake2D(x, y, 1, 1) mipmapLevel:0];
                    bool changed = !impl->has_traced_pixel;
                    for (unsigned channel = 0; channel < 4 && !changed; ++channel)
                        changed = std::abs(pixel[channel] - impl->last_traced_pixel[channel]) > 1.0f / 255.0f;
                    if (changed) {
                        LOG_INFO("Metal pixel trace draw={} target={:x} fs={} xy={},{} rgba={},{},{},{}",
                            impl->traced_pixel_draws, ctx.impl->guest_color.data.address(),
                            hex_string(fp->fragment_program->hash), x, y, pixel[0], pixel[1], pixel[2], pixel[3]);
                        if (!impl->dump_arm_pixel_shader.empty()
                            && impl->dump_arm_pixel_shader == hex_string(fp->fragment_program->hash)
                            && std::max({pixel[0], pixel[1], pixel[2]}) > impl->dump_arm_pixel_min_rgb)
                            impl->dump_pixel_armed = true;
                        if (!impl->dump_pixel_armed && !impl->dump_arm_pixel_rgb.empty()) {
                            float target_r = 0, target_g = 0, target_b = 0, tolerance = 0;
                            if (std::sscanf(impl->dump_arm_pixel_rgb.c_str(), "%f,%f,%f,%f",
                                    &target_r, &target_g, &target_b, &tolerance) == 4
                                && tolerance >= 0 && std::abs(pixel[0] - target_r) <= tolerance
                                && std::abs(pixel[1] - target_g) <= tolerance
                                && std::abs(pixel[2] - target_b) <= tolerance) {
                                impl->dump_draw_shader = hex_string(fp->fragment_program->hash);
                                impl->dump_pixel_armed = true;
                                LOG_INFO("Metal pixel-color draw capture armed: fs={} xy={},{}",
                                    impl->dump_draw_shader, x, y);
                            }
                        }
                        std::copy(std::begin(pixel), std::end(pixel), impl->last_traced_pixel);
                        impl->has_traced_pixel = true;
                    }
                }
                ++impl->traced_pixel_draws;
            }
        }
    }
}

std::vector<uint32_t> MetalState::dump_frame(DisplayState &display, uint32_t &width, uint32_t &height) {
    // Standalone renderer clients (including GPU validation) have no worker
    // thread, so their caller owns the context and can read it directly.
    if (!render_thread || !render_thread->joinable()) {
        if (context) finish(*static_cast<MetalContext *>(context));
        return read_display_frame(display, width, height);
    }
    // GXM commands and surface caches belong to the render thread. A UI hotkey
    // must not finish its in-flight command buffer or inspect that cache here.
    std::unique_lock lock(impl->screenshot_mutex);
    impl->screenshot_requested = true;
    impl->screenshot_frame.clear();
    if (!impl->screenshot_ready.wait_for(lock, std::chrono::seconds(3), [&] {
            return !impl->screenshot_requested;
        })) {
        impl->screenshot_requested = false;
        width = height = 0;
        return {};
    }
    width = impl->screenshot_width;
    height = impl->screenshot_height;
    return std::move(impl->screenshot_frame);
}
std::vector<uint32_t> MetalState::read_display_frame(DisplayState &display, uint32_t &width, uint32_t &height) {
    DisplayFrameInfo next;
    { std::lock_guard lock(display.display_info_mutex); next = display.next_rendered_frame; }
    const auto region = find_display_surface_region(impl->surfaces, next, res_multiplier, true);
    if (!region || (region->color.pixelFormat != MTLPixelFormatRGBA8Unorm
        && region->color.pixelFormat != MTLPixelFormatRGBA8Unorm_sRGB
        && region->color.pixelFormat != MTLPixelFormatRGBA16Float)) {
        width = height = 0; return {};
    }
    id<MTLTexture> color = region->color;
    width = region->width;
    height = region->height;
    const uint32_t line = region->line;
    const uint32_t rows = region->available_rows;
    std::vector<uint32_t> result(size_t(width) * height, 0xff000000u);
    auto *mem = context ? static_cast<MetalContext *>(context)->impl->mem : nullptr;
    for (const auto [begin, count] : {std::pair{0u, region->destination_line},
             std::pair{region->destination_line + rows, height - region->destination_line - rows}}) {
        const auto border = display_border_pixels(next, *region, mem, begin, count);
        std::copy(border.begin(), border.end(), result.begin() + size_t(begin) * width);
    }
    auto *destination = result.data() + size_t(region->destination_line) * width;
    if (color.pixelFormat == MTLPixelFormatRGBA16Float) {
        std::vector<__fp16> pixels(size_t(width) * rows * 4);
        [color getBytes:pixels.data() bytesPerRow:width * 8 fromRegion:MTLRegionMake2D(0, line, width, rows) mipmapLevel:0];
        auto *bytes = reinterpret_cast<uint8_t *>(destination);
        for (size_t i = 0; i < pixels.size(); ++i) {
            const float value = pixels[i];
            bytes[i] = std::isnan(value) ? 0 : uint8_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
        }
    } else {
        [color getBytes:destination bytesPerRow:width * 4 fromRegion:MTLRegionMake2D(0, line, width, rows) mipmapLevel:0];
    }
    return result;
}
void MetalState::render_frame(DisplayState &display, const GxmState &, MemState &mem) {
    @autoreleasepool {
        should_display = false;
        if (context) finish(*static_cast<MetalContext *>(context));
        bool capture_frame = false;
        {
            std::lock_guard lock(impl->screenshot_mutex);
            capture_frame = impl->screenshot_requested;
        }
        if (capture_frame) {
            uint32_t captured_width = 0, captured_height = 0;
            auto captured = read_display_frame(display, captured_width, captured_height);
            {
                std::lock_guard lock(impl->screenshot_mutex);
                if (impl->screenshot_requested) {
                    impl->screenshot_frame = std::move(captured);
                    impl->screenshot_width = captured_width;
                    impl->screenshot_height = captured_height;
                    impl->screenshot_requested = false;
                }
            }
            impl->screenshot_ready.notify_all();
        }
        if (!impl->layer || !frame) return;
        update_overlays();
        bool has_overlays = false;
        if (overlay_manager) {
            std::shared_lock lock(*overlay_manager);
            has_overlays = overlay_manager->has_visible();
        }
        const uint32_t width = frame->drawable_width(), height = frame->drawable_height();
        if (!width || !height) return;
        impl->layer.drawableSize = CGSizeMake(width, height);
        const int vsync = pending_vsync.exchange(-1);
        if (vsync >= 0) impl->layer.displaySyncEnabled = vsync != 0;
        DisplayFrameInfo next;
        { std::lock_guard lock(display.display_info_mutex); next = display.next_rendered_frame; }
        const bool has_frame = next.base && next.image_size.x > 0 && next.image_size.y > 0;
        if (!has_frame && !has_overlays) return;
        impl->screen_commands = [impl->device->command_queue() commandBuffer];
        impl->screen_commands.label = @"Vita3K screen";
        id<MTLTexture> source = nil;
        const auto region = has_frame
            ? find_display_surface_region(impl->surfaces, next, res_multiplier, false) : std::nullopt;
        if (impl->trace_display_pixel && region && ++impl->traced_display_pixels % 30 == 0
            && impl->traced_display_pixels <= 900
            && (region->color.pixelFormat == MTLPixelFormatRGBA8Unorm
                || region->color.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB)) {
            const uint32_t x = region->width / 2;
            const uint32_t y = region->line + region->available_rows / 2;
            uint8_t rgba[4]{};
            [region->color getBytes:rgba bytesPerRow:4
                fromRegion:MTLRegionMake2D(x, y, 1, 1) mipmapLevel:0];
            LOG_INFO("Metal display pixel: base={:#x} region={}x{} line={} row={} source={}x{} center={},{} rgba={},{},{},{}",
                next.base.address(), region->width, region->height, region->line, region->available_rows,
                uint32_t(region->color.width), uint32_t(region->color.height), x, y,
                rgba[0], rgba[1], rgba[2], rgba[3]);
            unsigned comparable = 0;
            for (const auto &[address, surface] : impl->surfaces) {
                if (!surface.color || surface.color.width != region->color.width
                    || surface.color.height != region->color.height
                    || surface.color.pixelFormat != region->color.pixelFormat) continue;
                if (comparable++ >= 8) break;
                const uint32_t probe_x = std::min<uint32_t>(region->width / 8, uint32_t(surface.color.width) - 1);
                const uint32_t probe_y = std::min<uint32_t>(region->height / 2, uint32_t(surface.color.height) - 1);
                uint8_t probe[4]{};
                [surface.color getBytes:probe bytesPerRow:4
                    fromRegion:MTLRegionMake2D(probe_x, probe_y, 1, 1) mipmapLevel:0];
                LOG_INFO("Metal display candidate pixel: selected={:#x} cached={:#x} revision={} xy={},{} rgba={},{},{},{}",
                    next.base.address(), address, surface.revision, probe_x, probe_y,
                    probe[0], probe[1], probe[2], probe[3]);
            }
        }
        if (impl->trace_batches && impl->traced_display_frames++ % 60 == 0 && impl->traced_display_frames < 600) {
            Address cached_address = 0;
            if (region) for (const auto &[address, surface] : impl->surfaces)
                if (surface.color == region->color) { cached_address = address; break; }
            LOG_INFO("Metal display selection: base={:#x} pitch={} size={}x{} cached={:#x}",
                next.base.address(), next.pitch, next.image_size.x, next.image_size.y, cached_address);
        }
        if (region) {
            source = region->color;
            if (region->destination_line || region->available_rows != region->height) {
                auto padded = make_texture(*impl->device, source.pixelFormat, region->width, region->height,
                    MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView);
                if (source.pixelFormat == MTLPixelFormatRGBA8Unorm || source.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB
                    || source.pixelFormat == MTLPixelFormatRGBA16Float) {
                    for (const auto [begin, count] : {std::pair{0u, region->destination_line},
                             std::pair{region->destination_line + region->available_rows,
                                 region->height - region->destination_line - region->available_rows}}) {
                        if (!count) continue;
                        const auto border = display_border_pixels(next, *region, &mem, begin, count);
                        if (source.pixelFormat == MTLPixelFormatRGBA16Float) {
                            std::vector<__fp16> half(border.size() * 4);
                            const auto *bytes = reinterpret_cast<const uint8_t *>(border.data());
                            for (size_t i = 0; i < half.size(); ++i) half[i] = float(bytes[i]) / 255.f;
                            [padded replaceRegion:MTLRegionMake2D(0, begin, region->width, count) mipmapLevel:0
                                withBytes:half.data() bytesPerRow:region->width * 8];
                        } else {
                            [padded replaceRegion:MTLRegionMake2D(0, begin, region->width, count) mipmapLevel:0
                                withBytes:border.data() bytesPerRow:region->width * 4];
                        }
                    }
                } else {
                    auto clear = [MTLRenderPassDescriptor renderPassDescriptor];
                    clear.colorAttachments[0].texture = padded;
                    clear.colorAttachments[0].loadAction = MTLLoadActionClear;
                    clear.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
                    clear.colorAttachments[0].storeAction = MTLStoreActionStore;
                    [[impl->screen_commands renderCommandEncoderWithDescriptor:clear] endEncoding];
                }
                auto blit = [impl->screen_commands blitCommandEncoder];
                [blit copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, region->line, 0)
                    sourceSize:MTLSizeMake(region->width, region->available_rows, 1) toTexture:padded destinationSlice:0
                    destinationLevel:0 destinationOrigin:MTLOriginMake(0, region->destination_line, 0)];
                [blit endEncoding];
                source = padded;
            } else if (region->line || region->width != source.width || region->available_rows != source.height) {
                if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                source = impl->caster->enqueue_subrectangle(source, uint32_t(source.width), uint32_t(source.height),
                    SurfaceRect{0, region->line, region->width, region->available_rows}, impl->screen_commands);
            }
        } else if (has_frame) {
            if (impl->trace_batches && impl->traced_display_fallbacks++ < 16) {
                LOG_INFO("Metal display fallback: base={:#x} pitch={} size={}x{} surfaces={}",
                    next.base.address(), next.pitch, next.image_size.x, next.image_size.y, impl->surfaces.size());
                unsigned reported = 0;
                for (const auto &[address, surface] : impl->surfaces) {
                    if (reported++ == 16) break;
                    LOG_INFO("Metal display candidate: base={:#x} stride={} size={}x{} format={:#x} native={}x{}",
                        address, surface.guest.strideInPixels, surface.guest.width, surface.guest.height,
                        uint32_t(surface.guest.colorFormat), uint32_t(surface.color.width), uint32_t(surface.color.height));
                }
            }
            source = make_texture(*impl->device, MTLPixelFormatRGBA8Unorm, next.image_size.x, next.image_size.y, MTLTextureUsageShaderRead);
            [source replaceRegion:MTLRegionMake2D(0, 0, next.image_size.x, next.image_size.y) mipmapLevel:0 withBytes:next.base.get(mem) bytesPerRow:next.pitch * 4];
        }
        // The UNORM drawable consumes already-encoded framebuffer bytes.
        if (source && source.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB) source = rgba8_gamma_view(source,false);
        if (!impl->screen) set_screen_filter("Bilinear");
        impl->drawable = [impl->layer nextDrawable];
        if (!impl->drawable) { impl->screen_commands = nil; return; }
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = impl->drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,1);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;

        double vw = width, vh = height;
        const bool pixel_perfect = fullscreen_hd_res_pixel_perfect && fullscreen
            && !(width % DEFAULT_RES_WIDTH) && !(height % (DEFAULT_RES_HEIGHT - 4));
        if (pixel_perfect) {
            // Match Vulkan's fullscreen layout, including its four-line crop
            // on common 1080p displays.
            vh = vw * double(DEFAULT_RES_HEIGHT) / DEFAULT_RES_WIDTH;
        } else if (!stretch_the_display_area) {
            const double ratio = double(DEFAULT_RES_WIDTH) / DEFAULT_RES_HEIGHT;
            if (vw / vh > ratio) vw = vh * ratio; else vh = vw / ratio;
        }
        const double x = (width-vw)/2, y = (height-vh)/2;
        display.viewport_x = x; display.viewport_y = y; display.viewport_w = vw; display.viewport_h = vh;
        display.viewport_drawable_w = width; display.viewport_drawable_h = height;
        source = impl->screen->prepare(impl->screen_commands, source,
            uint32_t(std::lround(vw)), uint32_t(std::lround(vh)));
        auto encoder = [impl->screen_commands renderCommandEncoderWithDescriptor:pass];
        impl->screen->render(encoder, source, MTLViewport{x,y,vw,vh,0,1});
        if (has_overlays) impl->overlay->render(encoder, *overlay_manager, MTLViewport{x,y,vw,vh,0,1});
        [encoder endEncoding];
    }
}
void MetalState::swap_window() {
    if (!impl->screen_commands || !impl->drawable) return;
    [impl->screen_commands presentDrawable:impl->drawable];
    std::string error;
    require(impl->device->submit_and_wait(impl->screen_commands, error), error);
    impl->screen_commands = nil; impl->drawable = nil;
}
} // namespace renderer::metal
