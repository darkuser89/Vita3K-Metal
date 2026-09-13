// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <renderer/metal/buffers.h>
#include <renderer/metal/overlay.h>
#include <renderer/metal/screen.h>
#include <renderer/metal/textures.h>
#include <renderer/metal/state.h>
#include <renderer/functions.h>
#include <shader/msl_recompiler.h>
#include <shader/metal_texture.h>
#include <shader/uniform_block.h>
#include <gxm/functions.h>
#include <config/state.h>
#include <display/state.h>
#include <overlay/display_manager.h>
#include <util/log.h>
#include <util/bytes.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace renderer::metal {
namespace {
void require(bool condition, const std::string &error) {
    if (!condition) throw std::runtime_error(error);
}
void append(std::string &key, uint32_t value) {
    for (int i = 0; i < 4; ++i) key.push_back(static_cast<char>(value >> (i * 8)));
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
MTLPixelFormat color_format(SceGxmColorBaseFormat format) {
    switch (format) {
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8:
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8: return MTLPixelFormatRGBA8Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8: return MTLPixelFormatRGBA8Snorm;
    case SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16: return MTLPixelFormatRGBA16Float;
    // Metal has no A2 + three unsigned 10-bit float attachment. Keep the
    // channels in an expanded float target, also usable by later texture reads.
    case SCE_GXM_COLOR_BASE_FORMAT_U2F10F10F10: return MTLPixelFormatRGBA16Float;
    case SCE_GXM_COLOR_BASE_FORMAT_F32F32: return MTLPixelFormatRG32Float;
    case SCE_GXM_COLOR_BASE_FORMAT_F32: return MTLPixelFormatR32Float;
    case SCE_GXM_COLOR_BASE_FORMAT_F16: return MTLPixelFormatR16Float;
    case SCE_GXM_COLOR_BASE_FORMAT_U8: return MTLPixelFormatR8Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_U16: return MTLPixelFormatR16Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8: return MTLPixelFormatRG8Unorm;
    case SCE_GXM_COLOR_BASE_FORMAT_F11F11F10: return MTLPixelFormatRG11B10Float;
    default: throw std::runtime_error("Metal: unsupported color surface format " + std::to_string(format));
    }
}
MTLVertexFormat attribute_format(SceGxmAttributeFormat f, uint32_t count) {
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
    static const MTLBlendFactor factors[] = {MTLBlendFactorZero, MTLBlendFactorOne, MTLBlendFactorSourceColor,
        MTLBlendFactorOneMinusSourceColor, MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha,
        MTLBlendFactorDestinationColor, MTLBlendFactorOneMinusDestinationColor, MTLBlendFactorDestinationAlpha,
        MTLBlendFactorOneMinusDestinationAlpha, MTLBlendFactorSourceAlphaSaturated};
    require(f < std::size(factors), "Metal: destination alpha saturate requires shader emulation");
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
    const auto x1 = std::max(x0, clamp_edge(std::ceil((double(r.region_clip_max.x) + 1) * scale), w));
    const auto y1 = std::max(y0, clamp_edge(std::ceil((double(r.region_clip_max.y) + 1) * scale), h));
    std::vector<MTLScissorRect> result;
    auto add = [&](uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
        if (width && height) result.push_back({x, y, width, height});
    };
    if (r.region_clip_mode == SCE_GXM_REGION_CLIP_OUTSIDE) add(x0, y0, x1 - x0, y1 - y0);
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

struct Surface {
    uint64_t revision = 0;
    id<MTLTexture> color;
    id<MTLTexture> multisample_color;
    bool multisample_dirty = true;
    uint32_t multisample_scale = 1;
    std::map<std::pair<uint32_t, bool>, id<MTLTexture>> rgba8_casts;
    std::map<std::array<uint32_t,4>,id<MTLTexture>> subrectangles;
    SceGxmColorSurface guest{};
    // Last CPU bytes imported or published. Compare against this baseline,
    // not a downsampled GPU image, to retain untouched high-resolution pixels.
    std::vector<uint8_t> cpu_snapshot;
};
static void update_cpu_snapshot(Surface &surface, std::span<const uint8_t> bytes,
    std::span<const SurfaceMemoryRange> ranges) {
    if (surface.cpu_snapshot.size() != bytes.size()) return;
    for (const auto &range : ranges)
        std::memcpy(surface.cpu_snapshot.data()+range.offset, bytes.data()+range.offset, range.size);
}
struct DepthSurface {
    std::map<std::array<uint32_t,4>,id<MTLTexture>> subrectangles;
    id<MTLTexture> texture;
    id<MTLTexture> snapshot;
    SceGxmDepthStencilSurface guest{};
    uint32_t width = 0, height = 0;
    SceGxmMultisampleMode multisample = SCE_GXM_MULTISAMPLE_NONE;
    std::vector<uint8_t> published_depth, published_stencil;
    float published_background_depth = 0;
    uint32_t published_background_stencil = 0;
    bool published = false;
};
struct MetalContext::Impl {
    MemState *mem = nullptr;
    id<MTLCommandBuffer> commands;
    id<MTLRenderCommandEncoder> encoder;
    id<MTLTexture> color;
    id<MTLTexture> depth;
    id<MTLTexture> transient_depth;
    id<MTLTexture> render_color;
    uint32_t samples = 1, sample_scale = 1;
    bool expanded_color = false, custom_samples = false;
    std::array<MTLSamplePosition,4> sample_positions{};
    std::pair<Address, Address> depth_key{};
    id<MTLTexture> mask;
    SceGxmColorSurface guest_color{};
    SceGxmDepthStencilSurface guest_depth{};
    std::optional<DepthMemoryLayout> depth_layout;
    bool scene_active=false;
    uint32_t width = 0, height = 0;
    bool mask_pass = false;
    bool depth_written = false;
    uint32_t pending_draws = 0;
    size_t pending_upload_bytes = 0;
    UploadBufferArena uploads;
};

static void begin_pass(MetalContext &ctx, bool mask, bool clear_depth = false) {
    if (ctx.impl->encoder) [ctx.impl->encoder endEncoding];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = mask ? ctx.impl->mask : ctx.impl->render_color;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (!mask && ctx.impl->samples > 1 && ctx.impl->color && !ctx.impl->expanded_color) {
        pass.colorAttachments[0].resolveTexture = ctx.impl->color;
        pass.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
    }
    if (ctx.impl->custom_samples) [pass setSamplePositions:ctx.impl->sample_positions.data() count:ctx.impl->samples];
    pass.depthAttachment.texture = pass.stencilAttachment.texture = ctx.impl->depth;
    pass.depthAttachment.loadAction = pass.stencilAttachment.loadAction = clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
    pass.depthAttachment.clearDepth = ctx.record.depth_stencil_surface.background_depth;
    pass.stencilAttachment.clearStencil = ctx.record.depth_stencil_surface.stencil;
    pass.depthAttachment.storeAction = pass.stencilAttachment.storeAction = MTLStoreActionStore;
    if (ctx.impl->custom_samples) pass.depthAttachment.storeAction = MTLStoreActionCustomSampleDepthStore;
    ctx.impl->encoder = [ctx.impl->commands renderCommandEncoderWithDescriptor:pass];
    ctx.impl->mask_pass = mask;
    ctx.impl->depth_written |= clear_depth;
    require(ctx.impl->encoder != nil, "Metal: cannot begin GXM render pass");
}
struct MetalState::Impl {
    bool cache_enabled = false;
    bool cache_reported = false;
    bool sync_draws = std::getenv("VITA3K_METAL_SYNC_DRAWS") != nullptr;
    bool trace_batches = std::getenv("VITA3K_METAL_TRACE_BATCHES") != nullptr;
    uint32_t traced_batches = 0;
    bool trace_textures = std::getenv("VITA3K_METAL_TRACE_TEXTURES") != nullptr;
    std::set<std::string> traced_textures;
    bool trace_draws = std::getenv("VITA3K_METAL_TRACE_DRAWS") != nullptr;
    std::set<std::string> traced_draws;
    std::filesystem::path dump_pipeline_dir = std::getenv("VITA3K_METAL_DUMP_PIPELINE_DIR") ? std::getenv("VITA3K_METAL_DUMP_PIPELINE_DIR") : "";
    uint32_t dumped_pipelines = 0;
    std::filesystem::path dump_surface_dir = std::getenv("VITA3K_METAL_DUMP_SURFACE_DIR") ? std::getenv("VITA3K_METAL_DUMP_SURFACE_DIR") : "";
    std::set<std::string> dumped_surfaces;
    std::filesystem::path dump_draw_dir = std::getenv("VITA3K_METAL_DUMP_DRAW_DIR") ? std::getenv("VITA3K_METAL_DUMP_DRAW_DIR") : "";
    std::string dump_draw_shader = std::getenv("VITA3K_METAL_DUMP_DRAW_SHADER") ? std::getenv("VITA3K_METAL_DUMP_DRAW_SHADER") : "";
    std::string dump_arm_shader = std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_SHADER") ? std::getenv("VITA3K_METAL_DUMP_DRAW_ARM_SHADER") : "";
    bool dump_armed = dump_arm_shader.empty();
    bool dump_attachments = std::getenv("VITA3K_METAL_DUMP_DRAW_ATTACHMENTS") != nullptr;
    std::string dump_vertex_shader = std::getenv("VITA3K_METAL_DUMP_VERTEX_SHADER") ? std::getenv("VITA3K_METAL_DUMP_VERTEX_SHADER") : "";
    bool draw_dumped = false;
    uint32_t dump_draw_matches = 0;
    uint32_t dump_draw_skip = std::getenv("VITA3K_METAL_DUMP_DRAW_SKIP")
        ? uint32_t(std::min<unsigned long>(std::strtoul(std::getenv("VITA3K_METAL_DUMP_DRAW_SKIP"), nullptr, 10), 1000000)) : 0;
    std::unique_ptr<Device> device;
    std::unique_ptr<OverlayRenderer> overlay;
    std::string gpu_name;
    std::map<Address, Surface> surfaces;
    struct RenderedImage {
        id<MTLTexture> texture, uploaded;
        std::vector<id<MTLTexture>> sources;
    };
    std::map<std::string, RenderedImage> rendered_images;
    std::map<std::pair<Address, Address>, DepthSurface> depth_surfaces;
    std::map<std::string, std::unique_ptr<CompiledProgram>> shaders;
    std::map<std::string, id<MTLRenderPipelineState>> pipelines;
    CAMetalLayer *layer;
    std::unique_ptr<ScreenRenderer> screen;
    std::unique_ptr<SurfaceCaster> caster;
    id<CAMetalDrawable> drawable;
    id<MTLCommandBuffer> screen_commands;
    std::vector<uint32_t> last_frame;
    uint32_t last_width = 0, last_height = 0;
};
MetalContext::MetalContext() : impl(std::make_unique<Impl>()) {}
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
    impl->gpu_name = impl->device->native_device().name.UTF8String;
    shader_version = "metal" + std::to_string(shader::metal::SHADER_ABI_VERSION);
    if (frame) {
        auto handle = frame->handle();
        auto *mac = std::get_if<MacOSDisplayHandle>(&handle);
        if (!mac || !mac->view) return false;
        NSView *view = (__bridge NSView *)mac->view;
        if (![view.layer isKindOfClass:[CAMetalLayer class]]) {
            LOG_ERROR("Metal: frame host must provide a CAMetalLayer"); return false;
        }
        impl->layer = (CAMetalLayer *)view.layer;
        impl->layer.device = impl->device->native_device();
        impl->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        impl->layer.framebufferOnly = YES;
        init_overlay_font_dirs();
    }
    LOG_INFO("Native Metal 3 renderer: {}", impl->gpu_name);
    return true;
}
void MetalState::set_app(const char *title_id, const char *self_name) {
    State::set_app(title_id, self_name);
    impl->device->configure_cache(impl->cache_enabled && !cache_path.empty()
        ? std::filesystem::path(shaders_path.string()) / "metal" : std::filesystem::path{});
    impl->cache_reported = false;
    if (frame) impl->overlay = std::make_unique<OverlayRenderer>(*impl->device, std::filesystem::path(static_assets.string()));
    if (!impl->device->cache_directory().empty())
        LOG_INFO("Metal persistent cache: {}", impl->device->cache_directory().string());
}
void MetalState::cleanup() {
    if (context) finish(*static_cast<MetalContext *>(context));
    if (impl->screen_commands) {
        std::string error;
        require(impl->device->submit_and_wait(impl->screen_commands, error), error);
        impl->screen_commands = nil;
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
    impl->surfaces.clear(); impl->depth_surfaces.clear(); impl->pipelines.clear(); impl->shaders.clear();
}
void MetalState::late_init(const Config &cfg, std::string_view game_id, MemState &) {
    impl->cache_enabled = cfg.shader_cache;
    texture_cache.backend = Backend::Metal;
    texture_cache.TextureCache::init(false, texture_folder(), game_id);
}
int MetalState::get_supported_filters() { return int(Filter::NEAREST) | int(Filter::BILINEAR); }
void MetalState::set_screen_filter(const std::string_view &filter) {
    if (!impl->screen) impl->screen = std::make_unique<ScreenRenderer>(*impl->device);
    impl->screen->set_filter(filter != "Nearest");
}
void MetalState::set_anisotropic_filtering(int value) { texture_cache.anisotropic_filtering = std::clamp(value, 1, 16); }
std::string_view MetalState::get_gpu_name() { return impl->gpu_name; }
void MetalState::precompile_shader(const ShadersHash &) {
    // Native variants need the draw's actual vertex layout and texture formats.
    // They are compiled and cached when that complete key first becomes available.
}
void MetalState::preclose_action() {
    // Called on the UI thread before stop_render_thread joins the GXM worker.
    // GPU completion is handled on the worker and needs no wake-up here.
    // cleanup() runs after the join; ending encoders here races active draws.
}
void MetalState::finish(MetalContext &ctx) {
    if (ctx.impl->encoder) { [ctx.impl->encoder endEncoding]; ctx.impl->encoder = nil; }
    if (ctx.impl->commands) {
        std::string error;
        require(impl->device->submit_and_wait(ctx.impl->commands, error), error);
        ctx.impl->commands = nil;
        ctx.impl->uploads.reset_after_completion();
        if (ctx.impl->samples > 1 && ctx.impl->expanded_color && ctx.impl->color) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            impl->caster->expand_multisample(ctx.impl->render_color,ctx.impl->color,ctx.impl->sample_scale);
        }
        if(impl->trace_batches && ctx.impl->pending_draws && impl->traced_batches++<64)
            LOG_INFO("Metal batch completed: draws={} input_bytes={}",ctx.impl->pending_draws,ctx.impl->pending_upload_bytes);
        ctx.impl->pending_draws = 0;
        ctx.impl->pending_upload_bytes = 0;
        if (ctx.impl->depth_written) {
            auto found = impl->depth_surfaces.find(ctx.impl->depth_key);
            if (found != impl->depth_surfaces.end()) {
                found->second.snapshot = nil;
                found->second.subrectangles.clear();
            }
            ctx.impl->depth_written = false;
        }
        // SetContext has already recorded the next scene's surfaces when it calls
        // finish. Read back the surface actually attached to this command buffer.
        const auto &surface = ctx.impl->guest_color;
        if (!disable_surface_sync && ctx.impl->mem && surface.data && ctx.impl->color
            && res_multiplier == 1 && surface.surfaceType == SCE_GXM_COLOR_SURFACE_LINEAR
            && surface.colorFormat == SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR) {
            [ctx.impl->color getBytes:surface.data.get(*ctx.impl->mem) bytesPerRow:surface.strideInPixels * 4
                fromRegion:MTLRegionMake2D(0, 0, surface.width, surface.height) mipmapLevel:0];
            const auto found=impl->surfaces.find(surface.data.address());
            if (found!=impl->surfaces.end()) {
                const size_t bytes=surface_memory_size(surface);
                const SurfaceMemoryRange all{0,bytes};
                update_cpu_snapshot(found->second,
                    {static_cast<const uint8_t *>(surface.data.get(*ctx.impl->mem)),bytes},{&all,1});
            }
        }
    }
}
static std::pair<std::span<uint8_t>,std::span<uint8_t>> depth_memory_spans(MemState &mem,
    const SceGxmDepthStencilSurface &surface,const DepthMemoryLayout &layout) {
    auto span=[&](Ptr<void> pointer,size_t bytes) -> std::span<uint8_t> {
        if(!bytes) return {};
        const uint64_t end=uint64_t(pointer.address())+bytes;
        require(pointer && end<=uint64_t(UINT32_MAX)-4095 && is_valid_addr_range(mem,pointer.address(),Address(end)),
            "Metal: depth/stencil allocation extends beyond mapped guest memory");
        return {static_cast<uint8_t *>(pointer.get(mem)),bytes};
    };
    return {span(surface.depth_data,layout.depth_size),span(surface.stencil_data,layout.stencil_size)};
}
void MetalState::end_scene(MetalContext &ctx) {
    @autoreleasepool {
    finish(ctx);
    if(!ctx.impl->scene_active) return;
    auto cached_depth=impl->depth_surfaces.find(ctx.impl->depth_key);
    if (cached_depth!=impl->depth_surfaces.end()) cached_depth->second.published=false;
    if(ctx.impl->guest_depth.force_store && ctx.impl->depth_layout && ctx.impl->mem) {
        const auto &layout=*ctx.impl->depth_layout;
        const auto [depth,stencil]=depth_memory_spans(*ctx.impl->mem,ctx.impl->guest_depth,layout);
        if(!depth.empty() || !stencil.empty()) {
            if(!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->store_depth_memory(ctx.impl->depth,ctx.impl->guest_depth,layout,ctx.impl->sample_scale,depth,stencil),
                "Metal: cannot publish guest depth/stencil storage");
            if (cached_depth!=impl->depth_surfaces.end() && cached_depth->second.texture==ctx.impl->depth) {
                auto &entry=cached_depth->second;
                entry.published_depth.assign(depth.begin(),depth.end());
                entry.published_stencil.assign(stencil.begin(),stencil.end());
                entry.published_background_depth=ctx.impl->guest_depth.background_depth;
                entry.published_background_stencil=ctx.impl->guest_depth.stencil;
                entry.published=true;
            }
        }
    }
    ctx.impl->scene_active=false;
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
    if (!read_surface_memory(found->second.color, surface, output)) return false;
    if (found->second.guest.strideInPixels == surface.strideInPixels
        && found->second.guest.surfaceType == surface.surfaceType) {
        const SurfaceMemoryRange all{0,bytes};
        update_cpu_snapshot(found->second, output, {&all,1});
    } else found->second.cpu_snapshot.clear();
    return true;
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
            || !write_surface_memory(surface.color,surface.guest,
                {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},{})) return false;
        updates.push_back(std::move(update));
    }
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
        require(write_surface_memory(surface.color,surface.guest,
            {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},update.ranges),
            "Metal: validated transfer destination could not be updated");
        update_cpu_snapshot(surface,
            {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},update.ranges);
        ++surface.revision;
        surface.rgba8_casts.clear(); surface.subrectangles.clear();
        if (surface.multisample_color && !surface.multisample_dirty) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->patch_multisample(surface.multisample_color,surface.guest,surface.multisample_scale,
                {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},update.ranges),
                "Metal: validated transfer could not preserve multisample storage");
        }

    }
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
            // The transfer descriptor supplies the complete Morton dimensions.
            // Nonzero origins require a separate allocation extent, which this
            // command does not currently carry; do not guess its layout.
            if (image.x || image.y || !std::has_single_bit(image.width) || !std::has_single_bit(image.height)) return false;
            const uint64_t begin=image.address.address(), end=begin+uint64_t(image.width)*image.height*bytes;
            if (end>uint64_t(UINT32_MAX)-4095 || !is_valid_addr_range(mem,Address(begin),Address(end))) return false;
            ranges.emplace_back(begin,end);
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
std::vector<SurfaceMemoryRange> surface_intersections(uint64_t address, size_t bytes, const TransferRanges &ranges) {
    std::vector<SurfaceMemoryRange> result;
    for (const auto &[begin,end] : ranges) {
        const auto first=std::max(begin,address), last=std::min(end,address+bytes);
        if (first<last) result.push_back({size_t(first-address),size_t(last-first)});
    }
    return result;
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
bool MetalState::transfer_image(MemState &mem, const SceGxmTransferImage &source, const SceGxmTransferImage &destination,
    SceGxmTransferType source_type, SceGxmTransferType destination_type,
    SceGxmTransferColorKeyMode mode, uint32_t key, uint32_t mask, bool downscale) {
    std::array<unsigned,4> component_bits{};
    if (downscale) {
        switch (source.format) {
        case SCE_GXM_TRANSFER_FORMAT_U8_R: component_bits={8,0,0,0}; break;
        case SCE_GXM_TRANSFER_FORMAT_U8U8_GR: component_bits={8,8,0,0}; break;
        case SCE_GXM_TRANSFER_FORMAT_U4U4U4U4_ABGR: component_bits={4,4,4,4}; break;
        case SCE_GXM_TRANSFER_FORMAT_U1U5U5U5_ABGR: component_bits={5,5,5,1}; break;
        case SCE_GXM_TRANSFER_FORMAT_U5U6U5_BGR: component_bits={5,6,5,0}; break;
        case SCE_GXM_TRANSFER_FORMAT_U8U8U8_BGR: component_bits={8,8,8,0}; break;
        case SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR: component_bits={8,8,8,8}; break;
        case SCE_GXM_TRANSFER_FORMAT_U2U10U10U10_ABGR: component_bits={10,10,10,2}; break;
        default: return false;
        }
    }
    const uint32_t bits=gxm::get_bits_per_pixel(source.format);
    const uint32_t ratio=downscale?2:1;
    if (source.format!=destination.format || !bits || bits%8 || bits>128
        || source.width!=uint64_t(destination.width)*ratio || source.height!=uint64_t(destination.height)*ratio
        || (mode!=SCE_GXM_TRANSFER_COLORKEY_NONE && mode!=SCE_GXM_TRANSFER_COLORKEY_PASS && mode!=SCE_GXM_TRANSFER_COLORKEY_REJECT)
        || (mode!=SCE_GXM_TRANSFER_COLORKEY_NONE && bits!=32)) return false;
    const size_t bytes=bits/8;
    const TransferLayout src{source,source_type,bytes}, dst{destination,destination_type,bytes};
    TransferRanges reads, destinations;
    if (!src.validate(mem,reads) || !dst.validate(mem,destinations)) return false;
    if (context) finish(*static_cast<MetalContext *>(context));
    struct Snapshot { uint64_t address; std::vector<uint8_t> data; };
    struct Update { Surface *surface; size_t bytes; };
    std::vector<Snapshot> snapshots;
    std::vector<Update> updates;
    // Preflight every overlapping cached representation before modifying RAM.
    for (auto &[address,surface] : impl->surfaces) {
        const size_t extent=cached_surface_extent(surface);
        const bool read=!surface_intersections(address,extent,reads).empty();
        const bool write=!surface_intersections(address,extent,destinations).empty();
        if (!read && !write) continue;
        const size_t size=surface_memory_size(surface.guest);
        const uint64_t end=uint64_t(address)+size;
        if (!size || end>uint64_t(UINT32_MAX)-4095 || !is_valid_addr_range(mem,address,Address(end))) return false;
        const auto *guest=static_cast<const uint8_t *>(surface.guest.data.get(mem));
        if (write) {
            if (!write_surface_memory(surface.color,surface.guest,{guest,size},{})) return false;
            updates.push_back({&surface,size});
        }
        if (read) {
            // Competing cache aliases need render-order tracking. Reject such
            // ambiguity instead of selecting a source by map iteration order.
            for (const auto &other : snapshots) {
                const auto first=std::max<uint64_t>(address,other.address), last=std::min(end,other.address+other.data.size());
                if (first<last && !surface_intersections(first,last-first,reads).empty()) return false;
            }
            Snapshot snapshot{address,std::vector<uint8_t>(guest,guest+size)};
            if (!read_surface_memory(surface.color,surface.guest,snapshot.data)) return false;
            snapshots.push_back(std::move(snapshot));
        }
    }
    // Snapshot the complete source rectangle before any destination writes:
    // overlapping copies never consume bytes already replaced by this command.
    std::vector<uint8_t> pixels(size_t(source.width)*source.height*bytes);
    for (uint32_t y=0;y<source.height;++y) for (uint32_t x=0;x<source.width;++x) {
        const uint64_t address=uint64_t(src.address(x,y));
        auto *pixel=pixels.data()+(size_t(y)*source.width+x)*bytes;
        std::memcpy(pixel,Ptr<uint8_t>(Address(address)).get(mem),bytes);
        for (const auto &snapshot : snapshots) {
            const auto first=std::max(address,snapshot.address), last=std::min(address+bytes,snapshot.address+snapshot.data.size());
            if (first<last) std::memcpy(pixel+first-address,snapshot.data.data()+first-snapshot.address,last-first);
        }
    }
    if (downscale) {
        std::vector<uint8_t> reduced(size_t(destination.width)*destination.height*bytes);
        for (uint32_t y=0;y<destination.height;++y) for (uint32_t x=0;x<destination.width;++x) {
            std::array<uint32_t,4> words{};
            for (unsigned dy=0;dy<2;++dy) for (unsigned dx=0;dx<2;++dx)
                std::memcpy(&words[dy*2+dx],pixels.data()+((size_t(y)*2+dy)*source.width+x*2+dx)*bytes,bytes);
            uint32_t output=0; unsigned shift=0;
            for (const unsigned width : component_bits) {
                if (!width) break;
                const uint32_t mask=(1u<<width)-1;
                uint32_t sum=0;
                for (const auto word : words) sum+=(word>>shift)&mask;
                // Average the four encoded components independently. Half ties
                // round upwards; packed words must never be averaged as scalars.
                // Hardware tie precision still needs a physical-Vita comparison.
                output|=((sum+2)/4)<<shift;
                shift+=width;
            }
            std::memcpy(reduced.data()+(size_t(y)*destination.width+x)*bytes,&output,bytes);
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
    for (uint32_t y=0;y<destination.height;++y) for (uint32_t x=0;x<destination.width;++x) {
        const auto *pixel=pixels.data()+(size_t(y)*destination.width+x)*bytes;
        if (!passes(pixel)) continue;
        const Address address=Address(dst.address(x,y));
        std::memcpy(Ptr<uint8_t>(address).get(mem),pixel,bytes);
        append_transfer_range(writes,address,bytes);
    }
    merge_transfer_ranges(writes);
    for (const auto &update : updates) {
        auto &surface=*update.surface;
        const auto ranges=surface_intersections(surface.guest.data.address(),update.bytes,writes);
        if (ranges.empty()) continue;
        require(write_surface_memory(surface.color,surface.guest,
            {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},ranges),
            "Metal: validated transfer copy destination could not be updated");
        update_cpu_snapshot(surface,
            {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},ranges);
        ++surface.revision;
        surface.rgba8_casts.clear(); surface.subrectangles.clear();
        if (surface.multisample_color && !surface.multisample_dirty) {
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            require(impl->caster->patch_multisample(surface.multisample_color,surface.guest,surface.multisample_scale,
                {static_cast<const uint8_t *>(surface.guest.data.get(mem)),update.bytes},ranges),
                "Metal: validated transfer could not preserve multisample storage");
        }

    }
    return true;
}
void MetalState::set_context(MetalContext &ctx, MemState &mem) {
    @autoreleasepool {
        if (context && context != &ctx) end_scene(*static_cast<MetalContext *>(context));
        end_scene(ctx);
        context = &ctx; ctx.impl->mem = &mem;
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
        require(samples==1 || (res_multiplier>=1 && std::floor(res_multiplier)==res_multiplier),"Metal: MSAA sample storage requires integer resolution scale");
        ctx.impl->samples=samples; ctx.impl->sample_scale=uint32_t(res_multiplier);
        ctx.impl->expanded_color=samples>1 && surface.data && !surface.downscale;
        ctx.impl->custom_samples=samples>1 && target->custom_multisample_locations;
        for(uint32_t i=0;i<samples && ctx.impl->custom_samples;++i)
            ctx.impl->sample_positions[i]={float((target->multisample_locations>>(i*8))&15)/16,
                float((target->multisample_locations>>(i*8+4))&15)/16};
        const uint32_t color_width=surface.data ? uint32_t(surface.width*res_multiplier) : target->width;
        const uint32_t color_height=surface.data ? uint32_t(surface.height*res_multiplier) : target->height;
        require(!ctx.impl->expanded_color || (!(surface.width%(samples/2)) && !(surface.height%2)),"Metal: invalid expanded MSAA color extent");
        const auto width=color_width/(ctx.impl->expanded_color ? samples/2 : 1);
        const auto height=color_height/(ctx.impl->expanded_color ? 2 : 1);
        require(width && height, "Metal: empty render target");
        ctx.impl->width = width; ctx.impl->height = height;
        if (surface.data) {
            auto &entry = impl->surfaces[surface.data.address()];
            ++entry.revision;
            entry.rgba8_casts.clear(); entry.subrectangles.clear();
            auto format = color_format(gxm::get_base_format(surface.colorFormat));
            if (surface.gamma) {
                require(surface.gamma == 1 && format == MTLPixelFormatRGBA8Unorm,
                    "Metal: color gamma currently requires RGB/BGR gamma on an RGB8 or RGBA8 target");
                format = MTLPixelFormatRGBA8Unorm_sRGB;
            }
            // A gamma interpretation change must preserve the stored bytes.
            if (entry.color && entry.color.width == color_width && entry.color.height == color_height
                && (entry.color.pixelFormat == MTLPixelFormatRGBA8Unorm || entry.color.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB)
                && (format == MTLPixelFormatRGBA8Unorm || format == MTLPixelFormatRGBA8Unorm_sRGB))
                entry.color = rgba8_gamma_view(entry.color, format == MTLPixelFormatRGBA8Unorm_sRGB);
            const bool new_color = !entry.color || entry.color.width != color_width || entry.color.height != color_height || entry.color.pixelFormat != format;
            if (new_color) {
                entry.multisample_color = nil;
                entry.color = make_texture(*impl->device, format, color_width, color_height, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView | MTLTextureUsageShaderWrite);
                entry.cpu_snapshot.clear();
            }
            const bool rgb = gxm::get_base_format(surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U8U8U8;
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
                        require(write_surface_memory(entry.color,surface,
                            {static_cast<const uint8_t *>(surface.data.get(mem)),bytes},{&all,1}),"Metal: RGB initial surface upload failed");
                    }
                }
            }
            // With synchronous surfaces guest memory is authoritative between scenes,
            // including CPU fills and transfers at an unchanged base address.
            if (!disable_surface_sync && res_multiplier >= 1) {
                const size_t bytes=surface_memory_size(surface);
                const uint64_t end=uint64_t(surface.data.address())+bytes;
                if (bytes && end<=uint64_t(UINT32_MAX)-4095 && is_valid_addr_range(mem,surface.data.address(),Address(end))) {
                    const std::span<const uint8_t> cpu{static_cast<const uint8_t *>(surface.data.get(mem)),bytes};
                    const bool same_layout=entry.guest.width==surface.width && entry.guest.height==surface.height
                        && entry.guest.strideInPixels==surface.strideInPixels && entry.guest.colorFormat==surface.colorFormat
                        && entry.guest.surfaceType==surface.surfaceType;
                    std::vector<SurfaceMemoryRange> changes;
                    if (!same_layout || entry.cpu_snapshot.size()!=bytes) changes.push_back({0,bytes});
                    else for (size_t at=0;at<bytes;) {
                        if (cpu[at]==entry.cpu_snapshot[at]) {++at;continue;}
                        const size_t first=at++;
                        while (at<bytes && cpu[at]!=entry.cpu_snapshot[at]) ++at;
                        changes.push_back({first,at-first});
                    }
                    if (!changes.empty()) {
                        require(write_surface_memory(entry.color,surface,cpu,changes),"Metal: CPU surface import failed");
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
                        entry.cpu_snapshot.assign(cpu.begin(),cpu.end());
                    }
                }
            }
            if (new_color || samples==1) entry.multisample_dirty=true;
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
                    impl->caster->seed_multisample(entry.color,entry.multisample_color,ctx.impl->sample_scale,ctx.impl->expanded_color);
                    entry.multisample_dirty=false;
                }
            }
            ctx.impl->render_color=samples>1 ? entry.multisample_color : entry.color;
            entry.guest = surface;
            ctx.impl->color = entry.color;
        } else ctx.impl->color = ctx.impl->render_color = nil;
        ctx.record.is_gamma_corrected = ctx.impl->color && ctx.impl->color.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB;
        const auto &ds = ctx.record.depth_stencil_surface;
        ctx.impl->guest_depth=ds;
        ctx.impl->depth_layout=(res_multiplier>=1 && std::floor(res_multiplier)==res_multiplier)
            ? depth_memory_layout(ds,uint32_t(width/res_multiplier),uint32_t(height/res_multiplier),target->multisample_mode):std::nullopt;
        if(!ctx.impl->depth_layout && !ds.disabled() && (ds.depth_data || ds.stencil_data) && (ds.force_load || ds.force_store))
            LOG_WARN_ONCE("Metal: guest depth memory encoding/scale is not implemented, format={:#x}",uint32_t(ds.get_format()));
        const bool backed_depth = !ds.disabled() && (ds.depth_data || ds.stencil_data) && (ds.force_load || ds.force_store);
        ctx.impl->depth_key = backed_depth ? std::make_pair(ds.depth_data.address(), ds.stencil_data.address()) : std::pair<Address,Address>{};
        id<MTLTexture> selected = backed_depth ? impl->depth_surfaces[ctx.impl->depth_key].texture : ctx.impl->transient_depth;
        bool new_depth = !selected || selected.width != width || selected.height != height || selected.sampleCount != samples;
        if (backed_depth && !new_depth) {
            const auto &entry = impl->depth_surfaces[ctx.impl->depth_key];
            new_depth = entry.guest.get_format() != ds.get_format() || entry.guest.get_type() != ds.get_type()
                || entry.guest.get_stride() != ds.get_stride() || entry.multisample != target->multisample_mode;
        }
        if (new_depth)
            selected = make_texture(*impl->device, MTLPixelFormatDepth32Float_Stencil8, width, height, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, samples);
        if (backed_depth) {
            auto &entry = impl->depth_surfaces[ctx.impl->depth_key];
            if (new_depth) { entry.snapshot = nil; entry.subrectangles.clear(); entry.published=false; }
            entry.texture = selected;
            entry.guest = ds;
            entry.width = uint32_t(width / res_multiplier);
            entry.height = uint32_t(height / res_multiplier);
            entry.multisample = target->multisample_mode;
        } else ctx.impl->transient_depth = selected;
        ctx.impl->depth = selected;
        bool loaded_depth=false;
        if(backed_depth && ds.force_load && ctx.impl->depth_layout) {
            const auto &layout=*ctx.impl->depth_layout;
            const auto [depth,stencil]=depth_memory_spans(mem,ds,layout);
            if(!depth.empty() || !stencil.empty()) {
                auto &entry=impl->depth_surfaces[ctx.impl->depth_key];
                // A store quantizes D16/D24 and reduces scaled depth/stencil to
                // the guest grid. Re-uploading our own unchanged bytes destroys
                // the exact depth needed by a following EQUAL material pass.
                // Retain the native attachment only while that published guest
                // image is unchanged; CPU/transfer edits still force a reload.
                const auto unchanged=[](const std::vector<uint8_t> &saved,std::span<const uint8_t> current) {
                    return saved.size()==current.size() && std::equal(saved.begin(),saved.end(),current.begin());
                };
                const bool reuse=entry.published && !new_depth
                    && entry.published_background_depth==ds.background_depth && entry.published_background_stencil==ds.stencil
                    && unchanged(entry.published_depth,depth) && unchanged(entry.published_stencil,stencil);
                if (!reuse) {
                    if(!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    require(impl->caster->load_depth_memory(selected,ds,layout,ctx.impl->sample_scale,depth,stencil),
                        "Metal: cannot load guest depth/stencil storage");
                    entry.snapshot=nil;entry.subrectangles.clear();
                }
                loaded_depth=true;
            }
        }
        ctx.impl->scene_active=true;
        if (!ctx.impl->mask || ctx.impl->mask.width!=width || ctx.impl->mask.height!=height || ctx.impl->mask.sampleCount!=samples) {
            ctx.impl->mask=make_texture(*impl->device,MTLPixelFormatRGBA8Unorm,width,height,
                MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead,samples);
            auto commands=[impl->device->command_queue() commandBuffer]; auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture=ctx.impl->mask;
            pass.colorAttachments[0].loadAction=MTLLoadActionClear;
            pass.colorAttachments[0].clearColor=MTLClearColorMake(1,1,1,1);
            pass.colorAttachments[0].storeAction=MTLStoreActionStore;
            auto encoder=[commands renderCommandEncoderWithDescriptor:pass]; [encoder endEncoding];
            std::string error; require(impl->device->submit_and_wait(commands,error),error);
        }
        ctx.impl->commands = [impl->device->command_queue() commandBuffer];
        begin_pass(ctx, false, !loaded_depth && (new_depth || !ctx.record.depth_stencil_surface.force_load));
    }
}
void set_uniform_buffer(MetalContext &ctx, const ShaderProgram &program, bool vertex, int block, uint32_t size, const uint8_t *data) {
    const auto offset = program.uniform_buffer_data_offsets.at(block);
    if (offset == uint32_t(-1)) return;
    const auto count = std::min<size_t>(size, program.uniform_buffer_sizes.at(block) * 4ull);
    require(data != nullptr && count, "Metal: empty uniform binding");
    ctx.uniforms[vertex ? 0 : 1].at(block) = {const_cast<uint8_t *>(data), count};
}
void set_viewport(MetalContext &ctx, float x, float y, float sx, float sy) {
    ctx.record.viewport_flip[0] = sx < 0 ? -1.0f : 1.0f;
    ctx.viewport = {x - std::abs(sx), y - std::abs(sy), 2 * std::abs(sx), 2 * std::abs(sy)};
}

struct MetalTextureCache::Impl {
    MetalState &state;
    std::array<id<MTLTexture>, TextureCacheSize> textures{};
    size_t current = 0;
    uint32_t width = 0, height = 0, mips = 1;
    bool cube = false;
    uint32_t gamma = 0;
    SceGxmTexture source{};
    std::filesystem::path dump_dir = std::getenv("VITA3K_METAL_DUMP_TEXTURE_DIR") ? std::getenv("VITA3K_METAL_DUMP_TEXTURE_DIR") : "";
    std::set<std::string> dumped_textures;
    explicit Impl(MetalState &state) : state(state) {}
};
MetalTextureCache::MetalTextureCache(MetalState &state) : impl(std::make_unique<Impl>(state)) {}
MetalTextureCache::~MetalTextureCache() = default;
void MetalTextureCache::cache_and_bind_image(const SceGxmTexture &texture, MemState &mem) {
    TextureCache::cache_and_bind_texture(texture_image_descriptor(texture),mem);
}
id<MTLTexture> current_texture(const MetalTextureCache &cache) {
    return cache.impl->textures[cache.impl->current];
}
void MetalTextureCache::select(size_t index, const SceGxmTexture &texture) {
    require(index < TextureCacheSize, "Metal: texture cache index out of bounds");
    impl->current = index;
    impl->source = texture;
    impl->width = gxm::get_width(texture); impl->height = gxm::get_height(texture);
    impl->mips = renderer::texture::get_upload_mip(texture.true_mip_count(), impl->width, impl->height);
    impl->cube = texture.texture_type() == SCE_GXM_TEXTURE_CUBE || texture.texture_type() == SCE_GXM_TEXTURE_CUBE_ARBITRARY;
    impl->gamma = texture.gamma_mode;
}
void MetalTextureCache::configure_texture(const SceGxmTexture &) { impl->textures[impl->current] = nil; }
uint64_t MetalTextureCache::additional_texture_hash(const SceGxmTexture &texture, const MemState &mem) const {
    const auto type = texture.texture_type();
    const bool cube = type == SCE_GXM_TEXTURE_CUBE || type == SCE_GXM_TEXTURE_CUBE_ARBITRARY;
    if (!texture.data_addr || (!cube && texture.true_mip_count() <= 1)) return 0;
    const size_t bytes = texture_storage_size(texture);
    const Address address = texture.data_addr << 2;
    const uint64_t end = uint64_t(address)+bytes;
    require(bytes && end <= uint64_t(UINT32_MAX)-4095 && is_valid_addr_range(mem,address,Address(end)),
        "Metal: texture mip/face storage extends beyond mapped guest memory");
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
void MetalTextureCache::import_configure_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height, bool srgb, uint16_t, uint16_t mips, bool swap_rb) {
    require(!swap_rb, "Metal: replacement channel swap not yet supported");
    impl->width = width; impl->height = height; impl->mips = mips;
    // Replacement metadata requests sRGB for all stored color channels,
    // independently of the guest descriptor's partial-channel gamma mode.
    impl->gamma = srgb ? (format == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8 ? 3 : 1) : 0;
    impl->textures[impl->current] = nil;
}
void MetalTextureCache::upload_texture_impl(SceGxmTextureBaseFormat format, uint32_t width, uint32_t height,
    uint32_t mip, const void *pixels, int face, uint32_t stride) {
    MTLPixelFormat native;
    uint32_t bytes;
    switch (format) {
#define TEX(gxm, mtl, size) case SCE_GXM_TEXTURE_BASE_FORMAT_##gxm: native = MTLPixelFormat##mtl; bytes = size; break
    TEX(U8, R8Unorm, 1); TEX(S8, R8Snorm, 1);
    TEX(U8U8, RG8Unorm, 2); TEX(S8S8, RG8Snorm, 2);
    TEX(U8U8U8U8, RGBA8Unorm, 4); TEX(S8S8S8S8, RGBA8Snorm, 4);
    TEX(U16, R16Unorm, 2); TEX(S16, R16Snorm, 2); TEX(F16, R16Float, 2);
    TEX(U16U16, RG16Unorm, 4); TEX(S16S16, RG16Snorm, 4); TEX(F16F16, RG16Float, 4);
    TEX(U16U16U16U16, RGBA16Unorm, 8); TEX(S16S16S16S16, RGBA16Snorm, 8); TEX(F16F16F16F16, RGBA16Float, 8);
    TEX(U32, R32Uint, 4); TEX(F32, R32Float, 4); TEX(F32M, R32Float, 4);
    TEX(U32U32, RG32Uint, 8); TEX(F32F32, RG32Float, 8); TEX(F11F11F10, RG11B10Float, 4);
#undef TEX
    default: throw std::runtime_error("Metal: unsupported decoded texture format " + std::to_string(format));
    }
    const uint32_t slice = face > 0 ? face - 1 : 0;
    if (!impl->dump_dir.empty() && !mip && format == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 && impl->dumped_textures.size() < 64) {
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
    std::vector<uint16_t> linear_pixels;
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
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native width:impl->width height:impl->height mipmapped:NO];
        desc.textureType = impl->cube ? MTLTextureTypeCube : MTLTextureType2D;
        desc.mipmapLevelCount = impl->mips;
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
        impl->textures[impl->current] = [impl->state.impl->device->native_device() newTextureWithDescriptor:desc];
    }
    id<MTLTexture> texture = impl->textures[impl->current];
    require(texture != nil, "Metal: texture upload allocation failed");
    const auto pitch = std::max(stride, width) * bytes;
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:mip slice:slice withBytes:pixels bytesPerRow:pitch bytesPerImage:pitch * height];
}

void MetalState::draw(MetalContext &ctx, MemState &mem, SceGxmPrimitiveType primitive, SceGxmIndexFormat index_format,
    const void *indices, uint32_t count, uint32_t instances) {
    // GXM commands run on a long-lived std::thread, outside Cocoa's event-loop
    // pool. Completed command buffers otherwise retain their draw resources
    // until thread exit. Persistent state owns its objects with strong ARC refs.
    @autoreleasepool {
        if (!count || !instances) return;
        if (primitive == SCE_GXM_PRIMITIVE_TRIANGLE_FAN && count < 3) return;
        require(index_format == SCE_GXM_INDEX_FORMAT_U16 || index_format == SCE_GXM_INDEX_FORMAT_U32, "Metal: invalid index format");
        require(ctx.impl->mem == &mem && ctx.current_render_target && indices, "Metal: draw outside a scene or without indices");
        auto &record = ctx.record;
        auto *vp = record.vertex_program.get(mem);
        auto *fp = record.fragment_program.get(mem);
        require(vp && fp, "Metal: missing GXM shader program");
        const bool triangles = primitive == SCE_GXM_PRIMITIVE_TRIANGLES
            || primitive == SCE_GXM_PRIMITIVE_TRIANGLE_STRIP || primitive == SCE_GXM_PRIMITIVE_TRIANGLE_FAN;
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
        const bool fragment_resources = !record.is_maskupdate && !fragment_disabled;
        if (!impl->dump_armed && fragment_resources && impl->dump_arm_shader == hex_string(fp->renderer_data->hash))
            impl->dump_armed = true;
        const bool capture_draw = !impl->dump_draw_dir.empty() && !impl->draw_dumped && impl->dump_armed
            && (impl->dump_draw_shader.empty() || impl->dump_draw_shader == hex_string(fp->renderer_data->hash))
            && (impl->dump_vertex_shader.empty() || impl->dump_vertex_shader == hex_string(vp->renderer_data->hash))
            && impl->dump_draw_matches++ >= impl->dump_draw_skip;
        std::ofstream draw_metadata;
        std::ofstream texture_metadata;
        auto dump_bytes = [&](const std::string &name, const void *bytes, size_t size) {
            std::ofstream out(impl->dump_draw_dir/name, std::ios::binary);
            out.write(static_cast<const char *>(bytes), size);
            require(bool(out), "Metal: cannot write draw diagnostic " + name);
        };
        // Optional attachment probes are separate from draw.txt so older replay
        // readers remain usable. Snapshot only completed single-sample images.
        auto dump_attachment = [&](const char *name, id<MTLTexture> source) {
            if (!capture_draw || !impl->dump_attachments || !source) return;
            std::filesystem::create_directories(impl->dump_draw_dir);
            std::ofstream metadata(impl->dump_draw_dir/"attachments.txt",std::ios::app);
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
        const auto clip = scissors(record, ctx.impl->width, ctx.impl->height, res_multiplier);
        if (clip.empty()) return;
        if (capture_draw) finish(ctx); // Publish every producer before diagnostic texture reads.
        dump_attachment("color-before",ctx.impl->color);
        ctx.shader_hints.metal_samples = ctx.impl->samples;
        ctx.shader_hints.metal_missing_vertex_outputs = fragment_resources
            ? (uint32_t(gxp::get_fragment_inputs(*fp->program.get(mem)))
                  & ~uint32_t(gxp::get_vertex_outputs(*vp->program.get(mem))) & 0x3ffeu)
            : 0;
        ctx.shader_hints.metal_mip_sampling = true;
        ctx.shader_hints.attributes = &vp->attributes;
        ctx.shader_hints.color_format = record.color_surface.colorFormat;
        ctx.shader_hints.metal_output_register_size = record.color_surface.outputRegisterSize;
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
            std::string key = hex_string(vertex ? vp->renderer_data->hash : fp->renderer_data->hash);
            append(key, vertex);
            if (vertex) append(key, ctx.shader_hints.metal_missing_vertex_outputs);
            append(key, ctx.shader_hints.metal_mip_sampling);
            for (bool feature : {features.support_shader_interlock, features.support_texture_barrier,
                     features.direct_fragcolor, features.spirv_shader, features.support_get_texture_sub_image,
                     features.preserve_f16_nan_as_u16, features.support_unknown_format, features.support_rgb_attributes,
                     features.use_mask_bit, true /* native uniform addresses */, features.support_scaled_attribute_formats,
                     features.use_texture_viewport}) append(key, feature);
            append(key, ctx.shader_hints.color_format);
            append(key, ctx.shader_hints.metal_output_register_size);
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
        std::string error;
        auto compile = [&](bool vertex) -> CompiledProgram & {
            auto key = shader_key(vertex);
            auto &cached = impl->shaders[key];
            if (!cached) {
                const auto *gxp = vertex ? vp->program.get(mem) : fp->program.get(mem);
                // The generic memory-mapping feature also changes GXM vertex upload
                // and visibility handling. Keep those paths independent while Metal
                // gives shaders native addresses for their original uniform buffers.
                auto shader_features = features;
                shader_features.enable_memory_mapping = true;
                const bool gamma = !vertex && !record.is_maskupdate && record.is_gamma_corrected;
                if (auto msl = impl->device->load_cached_program(key)) {
                    cached = impl->device->compile(*msl, gamma, error);
                    if (!cached) LOG_WARN("Metal: cached shader failed to compile; rebuilding: {}", error);
                }
                if (!cached) {
                    auto msl = !vertex && fragment_disabled ? depth_only_program(ctx.impl->samples)
                        : shader::metal::convert_gxp(*gxp, hex_string(vertex ? vp->renderer_data->hash : fp->renderer_data->hash), shader_features, ctx.shader_hints, !vertex && record.is_maskupdate);
                    cached = impl->device->compile(msl, gamma, error);
                    require(bool(cached), error);
                    impl->device->store_cached_program(key, msl);
                }
                ++shaders_count_compiled;
            }
            return *cached;
        };
        auto &vs = compile(true); auto &fs = compile(false);
        const bool synchronous = impl->sync_draws || vs.writes_guest_memory || fs.writes_guest_memory;
        size_t captured_texture_bytes = 0;
        const auto find_subrectangle = [&](const SceGxmTexture &texture) {
            auto result=impl->surfaces.end();
            for (auto it=impl->surfaces.begin();it!=impl->surfaces.end();++it) {
                if (!surface_subrectangle(it->second.guest,texture)) continue;
                require(result==impl->surfaces.end(),"Metal: ambiguous color subrectangle aliases");
                result=it;
            }
            return result;
        };
        // A mip chain or cube can span several independently rendered surfaces.
        // Publish incompatible aliases and preserve native pixels for matching levels.
        std::array<id<MTLTexture>, SCE_GXM_MAX_TEXTURE_UNITS*2> prepared_images{};
        std::array<shader::metal::TextureMipInfo,SCE_GXM_MAX_TEXTURE_UNITS*2> texture_mip_info{};
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            const uint32_t slot=index%SCE_GXM_MAX_TEXTURE_UNITS;
            if (!vertex && !fragment_resources) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->renderer_data):static_cast<const ShaderProgram &>(*fp->renderer_data);
            if (!program.textures_used[slot]) continue;
            const auto &texture=ctx.textures[index];
            const bool cube=((vertex ? vs.cube_texture_mask : fs.cube_texture_mask)&(1u<<slot))!=0;
            if (!cube && (texture.texture_type()==SCE_GXM_TEXTURE_LINEAR_STRIDED
                || texture::get_upload_mip(texture.true_mip_count(),gxm::get_width(texture),gxm::get_height(texture))<=1)) continue;
            const auto upload=cube?cube_texture_descriptor(texture):texture;
            const Address address=texture.data_addr<<2;
            const size_t storage=texture_storage_size(upload), face_stride=cube?storage/6:storage;
            const uint64_t end=uint64_t(address)+storage;
            require(address && end<=uint64_t(UINT32_MAX)-4095 && is_valid_addr_range(mem,address,Address(end)),
                "Metal: texture subresources extend beyond mapped guest memory");
            std::vector<Surface *> overlaps;
            for (auto &[base,surface]:impl->surfaces) {
                const size_t bytes=surface_memory_size(surface.guest);
                if (bytes && uint64_t(base)<end && uint64_t(base)+bytes>address) overlaps.push_back(&surface);
            }
            // Without a color producer, retain the ordinary 2D path, including
            // its depth-surface aliases. Cubes still need their six-face upload.
            if (overlaps.empty() && !cube) continue;
            if (overlaps.empty()) {
                texture_cache.cache_and_bind_image(upload,mem);
                prepared_images[index]=sampling_view(current_texture(texture_cache),gxm::get_format(texture));
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
            uint32_t scale=1;
            for (auto *surface:overlaps) {
                bool matched=false;
                SceGxmTextureFormat surface_texture_format{};
                const bool same_format=gxm::convert_color_format_to_texture_format(surface->guest.colorFormat,surface_texture_format)
                    && gxm::get_base_format(surface_texture_format)==base_format;
                const bool can_cast=surface_format_cast_supported(surface->guest.colorFormat,base_format);
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
                            require(surface->color.width%w==0 && surface->color.height%h==0 && surface->color.height/h==surface->color.width/w,
                                "Metal: incompatible rendered texture resolution");
                            scale=std::max(scale,uint32_t(surface->color.width/w));
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
            auto uploaded=sampling_view(base_image,gxm::get_format(texture));
            if (rendered.empty()) { prepared_images[index]=uploaded;continue; }
            if (!cube) for (uint32_t mip=0;mip<mips;++mip)
                mip_info.control[0]|=mip_info.sizes[mip]!=std::array<uint32_t,2>{std::max(1u,uint32_t(uploaded.width)*scale>>mip),std::max(1u,uint32_t(uploaded.height)*scale>>mip)};

            const auto image_descriptor=texture_image_descriptor(upload);
            std::string image_key(reinterpret_cast<const char *>(&image_descriptor),sizeof(image_descriptor));
            const auto append_wide=[&](uint64_t value) { append(image_key,uint32_t(value));append(image_key,uint32_t(value>>32)); };
            append_wide(reinterpret_cast<uintptr_t>((__bridge void *)base_image));
            append(image_key,scale);
            for (const auto &face:rendered) {
                append_wide(reinterpret_cast<uintptr_t>((__bridge void *)face.surface->color));
                append_wide(face.surface->revision);append(image_key,face.face);append(image_key,face.mip);
            }
            if (auto found=impl->rendered_images.find(image_key);found!=impl->rendered_images.end()) {
                prepared_images[index]=found->second.texture;continue;
            }
            finish(ctx);
            if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
            std::vector<CubeSurface> sources;
            for (const auto &face:rendered) {
                auto native=face.surface->color;
                const SceGxmColorFormat *format=&face.surface->guest.colorFormat;
                if (face.cast) {
                    native=impl->caster->surface_format_cast(native,*format,base_format);
                    require(native!=nil,"Metal: incompatible rendered texture format cast");
                    format=nullptr; // The cast reconstructs guest bytes, before texture swizzling.
                }
                if (native.pixelFormat==MTLPixelFormatRGBA8Unorm_sRGB || texture.gamma_mode) {
                    native=impl->caster->rgba8_surface_sampling(native,format?*format:SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,texture.gamma_mode);format=nullptr;
                }
                sources.push_back({sampling_view(native,gxm::get_format(texture),format),face.face,face.mip});
            }
            prepared_images[index]=impl->caster->texture_snapshot(uploaded,sources,scale);
            if (impl->rendered_images.size()>=4) impl->rendered_images.erase(impl->rendered_images.begin());
            auto &cached=impl->rendered_images[image_key];
            cached.texture=prepared_images[index];cached.uploaded=base_image;
            for (const auto &face:rendered) cached.sources.push_back(face.surface->color);
            if (impl->trace_textures) LOG_INFO("Metal rendered texture assembled: address={:#x} faces_mips={} scale={}",address,rendered.size(),scale);
        }
        // Prepare separate images before opening/binding this draw's encoder.
        // This also handles feedback from an offset inside the active target.
        for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
            const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
            if (!vertex && !fragment_resources) continue;
            const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->renderer_data):static_cast<const ShaderProgram &>(*fp->renderer_data);
            if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
            if (prepared_images[index]) continue;
            const auto &texture=ctx.textures[index]; auto found=find_subrectangle(texture);
            if (found==impl->surfaces.end()) continue;
            auto &entry=found->second; const auto rect=*surface_subrectangle(entry.guest,texture);
            if (!rect.x && !rect.y && rect.width==entry.guest.width && rect.height==entry.guest.height) continue;
            auto &image=entry.subrectangles[{rect.x,rect.y,rect.width,rect.height}];
            if (!image) {
                finish(ctx);
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                image=impl->caster->color_subrectangle(entry.color,entry.guest,rect);
            }
        }
        id<MTLTexture> color_feedback = nil;
        if (ctx.impl->color && !record.is_maskupdate) {
            for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
                const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
                if (!vertex && !fragment_resources) continue;
                const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->renderer_data):static_cast<const ShaderProgram &>(*fp->renderer_data);
                if (!program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS]) continue;
                if (prepared_images[index]) continue;
                const auto &texture=ctx.textures[index];
                const Address address=texture.data_addr<<2;
                const auto base=gxm::get_base_format(gxm::get_format(texture));
                const bool word_alias=ctx.impl->color.pixelFormat==MTLPixelFormatRG32Float
                    && (base==SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 || base==SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8)
                    && uint64_t(address)==uint64_t(ctx.impl->guest_color.data.address())+4;
                if (address==ctx.impl->guest_color.data.address() || word_alias) {
                    finish(ctx);
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    color_feedback=impl->caster->color_snapshot(ctx.impl->color);
                    break;
                }
            }
        }
        // If this draw samples its own newly cleared depth, publish the clear
        // before taking a snapshot. This happens before any encoder bindings.
        if (ctx.impl->depth_written && ctx.impl->depth_key.first) {
            for (uint32_t index=0;index<SCE_GXM_MAX_TEXTURE_UNITS*2;++index) {
                const bool vertex=index>=SCE_GXM_MAX_TEXTURE_UNITS;
                if(!vertex && !fragment_resources) continue;
                const auto &program=vertex?static_cast<const ShaderProgram &>(*vp->renderer_data):static_cast<const ShaderProgram &>(*fp->renderer_data);
                const auto found=impl->depth_surfaces.find(ctx.impl->depth_key);
                if(!prepared_images[index] && program.textures_used[index%SCE_GXM_MAX_TEXTURE_UNITS] && found!=impl->depth_surfaces.end()
                    && depth_subrectangle(found->second.guest,found->second.width,found->second.height,
                        found->second.multisample,ctx.textures[index])) { finish(ctx);break; }
            }
        }
        if (!ctx.impl->encoder && ctx.impl->samples>1 && ctx.impl->guest_color.data) {
            auto &surface=impl->surfaces.at(ctx.impl->guest_color.data.address());
            if (surface.multisample_dirty) {
                if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                impl->caster->seed_multisample(surface.color,ctx.impl->render_color,ctx.impl->sample_scale,ctx.impl->expanded_color);
                surface.multisample_dirty=false;
            }
        }
        if (!ctx.impl->encoder) {
            ctx.impl->commands = [impl->device->command_queue() commandBuffer];
            begin_pass(ctx, record.is_maskupdate);
        }
        if (ctx.impl->mask_pass != record.is_maskupdate) begin_pass(ctx, record.is_maskupdate);
        if (capture_draw) {
            std::filesystem::create_directories(impl->dump_draw_dir);
            draw_metadata.open(impl->dump_draw_dir/"draw.txt");
            texture_metadata.open(impl->dump_draw_dir/"textures.txt");
            texture_metadata << "version 2\n";
            draw_metadata.precision(17);
            draw_metadata << "version 1\nscale " << res_multiplier << "\nsize " << ctx.impl->width << ' ' << ctx.impl->height
                          << "\nfragment " << hex_string(fp->renderer_data->hash)
                          << "\nvertex " << hex_string(vp->renderer_data->hash)
                          << "\noutput_register_size " << record.color_surface.outputRegisterSize << '\n';
            auto shader_features = features;
            shader_features.enable_memory_mapping = true;
            auto msl = shader::metal::convert_gxp(*vp->program.get(mem), hex_string(vp->renderer_data->hash), shader_features, ctx.shader_hints, false);
            dump_bytes("vertex.metal", msl.source.data(), msl.source.size());
            const auto *vertex_gxp = vp->program.get(mem), *fragment_gxp = fp->program.get(mem);
            dump_bytes("vertex.gxp", vertex_gxp, vertex_gxp->size);
            dump_bytes("fragment.gxp", fragment_gxp, fragment_gxp->size);
            auto fragment_msl = fragment_disabled ? depth_only_program(ctx.impl->samples)
                : shader::metal::convert_gxp(*fragment_gxp, hex_string(fp->renderer_data->hash),
                    shader_features, ctx.shader_hints, record.is_maskupdate);
            dump_bytes("fragment.metal", fragment_msl.source.data(), fragment_msl.source.size());
            draw_metadata << "entry " << msl.entry_point << '\n';
            for (uint32_t slot=0;slot<SCE_GXM_MAX_TEXTURE_UNITS;++slot)
                if (vp->renderer_data->textures_used[slot]) draw_metadata << "vertex_texture " << slot << '\n';
        }
        MTLVertexDescriptor *layout = [MTLVertexDescriptor vertexDescriptor];
        std::string key = shader_key(true) + shader_key(false);
        uint32_t streams_used = 0;
        std::array<uint32_t, SCE_GXM_MAX_VERTEX_STREAMS> stream_extents{};
        for (const auto &a : vp->attributes) {
            auto it = vp->renderer_data->attribute_infos.find(a.regIndex);
            if (it == vp->renderer_data->attribute_infos.end()) continue;
            const auto &info = it->second;
            uint32_t components = a.componentCount;
            auto f = a.format;
            if (info.regformat) {
                components = info.component_count;
                switch (info.gxm_type) {
                case SCE_GXM_PARAMETER_TYPE_U8: case SCE_GXM_PARAMETER_TYPE_S8: f = SCE_GXM_ATTRIBUTE_FORMAT_U8; break;
                case SCE_GXM_PARAMETER_TYPE_C10: f = SCE_GXM_ATTRIBUTE_FORMAT_U8; components = (components * 10 + 7) / 8; break;
                case SCE_GXM_PARAMETER_TYPE_U16: case SCE_GXM_PARAMETER_TYPE_S16: case SCE_GXM_PARAMETER_TYPE_F16: f = SCE_GXM_ATTRIBUTE_FORMAT_U16; break;
                default: f = SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED; break;
                }
            }
            const uint32_t element_size = gxm::attribute_format_size(f);
            require(a.streamIndex < vp->streams.size() && a.streamIndex < SCE_GXM_MAX_VERTEX_STREAMS, "Metal: invalid vertex stream");
            stream_extents[a.streamIndex] = std::max(stream_extents[a.streamIndex], uint32_t(a.offset + components * element_size));
            for (uint32_t element = 0; element < (components + 3) / 4; ++element) {
                const auto location = info.location + element;
                require(location < 31, "Metal: attribute location exceeds hardware limit");
                const auto n = std::min(components - 4 * element, 4u);
                layout.attributes[location].format = attribute_format(f, n);
                layout.attributes[location].offset = a.offset + 4 * element * element_size;
                layout.attributes[location].bufferIndex = shader::metal::VERTEX_STREAM_BUFFER_BASE + a.streamIndex;
                append(key, location); append(key, layout.attributes[location].format);
                append(key, layout.attributes[location].offset); append(key, a.streamIndex);
                if (capture_draw) draw_metadata << "attribute " << location << ' ' << layout.attributes[location].format
                    << ' ' << layout.attributes[location].offset << ' ' << layout.attributes[location].bufferIndex << '\n';
            }
            streams_used |= 1u << a.streamIndex;
        }
        require(vp->streams.size() <= SCE_GXM_MAX_VERTEX_STREAMS, "Metal: too many vertex streams");
        for (uint32_t stream = 0; stream < vp->streams.size(); ++stream) if (streams_used & (1u << stream)) {
            auto *binding = layout.layouts[shader::metal::VERTEX_STREAM_BUFFER_BASE + stream];
            configure_vertex_stream(binding, vp->streams[stream].stride, stream_extents[stream],
                gxm::is_stream_instancing(static_cast<SceGxmIndexSource>(vp->streams[stream].indexSource)));
            append(key, binding.stride); append(key, binding.stepFunction);
            if (capture_draw) draw_metadata << "layout " << shader::metal::VERTEX_STREAM_BUFFER_BASE+stream << ' '
                << binding.stride << ' ' << binding.stepFunction << ' ' << binding.stepRate << '\n';
        }
        SceGxmBlendInfo mask_blend{};
        mask_blend.colorMask = SCE_GXM_COLOR_MASK_ALL;
        const auto blend = fragment_disabled ? SceGxmBlendInfo{}
            : record.is_maskupdate ? mask_blend : static_cast<MetalFragmentProgram &>(*fp->renderer_data).blend;
        id<MTLTexture> attachment = record.is_maskupdate ? ctx.impl->mask : ctx.impl->render_color;
        const bool rgb_target = !record.is_maskupdate && gxm::get_base_format(record.color_surface.colorFormat) == SCE_GXM_COLOR_BASE_FORMAT_U8U8U8;
        const bool alpha_target = !record.is_maskupdate && record.color_surface.colorFormat == SCE_GXM_COLOR_FORMAT_U8_A;
        append(key, attachment ? attachment.pixelFormat : MTLPixelFormatInvalid);
        append(key, rgb_target);
        append(key, alpha_target);
        append(key, ctx.impl->samples);
        append(key, blend.colorMask); append(key, blend.colorFunc); append(key, blend.alphaFunc);
        append(key, blend.colorSrc); append(key, blend.colorDst); append(key, blend.alphaSrc); append(key, blend.alphaDst);
        auto &pipeline = impl->pipelines[key];
        if (!pipeline) {
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
                    case SCE_GXM_BLEND_FACTOR_SRC_ALPHA_SATURATE: return MTLBlendFactorOne;
                    default: return blend_factor(factor);
                    }
                };
                color.sourceRGBBlendFactor = alpha_factor(blend.alphaSrc);
                color.destinationRGBBlendFactor = alpha_factor(blend.alphaDst);
            }
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
                    write_gxp("vertex.gxp",vp->program.get(mem)); write_gxp("fragment.gxp",fp->program.get(mem));
                    std::ofstream meta(dir/"pipeline.txt");
                    meta<<"vertex "<<hex_string(vp->renderer_data->hash)<<"\nfragment "<<hex_string(fp->renderer_data->hash)
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
            pipeline = impl->device->create_pipeline(desc, error); require(pipeline != nil, error);
        }
        id<MTLRenderCommandEncoder> encoder = ctx.impl->encoder;
        [encoder setRenderPipelineState:pipeline];
        MTLDepthStencilDescriptor *depth = [MTLDepthStencilDescriptor new];
        const bool depth_enabled = !record.is_maskupdate && !record.depth_stencil_surface.disabled();
        require(!depth_enabled || !two_sided || cull_front || cull_back
            || (record.front_depth_func == record.back_depth_func && record.front_depth_write_mode == record.back_depth_write_mode),
            "Metal: differing front/back depth state requires per-primitive routing");
        const bool back_depth = two_sided && cull_front;
        const auto depth_func = back_depth ? record.back_depth_func : record.front_depth_func;
        const auto depth_write_mode = back_depth ? record.back_depth_write_mode : record.front_depth_write_mode;
        depth.depthCompareFunction = depth_enabled ? static_cast<MTLCompareFunction>(uint32_t(depth_func) >> 22) : MTLCompareFunctionAlways;
        depth.depthWriteEnabled = depth_enabled && depth_write_mode == SCE_GXM_DEPTH_WRITE_ENABLED;
        ctx.impl->depth_written |= depth.depthWriteEnabled;
        if (depth_enabled) {
            depth.frontFaceStencil = stencil_desc(record.front_stencil_state_op, record.front_stencil_state_values);
            depth.backFaceStencil = record.two_sided == SCE_GXM_TWO_SIDED_DISABLED ? depth.frontFaceStencil : stencil_desc(record.back_stencil_state_op, record.back_stencil_state_values);
        }
        [encoder setDepthStencilState:[impl->device->native_device() newDepthStencilStateWithDescriptor:depth]];
        [encoder setStencilFrontReferenceValue:record.front_stencil_state_values.ref backReferenceValue:record.two_sided == SCE_GXM_TWO_SIDED_DISABLED ? record.front_stencil_state_values.ref : record.back_stencil_state_values.ref];
        [encoder setDepthBias:record.depth_bias_unit slopeScale:record.depth_bias_slope clamp:0];
        // Use the same transformed front face as GL/Vulkan. Swapping both
        // winding and cull mode preserves which triangles disappear, but
        // reverses front_facing and the per-face fragment/stencil state.
        [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
        [encoder setCullMode:record.cull_mode == SCE_GXM_CULL_NONE ? MTLCullModeNone : (record.cull_mode == SCE_GXM_CULL_CW ? MTLCullModeBack : MTLCullModeFront)];
        // Polygon rasterization mode affects triangles, not native point/line
        // primitives. Sly has a vertex program with a point-size output; point
        // draws using it must not fail because a point polygon mode is set.
        const bool wireframe = record.front_polygon_mode == SCE_GXM_POLYGON_MODE_LINE
            || record.front_polygon_mode == SCE_GXM_POLYGON_MODE_TRIANGLE_LINE;
        if (triangles && record.front_polygon_mode != SCE_GXM_POLYGON_MODE_TRIANGLE_FILL && !wireframe)
            throw std::runtime_error(fmt::format("Metal: triangle polygon mode {:#x} requires conversion (primitive={:#x}, vertex={}, fragment={})",
                uint32_t(record.front_polygon_mode), uint32_t(primitive),
                hex_string(vp->renderer_data->hash), hex_string(fp->renderer_data->hash)));
        [encoder setTriangleFillMode:wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
        const auto &v = ctx.viewport;
        MTLViewport viewport = record.viewport_flat ? MTLViewport{0, 0, double(ctx.impl->width), double(ctx.impl->height), 0, 1}
            : MTLViewport{v[0] * res_multiplier, v[1] * res_multiplier, v[2] * res_multiplier, v[3] * res_multiplier, 0, 1};
        [encoder setViewport:viewport];
        if (impl->trace_draws && impl->traced_draws.size() < 4096) {
            const auto detail = fmt::format("vertex={} fragment={} color={:#x} fmt={:#x} size={}x{} primitive={:#x} count={} instances={} mask={} viewport={},{},{},{} clips={} cull={} depth_enabled={} depth_func={} depth_write={} blend={},{},{},{},{},{} color_mask={:#x}",
                hex_string(vp->renderer_data->hash), hex_string(fp->renderer_data->hash),
                uint32_t(record.color_surface.data.address()), uint32_t(record.color_surface.colorFormat),
                ctx.impl->width, ctx.impl->height, uint32_t(primitive), count, instances, record.is_maskupdate,
                viewport.originX, viewport.originY, viewport.width, viewport.height, clip.size(),
                uint32_t(record.cull_mode), depth_enabled, uint32_t(depth_func), uint32_t(depth_write_mode),
                uint32_t(blend.colorFunc), uint32_t(blend.alphaFunc), uint32_t(blend.colorSrc), uint32_t(blend.colorDst),
                uint32_t(blend.alphaSrc), uint32_t(blend.alphaDst), uint32_t(blend.colorMask));
            if (impl->traced_draws.insert(detail).second) LOG_INFO("Metal draw binding: {}", detail);
        }
        if (capture_draw) draw_metadata << "viewport " << viewport.originX << ' ' << viewport.originY << ' '
            << viewport.width << ' ' << viewport.height << ' ' << viewport.znear << ' ' << viewport.zfar << '\n';
        // MSL constant structures round their size to the largest member alignment.
        // Keep padding local to Metal so shared GL/Vulkan uniform layouts are untouched.
        shader::RenderVertUniformBlockExtended vertex_info{};
        shader::RenderFragUniformBlockExtended fragment_info{};
        auto &vert = vertex_info.base_block;
        vert.viewport_flip = {record.viewport_flat ? 1.0f : record.viewport_flip[0], record.viewport_flat ? -1.0f : -record.viewport_flip[1], 1, 1};
        vert.viewport_flag = record.viewport_flat ? 0 : 1;
        vert.screen_width = ctx.impl->width / res_multiplier; vert.screen_height = ctx.impl->height / res_multiplier;
        vert.z_offset = record.z_offset; vert.z_scale = record.z_scale;
        auto &frag = fragment_info.base_block;
        frag.res_multiplier = res_multiplier; frag.writing_mask = record.writing_mask;
        frag.front_disabled = record.front_side_fragment_program_mode == SCE_GXM_FRAGMENT_PROGRAM_DISABLED;
        frag.back_disabled = record.two_sided == SCE_GXM_TWO_SIDED_DISABLED ? frag.front_disabled : record.back_side_fragment_program_mode == SCE_GXM_FRAGMENT_PROGRAM_DISABLED;
        std::vector<GuestBufferRange> guest_ranges;
        const ShaderProgram *programs[] = {vp->renderer_data.get(), fp->renderer_data.get()};
        for (uint32_t stage = 0; stage < 2; ++stage) {
            if (stage == 1 && !fragment_resources) continue;
            const auto &program = *programs[stage];
            for (uint32_t block = 0; block < program.buffer_count; ++block) {
                if (!program.uniform_buffer_sizes.at(block)) continue;
                const auto &binding = ctx.uniforms[stage].at(block);
                require(binding.data && binding.size, "Metal: shader uses an unbound uniform buffer");
                guest_ranges.push_back({binding.data, binding.size});
            }
        }
        GuestBufferBindings native_buffers(*impl->device, guest_ranges, mem.host_page_size, !synchronous, &ctx.impl->uploads);
        size_t draw_upload_bytes=native_buffers.allocated_bytes();
        native_buffers.make_resident(encoder);
        vertex_info.set_buffer_count(vp->renderer_data->buffer_count);
        fragment_info.set_buffer_count(fragment_resources ? fp->renderer_data->buffer_count : 0);
        vertex_info.set_texture_count(vp->renderer_data->texture_count);
        fragment_info.set_texture_count(fragment_resources ? fp->renderer_data->texture_count : 0);
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
                if (capture_draw && stage == 0) {
                    dump_bytes(fmt::format("uniform-{}.bin",block),binding.data,binding.size);
                    draw_metadata << "uniform " << block << ' ' << binding.size << '\n';
                } else if (capture_draw) {
                    dump_bytes(fmt::format("fragment-uniform-{}.bin",block),binding.data,binding.size);
                    texture_metadata << "fragment_uniform " << block << ' ' << binding.size << '\n';
                }
            }
        }
        if (capture_draw) {
            dump_bytes("vertex-info.bin",&vertex_info.base_block,sizeof(vertex_info.base_block));
            draw_metadata << "buffer_count " << vertex_info.buffer_count << '\n';
        }
        [encoder setFragmentTexture:record.is_maskupdate ? nil : ctx.impl->mask atIndex:shader::metal::MASK_TEXTURE];
        for (uint32_t stream = 0; stream < vp->streams.size(); ++stream) if (streams_used & (1u << stream)) {
            const auto &data = record.vertex_streams[stream];
            require(data.data && data.size, "Metal: empty vertex stream");
            const auto stride = vp->streams[stream].stride;
            const auto aligned = align(stride, 4);
            std::vector<uint8_t> packed;
            const void *bytes = data.data.get(mem); size_t length = data.size;
            if (stride && stride != aligned) {
                const size_t elements = (data.size + stride - 1) / stride;
                packed.resize(elements * aligned);
                for (size_t i = 0; i < elements; ++i) std::memcpy(packed.data() + i * aligned, data.data.get(mem) + i * stride, std::min<size_t>(stride, data.size - i * stride));
                bytes = packed.data(); length = packed.size();
            }
            const auto buffer = ctx.impl->uploads.allocate(*impl->device, length);
            std::memcpy(static_cast<uint8_t *>(buffer.buffer.contents) + buffer.offset, bytes, length);
            draw_upload_bytes+=length;
            [encoder setVertexBuffer:buffer.buffer offset:buffer.offset atIndex:shader::metal::VERTEX_STREAM_BUFFER_BASE + stream];
            if (capture_draw) {
                dump_bytes(fmt::format("stream-{}.bin",shader::metal::VERTEX_STREAM_BUFFER_BASE+stream),bytes,length);
                draw_metadata << "stream " << shader::metal::VERTEX_STREAM_BUFFER_BASE+stream << ' ' << length << '\n';
            }
        }
        for (uint32_t index = 0; index < SCE_GXM_MAX_TEXTURE_UNITS * 2; ++index) {
            const bool vertex = index >= SCE_GXM_MAX_TEXTURE_UNITS;
            if (!fragment_resources && !vertex) continue;
            const uint32_t slot = index % SCE_GXM_MAX_TEXTURE_UNITS;
            const auto &program = vertex ? static_cast<ShaderProgram &>(*vp->renderer_data) : static_cast<ShaderProgram &>(*fp->renderer_data);
            if (!program.textures_used[slot]) continue;
            const auto &texture = ctx.textures[index];
            const Address texture_address = texture.data_addr << 2;
            const auto texture_base = gxm::get_base_format(gxm::get_format(texture));
            const bool prepared = prepared_images[index] != nil;
            auto surface = prepared ? impl->surfaces.end() : impl->surfaces.find(texture_address);
            const auto subrectangle_surface=prepared ? impl->surfaces.end() : find_subrectangle(texture);
            if (subrectangle_surface!=impl->surfaces.end()) surface=subrectangle_surface;
            // Uncharted's signed normal/gloss alias starts at the second word
            // of its interleaved RG32 color/normal target.
            if (!prepared && surface == impl->surfaces.end() && texture_address >= 4
                && (texture_base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                    || texture_base == SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8)) {
                auto previous = impl->surfaces.find(texture_address - 4);
                if (previous != impl->surfaces.end() && previous->second.color.pixelFormat == MTLPixelFormatRG32Float)
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
            if (surface != impl->surfaces.end()) {
                native=surface->second.color;
                if (const auto rect=surface_subrectangle(surface->second.guest,texture)) {
                    auto crop=surface->second.subrectangles.find({rect->x,rect->y,rect->width,rect->height});
                    if (crop!=surface->second.subrectangles.end()) native=crop->second;
                }
            }
            else if (!native) {
                const auto base = gxm::get_base_format(gxm::get_format(texture));
                if (base == SCE_GXM_TEXTURE_BASE_FORMAT_X8U24 || base == SCE_GXM_TEXTURE_BASE_FORMAT_F32
                    || base == SCE_GXM_TEXTURE_BASE_FORMAT_U16) {
                    for (auto &[key, entry] : impl->depth_surfaces) {
                        const auto rect=depth_subrectangle(entry.guest,entry.width,entry.height,entry.multisample,texture);
                        if (!rect) continue;
                        if (!entry.snapshot) {
                            if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                            entry.snapshot = impl->caster->depth_snapshot(entry.texture, base == SCE_GXM_TEXTURE_BASE_FORMAT_U16, uint32_t(res_multiplier));
                        }
                        // Publish all stored depth samples in the expanded guest sampling grid.
                        native = entry.snapshot;
                        const uint32_t memory_width=entry.width*(entry.multisample==SCE_GXM_MULTISAMPLE_4X ? 2 : 1);
                        const uint32_t memory_height=entry.height*(entry.multisample!=SCE_GXM_MULTISAMPLE_NONE ? 2 : 1);
                        if (rect->x || rect->y || rect->width!=memory_width || rect->height!=memory_height) {
                            auto &crop=entry.subrectangles[{rect->x,rect->y,rect->width,rect->height}];
                            if (!crop) crop=impl->caster->snapshot_subrectangle(entry.snapshot,memory_width,memory_height,*rect);
                            native=crop;
                        }
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
                        const size_t bytes = cube_texture_storage_size(upload);
                        const uint64_t end = uint64_t(texture_address)+bytes;
                        require(texture_address && end<=uint64_t(UINT32_MAX)-4095
                            && is_valid_addr_range(mem,texture_address,Address(end)),"Metal: texture subresources extend beyond mapped guest memory");
                        if (upload.texture_type()!=texture.texture_type())
                            LOG_INFO_ONCE("Metal: cube sampler resolves six swizzled faces at {:#x}, mip_count={}, storage_bytes={}",
                                texture_address,uint32_t(texture.mip_count),bytes);
                    }
                    texture_cache.cache_and_bind_image(upload, mem);
                    native = current_texture(texture_cache);
                }
            }
            require(native != nil, "Metal: missing sampled texture");
            if (native == ctx.impl->color && !record.is_maskupdate) {
                require(color_feedback != nil, "Metal: missing color feedback snapshot");
                native = color_feedback;
            }
            const SceGxmColorFormat *rendered_format = surface != impl->surfaces.end() ? &surface->second.guest.colorFormat : nullptr;
            if (surface != impl->surfaces.end() && native.pixelFormat == MTLPixelFormatRG32Float
                && (texture_base == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8
                    || texture_base == SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8)) {
                auto &entry = surface->second;
                const uint32_t word_offset = (texture_address - surface->first)/4;
                const bool signed_normalized = texture_base == SCE_GXM_TEXTURE_BASE_FORMAT_S8S8S8S8;
                require(gxm::get_width(texture) == entry.guest.width*2 && gxm::get_height(texture) == entry.guest.height
                    && entry.guest.surfaceType == SCE_GXM_COLOR_SURFACE_LINEAR
                    && (texture.texture_type() == SCE_GXM_TEXTURE_LINEAR || texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED)
                    && entry.guest.strideInPixels == entry.guest.width
                    && (texture.texture_type() == SCE_GXM_TEXTURE_LINEAR_STRIDED
                        ? gxm::get_stride_in_bytes(texture) : align(gxm::get_width(texture), 8)*4) == entry.guest.strideInPixels*8
                    && res_multiplier >= 1 && std::floor(res_multiplier) == res_multiplier,
                    "Metal: RG32/RGBA8 alias requires equal row size and integer resolution scale");
                auto &cast = entry.rgba8_casts[{word_offset, signed_normalized}];
                if (!cast) {
                    if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                    cast = impl->caster->rgba8_from_rg32(native, uint32_t(res_multiplier),
                        (entry.guest.colorFormat & SCE_GXM_COLOR_SWIZZLE_MASK) == SCE_GXM_COLOR_SWIZZLE2_RG,
                        word_offset, signed_normalized);
                }
                native = cast;
                // A packed two-word target has a half-texel bias in the guest
                // sampling grid. Preserve that word selection at native scale
                // while retaining each native pixel's independent data.
                const std::pair<float,float> offset{packed_alias_x_offset(res_multiplier,uint32_t(native.width)),0};
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
            bool cast_memory=false;
            if (rendered_format) {
                SceGxmTextureFormat source_format{};
                const bool same_format=gxm::convert_color_format_to_texture_format(*rendered_format,source_format)
                    && gxm::get_base_format(source_format)==texture_base;
                if (!same_format && surface_format_cast_supported(*rendered_format,texture_base)) {
                    const auto &entry = surface->second;
                    require(surface_subrectangle(surface->second.guest,texture).has_value(),
                        fmt::format("Metal: format alias has incompatible extent, stride or memory layout: color={:#x} fmt={:#x} size={}x{} stride={} type={} texture={:#x} fmt={:#x} size={}x{} stride={} type={:#x}",
                            uint32_t(entry.guest.data.address()),uint32_t(entry.guest.colorFormat),entry.guest.width,entry.guest.height,entry.guest.strideInPixels,uint32_t(entry.guest.surfaceType),
                            texture_address,uint32_t(gxm::get_format(texture)),gxm::get_width(texture),gxm::get_height(texture),texture.texture_type()==SCE_GXM_TEXTURE_LINEAR_STRIDED?gxm::get_stride_in_bytes(texture):0,uint32_t(texture.texture_type())));
                    if (!impl->caster) impl->caster=std::make_unique<SurfaceCaster>(*impl->device);
                    native=impl->caster->surface_format_cast(native,*rendered_format,texture_base);
                    require(native!=nil,"Metal: cannot reinterpret sampled surface format");
                    rendered_format=nullptr;cast_memory=true;
                }
            }
            if ((rendered_format || cast_memory) && (native.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB || texture.gamma_mode)) {
                if (!impl->caster) impl->caster = std::make_unique<SurfaceCaster>(*impl->device);
                // Decode guest memory channels before applying the texture's
                // swizzle, including color formats that relocate alpha.
                native = impl->caster->rgba8_surface_sampling(native, rendered_format ? *rendered_format : SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR, texture.gamma_mode);
                rendered_format = nullptr;
            }
            if (!prepared) native = sampling_view(native, gxm::get_format(texture),
                rendered_format);
            if (capture_draw) {
                const char *stage=vertex ? "vertex" : "fragment";
                const bool is_cube=native.textureType==MTLTextureTypeCube;
                const uint32_t faces=is_cube ? 6 : 1;
                const uint32_t levels=uint32_t(native.mipmapLevelCount);
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
            const auto linear_filter=[](uint32_t filter) { return filter==SCE_GXM_TEXTURE_FILTER_LINEAR || filter==SCE_GXM_TEXTURE_FILTER_MIPMAP_LINEAR; };
            auto &mip_info=texture_mip_info[index];
            mip_info.control[1]=uint32_t(linear_filter(texture.min_filter))
                |(uint32_t(linear_filter(texture.mag_filter))<<1)|(texture.mip_filter<<2)
                |(texture.uaddr_mode<<3)|(texture.vaddr_mode<<6)
                |((texture.lod_min0|(texture.lod_min1<<2))<<9);
            mip_info.control[2]=effective_sampler_anisotropy(texture,texture_cache.anisotropic_filtering);
            auto sampling = make_sampler(*impl->device, texture, texture_cache.anisotropic_filtering);
            if (vertex) { [encoder setVertexTexture:native atIndex:slot]; [encoder setVertexSamplerState:sampling atIndex:slot]; }
            else { [encoder setFragmentTexture:native atIndex:slot]; [encoder setFragmentSamplerState:sampling atIndex:slot]; }
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
        std::vector<uint32_t> fan_indices;
        size_t native_count = count;
        auto index_size = index_format == SCE_GXM_INDEX_FORMAT_U16 ? 2u : 4u;
        switch (primitive) {
        case SCE_GXM_PRIMITIVE_TRIANGLES: type = MTLPrimitiveTypeTriangle; break;
        case SCE_GXM_PRIMITIVE_TRIANGLE_STRIP: type = MTLPrimitiveTypeTriangleStrip; break;
        case SCE_GXM_PRIMITIVE_LINES: type = MTLPrimitiveTypeLine; break;
        case SCE_GXM_PRIMITIVE_POINTS: type = MTLPrimitiveTypePoint; break;
        case SCE_GXM_PRIMITIVE_TRIANGLE_FAN: {
            // Metal has no fan topology. Keep the original anchor and winding,
            // including full-width guest indices, in an equivalent triangle list.
            type = MTLPrimitiveTypeTriangle;
            native_count = size_t(count - 2) * 3;
            fan_indices.reserve(native_count);
            const auto at = [&](uint32_t i) -> uint32_t {
                return index_size == 2 ? static_cast<const uint16_t *>(indices)[i] : static_cast<const uint32_t *>(indices)[i];
            };
            const auto anchor = at(0);
            for (uint32_t i = 2; i < count; ++i) {
                fan_indices.push_back(anchor);
                fan_indices.push_back(at(i - 1));
                fan_indices.push_back(at(i));
            }
            indices = fan_indices.data();
            index_size = 4;
            break;
        }
        default: throw std::runtime_error("Metal: primitive conversion required");
        }
        const auto index_buffer = ctx.impl->uploads.allocate(*impl->device, native_count * index_size);
        std::memcpy(static_cast<uint8_t *>(index_buffer.buffer.contents) + index_buffer.offset, indices, native_count * index_size);
        if (capture_draw) {
            dump_bytes("indices.bin",indices,native_count*index_size);
            draw_metadata << "draw " << type << ' ' << index_size << ' ' << native_count << ' ' << instances << '\n';
            require(bool(draw_metadata) && bool(texture_metadata),"Metal: cannot write draw metadata");
            impl->draw_dumped = true;
            LOG_INFO("Metal vertex draw saved: {}",impl->dump_draw_dir.string());
        }
        for (const auto &rect : clip) {
            [encoder setScissorRect:rect];
            [encoder drawIndexedPrimitives:type indexCount:native_count indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32 indexBuffer:index_buffer.buffer indexBufferOffset:index_buffer.offset instanceCount:instances];
        }
        // Read-only shaders own snapshots retained by the command buffer. Guest
        // memory writers still finish before CPU processing can observe/reuse it.
        // Limit retained resources even when a guest emits an unusually long scene.
        ++ctx.impl->pending_draws;
        ctx.impl->pending_upload_bytes+=draw_upload_bytes+native_count*index_size;
        if (!record.is_maskupdate && ctx.impl->guest_color.data) {
            auto surface=impl->surfaces.find(ctx.impl->guest_color.data.address());
            if (surface!=impl->surfaces.end()) { ++surface->second.revision; surface->second.rgba8_casts.clear(); surface->second.subrectangles.clear(); }
        }
        if(synchronous || ctx.impl->pending_draws>=64 || ctx.impl->pending_upload_bytes>=32*1024*1024)
            finish(ctx);
        if (capture_draw && impl->dump_attachments) {
            finish(ctx);
            dump_attachment("color-after",ctx.impl->color);
        }
    }
}

std::vector<uint32_t> MetalState::dump_frame(DisplayState &display, uint32_t &width, uint32_t &height) {
    if (context) finish(*static_cast<MetalContext *>(context));
    DisplayFrameInfo next;
    { std::lock_guard lock(display.display_info_mutex); next = display.next_rendered_frame; }
    auto it = impl->surfaces.find(next.base.address());
    if (it == impl->surfaces.end() || (it->second.color.pixelFormat != MTLPixelFormatRGBA8Unorm
        && it->second.color.pixelFormat != MTLPixelFormatRGBA8Unorm_sRGB
        && it->second.color.pixelFormat != MTLPixelFormatRGBA16Float)) {
        width = height = 0; return {};
    }
    id<MTLTexture> color = it->second.color;
    width = color.width; height = color.height;
    std::vector<uint32_t> result(size_t(width) * height);
    if (color.pixelFormat == MTLPixelFormatRGBA16Float) {
        std::vector<__fp16> pixels(size_t(width) * height * 4);
        [color getBytes:pixels.data() bytesPerRow:width * 8 fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0];
        auto *bytes = reinterpret_cast<uint8_t *>(result.data());
        for (size_t i = 0; i < pixels.size(); ++i) {
            const float value = pixels[i];
            bytes[i] = std::isnan(value) ? 0 : uint8_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
        }
    } else {
        [color getBytes:result.data() bytesPerRow:width * 4 fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0];
    }
    return result;
}
void MetalState::render_frame(DisplayState &display, const GxmState &, MemState &mem) {
    @autoreleasepool {
        should_display = false;
        if (context) finish(*static_cast<MetalContext *>(context));
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
        id<MTLTexture> source = nil;
        auto it = impl->surfaces.find(next.base.address());
        if (it != impl->surfaces.end()) source = it->second.color;
        else if (has_frame) {
            source = make_texture(*impl->device, MTLPixelFormatRGBA8Unorm, next.image_size.x, next.image_size.y, MTLTextureUsageShaderRead);
            [source replaceRegion:MTLRegionMake2D(0, 0, next.image_size.x, next.image_size.y) mipmapLevel:0 withBytes:next.base.get(mem) bytesPerRow:next.pitch * 4];
        }
        // The UNORM drawable consumes already-encoded framebuffer bytes.
        if (source && source.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB) source = rgba8_gamma_view(source,false);
        if (!impl->screen) set_screen_filter("Bilinear");
        impl->drawable = [impl->layer nextDrawable];
        if (!impl->drawable) return;
        impl->screen_commands = [impl->device->command_queue() commandBuffer];
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = impl->drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,1);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;

        auto encoder = [impl->screen_commands renderCommandEncoderWithDescriptor:pass];
        double vw = width, vh = height;
        if (!stretch_the_display_area) {
            const double ratio = has_frame ? double(next.image_size.x) / next.image_size.y : 960.0 / 544.0;
            if (vw / vh > ratio) vw = vh * ratio; else vh = vw / ratio;
        }
        const double x = (width-vw)/2, y = (height-vh)/2;
        display.viewport_x = x; display.viewport_y = y; display.viewport_w = vw; display.viewport_h = vh;
        display.viewport_drawable_w = width; display.viewport_drawable_h = height;
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
