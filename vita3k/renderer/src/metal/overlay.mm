// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// Copyright RPCS3
// SPDX-License-Identifier: GPL-2.0-or-later
// Overlay constants/effect mapping adapted from the existing Vita3K overlay renderer.
#include <renderer/metal/overlay.h>
#include <overlay/display_manager.h>
#include <overlay/controls.h>
#include <overlay/font.h>
#include <overlay/shader_precompile_progress.h>
#include <util/log.h>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace renderer::metal {
namespace {
struct alignas(16) Constants {
    float ui_scale[4], albedo[4], viewport[4], clip_bounds[4];
    uint32_t vertex_config, fragment_config;
    float timestamp, blur_intensity;
    float sdf_params[4], sdf_origin[4], sdf_border_color[4];
};
static_assert(sizeof(Constants) == 128);
std::vector<uint32_t> read_shader(const std::filesystem::path &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto size = f.tellg();
    if (!f || size < 20 || size > 1024 * 1024 || size % 4)
        throw std::runtime_error("Metal: invalid overlay shader " + path.string());
    std::vector<uint32_t> words(size / 4);
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(words.data()), size))
        throw std::runtime_error("Metal: cannot read overlay shader");
    return words;
}
}
struct OverlayRenderer::Impl {
    Device &device;
    id<MTLRenderPipelineState> pipeline;
    id<MTLSamplerState> sampler;
    id<MTLTexture> white, white_array;
    std::map<const overlay::font *, id<MTLTexture>> fonts;
    struct CachedImage {
        int width = 0, height = 0, channels = 0;
        std::vector<uint8_t> pixels;
        id<MTLTexture> texture = nil;
        uint64_t last_used = 0;
    };
    std::map<const overlay::image_info_base *, CachedImage> images;
    uint64_t image_frame = 0;
    overlay::resource_config resources;
    bool resources_loaded = false;
    size_t last_view_count = SIZE_MAX, last_command_count = SIZE_MAX, last_nontransparent_count = SIZE_MAX;
    explicit Impl(Device &device) : device(device) {}
    id<MTLTexture> texture(MTLPixelFormat format, uint32_t w, uint32_t h, uint32_t layers = 0) {
        auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
        desc.storageMode = MTLStorageModeShared; desc.usage = MTLTextureUsageShaderRead;
        if (layers) { desc.textureType = MTLTextureType2DArray; desc.arrayLength = layers; }
        auto image = [device.native_device() newTextureWithDescriptor:desc];
        if (!image) throw std::runtime_error("Metal: overlay texture allocation failed");
        return image;
    }
    id<MTLTexture> image(const overlay::image_info_base *info) {
        if (!info || !info->get_data() || info->w <= 0 || info->h <= 0) return white;
        if (info->channels != 1 && info->channels != 4)
            throw std::runtime_error("Metal: invalid overlay image channels");
        const size_t bytes = size_t(info->w) * info->h * info->channels;
        const uint8_t *data = info->get_data();
        auto &cached = images[info];
        cached.last_used = image_frame;
        if (cached.texture && cached.width == info->w && cached.height == info->h
            && cached.channels == info->channels && cached.pixels.size() == bytes
            && std::memcmp(cached.pixels.data(), data, bytes) == 0) {
            info->dirty = false;
            return cached.texture;
        }
        // A changed image gets a new resource: an older command buffer may
        // still be sampling the previous texture asynchronously.
        auto result = texture(info->channels == 4 ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatR8Unorm, info->w, info->h);
        [result replaceRegion:MTLRegionMake2D(0,0,info->w,info->h) mipmapLevel:0 withBytes:data bytesPerRow:info->w * info->channels];
        cached.width = info->w;
        cached.height = info->h;
        cached.channels = info->channels;
        cached.pixels.assign(data, data + bytes);
        cached.texture = result;
        info->dirty = false;
        return result;
    }
    id<MTLTexture> font(const overlay::font *font) {
        const auto dims = font->get_glyph_data_dimensions();
        if (!dims.depth) return white_array;
        auto &cached = fonts[font];
        if (cached && cached.arrayLength == dims.depth) return cached;
        const auto &data = font->get_glyph_data();
        const size_t page = size_t(dims.width) * dims.height;
        if (data.size() < page * dims.depth) throw std::runtime_error("Metal: truncated font atlas");
        cached = texture(MTLPixelFormatR8Unorm, dims.width, dims.height, dims.depth);
        for (uint32_t layer = 0; layer < dims.depth; ++layer)
            [cached replaceRegion:MTLRegionMake2D(0,0,dims.width,dims.height) mipmapLevel:0 slice:layer withBytes:data.data() + layer*page bytesPerRow:dims.width bytesPerImage:page];
        return cached;
    }
    void draw(id<MTLRenderCommandEncoder> encoder, const overlay::compiled_resource::command &draw_cmd, MTLViewport viewport) {
        if (draw_cmd.verts.empty()) return;
        const auto &config = draw_cmd.config;
        const float viewport_w = viewport.width, viewport_h = viewport.height, viewport_x = viewport.originX, viewport_y = viewport.originY;
        id<MTLTexture> image2d = white, atlas = white_array;
        if (config.font_ref) atlas = font(config.font_ref);
        else if (config.texture_ref == overlay::raw_image && config.external_data_ref)
            image2d = image(static_cast<const overlay::image_info_base *>(config.external_data_ref));
        else if (config.texture_ref > 0 && config.texture_ref < overlay::raw_image) {
            if (!resources_loaded) { resources.load_files(); resources_loaded = true; }
            const size_t index = config.texture_ref - 1;
            if (index < resources.texture_raw_data.size()) image2d = image(resources.texture_raw_data[index].get());
        }
        [encoder setFragmentTexture:image2d atIndex:0];
        [encoder setFragmentTexture:atlas atIndex:1];
    Constants pc{};

    pc.ui_scale[0] = 960.f;
    pc.ui_scale[1] = 544.f;
    pc.ui_scale[2] = 1.f;
    pc.ui_scale[3] = 1.f;

    pc.albedo[0] = config.color.r;
    pc.albedo[1] = config.color.g;
    pc.albedo[2] = config.color.b;
    pc.albedo[3] = config.color.a;

    pc.viewport[0] = viewport_w;
    pc.viewport[1] = viewport_h;
    pc.viewport[2] = viewport_x;
    pc.viewport[3] = viewport_y;

    pc.clip_bounds[0] = config.clip_rect.x1;
    pc.clip_bounds[1] = config.clip_rect.y1;
    pc.clip_bounds[2] = config.clip_rect.x2;
    pc.clip_bounds[3] = config.clip_rect.y2;

    uint32_t vert_cfg = 0;
    if (config.disable_vertex_snap)
        vert_cfg |= 1u;
    pc.vertex_config = vert_cfg;

    uint32_t frag_cfg = 0;
    if (config.clip_region)
        frag_cfg |= 1u;
    if (config.pulse_glow)
        frag_cfg |= 2u;

    uint32_t sampler_mode = 0;
    if (config.font_ref) {
        sampler_mode = 2;
    } else if (config.texture_ref == overlay::font_file) {
        sampler_mode = 1;
    } else if (config.texture_ref != overlay::image_resource_none
        && config.texture_ref != overlay::game_icon
        && config.texture_ref != overlay::backbuffer) {
        sampler_mode = 3;
    }
    frag_cfg |= (sampler_mode & 3u) << 2u;

    const bool is_sdf = config.active_effect == overlay::compiled_resource::effect_type::sdf
        && config.effect.sdf.func != overlay::sdf_function::none;
    const bool is_gloss = config.active_effect == overlay::compiled_resource::effect_type::gloss;
    const bool is_btn_gloss = config.active_effect == overlay::compiled_resource::effect_type::btn_gloss;
    const bool has_sdf_btn_gloss = is_sdf && config.effect.sdf.btn_gloss_height > 0.f;

    uint32_t sdf_type = is_sdf ? static_cast<uint32_t>(config.effect.sdf.func) : 0u;
    frag_cfg |= (sdf_type & 3u) << 4u;
    if (is_gloss)
        frag_cfg |= 64u;
    if (is_btn_gloss || has_sdf_btn_gloss)
        frag_cfg |= 128u;
    pc.fragment_config = frag_cfg;

    pc.timestamp = config.get_sinus_value();

    pc.blur_intensity = static_cast<float>(config.blur_strength);

    // SDF parameters - transform from virtual space to viewport pixel space
    if (is_sdf) {
        auto sdf = config.effect.sdf;
        // Metal window coordinates have a downward Y axis.
        overlay::areaf target_vp;
        target_vp.x1 = viewport_x;
        target_vp.y1 = viewport_y;
        target_vp.x2 = viewport_x + viewport_w;
        target_vp.y2 = viewport_y + viewport_h;
        sdf.transform(target_vp, { 960.f, 544.f });

        pc.sdf_params[0] = sdf.hx;
        pc.sdf_params[1] = sdf.hy;
        pc.sdf_params[2] = sdf.br;
        pc.sdf_params[3] = sdf.bw;
        pc.sdf_origin[0] = sdf.cx;
        pc.sdf_origin[1] = sdf.cy;
        // Pack btn_gloss params into sdf_origin[2..3] when both SDF and btn_gloss active
        if (has_sdf_btn_gloss) {
            pc.sdf_origin[2] = sdf.btn_gloss_height;
            pc.sdf_origin[3] = std::round(sdf.btn_gloss_opacity * 100.f) + sdf.btn_gloss_bottom_opacity;
        }
        pc.sdf_border_color[0] = sdf.border_color.r;
        pc.sdf_border_color[1] = sdf.border_color.g;
        pc.sdf_border_color[2] = sdf.border_color.b;
        pc.sdf_border_color[3] = sdf.border_color.a;
    } else if (is_gloss) {
        const auto &g = config.effect.gloss;
        pc.sdf_params[0] = g.height;
        pc.sdf_params[1] = g.feather;
        pc.sdf_params[2] = g.opacity;
    } else if (is_btn_gloss) {
        const auto &bg = config.effect.btn_gloss;
        pc.sdf_params[0] = bg.height;
        pc.sdf_params[1] = bg.curve_lift;
        pc.sdf_params[2] = bg.opacity;
        pc.sdf_params[3] = bg.border_radius_frac;
        pc.sdf_origin[0] = bg.aspect;
        pc.sdf_origin[1] = bg.bottom_opacity;
    }


        [encoder setVertexBytes:&pc length:sizeof(pc) atIndex:0];
        [encoder setFragmentBytes:&pc length:sizeof(pc) atIndex:0];
        std::vector<overlay::vertex> converted;
        const auto *vertices = &draw_cmd.verts;
        MTLPrimitiveType type = MTLPrimitiveTypeTriangleStrip;
        const auto count = draw_cmd.verts.size();
        switch (config.primitives) {
        case overlay::primitive_type::quad_list: break;
        case overlay::primitive_type::triangle_strip: break;
        case overlay::primitive_type::line_list: type = MTLPrimitiveTypeLine; break;
        case overlay::primitive_type::line_strip: type = MTLPrimitiveTypeLineStrip; break;
        case overlay::primitive_type::triangle_fan:
            type = MTLPrimitiveTypeTriangle;
            for (size_t i = 2; i < count; ++i) {
                converted.push_back(draw_cmd.verts[0]); converted.push_back(draw_cmd.verts[i-1]); converted.push_back(draw_cmd.verts[i]);
            }
            vertices = &converted; break;
        }
        if (vertices->empty()) return;
        const size_t vertex_bytes = vertices->size() * sizeof(overlay::vertex);
        if (vertex_bytes <= 4096) {
            // Most UI commands are a single quad. Inline bytes avoid a new
            // MTLBuffer allocation and a resource lifetime per command.
            [encoder setVertexBytes:vertices->data() length:vertex_bytes atIndex:1];
        } else {
            auto buffer = [device.native_device() newBufferWithBytes:vertices->data()
                length:vertex_bytes options:MTLResourceStorageModeShared];
            if (!buffer) throw std::runtime_error("Metal: overlay vertex allocation failed");
            [encoder setVertexBuffer:buffer offset:0 atIndex:1];
        }
        if (config.primitives == overlay::primitive_type::quad_list) {
            for (size_t i = 0; i + 3 < count; i += 4) [encoder drawPrimitives:type vertexStart:i vertexCount:4];
        } else [encoder drawPrimitives:type vertexStart:0 vertexCount:vertices->size()];
    }
};
OverlayRenderer::OverlayRenderer(Device &device, const std::filesystem::path &assets) : impl(std::make_unique<Impl>(device)) {
    std::string error;
    const auto path = assets / "shaders-builtin" / "overlay";
    auto vs = device.compile(shader::metal::convert_overlay_spirv(read_shader(path / "overlay.vert.spv")), false, error);
    if (!vs) throw std::runtime_error(error);
    auto fs = device.compile(shader::metal::convert_overlay_spirv(read_shader(path / "overlay.frag.spv")), false, error);
    if (!fs) throw std::runtime_error(error);
    auto desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = vs->function; desc.fragmentFunction = fs->function;
    auto layout = [MTLVertexDescriptor vertexDescriptor];
    layout.attributes[0].format = MTLVertexFormatFloat4; layout.attributes[0].bufferIndex = 1;
    layout.layouts[1].stride = sizeof(overlay::vertex); desc.vertexDescriptor = layout;
    auto color = desc.colorAttachments[0]; color.pixelFormat = MTLPixelFormatBGRA8Unorm;
    color.blendingEnabled = YES;
    color.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha; color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    color.sourceAlphaBlendFactor = MTLBlendFactorZero; color.destinationAlphaBlendFactor = MTLBlendFactorOne;
    impl->pipeline = device.create_pipeline(desc, error);
    if (!impl->pipeline) throw std::runtime_error(error);
    auto sampling = [MTLSamplerDescriptor new];
    sampling.minFilter = sampling.magFilter = MTLSamplerMinMagFilterLinear;
    sampling.sAddressMode = sampling.tAddressMode = MTLSamplerAddressModeClampToEdge;
    impl->sampler = [device.native_device() newSamplerStateWithDescriptor:sampling];
    impl->white = impl->texture(MTLPixelFormatRGBA8Unorm,1,1);
    impl->white_array = impl->texture(MTLPixelFormatR8Unorm,1,1,1);
    const uint32_t pixel = 0xffffffff;
    [impl->white replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&pixel bytesPerRow:4];
    [impl->white_array replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 slice:0 withBytes:&pixel bytesPerRow:1 bytesPerImage:1];
}
OverlayRenderer::~OverlayRenderer() = default;
void OverlayRenderer::render(id<MTLRenderCommandEncoder> encoder, const overlay::display_manager &manager, MTLViewport viewport) {
    ++impl->image_frame;
    struct Prepared { std::shared_ptr<overlay::overlay> view; overlay::compiled_resource resource; };
    std::vector<Prepared> prepared;
    {
        std::shared_lock lock(manager);
        for (const auto &view : manager.get_views()) if (view && view->visible.load()) {
            view->set_render_viewport(std::min<double>(viewport.width, UINT16_MAX), std::min<double>(viewport.height, UINT16_MAX));
            prepared.push_back({view, view->get_compiled()});
        }
    }
    // A loading background can outlive precompilation until the first guest
    // framebuffer arrives. Dialogs raised before that frame must remain visible.
    std::stable_partition(prepared.begin(), prepared.end(), [](const Prepared &view) {
        return view.view->type_index == overlay::get_overlay_type_id<overlay::shader_precompile_progress>();
    });
    size_t command_count = 0, nontransparent_count = 0;
    for (const auto &view : prepared) {
        command_count += view.resource.draw_commands.size();
        for (const auto &command : view.resource.draw_commands)
            nontransparent_count += command.config.color.a > 0.01f && !command.verts.empty();
    }
    if (prepared.size() != impl->last_view_count || command_count != impl->last_command_count
        || nontransparent_count != impl->last_nontransparent_count) {
        LOG_TRACE("Metal overlays: {} visible views, {} draw commands, {} nontransparent draws", prepared.size(), command_count, nontransparent_count);
        impl->last_view_count = prepared.size();
        impl->last_command_count = command_count;
        impl->last_nontransparent_count = nontransparent_count;
    }
    [encoder setRenderPipelineState:impl->pipeline];
    [encoder setViewport:viewport];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setFragmentSamplerState:impl->sampler atIndex:0];
    [encoder setFragmentSamplerState:impl->sampler atIndex:1];
    for (auto &view : prepared) {
        for (const auto &command : view.resource.draw_commands) impl->draw(encoder, command, viewport);
        view.view->update(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }
    // Dialogs and loading backgrounds can disappear between frames. Bound
    // retained CPU copies and GPU textures to images used recently.
    for (auto image = impl->images.begin(); image != impl->images.end();) {
        if (impl->image_frame - image->second.last_used > 2)
            image = impl->images.erase(image);
        else ++image;
    }
}
} // namespace renderer::metal
