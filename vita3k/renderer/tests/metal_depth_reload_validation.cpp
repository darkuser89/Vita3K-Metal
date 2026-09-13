// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/state.h>
#include <display/state.h>
#include <renderer/functions.h>
#include <renderer/driver_functions.h>
#include <mem/state.h>
#include <config/state.h>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}
static Ptr<const SceGxmProgram> load(MemState &mem, const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(bool(file), "Cannot open GXP fixture");
    std::streamsize size = file.tellg();
    check(size >= sizeof(SceGxmProgram) && size < 64 * 1024 * 1024, "Invalid GXP fixture length");
    Ptr<SceGxmProgram> result(alloc(mem, size, "Metal GXP fixture"));
    check(bool(result), "Cannot allocate guest GXP");
    file.seekg(0);
    check(bool(file.read(reinterpret_cast<char *>(result.get(mem)), size)), "Cannot read GXP");
    check(result.get(mem)->magic == 0x00505847 && result.get(mem)->size <= size, "Invalid GXP fixture");
    return result.cast<const SceGxmProgram>();
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: metal-depth-reload-validation Sly-e007896-vertex.gxp Sly-028582-fragment.gxp\n";
        return 2;
    }
    try {
        MemState mem;
        check(init(mem, false), "Guest memory init failed");
        auto owner = std::make_unique<renderer::metal::MetalContext>();
        renderer::metal::MetalState state;
        check(state.init(), "Metal init failed");
        auto &ctx = *owner;
        state.context = &ctx;
        Config config;
        auto execute = [&](renderer::GXMState which, auto... args) {
            renderer::Command c{};
            renderer::CommandHelper w(&c);
            check(w.push(which) && (w.push(args) && ...), "State payload failed");
            renderer::CommandHelper r(&c);
            renderer::cmd_handle_set_state(state, mem, config, r, state.features, &ctx);
        };
        Ptr<SceGxmVertexProgram> va(alloc(mem, sizeof(SceGxmVertexProgram), "Point vertex program"));
        Ptr<SceGxmFragmentProgram> fa(alloc(mem, sizeof(SceGxmFragmentProgram), "Point fragment program"));
        auto *vp = new (va.get(mem)) SceGxmVertexProgram{};
        auto *fp = new (fa.get(mem)) SceGxmFragmentProgram{};
        vp->program = load(mem, argv[1]);
        fp->program = load(mem, argv[2]);
        for (unsigned i = 0; i < 3; ++i) {
            SceGxmVertexAttribute a{};
            a.streamIndex = 0;
            a.offset = i * 16;
            a.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
            a.componentCount = 4;
            a.regIndex = i * 4;
            vp->attributes.push_back(a);
        }
        vp->streams.push_back({48, SCE_GXM_INDEX_SOURCE_EACH_VERTEX_16BIT});
        check(renderer::create(vp->renderer_data, state, *vp->program.get(mem), state.gxp_ptr_map, vp->attributes),
              "Vertex creation");
        check(renderer::create(fp->renderer_data, state, *fp->program.get(mem), nullptr, state.gxp_ptr_map),
              "Fragment creation");
        renderer::metal::MetalRenderTarget target;
        target.width = target.height = 64;
        target.multisample_mode = SCE_GXM_MULTISAMPLE_NONE;
        ctx.current_render_target = &target;
        Ptr<uint32_t> pixels(alloc(mem, 64 * 64 * 4, "Point framebuffer"));
        ctx.record.color_surface = {};
        ctx.record.color_surface.data = pixels.cast<void>();
        ctx.record.color_surface.width = ctx.record.color_surface.height = ctx.record.color_surface.strideInPixels = 64;
        ctx.record.color_surface.colorFormat = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
        ctx.record.color_surface.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
        ctx.record.depth_stencil_surface.depth_data = Ptr<void>();
        float data[36] = {};
        for (unsigned i = 0; i < 3; ++i) {
            float center = i ? 48.5f : 16.5f;
            data[i * 12] = center / 32 - 1;
            data[i * 12 + 1] = (i == 2 ? 16.5f : center) / 32 - 1;
            data[i * 12 + 2] = .5;
            data[i * 12 + 3] = 1;
        }
        Ptr<float> vertices(alloc(mem, sizeof(data), "Point vertices"));
        std::memcpy(vertices.get(mem), data, sizeof(data));
        ctx.record.vertex_streams[0] = {vertices.cast<const uint8_t>(), sizeof(data)};
        float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        Ptr<float> matrix(alloc(mem, sizeof(identity), "Point transform"));
        std::memcpy(matrix.get(mem), identity, sizeof(identity));
        float rgba[] = {.25, .5, .75, 1};
        Ptr<float> color(alloc(mem, 16, "Point color"));
        std::memcpy(color.get(mem), rgba, 16);
        Ptr<uint16_t> indices(alloc(mem, 6, "Point indices"));
        indices.get(mem)[0] = 0;
        indices.get(mem)[1] = 1;
        indices.get(mem)[2] = 2;
        execute(renderer::GXMState::Program, va.cast<void>(), false);
        execute(renderer::GXMState::Program, fa.cast<void>(), true);
        execute(renderer::GXMState::Viewport, false, 32.f, 32.f, 0.f, 32.f, 32.f, 1.f);
        execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0), uint32_t(64), uint32_t(0),
                uint32_t(64));
        execute(renderer::GXMState::UniformBuffer, matrix.cast<uint8_t>(), true, 0, uint32_t(64));
        execute(renderer::GXMState::UniformBuffer, color.cast<uint8_t>(), false, 0, uint32_t(16));
        auto draw = [&](SceGxmPrimitiveType primitive, uint32_t count) {
            renderer::Command c{};
            renderer::CommandHelper w(&c);
            auto format = SCE_GXM_INDEX_FORMAT_U16;
            auto index_ptr = indices.cast<const void>();
            uint32_t instances = 1;
            w.push(primitive);
            w.push(format);
            w.push(index_ptr);
            w.push(count);
            w.push(instances);
            renderer::CommandHelper r(&c);
            renderer::cmd_handle_draw(state, mem, config, r, state.features, &ctx);
            state.finish(ctx);
        };
        // Nonconstant depth exposes both packed-depth rounding and the loss of
        // per-pixel depth when a scaled attachment is stored at guest resolution.
        const float positions[3][4] = {{-1, -1, .137f, 1}, {3, -1, .817f, 1}, {-1, 3, .537f, 1}};
        for (unsigned i = 0; i < 3; ++i)
            std::memcpy(vertices.get(mem) + i * 12, positions[i], 16);
        Ptr<void> depth_memory(alloc(mem, 64 * 64 * 4, "Depth reload guest memory"));
        DisplayState display;
        display.next_rendered_frame.base = pixels.cast<void>();
        auto count_pixels = [&](bool white) {
            uint32_t w = 0, h = 0;
            auto image = state.dump_frame(display, w, h);
            check(w == 64 * state.res_multiplier && h == w && image.size() == size_t(w) * h, "Capture extent");
            size_t bad = 0;
            const auto *bytes = reinterpret_cast<const uint8_t *>(image.data());
            for (size_t i = 0; i < image.size(); ++i) {
                bool mismatch = false;
                for (unsigned c = 0; c < 4; ++c)
                    mismatch |= bytes[i * 4 + c] != (white ? 255 : 0);
                bad += mismatch;
            }
            return bad;
        };
        size_t failures = 0;
        for (auto format : {SCE_GXM_DEPTH_STENCIL_FORMAT_D16, SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
                            SCE_GXM_DEPTH_STENCIL_FORMAT_DF32}) {
            for (unsigned scale : {1u, 2u, 3u}) {
                state.end_scene(ctx);
                state.res_multiplier = scale;
                auto &ds = ctx.record.depth_stencil_surface;
                ds = {};
                ds.set_format(format);
                ds.set_stride(64);
                ds.depth_data = depth_memory;
                ds.background_depth = 1;
                ds.force_store = 1;
                ctx.record.front_polygon_mode = SCE_GXM_POLYGON_MODE_TRIANGLE_FILL;
                ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
                ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_ENABLED;
                std::fill_n(color.get(mem), 4, 0.f);
                state.set_context(ctx, mem);
                draw(SCE_GXM_PRIMITIVE_TRIANGLES, 3);
                state.end_scene(ctx);
                ds.force_load = 1;
                ds.force_store = 0;
                ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_EQUAL;
                ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_DISABLED;
                std::fill_n(color.get(mem), 4, 1.f);
                state.set_context(ctx, mem);
                draw(SCE_GXM_PRIMITIVE_TRIANGLES, 3);
                auto bad = count_pixels(true);
                failures += bad;
                std::cout << "format=" << uint32_t(format) << " scale=" << scale
                          << " unchanged-memory EQUAL failures=" << bad << '\n';
                state.end_scene(ctx);
                // A CPU write must remain authoritative. Clear both attachments
                // with a depth-disabled black draw, then load the changed memory.
                ds.force_load = 0;
                ds.force_store = 1;
                ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
                ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_ENABLED;
                std::fill_n(color.get(mem), 4, 0.f);
                state.set_context(ctx, mem);
                draw(SCE_GXM_PRIMITIVE_TRIANGLES, 3);
                state.end_scene(ctx);
                std::memset(depth_memory.get(mem), 0, 64 * 64 * 4);
                ds.force_load = 1;
                ds.force_store = 0;
                ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_EQUAL;
                ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_DISABLED;
                std::fill_n(color.get(mem), 4, 1.f);
                state.set_context(ctx, mem);
                draw(SCE_GXM_PRIMITIVE_TRIANGLES, 3);
                auto cpu_bad = count_pixels(false);
                failures += cpu_bad;
                std::cout << "CPU-modified depth failures=" << cpu_bad << '\n';
                state.end_scene(ctx);
                // Unsaved native writes must not be mistaken for a published
                // image just because the old guest bytes remain unchanged.
                ds.force_load = 0;
                ds.force_store = 1;
                ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
                ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_ENABLED;
                std::fill_n(color.get(mem), 4, 0.f);
                state.set_context(ctx, mem);
                draw(SCE_GXM_PRIMITIVE_TRIANGLES, 3);
                state.end_scene(ctx);
                ds.force_load = 1;
                ds.force_store = 0;
                execute(renderer::GXMState::Viewport, false, 32.f, 32.f, .9f, 32.f, 32.f, 0.f);
                state.set_context(ctx, mem);
                draw(SCE_GXM_PRIMITIVE_TRIANGLES, 3);
                state.end_scene(ctx);
                ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_EQUAL;
                ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_DISABLED;
                std::fill_n(color.get(mem), 4, 1.f);
                state.set_context(ctx, mem);
                draw(SCE_GXM_PRIMITIVE_TRIANGLES, 3);
                auto unsaved_bad = count_pixels(false);
                failures += unsaved_bad;
                std::cout << "unsaved-native-depth failures=" << unsaved_bad << '\n';
                state.end_scene(ctx);
                execute(renderer::GXMState::Viewport, false, 32.f, 32.f, 0.f, 32.f, 32.f, 1.f);
            }
        }
        check(failures == 0, "Depth reload lost GPU precision or ignored CPU memory");
        std::cout << "PASS native depth reuse and CPU modification across three formats and three scales\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
