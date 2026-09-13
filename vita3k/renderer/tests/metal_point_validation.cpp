// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/state.h>
#include <renderer/functions.h>
#include <renderer/driver_functions.h>
#include <mem/state.h>
#include <config/state.h>
#include <display/state.h>
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
        std::cerr << "usage: metal-point-validation Sly-e007896-vertex.gxp Sly-028582-fragment.gxp\n";
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
        unsigned runs = 0;
        for (auto mode : {SCE_GXM_POLYGON_MODE_POINT, SCE_GXM_POLYGON_MODE_POINT_10UV, SCE_GXM_POLYGON_MODE_POINT_01UV,
                          SCE_GXM_POLYGON_MODE_TRIANGLE_POINT, SCE_GXM_POLYGON_MODE_LINE,
                          SCE_GXM_POLYGON_MODE_TRIANGLE_LINE, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL}) {
            std::memset(pixels.get(mem), 0, 64 * 64 * 4);
            ctx.record.front_polygon_mode = mode;
            state.set_context(ctx, mem);
            draw(SCE_GXM_PRIMITIVE_POINTS, 2);
            unsigned colored = 0;
            const uint8_t *p = (const uint8_t *)pixels.get(mem);
            for (unsigned y = 0; y < 64; ++y)
                for (unsigned x = 0; x < 64; ++x) {
                    bool expected = (x == 16 && y == 16) || (x == 48 && y == 48);
                    const int want[] = {64, 128, 191, 255};
                    for (unsigned k = 0; k < 4; ++k)
                        check(std::abs(int(p[(y * 64 + x) * 4 + k]) - (expected ? want[k] : 0)) <= 1,
                              "Point pixel mismatch x=" + std::to_string(x) + " y=" + std::to_string(y));
                    colored += p[(y * 64 + x) * 4 + 3] != 0;
                }
            check(colored == 2, "Point coverage mismatch");
            ++runs;
        }
        std::vector<uint8_t> line_reference, wireframe_reference;
        for (auto primitive : {SCE_GXM_PRIMITIVE_LINES, SCE_GXM_PRIMITIVE_TRIANGLES}) {
            for (auto mode : {SCE_GXM_POLYGON_MODE_TRIANGLE_LINE, SCE_GXM_POLYGON_MODE_LINE, SCE_GXM_POLYGON_MODE_POINT,
                              SCE_GXM_POLYGON_MODE_POINT_10UV, SCE_GXM_POLYGON_MODE_POINT_01UV,
                              SCE_GXM_POLYGON_MODE_TRIANGLE_POINT, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL}) {
                if (primitive == SCE_GXM_PRIMITIVE_TRIANGLES && mode != SCE_GXM_POLYGON_MODE_LINE &&
                    mode != SCE_GXM_POLYGON_MODE_TRIANGLE_LINE)
                    continue;
                std::memset(pixels.get(mem), 0, 64 * 64 * 4);
                ctx.record.front_polygon_mode = mode;
                state.set_context(ctx, mem);
                draw(primitive, primitive == SCE_GXM_PRIMITIVE_LINES ? 2 : 3);
                const auto *bytes = reinterpret_cast<const uint8_t *>(pixels.get(mem));
                auto &reference = primitive == SCE_GXM_PRIMITIVE_LINES ? line_reference : wireframe_reference;
                unsigned coverage = 0;
                for (unsigned i = 0; i < 64 * 64; ++i)
                    coverage += bytes[i * 4 + 3] != 0;
                check(coverage > 20 && coverage < 130, "Line coverage outside expected bounds");
                check(bytes[(24 * 64 + 40) * 4 + 3] == 0, "Wireframe unexpectedly filled the triangle interior");
                if (reference.empty())
                    reference.assign(bytes, bytes + 64 * 64 * 4);
                else
                    check(std::memcmp(reference.data(), bytes, reference.size()) == 0,
                          "Polygon mode changed native line coverage or wireframe alias output");
            }
        }
        std::cout << "PASS " << runs
                  << " native point draws, all seven polygon modes, original Sly point-size shader, exact two-pixel "
                     "coverage and guest readback; 7 native line draws and 2 identical triangle wireframe modes\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
