// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <cmath>
#include <renderer/metal/state.h>
#include <config/state.h>
#include <display/state.h>
#include <fstream>
#include <iostream>
#include <mem/state.h>
#include <renderer/driver_functions.h>
#include <renderer/functions.h>
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
        std::cerr << "Usage: metal-face-state-validation <Sly25188 vertex.gxp> <Sly028582 fragment.gxp> "
                     "\n";
        return 2;
    }
    try {
        MemState mem;
        check(init(mem, false), "Cannot initialize guest memory");
        renderer::metal::MetalContext ctx;
        renderer::metal::MetalState state;
        check(state.init(), "Cannot initialize Metal");
        state.context = &ctx;
        Config config;
        auto execute = [&](renderer::GXMState code, auto... args) {
            renderer::Command cmd{};
            renderer::CommandHelper writer(&cmd);
            check(writer.push(code) && (writer.push(args) && ...), "State command overflow");
            renderer::CommandHelper reader(&cmd);
            renderer::cmd_handle_set_state(state, mem, config, reader, state.features, &ctx);
        };
        Ptr<SceGxmVertexProgram> va(alloc(mem, sizeof(SceGxmVertexProgram), "Disable vertex"));
        Ptr<SceGxmFragmentProgram> fa(alloc(mem, sizeof(SceGxmFragmentProgram), "Disable fragment"));
        auto *vp = new (va.get(mem)) SceGxmVertexProgram{};
        auto *fp = new (fa.get(mem)) SceGxmFragmentProgram{};
        vp->program = load(mem, argv[1]);
        fp->program = load(mem, argv[2]);
        SceGxmVertexAttribute attr{};
        attr.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
        attr.componentCount = 4;
        vp->attributes.push_back(attr);
        vp->streams.push_back({16, SCE_GXM_INDEX_SOURCE_EACH_VERTEX_16BIT});
        check(renderer::create(vp->renderer_data, state, *vp->program.get(mem), state.gxp_ptr_map, vp->attributes),
              "Vertex creation failed");
        check(renderer::create(fp->renderer_data, state, *fp->program.get(mem), nullptr, state.gxp_ptr_map),
              "Fragment creation failed");
        execute(renderer::GXMState::Program, va.cast<void>(), false);
        execute(renderer::GXMState::Program, fa.cast<void>(), true);
        float vertices[] = {-1, -1, 1, 1, 3, -1, 1, 1, -1, 3, 1, 1};
        Ptr<float> stream(alloc(mem, sizeof(vertices), "Disable vertices"));
        std::memcpy(stream.get(mem), vertices, sizeof(vertices));
        ctx.record.vertex_streams[0] = {stream.cast<const uint8_t>(), sizeof(vertices)};
        Ptr<float> uniform(alloc(mem, 16, "Disable color"));
        float rgba[] = {.25f, .5f, .75f, 1};
        std::memcpy(uniform.get(mem), rgba, 16);
        Ptr<uint16_t> indices(alloc(mem, 6, "Disable indices"));
        indices.get(mem)[0] = 0;
        auto draw_type = SCE_GXM_PRIMITIVE_TRIANGLES;
        auto draw = [&] { state.draw(ctx, mem, draw_type, SCE_GXM_INDEX_FORMAT_U16, indices.get(mem), 3, 1); };
        Ptr<uint32_t> notification(alloc(mem, 4, "Disable notification"));
        auto end = [&] {
            *notification.get(mem) = 0;
            renderer::Command cmd{};
            renderer::CommandHelper writer(&cmd);
            SceGxmNotification v{}, f{notification, 0x51a7u};
            writer.push(v);
            writer.push(f);
            renderer::CommandHelper reader(&cmd);
            renderer::cmd_handle_sync_surface_data(state, mem, config, reader, state.features, &ctx);
            check(*notification.get(mem) == 0x51a7u, "Missing EndScene notification");
        };
        renderer::metal::MetalRenderTarget target;
        ctx.current_render_target = &target;
        unsigned cases = 0;
        size_t values = 0;
        for (auto topology :
             {SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP, SCE_GXM_PRIMITIVE_TRIANGLE_FAN})
            for (unsigned scale : {1u, 2u})
                for (auto samples : {SCE_GXM_MULTISAMPLE_NONE, SCE_GXM_MULTISAMPLE_2X, SCE_GXM_MULTISAMPLE_4X}) {
                    draw_type = topology;
                    state.res_multiplier = scale;
                    target.width = target.height = 32 * scale;
                    target.multisample_mode = samples;
                    unsigned sx = samples == SCE_GXM_MULTISAMPLE_4X ? 2 : 1,
                             sy = samples == SCE_GXM_MULTISAMPLE_NONE ? 1 : 2;
                    auto &color = ctx.record.color_surface;
                    color = {};
                    color.width = color.height = color.strideInPixels = 32;
                    color.downscale = samples != SCE_GXM_MULTISAMPLE_NONE;
                    color.colorFormat = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
                    color.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
                    color.data = Ptr<void>(alloc(mem, 32 * 32 * 4, "Face color"));
                    auto &ds = ctx.record.depth_stencil_surface;
                    ds = {};
                    ds.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8);
                    ds.set_type(SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR);
                    ds.set_stride(32 * sx);
                    ds.force_store = 1;
                    ds.depth_data = Ptr<void>(alloc(mem, 32 * 32 * sx * sy * 4, "Face depth"));
                    ds.stencil_data = Ptr<void>(alloc(mem, 32 * 32 * sx * sy, "Face stencil"));
                    execute(renderer::GXMState::UniformBuffer, uniform.cast<uint8_t>(), false, 0, uint32_t(16));
                    execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0), uint32_t(32),
                            uint32_t(0), uint32_t(32));
                    execute(renderer::GXMState::TwoSided, SCE_GXM_TWO_SIDED_ENABLED);
                    for (float vx : {16.f, -16.f})
                        for (float vy : {16.f, -16.f})
                            for (bool reverse : {false, true}) {
                                indices.get(mem)[1] = reverse ? 2 : 1;
                                indices.get(mem)[2] = reverse ? 1 : 2;
                                execute(renderer::GXMState::Viewport, false, 16.f, 16.f, 0.f, vx, vy, .25f);
                                unsigned surviving_sides = 0;
                                for (auto cull : {SCE_GXM_CULL_CW, SCE_GXM_CULL_CCW}) {
                                    const bool back = cull == SCE_GXM_CULL_CCW;
                                    execute(renderer::GXMState::CullMode, cull);
                                    bool visible = false;
                                    for (unsigned variant = 0; variant < 5; variant++) {
                                        std::memset(color.data.get(mem), 0x35, 32 * 32 * 4);
                                        ds.background_depth = .75f;
                                        ds.stencil = 0x25;
                                        ctx.record.front_depth_func = ctx.record.back_depth_func =
                                            SCE_GXM_DEPTH_FUNC_ALWAYS;
                                        ctx.record.front_depth_write_mode = ctx.record.back_depth_write_mode =
                                            SCE_GXM_DEPTH_WRITE_ENABLED;
                                        execute(renderer::GXMState::FragmentProgramEnable, true,
                                                SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
                                        execute(renderer::GXMState::FragmentProgramEnable, false,
                                                SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
                                        for (bool front : {true, false})
                                            execute(renderer::GXMState::StencilFunc, front, SCE_GXM_STENCIL_FUNC_ALWAYS,
                                                    SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
                                                    SCE_GXM_STENCIL_OP_REPLACE, uint8_t(255), uint8_t(255));
                                        ctx.record.front_stencil_state_values.ref = 0x71;
                                        ctx.record.back_stencil_state_values.ref = 0xb2;
                                        auto &selected_func =
                                            back ? ctx.record.back_depth_func : ctx.record.front_depth_func;
                                        auto &unused_func =
                                            back ? ctx.record.front_depth_func : ctx.record.back_depth_func;
                                        auto &selected_write =
                                            back ? ctx.record.back_depth_write_mode : ctx.record.front_depth_write_mode;
                                        auto &unused_write =
                                            back ? ctx.record.front_depth_write_mode : ctx.record.back_depth_write_mode;
                                        if (variant == 1 || variant == 3 || variant == 4)
                                            unused_func = SCE_GXM_DEPTH_FUNC_NEVER;
                                        if (variant == 2)
                                            selected_func = SCE_GXM_DEPTH_FUNC_NEVER;
                                        if (variant == 3)
                                            selected_write = SCE_GXM_DEPTH_WRITE_DISABLED;
                                        if (variant == 1 || variant == 4)
                                            unused_write = SCE_GXM_DEPTH_WRITE_DISABLED;
                                        if (variant == 4)
                                            execute(renderer::GXMState::FragmentProgramEnable, !back,
                                                    SCE_GXM_FRAGMENT_PROGRAM_DISABLED);
                                        state.set_context(ctx, mem);
                                        draw();
                                        end();
                                        check(state.sync_surface(mem, color), "Face color readback failed");
                                        const auto *actual = static_cast<const uint8_t *>(color.data.get(mem));
                                        if (variant == 0) {
                                            visible = actual[0] != 0x35;
                                            surviving_sides += visible;
                                        }
                                        const bool passes = visible && variant != 2;
                                        const std::string detail =
                                            " case=" + std::to_string(cases) + " variant=" + std::to_string(variant) +
                                            " back=" + std::to_string(back) + " reverse=" + std::to_string(reverse) +
                                            " vx=" + std::to_string(vx) + " vy=" + std::to_string(vy);
                                        for (size_t i = 0; i < 32 * 32 * 4; i++) {
                                            int expected =
                                                passes && variant != 4 ? std::lround(rgba[i % 4] * 255) : 0x35;
                                            check(std::abs(int(actual[i]) - expected) <=
                                                      (passes && variant != 4 ? 1 : 0),
                                                  "Face color mismatch" + detail);
                                            ++values;
                                        }
                                        for (size_t i = 0; i < 32 * 32 * sx * sy; i++) {
                                            float expected = passes && variant != 3 ? .25f : .75f;
                                            check(static_cast<float *>(ds.depth_data.get(mem))[i] == expected,
                                                  "Wrong effective face depth" + detail);
                                            check(static_cast<uint8_t *>(ds.stencil_data.get(mem))[i] ==
                                                      (passes ? (back ? 0xb2 : 0x71) : 0x25),
                                                  "Wrong effective face stencil" + detail);
                                            values += 2;
                                        }
                                        ++cases;
                                    }
                                }
                                check(surviving_sides == 1, "Cull calibration did not select exactly one face");
                            }
                    // Depth state is irrelevant in a mask update and with a disabled
                    // depth surface, even with both face windings potentially present.
                    execute(renderer::GXMState::CullMode, SCE_GXM_CULL_NONE);
                    execute(renderer::GXMState::Viewport, false, 16.f, 16.f, 0.f, 16.f, 16.f, .25f);
                    execute(renderer::GXMState::FragmentProgramEnable, true, SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
                    execute(renderer::GXMState::FragmentProgramEnable, false, SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
                    ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_NEVER;
                    ctx.record.back_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
                    ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_DISABLED;
                    ctx.record.back_depth_write_mode = SCE_GXM_DEPTH_WRITE_ENABLED;
                    for (bool mask : {false, true}) {
                        auto saved = ds;
                        if (!mask)
                            ds = {};
                        state.set_context(ctx, mem);
                        fp->is_maskupdate = mask;
                        execute(renderer::GXMState::Program, fa.cast<void>(), true);
                        if (mask)
                            execute(renderer::GXMState::StencilFunc, true, SCE_GXM_STENCIL_FUNC_ALWAYS,
                                    SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
                                    uint8_t(255), uint8_t(255));
                        draw();
                        end();
                        fp->is_maskupdate = false;
                        execute(renderer::GXMState::Program, fa.cast<void>(), true);
                        ds = saved;
                        ++cases;
                    }
                }
        state.context = nullptr;
        vp->~SceGxmVertexProgram();
        fp->~SceGxmFragmentProgram();
        std::cout << "PASS " << cases << " effective face state cases, " << values
                  << " color/depth/stencil values; both cull sides, windings, viewport reflections, 1/2/4samples, "
                     "1x/2x, differing depth/write and fragment modes, disabled depth and mask updates\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
