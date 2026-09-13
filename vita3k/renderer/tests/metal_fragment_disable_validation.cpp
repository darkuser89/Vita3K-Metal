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
    if (argc != 3 && argc != 6) {
        std::cerr << "Usage: metal-fragment-disable-validation <Sly25188 vertex.gxp> <Sly028582 fragment.gxp> "
                     "[Sly/SAO/Unit13 disabled fragments]\n";
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
        std::vector<Ptr<SceGxmFragmentProgram>> disabled_programs;
        for (int arg = 3; arg < argc; arg++) {
            Ptr<SceGxmFragmentProgram> address(alloc(mem, sizeof(SceGxmFragmentProgram), "Focused disabled fragment"));
            auto *program = new (address.get(mem)) SceGxmFragmentProgram{};
            program->program = load(mem, argv[arg]);
            check(
                renderer::create(program->renderer_data, state, *program->program.get(mem), nullptr, state.gxp_ptr_map),
                "Focused fragment creation failed");
            disabled_programs.push_back(address);
        }
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
        auto draw = [&] {
            state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, indices.get(mem), 3, 1);
        };
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
        size_t components = 0;
        for (unsigned scale : {1u, 2u})
            for (auto samples : {SCE_GXM_MULTISAMPLE_NONE, SCE_GXM_MULTISAMPLE_2X, SCE_GXM_MULTISAMPLE_4X})
                for (bool reverse : {false, true})
                    for (unsigned mask_mode : {0u, 1u, 2u})
                        for (bool write : {true, false}) {
                            state.res_multiplier = scale;
                            target.width = target.height = 32 * scale;
                            target.multisample_mode = samples;
                            unsigned sx = samples == SCE_GXM_MULTISAMPLE_4X ? 2 : 1,
                                     sy = samples == SCE_GXM_MULTISAMPLE_NONE ? 1 : 2;
                            indices.get(mem)[1] = reverse ? 2 : 1;
                            indices.get(mem)[2] = reverse ? 1 : 2;
                            auto &color = ctx.record.color_surface;
                            color = {};
                            color.width = color.height = color.strideInPixels = 32;
                            color.downscale = samples != SCE_GXM_MULTISAMPLE_NONE;
                            color.colorFormat = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
                            color.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
                            color.data = Ptr<void>(alloc(mem, 32 * 32 * 4, "Disable color target"));
                            auto &ds = ctx.record.depth_stencil_surface;
                            ds = {};
                            ds.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8);
                            ds.set_type(SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR);
                            ds.set_stride(32 * sx);
                            ds.force_store = 1;
                            ds.depth_data = Ptr<void>(alloc(mem, 32 * 32 * sx * sy * 4, "Disable depth"));
                            ds.stencil_data = Ptr<void>(alloc(mem, 32 * 32 * sx * sy, "Disable stencil"));
                            // Disabled -> enabled -> disabled uses the same target/program cache.
                            for (unsigned variant : {0u, 1u, 2u}) {
                                bool disabled = variant != 1;
                                execute(renderer::GXMState::TwoSided,
                                        variant == 2 ? SCE_GXM_TWO_SIDED_ENABLED : SCE_GXM_TWO_SIDED_DISABLED);
                                execute(renderer::GXMState::FragmentProgramEnable, true,
                                        disabled ? SCE_GXM_FRAGMENT_PROGRAM_DISABLED
                                                 : SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
                                execute(renderer::GXMState::FragmentProgramEnable, false,
                                        variant == 2 ? SCE_GXM_FRAGMENT_PROGRAM_DISABLED
                                                     : SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
                                execute(renderer::GXMState::UniformBuffer, uniform.cast<uint8_t>(), false, 0,
                                        uint32_t(16));
                                std::memset(color.data.get(mem), 0x35, 32 * 32 * 4);
                                ds.background_depth = .75f;
                                ds.stencil = 0x25;
                                state.set_context(ctx, mem);
                                fp->is_maskupdate = true;
                                execute(renderer::GXMState::Program, fa.cast<void>(), true);
                                execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0),
                                        uint32_t(32), uint32_t(0), uint32_t(32));
                                execute(renderer::GXMState::Viewport, false, 16.f, 16.f, 0.f, 16.f, 16.f, .25f);
                                execute(renderer::GXMState::StencilFunc, true, SCE_GXM_STENCIL_FUNC_ALWAYS,
                                        SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
                                        uint8_t(255), uint8_t(255));
                                draw();
                                if (mask_mode) {
                                    execute(renderer::GXMState::StencilFunc, true, SCE_GXM_STENCIL_FUNC_NEVER,
                                            SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
                                            uint8_t(255), uint8_t(255));
                                    if (mask_mode == 1)
                                        execute(renderer::GXMState::Viewport, false, 8.f, 16.f, 0.f, 8.f, 16.f, .25f);
                                    draw();
                                }
                                fp->is_maskupdate = false;
                                execute(renderer::GXMState::Program, fa.cast<void>(), true);
                                execute(renderer::GXMState::Viewport, false, 16.f, 16.f, 0.f, 16.f, 16.f, .25f);
                                execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0),
                                        uint32_t(32), uint32_t(0), uint32_t(32));
                                ctx.record.front_depth_func = ctx.record.back_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
                                ctx.record.front_depth_write_mode = ctx.record.back_depth_write_mode =
                                    write ? SCE_GXM_DEPTH_WRITE_ENABLED : SCE_GXM_DEPTH_WRITE_DISABLED;
                                for (bool front : {true, false}) {
                                    execute(renderer::GXMState::StencilFunc, front, SCE_GXM_STENCIL_FUNC_ALWAYS,
                                            SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
                                            SCE_GXM_STENCIL_OP_REPLACE, uint8_t(255), uint8_t(255));
                                }
                                ctx.record.front_stencil_state_values.ref = ctx.record.back_stencil_state_values.ref =
                                    0x7b;
                                // A disabled guest fragment must not read its uniform at all.
                                if (variant == 2) {
                                    ctx.uniforms[1][0] = {};
                                    if (!disabled_programs.empty())
                                        execute(renderer::GXMState::Program,
                                                disabled_programs[(cases / 3) % disabled_programs.size()].cast<void>(),
                                                true);
                                }
                                draw();
                                end();
                                check(state.sync_surface(mem, color), "Color readback failed");
                                const auto label =
                                    " case=" + std::to_string(cases) + " variant=" + std::to_string(variant) +
                                    " mask=" + std::to_string(mask_mode) + " scale=" + std::to_string(scale) +
                                    " samples=" + std::to_string(samples);
                                auto visible = [&](unsigned x) {
                                    return mask_mode == 0 || (mask_mode == 1 && x >= 16);
                                };
                                for (unsigned y = 0; y < 32; y++)
                                    for (unsigned x = 0; x < 32; x++)
                                        for (unsigned c = 0; c < 4; c++) {
                                            unsigned expected =
                                                !disabled && visible(x) ? std::lround(rgba[c] * 255) : 0x35;
                                            unsigned actual =
                                                static_cast<uint8_t *>(color.data.get(mem))[(y * 32 + x) * 4 + c];
                                            check(std::abs(int(actual) - int(expected)) <=
                                                      (!disabled && visible(x) ? 1 : 0),
                                                  "Color changed unexpectedly" + label + " x=" + std::to_string(x) +
                                                      " y=" + std::to_string(y) + " c=" + std::to_string(c) +
                                                      " actual=" + std::to_string(actual) +
                                                      " expected=" + std::to_string(expected));
                                            ++components;
                                        }
                                for (unsigned y = 0; y < 32 * sy; y++)
                                    for (unsigned x = 0; x < 32 * sx; x++) {
                                        size_t at = y * 32 * sx + x;
                                        float expected = visible(x / sx) && write ? .25f : .75f;
                                        float actual = static_cast<float *>(ds.depth_data.get(mem))[at];
                                        check(actual == expected, "Disabled fragment lost depth write" + label +
                                                                      " actual=" + std::to_string(actual) +
                                                                      " expected=" + std::to_string(expected));
                                        uint8_t stencil = static_cast<uint8_t *>(ds.stencil_data.get(mem))[at];
                                        check(stencil == (visible(x / sx) ? 0x7b : 0x25),
                                              "Disabled fragment lost stencil write" + label);
                                        components += 2;
                                    }
                                ++cases;
                            }
                        }
        // Preserve native detail when an unchanged CPU image is rebound, then
        // update only a changed guest byte. A whole-image upload would flatten
        // the deliberately one-native-pixel-wide stripe into a two-pixel stripe.
        state.res_multiplier = 2;
        target.width = target.height = 64;
        target.multisample_mode = SCE_GXM_MULTISAMPLE_NONE;
        ctx.record.depth_stencil_surface = {};
        auto &surface = ctx.record.color_surface;
        surface = {};
        surface.width = surface.height = surface.strideInPixels = 32;
        surface.colorFormat = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
        surface.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
        surface.data = Ptr<void>(alloc(mem, 32 * 32 * 4, "CPU import detail target"));
        std::memset(surface.data.get(mem), 0x35, 32 * 32 * 4);
        execute(renderer::GXMState::TwoSided, SCE_GXM_TWO_SIDED_DISABLED);
        execute(renderer::GXMState::FragmentProgramEnable, true, SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
        execute(renderer::GXMState::Program, fa.cast<void>(), true);
        execute(renderer::GXMState::UniformBuffer, uniform.cast<uint8_t>(), false, 0, uint32_t(16));
        execute(renderer::GXMState::Viewport, false, 16.f, 16.f, 0.f, 16.f, 16.f, .25f);
        state.set_context(ctx, mem);
        fp->is_maskupdate = true;
        execute(renderer::GXMState::Program, fa.cast<void>(), true);
        execute(renderer::GXMState::StencilFunc, true, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, uint8_t(255), uint8_t(255));
        draw();
        fp->is_maskupdate = false;
        execute(renderer::GXMState::Program, fa.cast<void>(), true);
        execute(renderer::GXMState::Viewport, false, .25f, 16.f, 0.f, .25f, 16.f, .25f);
        draw();
        end();
        check(state.sync_surface(mem, surface), "Initial native-detail readback failed");
        DisplayState display;
        display.next_rendered_frame.base = surface.data;
        auto frame = [&] {
            uint32_t w = 0, h = 0;
            auto data = state.dump_frame(display, w, h);
            check(w == 64 && h == 64, "Lost native 2x extent");
            return data;
        };
        const auto original = frame();
        check(original[0] != original[1], "Native detail fixture did not produce a one-pixel stripe");
        for (unsigned iteration = 0; iteration < 3; iteration++) {
            if (iteration == 1)
                static_cast<uint8_t *>(surface.data.get(mem))[(5 * 32 + 7) * 4 + 1] = 0xa9;
            if (iteration == 2) {
                SceGxmTransferImage fill{};
                fill.address = surface.data.cast<uint8_t>();
                fill.format = SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR;
                fill.x = 12;
                fill.y = 9;
                fill.width = fill.height = 1;
                fill.stride = 32 * 4;
                check(state.transfer_fill(mem, fill, 0x82634120), "CPU snapshot transfer fill failed");
            }
            execute(renderer::GXMState::FragmentProgramEnable, true, SCE_GXM_FRAGMENT_PROGRAM_DISABLED);
            state.set_context(ctx, mem);
            draw();
            end();
            const auto actual = frame();
            auto expected = original;
            if (iteration >= 1)
                for (unsigned y = 10; y < 12; y++)
                    for (unsigned x = 14; x < 16; x++)
                        reinterpret_cast<uint8_t *>(expected.data())[(y * 64 + x) * 4 + 1] = 0xa9;
            if (iteration == 2)
                for (unsigned y = 18; y < 20; y++)
                    for (unsigned x = 24; x < 26; x++)
                        expected[y * 64 + x] = 0x82634120;
            check(actual == expected, "CPU import flattened untouched native pixels or lost a byte/transfer update");
            check(state.sync_surface(mem, surface), "Native-detail snapshot publication failed");
        }
        std::cout << "PASS 2x CPU import preserves a one-native-pixel stripe across unchanged scenes, byte updates, "
                     "transfer fill and explicit publication\n";
        for (auto address : disabled_programs)
            address.get(mem)->~SceGxmFragmentProgram();
        state.context = nullptr;
        vp->~SceGxmVertexProgram();
        fp->~SceGxmFragmentProgram();
        std::cout << "PASS " << cases << " fragment-disable GXM cases, " << components
                  << " color/depth/stencil values; single/two-sided, enabled transitions, unbound inactive uniform, "
                     "masks, 1/2/4 samples, 1x/2x, both windings, EndScene\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
