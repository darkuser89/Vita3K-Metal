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
        std::cerr << "Usage: metal-msaa-scene-validation <Sly25188 vertex.gxp> <Sly028582 fragment.gxp>\n";
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
        Ptr<SceGxmVertexProgram> va(alloc(mem, sizeof(SceGxmVertexProgram), "MSAA scene vertex"));
        Ptr<SceGxmFragmentProgram> fa(alloc(mem, sizeof(SceGxmFragmentProgram), "MSAA scene fragment"));
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
        float vertices[] = {-1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1};
        Ptr<float> stream(alloc(mem, sizeof(vertices), "MSAA scene vertices"));
        std::memcpy(stream.get(mem), vertices, sizeof(vertices));
        ctx.record.vertex_streams[0] = {stream.cast<const uint8_t>(), sizeof(vertices)};
        Ptr<float> uniform(alloc(mem, 16, "MSAA scene color"));
        execute(renderer::GXMState::UniformBuffer, uniform.cast<uint8_t>(), false, 0, uint32_t(16));
        Ptr<uint16_t> indices(alloc(mem, 8, "MSAA scene indices"));
        for (unsigned i = 0; i < 4; i++)
            indices.get(mem)[i] = i;
        renderer::metal::MetalRenderTarget target;
        ctx.current_render_target = &target;
        ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
        execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0), uint32_t(32), uint32_t(0),
                uint32_t(32));
        auto draw = [&](float r, float g, float b, float edge) {
            float color[] = {r, g, b, 1};
            std::memcpy(uniform.get(mem), color, 16);
            execute(renderer::GXMState::Viewport, false, edge / 2, 16.f, 0.f, edge / 2, 16.f, .25f);
            state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLE_FAN, SCE_GXM_INDEX_FORMAT_U16, indices.get(mem), 4, 1);
        };
        DisplayState display;
        auto capture = [&](const SceGxmColorSurface &surface) {
            display.next_rendered_frame.base = surface.data;
            uint32_t w = 0, h = 0;
            auto pixels = state.dump_frame(display, w, h);
            check(w == surface.width * state.res_multiplier && h == surface.height * state.res_multiplier,
                  "Wrong native frame extent");
            return pixels;
        };
        unsigned cases = 0;
        size_t pixels_checked = 0;
        for (auto samples : {SCE_GXM_MULTISAMPLE_2X, SCE_GXM_MULTISAMPLE_4X})
            for (unsigned scale : {1u, 2u})
                for (bool expanded : {false, true})
                    for (bool gamma : {false, true})
                        for (bool argb : {false, true})
                            for (unsigned cpu_edit : {0u, 1u, 2u}) {
                                state.res_multiplier = scale;
                                target.width = target.height = 32 * scale;
                                target.multisample_mode = samples;
                                const unsigned sx = expanded && samples == SCE_GXM_MULTISAMPLE_4X ? 2 : 1,
                                               sy = expanded ? 2 : 1;
                                const float edge = 16.f - .5f / scale;
                                std::vector<uint32_t> reference;
                                for (unsigned path : {0u, 1u, 2u}) {
                                    auto &surface = ctx.record.color_surface;
                                    surface = {};
                                    surface.width = surface.strideInPixels = 32 * sx;
                                    surface.height = 32 * sy;
                                    surface.colorFormat =
                                        argb ? SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB : SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
                                    surface.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
                                    surface.downscale = !expanded;
                                    surface.gamma = gamma;
                                    surface.data = Ptr<void>(
                                        alloc(mem, surface.width * surface.height * 4, "MSAA scene color target"));
                                    std::memset(surface.data.get(mem), 0, surface.width * surface.height * 4);
                                    state.set_context(ctx, mem);
                                    draw(0, 0, 1, 32);
                                    draw(1, 0, 0, edge);
                                    state.finish(ctx);
                                    const auto pre = capture(surface);
                                    if (!expanded) {
                                        const uint8_t red = reinterpret_cast<const uint8_t *>(
                                            pre.data())[((8 * scale) * (32 * scale) + 16 * scale - 1) * 4];
                                        check(red > 0 && red < 255,
                                              "Fixture failed to produce distinct samples at its fractional edge");
                                    }
                                    check(state.sync_surface(mem, surface), "Initial sample readback failed");
                                    const auto active = surface;
                                    // Edit either an unrelated pixel or just one
                                    // component of the pixel containing the edge.
                                    const unsigned edit_x = cpu_edit == 2 ? 15 : 30;
                                    const size_t modified = ((5 * sy) * surface.width + edit_x * sx) * 4 + 2;
                                    if (cpu_edit) {
                                        if (path == 0) {
                                            SceGxmTransferImage fill{};
                                            fill.address = surface.data.cast<uint8_t>() + modified;
                                            fill.format = SCE_GXM_TRANSFER_FORMAT_U8_R;
                                            fill.width = fill.height = fill.stride = 1;
                                            check(state.transfer_fill(mem, fill, 0x49),
                                                  "Reference byte transfer failed");
                                        } else
                                            static_cast<uint8_t *>(surface.data.get(mem))[modified] = 0x49;
                                    }
                                    if (path == 2) {
                                        surface.data = Ptr<void>(
                                            alloc(mem, surface.width * surface.height * 4, "MSAA intervening target"));
                                        std::memset(surface.data.get(mem), 0, surface.width * surface.height * 4);
                                        state.set_context(ctx, mem);
                                        draw(1, 1, 0, 32);
                                        state.finish(ctx);
                                        surface = active;
                                    }
                                    if (path)
                                        state.set_context(ctx, mem);
                                    draw(0, 1, 0, edge);
                                    state.finish(ctx);
                                    const auto actual = capture(surface);
                                    if (path == 0)
                                        reference = actual;
                                    else {
                                        check(actual.size() == reference.size(), "Different frame extents");
                                        for (size_t i = 0; i < actual.size(); i++) {
                                            check(actual[i] == reference[i],
                                                  "MSAA samples changed across scene boundary case=" +
                                                      std::to_string(cases) + " path=" + std::to_string(path) +
                                                      " samples=" + std::to_string(samples) + " scale=" +
                                                      std::to_string(scale) + " expanded=" + std::to_string(expanded) +
                                                      " gamma=" + std::to_string(gamma) + " argb=" +
                                                      std::to_string(argb) + " cpu_edit=" + std::to_string(cpu_edit) +
                                                      " pixel=" + std::to_string(i) +
                                                      " actual=" + std::to_string(actual[i]) +
                                                      " expected=" + std::to_string(reference[i]));
                                            ++pixels_checked;
                                        }
                                    }
                                    ++cases;
                                }
                            }
        state.context = nullptr;
        vp->~SceGxmVertexProgram();
        fp->~SceGxmFragmentProgram();
        std::cout << "PASS " << cases << " GXM MSAA scene paths; " << pixels_checked
                  << " exact native pixel comparisons; "
                     "2/4samples,1x/2x,resolved/expanded,linear/sRGB,ABGR/ARGB,unchanged/CPUbyte,same-target/A-B-A\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
