// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Declare renderer::Context before the generic dispatcher header.
#include <renderer/metal/state.h>

#include <cmath>
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
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: metal-native-color-validation <Sly 3a2fc0 vertex.gxp> <Stealth 72c63c fragment.gxp> [Unit13 f4c6949a fragment.gxp]\n";
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
        state.late_init(config, "metal-native-color-validation", mem);
        auto execute = [&](renderer::GXMState code, auto... args) {
            renderer::Command command{};
            renderer::CommandHelper writer(&command);
            check(writer.push(code) && (writer.push(args) && ...), "Cannot encode state");
            renderer::CommandHelper reader(&command);
            renderer::cmd_handle_set_state(state, mem, config, reader, state.features, &ctx);
        };
        Ptr<SceGxmVertexProgram> vp_addr(alloc(mem, sizeof(SceGxmVertexProgram), "Native color vertex"));
        Ptr<SceGxmFragmentProgram> fp_addr(alloc(mem, sizeof(SceGxmFragmentProgram), "Native color fragment"));
        auto *vp = new (vp_addr.get(mem)) SceGxmVertexProgram{};
        auto *fp = new (fp_addr.get(mem)) SceGxmFragmentProgram{};
        vp->program = load(mem, argv[1]);
        fp->program = load(mem, argv[2]);
        const auto &gxp = *vp->program.get(mem);
        for (uint32_t i = 0; i < gxp.parameter_count; ++i) {
            const auto &param = gxp.program_parameters()[i];
            if (param.category != SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE)
                continue;
            const std::string name = param.name();
            const unsigned offset = name == "IN.pos" ? 0 : name == "IN.color" ? 16
                : name == "IN.uv0"                                            ? 32
                                                                              : 999;
            check(offset != 999, "Unexpected vertex input: " + name);
            SceGxmVertexAttribute a{};
            a.offset = offset;
            a.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
            a.componentCount = param.component_count;
            a.regIndex = param.resource_index;
            vp->attributes.push_back(a);
        }
        check(vp->attributes.size() == 3, "Expected position/color/UV attributes");
        vp->streams.push_back({ 48, SCE_GXM_INDEX_SOURCE_EACH_VERTEX_16BIT });
        check(renderer::create(vp->renderer_data, state, gxp, state.gxp_ptr_map, vp->attributes), "Cannot create vertex program");
        check(renderer::create(fp->renderer_data, state, *fp->program.get(mem), nullptr, state.gxp_ptr_map), "Cannot create fragment program");
        execute(renderer::GXMState::Program, vp_addr.cast<void>(), false);
        execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
        const float vertices[] = { -1, -1, 0, 1, 1, 1, 1, 1, .5, .5, 0, 0,
            3, -1, 0, 1, 1, 1, 1, 1, .5, .5, 0, 0, -1, 3, 0, 1, 1, 1, 1, 1, .5, .5, 0, 0 };
        Ptr<float> stream(alloc(mem, sizeof(vertices), "Native color vertices"));
        std::memcpy(stream.get(mem), vertices, sizeof(vertices));
        ctx.record.vertex_streams[0] = { stream.cast<const uint8_t>(), sizeof(vertices) };
        const float matrix[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        Ptr<float> uniform(alloc(mem, sizeof(matrix), "Native color identity"));
        std::memcpy(uniform.get(mem), matrix, sizeof(matrix));
        execute(renderer::GXMState::UniformBuffer, uniform.cast<uint8_t>(), true, 0, uint32_t(sizeof(matrix)));
        Ptr<uint16_t> indices(alloc(mem, 6, "Native color indices"));
        for (unsigned i = 0; i < 3; ++i)
            indices.get(mem)[i] = i;
        renderer::metal::MetalRenderTarget target;
        target.width = target.height = 128;
        ctx.current_render_target = &target;
        ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
        ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_DISABLED;
        ctx.record.front_stencil_state_op.func = SCE_GXM_STENCIL_FUNC_ALWAYS;
        Ptr<uint8_t> fragmentUniform(alloc(mem, 16, "Native color uniforms"));
        execute(renderer::GXMState::UniformBuffer, fragmentUniform, false, 0, uint32_t(16));
        auto &surface = ctx.record.color_surface;
        surface = {};
        surface.width = surface.height = surface.strideInPixels = 8;
        surface.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
        surface.data = Ptr<void>(alloc(mem, 8 * 8 * 8, "Native color output"));
        target.width = target.height = 8;
        unsigned passed = 0;
        for (unsigned scale : { 1u, 2u })
            for (auto mode : { SCE_GXM_MULTISAMPLE_NONE, SCE_GXM_MULTISAMPLE_2X, SCE_GXM_MULTISAMPLE_4X })
                for (auto format : { SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR, SCE_GXM_COLOR_FORMAT_F32F32_GR,
                         SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR, SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB }) {
                    const bool paired = format == SCE_GXM_COLOR_FORMAT_F32F32_GR;
                    state.res_multiplier = scale;
                    target.multisample_mode = mode;
                    surface.colorFormat = format;
                    surface.downscale = mode != SCE_GXM_MULTISAMPLE_NONE;
                    execute(renderer::GXMState::Viewport, false, 4.f, 4.f, 0.f, 4.f, 4.f, 1.f);
                    execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0), uint32_t(8), uint32_t(0), uint32_t(8));
                    for (unsigned bits : { 32u, 64u, 32u })
                        for (unsigned pass = 0; pass < 4; ++pass) {
                            if (paired && bits != 64) continue;
                            surface.outputRegisterSize = bits == 32 ? SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT : SCE_GXM_OUTPUT_REGISTER_SIZE_64BIT;
                            // Finite float bit patterns for the paired target:
                            // native float MSAA resolve is exercised here too.
                            const uint8_t color[] = { uint8_t(pass * 53), uint8_t(31 + pass * 29), uint8_t(73 + pass * 17),
                                uint8_t(paired ? 32+pass*11 : 255-pass*41),
                                uint8_t(23+pass*17), uint8_t(177-pass*31), uint8_t(pass*43), uint8_t(48+pass*9) };
                            std::memset(fragmentUniform.get(mem), 0, 16);
                            if (paired)
                                std::memcpy(fragmentUniform.get(mem), color, 8);
                            else if (bits == 32)
                                std::memcpy(fragmentUniform.get(mem), color, 4);
                            else {
                                __fp16 halves[4];
                                for (unsigned c = 0; c < 4; ++c)
                                    halves[c] = float(color[c]) / 255;
                                std::memcpy(fragmentUniform.get(mem), halves, 8);
                            }
                            state.set_context(ctx, mem);
                            state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, indices.get(mem), 3, 1);
                            check(state.sync_surface(mem, surface), "Native color output sync failed");
                            const auto *actual = static_cast<const uint8_t *>(surface.data.get(mem));
                            for (unsigned i = 0; i < 8 * 8 * (paired ? 8 : 4); ++i) {
                                unsigned c = i % (paired ? 8 : 4);
                                if (format == SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB && (c == 0 || c == 2))
                                    c = 2 - c;
                                check(std::abs(int(actual[i]) - int(color[c])) <= 1, "Native color pixel mismatch bits=" + std::to_string(bits) + " format=" + std::to_string(format) + " scale=" + std::to_string(scale) + " samples=" + std::to_string(mode) + " byte=" + std::to_string(i) + " actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(color[c]));
                            }
                            ++passed;
                        }
                }
        // F16 render targets can contain packed bytes rather than numerical
        // colors. Exercise output, scale, channel order, and guest readback
        // with subnormals, infinities and distinct signed NaN payloads.
        const uint16_t patterns[] = {0, 0x8000, 1, 0x03ff, 0x0400, 0x3555, 0x3c00, 0x7bff,
                                    0x7c00, 0xfc00, 0x7c01, 0x7d55, 0x7e00, 0x7f7f, 0xfe55, 0xffff};
        unsigned half_passed = 0;
        for (int shader_index = 2; shader_index < argc; ++shader_index) {
            if (shader_index != 2) {
                fp->program = load(mem, argv[shader_index]);
                check(renderer::create(fp->renderer_data, state, *fp->program.get(mem), nullptr, state.gxp_ptr_map),
                      "Cannot create Unit 13 fragment program");
                execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
            }
            for (unsigned scale : {1u, 2u})
                for (auto format : {SCE_GXM_COLOR_FORMAT_F16F16F16F16_ABGR, SCE_GXM_COLOR_FORMAT_F16F16F16F16_ARGB}) {
                    state.res_multiplier = scale;
                    target.multisample_mode = SCE_GXM_MULTISAMPLE_NONE;
                    surface.colorFormat = format;
                    surface.outputRegisterSize = SCE_GXM_OUTPUT_REGISTER_SIZE_64BIT;
                    surface.downscale = false;
                    execute(renderer::GXMState::Viewport, false, 4.f, 4.f, 0.f, 4.f, 4.f, 1.f);
                    for (unsigned pass = 0; pass < 16; ++pass) {
                        uint16_t expected[4];
                        for (unsigned c = 0; c < 4; ++c)
                            expected[c] = patterns[(pass + c * 5) % 16];
                        std::memset(fragmentUniform.get(mem), 0, 16);
                        std::memcpy(fragmentUniform.get(mem), expected, sizeof(expected));
                        state.set_context(ctx, mem);
                        state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, indices.get(mem), 3, 1);
                        check(state.sync_surface(mem, surface), "Half attachment readback failed");
                        const auto *actual = static_cast<const uint16_t *>(surface.data.get(mem));
                        for (unsigned i = 0; i < 64 * 4; ++i) {
                            unsigned c = i % 4;
                            if (format == SCE_GXM_COLOR_FORMAT_F16F16F16F16_ARGB && (c == 0 || c == 2))
                                c = 2 - c;
                            check(actual[i] == expected[c], "Half payload mismatch shader=" + std::to_string(shader_index)
                                + " scale=" + std::to_string(scale) + " format=" + std::to_string(format)
                                + " pass=" + std::to_string(pass) + " component=" + std::to_string(i)
                                + " actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected[c]));
                        }
                        ++half_passed;
                    }
                }
        }
        state.context = nullptr;
        vp->~SceGxmVertexProgram();
        fp->~SceGxmFragmentProgram();
        std::cout << "PASS " << passed << " native-color GXM draws: same shader/target 32->64->32 registers, RGBA8->RG32F->RGBA8 format changes, ABGR/ARGB, 1x/2x, 1/2/4 samples, every guest component\n";
        std::cout << "PASS " << half_passed << " F16 attachment GXM draws with exact finite/NaN/Inf payloads, ABGR/ARGB, 1x/2x and guest readback\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
