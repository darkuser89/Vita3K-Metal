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
    if (argc != 3) {
        std::cerr << "Usage: metal-lod-gxm-validation <Sly 3a2fc0 vertex.gxp> <Stealth 3b6ed069 fragment.gxp>\n";
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
        state.late_init(config, "metal-lod-gxm-validation", mem);
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
        const float vertices[] = { -1, -1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            3, -1, 0, 1, 1, 1, 1, 1, 2, 0, 0, 0, -1, 3, 0, 1, 1, 1, 1, 1, 0, 2, 0, 0 };
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
        auto &surface = ctx.record.color_surface;
        surface = {};
        surface.width = surface.height = surface.strideInPixels = 8;
        surface.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
        surface.data = Ptr<void>(alloc(mem, 8 * 8 * 8, "Native color output"));
        target.width = target.height = 8;
        surface.colorFormat = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
        surface.outputRegisterSize = SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT;
        Ptr<uint8_t> pixels(alloc(mem, 64 * 64 * 8, "LOD mip chain"));
        auto *destination = pixels.get(mem);
        for (unsigned size = 64, level = 0; size; size >>= 1, ++level)
            for (unsigned i = 0; i < size * size; ++i) {
                *destination++ = level * 32;
                *destination++ = (6 - level) * 32;
                *destination++ = 64;
                *destination++ = 255;
            }
        auto &texture = ctx.textures[0];
        texture = {};
        texture.type = SCE_GXM_TEXTURE_SWIZZLED >> 29;
        texture.width_base2 = texture.height_base2 = 6;
        texture.base_format = SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 >> 24;
        texture.swizzle_format = uint32_t(SCE_GXM_TEXTURE_SWIZZLE4_ABGR) >> 12;
        texture.data_addr = pixels.address() >> 2;
        texture.mip_count = 6;
        texture.min_filter = texture.mag_filter = SCE_GXM_TEXTURE_FILTER_LINEAR;
        texture.uaddr_mode = texture.vaddr_mode = SCE_GXM_TEXTURE_ADDR_CLAMP;
        unsigned passed = 0;
        for (unsigned scale : {1u, 2u}) {
            state.res_multiplier = scale;
            execute(renderer::GXMState::Viewport, false, 4.f, 4.f, 0.f, 4.f, 4.f, 1.f);
            execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0), uint32_t(8), uint32_t(0), uint32_t(8));
            for (unsigned encoded : {31u, 39u, 23u, 33u, 29u, 31u})
                for (unsigned minimum : {0u, 4u, 0u})
                    for (unsigned linear : {0u, 1u}) {
                        texture.lod_bias = encoded;
                        texture.lod_min0 = minimum & 3;
                        texture.lod_min1 = minimum >> 2;
                        texture.mip_filter = linear;
                        state.set_context(ctx, mem);
                        state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, indices.get(mem), 3, 1);
                        check(state.sync_surface(mem, surface), "LOD output synchronization failed");
                        const float lod = std::max(3.f - float(scale == 2) + (float(encoded) - 31) / 8.f, float(minimum));
                        const float selected = linear ? lod : std::floor(lod + .5f);
                        const uint8_t expected[] = {uint8_t(selected * 32), uint8_t((6 - selected) * 32), 64, 255};
                        const auto *actual = static_cast<const uint8_t *>(surface.data.get(mem));
                        for (unsigned i = 0; i < 8 * 8 * 4; ++i)
                            check(std::abs(int(actual[i]) - int(expected[i % 4])) <= 1,
                                "LOD GXM mismatch scale=" + std::to_string(scale) + " bias=" + std::to_string(encoded)
                                + " min=" + std::to_string(minimum) + " byte=" + std::to_string(i)
                                + " actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected[i % 4]));
                        ++passed;
                    }
        }
        state.context = nullptr;
        vp->~SceGxmVertexProgram();
        fp->~SceGxmFragmentProgram();
        std::cout << "PASS " << passed << " GXM LOD draws: same shader/texture bias0,+1,-1,+0.25,-0.25,0; minimum0,4,0; two mip filters; scale1/2; every guest component\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
