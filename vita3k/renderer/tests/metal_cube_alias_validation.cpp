// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Declare renderer::Context before the generic dispatcher header.
#include <renderer/metal/state.h>

#include <bit>
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
    if (argc < 4 || argc > 14) {
        std::cerr << "Usage: metal-cube-alias-validation <Unit13 0b2fe8 VS> <Stealth 72c63c fill FS> <Unit13 80e37b "
                     "reflection FS> [texture swizzle 0..7] [surface swizzle 0..3] [surface sRGB 0/1] [texture gamma "
                     "0/1/3] [rgba8|f32 source] [cube|2d|crop|swizzled|tiled|mip-linear|mip-swizzled|mip-tiled] "
                     "[square|rectangle|npot] [point|point-gradient|linear-gradient] [repeat|mirror|clamp] [anisotropy "
                     "1..16]\n";
        return 2;
    }
    try {
        const unsigned swizzle = argc >= 5 ? std::stoul(argv[4]) : 0;
        const unsigned surface_swizzle = argc >= 6 ? std::stoul(argv[5]) : 0;
        const unsigned surface_gamma = argc >= 7 ? std::stoul(argv[6]) : 0;
        const unsigned texture_gamma = argc >= 8 ? std::stoul(argv[7]) : 0;
        const std::string source_kind = argc >= 9 ? argv[8] : "rgba8";
        const bool float_source = source_kind == "f32";
        const std::string texture_mode = argc >= 10 ? argv[9] : "cube";
        check(texture_mode == "cube" || texture_mode == "2d" || texture_mode == "crop" || texture_mode == "swizzled" ||
                  texture_mode == "tiled" || texture_mode == "mip-linear" || texture_mode == "mip-swizzled" ||
                  texture_mode == "mip-tiled",
              "Invalid texture mode");
        const bool sample_2d = texture_mode != "cube", crop = texture_mode == "crop";
        const bool mip_chain = texture_mode.starts_with("mip-");
        const std::string shape = argc >= 11 ? argv[10] : "square";
        check(shape == "square" || shape == "rectangle" || shape == "npot", "Invalid shape");
        check(shape == "square" || mip_chain, "Non-square shape requires a mip-chain mode");
        check(shape != "npot" || texture_mode != "mip-swizzled",
              "NPOT Morton render surfaces require a separate layout fixture");
        const unsigned base_width = shape == "npot" ? 15 : 16, base_height = shape == "npot"        ? 9
                                                                             : shape == "rectangle" ? 8
                                                                                                    : 16;
        const std::string filtering = argc >= 12 ? argv[11] : "point";
        const bool linear = filtering == "linear-gradient";
        const bool gradient = linear || filtering == "point-gradient";
        const unsigned anisotropy = argc >= 14 ? std::stoul(argv[13]) : 1;
        check(anisotropy >= 1 && anisotropy <= 16, "Invalid anisotropy");
        const std::string addressing = argc >= 13 ? argv[12] : "repeat";
        check(addressing == "repeat" || addressing == "mirror" || addressing == "clamp", "Invalid address mode");
        check(gradient || addressing == "repeat", "Address variations require gradient reference");
        const unsigned address_mode = addressing == "repeat"   ? SCE_GXM_TEXTURE_ADDR_REPEAT
                                      : addressing == "mirror" ? SCE_GXM_TEXTURE_ADDR_MIRROR
                                                               : SCE_GXM_TEXTURE_ADDR_CLAMP;
        const auto address_coordinate = [&](int coordinate, int size) {
            if (addressing == "clamp")
                return std::clamp(coordinate, 0, size - 1);
            const int period = addressing == "mirror" ? size * 2 : size;
            const int phase = (coordinate % period + period) % period;
            return phase < size ? phase : period - 1 - phase;
        };
        check(filtering == "point" || gradient, "Invalid filtering mode");
        check(!gradient || (texture_mode == "mip-linear" && !float_source && !swizzle && !surface_swizzle &&
                            !surface_gamma && !texture_gamma),
              "Gradient reference requires linear RGBA8 mip storage with identity color conversion");
        const auto ram_pixel = [](unsigned x, unsigned y, unsigned w, unsigned h) {
            return std::array<uint8_t, 4>{uint8_t(w > 1 ? x * 255 / (w - 1) : 37),
                                          uint8_t(h > 1 ? y * 255 / (h - 1) : 91), uint8_t((x + y) % 3 * 100), 255};
        };
        const bool tiled = texture_mode == "tiled" || texture_mode == "mip-tiled";
        const bool swizzled = texture_mode == "swizzled" || texture_mode == "mip-swizzled";
        check(source_kind == "rgba8" || float_source, "Source must be rgba8 or f32");
        check(!float_source || (!surface_swizzle && !surface_gamma), "F32 source requires identity/linear surface");
        check(swizzle < 8 && surface_swizzle < 4 && surface_gamma < 2 &&
                  (texture_gamma == 0 || texture_gamma == 1 || texture_gamma == 3),
              "Invalid swizzle or gamma argument");
        const auto surface_format =
            float_source ? SCE_GXM_COLOR_FORMAT_F32_R
                         : static_cast<SceGxmColorFormat>(SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR | (surface_swizzle << 20));
        // Logical RGBA -> guest memory byte order for the four surface formats.
        constexpr unsigned memory_channels[4][4] = {{0, 1, 2, 3}, {2, 1, 0, 3}, {3, 2, 1, 0}, {3, 0, 1, 2}};
        // GXM four-component formats are named from most to least significant
        // storage byte; expand their logical RGB independently of Metal views.
        constexpr unsigned channels[8][4] = {{0, 1, 2, 3}, {2, 1, 0, 3}, {3, 2, 1, 0}, {1, 2, 3, 0},
                                             {0, 1, 2, 4}, {2, 1, 0, 4}, {3, 2, 1, 4}, {1, 2, 3, 4}};
        MemState mem;
        check(init(mem, false), "Cannot initialize guest memory");
        renderer::metal::MetalContext ctx;
        renderer::metal::MetalState state;
        check(state.init(), "Cannot initialize Metal");
        state.context = &ctx;
        Config config;
        state.late_init(config, "metal-cube-alias-validation", mem);
        state.set_anisotropic_filtering(anisotropy);
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
            const unsigned offset = name == "iPosition"   ? 0
                                    : name == "iTexCoord" ? 16
                                    : name == "iNormal"   ? 32
                                    : name == "iTangent"  ? 48
                                                          : 999;
            check(offset != 999, "Unexpected vertex input: " + name);
            SceGxmVertexAttribute a{};
            a.offset = offset;
            a.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
            a.componentCount = param.component_count;
            a.regIndex = param.resource_index;
            vp->attributes.push_back(a);
        }
        check(vp->attributes.size() == 4, "Expected Unit 13 position/UV/normal/tangent attributes");
        vp->streams.push_back({64, SCE_GXM_INDEX_SOURCE_EACH_VERTEX_16BIT});
        check(renderer::create(vp->renderer_data, state, gxp, state.gxp_ptr_map, vp->attributes),
              "Cannot create vertex program");
        check(renderer::create(fp->renderer_data, state, *fp->program.get(mem), nullptr, state.gxp_ptr_map),
              "Cannot create fragment program");
        execute(renderer::GXMState::Program, vp_addr.cast<void>(), false);
        execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
        Ptr<SceGxmFragmentProgram> reflection_addr(
            alloc(mem, sizeof(SceGxmFragmentProgram), "Cube reflection program"));
        auto *reflection = new (reflection_addr.get(mem)) SceGxmFragmentProgram{};
        reflection->program = load(mem, argv[3]);
        check(renderer::create(reflection->renderer_data, state, *reflection->program.get(mem), nullptr,
                               state.gxp_ptr_map),
              "Cannot create reflection program");
        const float positions[3][4] = {{-1, -1, 0, 1}, {3, -1, 0, 1}, {-1, 3, 0, 1}};
        Ptr<float> stream(alloc(mem, 3 * 64, "Cube vertices"));
        std::memset(stream.get(mem), 0, 3 * 64);
        auto *v = stream.get(mem);
        for (unsigned i = 0; i < 3; ++i) {
            std::memcpy(v + i * 16, positions[i], 16);
            v[i * 16 + 4] = v[i * 16 + 5] = .5f;
            v[i * 16 + 8] = v[i * 16 + 9] = 0;
            v[i * 16 + 10] = 1;
        }
        ctx.record.vertex_streams[0] = {stream.cast<const uint8_t>(), 3 * 64};
        const float matrix[32] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        Ptr<float> uniform(alloc(mem, sizeof(matrix), "Cube identity"));
        std::memcpy(uniform.get(mem), matrix, sizeof(matrix));
        execute(renderer::GXMState::UniformBuffer, uniform.cast<uint8_t>(), true, 0, uint32_t(sizeof(matrix)));
        Ptr<uint8_t> fill_uniform(alloc(mem, 16, "Cube fill"));
        Ptr<uint32_t> reflection_uniform(alloc(mem, 16, "Cube reflection material"));
        const uint32_t material[4] = {0x3c00, 0, sample_2d ? 0x3c000000u : 0u, 0x3c00};
        std::memcpy(reflection_uniform.get(mem), material, 16);
        execute(renderer::GXMState::UniformBuffer, reflection_uniform.cast<uint8_t>(), false, 2, uint32_t(16));
        Ptr<uint16_t> indices(alloc(mem, 6, "Cube indices"));
        for (unsigned i = 0; i < 3; ++i)
            indices.get(mem)[i] = i;
        renderer::metal::MetalRenderTarget target;
        target.width = target.height = 16;
        ctx.current_render_target = &target;
        ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
        ctx.record.front_depth_write_mode = SCE_GXM_DEPTH_WRITE_DISABLED;
        ctx.record.front_stencil_state_op.func = SCE_GXM_STENCIL_FUNC_ALWAYS;
        ctx.record.depth_stencil_surface = {};
        ctx.record.depth_stencil_surface.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_DF32);
        state.disable_surface_sync = true;
        Ptr<uint8_t> base(alloc(mem, 64, "Cube base color"));
        std::memset(base.get(mem), 255, 64);
        SceGxmTexture base_desc{};
        base_desc.type = SCE_GXM_TEXTURE_LINEAR >> 29;
        base_desc.width = base_desc.height = 0;
        base_desc.mip_count = 15;
        base_desc.lod_bias = 31;
        const auto format = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
        base_desc.base_format = (uint32_t(format) >> 24) & 31;
        base_desc.format0 = uint32_t(format) >> 31;
        base_desc.swizzle_format = (uint32_t(format) >> 12) & 7;
        base_desc.data_addr = base.address() >> 2;
        execute(renderer::GXMState::Texture, uint32_t(0), base_desc);
        if (sample_2d) {
            // Zero reflection makes the original Unit 13 material shader return
            // the aliased 2D base color with its own alpha and no cancellation.
            std::memset(base.get(mem), 0, 64);
            auto zero_cube = base_desc;
            zero_cube.type = SCE_GXM_TEXTURE_CUBE >> 29;
            zero_cube.width_base2 = zero_cube.height_base2 = 0;
            execute(renderer::GXMState::Texture, uint32_t(1), zero_cube);
        }
        Ptr<uint8_t> output(alloc(mem, 8 * 8 * 8, "Cube output"));
        unsigned draws = 0, compared = 0;
        // The fill shader promotes its first F16 component to the F32 target.
        // These finite patterns keep producer MSAA resolves well-defined.
        const uint8_t stripe[4] = {241, 29, 83, uint8_t(float_source ? 63 : 255)};
        for (auto source_mode : {SCE_GXM_MULTISAMPLE_NONE, SCE_GXM_MULTISAMPLE_2X, SCE_GXM_MULTISAMPLE_4X})
            for (unsigned scale : {1u, 2u})
                for (unsigned render_mask : {1u, 63u})
                    for (bool mipmaps : {false, true}) {
                        state.res_multiplier = scale;
                        const unsigned levels =
                            mipmaps ? (mip_chain ? std::bit_width(std::min(base_width, base_height)) : 2) : 1;
                        const auto pitch = [&](unsigned size) {
                            return tiled ? 32u : (sample_2d && !swizzled ? ((size + 7) & ~7u) : size);
                        };
                        unsigned offsets[5]{}, storage = 0;
                        for (unsigned mip = 0; mip < levels; ++mip) {
                            offsets[mip] = storage;
                            const unsigned layout_w = (mipmaps ? std::bit_ceil(base_width) : base_width) >> mip;
                            const unsigned layout_h = (mipmaps ? std::bit_ceil(base_height) : base_height) >> mip;
                            storage += pitch(layout_w) * (tiled ? 32u : layout_h) * 4;
                        }
                        const unsigned face_stride =
                            mip_chain ? storage : (tiled ? (mipmaps ? 8192 : 4096) : (mipmaps ? 2048 : 1024));
                        const auto rendered = [&](unsigned face, unsigned mip) {
                            return (render_mask & (1u << face)) && !(mip_chain && (face & 1) && (mip & 1));
                        };
                        Ptr<uint8_t> cube(alloc(mem, 6 * face_stride, "Cube faces"));
                        std::memset(cube.get(mem), 0xcd, 6 * face_stride);
                        uint8_t colors[6][5][4];
                        for (unsigned face = 0; face < 6; ++face)
                            for (unsigned mip = 0; mip < levels; ++mip) {
                                const unsigned size = base_width >> mip, height = base_height >> mip,
                                               offset = face * face_stride + offsets[mip];
                                const uint8_t cpu_color[4] = {uint8_t(20 + face * 31), uint8_t(40 + mip * 80),
                                                              uint8_t(200 - face * 23),
                                                              uint8_t(float_source ? 62 : 255)};
                                for (unsigned pixel = 0; pixel < (pitch(size) * (tiled ? 32u : height)); ++pixel)
                                    std::memcpy(cube.get(mem) + offset + pixel * 4, cpu_color, 4);
                                if (gradient && !rendered(face, mip))
                                    for (unsigned y = 0; y < height; ++y)
                                        for (unsigned x = 0; x < size; ++x) {
                                            const auto pixel = ram_pixel(x, y, size, height);
                                            std::memcpy(cube.get(mem) + offset + (y * pitch(size) + x) * 4,
                                                        pixel.data(), 4);
                                        }
                                std::memcpy(colors[face][mip], cpu_color, 4);
                                if (!rendered(face, mip))
                                    continue;
                                colors[face][mip][0] += 37;
                                colors[face][mip][2] -= 17;
                                auto &surface = ctx.record.color_surface;
                                surface = {};
                                surface.width = size;
                                surface.height = height;
                                surface.strideInPixels = pitch(size);
                                surface.colorFormat = surface_format;
                                surface.gamma = surface_gamma;
                                surface.outputRegisterSize = SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT;
                                surface.surfaceType = tiled ? SCE_GXM_COLOR_SURFACE_TILED
                                                            : (sample_2d && !swizzled ? SCE_GXM_COLOR_SURFACE_LINEAR
                                                                                      : SCE_GXM_COLOR_SURFACE_SWIZZLED);
                                surface.downscale = source_mode != SCE_GXM_MULTISAMPLE_NONE;
                                target.multisample_mode = source_mode;
                                surface.data = Ptr<void>(cube.address() + offset);
                                execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
                                execute(renderer::GXMState::UniformBuffer, fill_uniform, false, 0, uint32_t(16));
                                std::memset(fill_uniform.get(mem), 0, 16);
                                std::memcpy(fill_uniform.get(mem), colors[face][mip], 4);
                                execute(renderer::GXMState::Viewport, false, size * .5f, height * .5f, 0.f, size * .5f,
                                        height * .5f, 1.f);
                                execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0),
                                        uint32_t(size), uint32_t(0), uint32_t(height));
                                state.set_context(ctx, mem);
                                state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                                           indices.get(mem), 3, 1);
                                if (scale == 2) {
                                    // One native pixel wide, half a guest pixel. A guest-sized
                                    // readback/reupload would incorrectly widen this stripe.
                                    std::memcpy(fill_uniform.get(mem), stripe, 4);
                                    execute(renderer::GXMState::Viewport, false, .25f, height * .5f, 0.f, .25f,
                                            height * .5f, 1.f);
                                    state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                                               indices.get(mem), 3, 1);
                                }
                            }
                        SceGxmTexture cube_desc = base_desc;
                        cube_desc.swizzle_format = swizzle;
                        cube_desc.gamma_mode = texture_gamma;
                        cube_desc.type = SCE_GXM_TEXTURE_CUBE >> 29;
                        cube_desc.width_base2 = cube_desc.height_base2 = 4;
                        cube_desc.data_addr = cube.address() >> 2;
                        cube_desc.mip_count = mipmaps ? 1 : 15;
                        for (unsigned revision = 0; revision < 2; ++revision) {
                            if (revision) {
                                // Refresh only the first face's last mip; other cached
                                // faces and RAM-only faces must keep their own values.
                                const unsigned mip = levels - 1, size = base_width >> mip, height = base_height >> mip;
                                colors[0][mip][1] = 211;
                                auto &surface = ctx.record.color_surface;
                                surface = {};
                                surface.width = size;
                                surface.height = height;
                                surface.strideInPixels = pitch(size);
                                surface.colorFormat = surface_format;
                                surface.gamma = surface_gamma;
                                surface.outputRegisterSize = SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT;
                                surface.surfaceType = tiled ? SCE_GXM_COLOR_SURFACE_TILED
                                                            : (sample_2d && !swizzled ? SCE_GXM_COLOR_SURFACE_LINEAR
                                                                                      : SCE_GXM_COLOR_SURFACE_SWIZZLED);
                                surface.downscale = source_mode != SCE_GXM_MULTISAMPLE_NONE;
                                target.multisample_mode = source_mode;
                                surface.data = Ptr<void>(cube.address() + offsets[mip]);
                                execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
                                execute(renderer::GXMState::UniformBuffer, fill_uniform, false, 0, uint32_t(16));
                                std::memcpy(fill_uniform.get(mem), colors[0][mip], 4);
                                execute(renderer::GXMState::Viewport, false, size * .5f, height * .5f, 0.f, size * .5f,
                                        height * .5f, 1.f);
                                state.set_context(ctx, mem);
                                state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                                           indices.get(mem), 3, 1);
                            }
                            // Exercise all LODs of one unchanged image consecutively.
                            for (unsigned face = 0; face < 6; ++face)
                                for (unsigned mip = 0; mip < levels; ++mip)
                                    for (unsigned sample = 0; sample < (gradient ? 8u : 3u); ++sample) {
                                        const unsigned size = base_width >> mip;
                                        const unsigned view_size = crop ? size / 2 : size;
                                        const unsigned crop_x = crop && (face & 1) ? 2 : 0, crop_y = crop ? 1 : 0;
                                        float u = sample == 0 ? .5f : (sample == 1 ? .25f : .75f) / view_size, t = 0,
                                              sc = 2 * u - 1;
                                        if (gradient && sample >= 3)
                                            u = sample == 3   ? 1.25f / view_size
                                                : sample == 4 ? -.25f / view_size
                                                              : 1 + .25f / view_size;
                                        if (gradient && sample >= 6)
                                            u = (sample == 6 ? .49f : .51f) / view_size;
                                        const float direction[6][3] = {{1, -t, -sc}, {-1, -t, sc}, {sc, 1, t},
                                                                       {sc, -1, -t}, {sc, -t, 1},  {-sc, -t, -1}};
                                        for (unsigned i = 0; i < 3; ++i) {
                                            if (sample_2d) {
                                                v[i * 16 + 4] = u;
                                                v[i * 16 + 5] = .5f;
                                            }
                                            v[i * 16 + 12] = -direction[face][0];
                                            v[i * 16 + 13] = -direction[face][1];
                                            v[i * 16 + 14] = direction[face][2];
                                            v[i * 16 + 15] = 1;
                                        }
                                        auto &surface = ctx.record.color_surface;
                                        surface = {};
                                        surface.width = surface.height = surface.strideInPixels = 8;
                                        surface.colorFormat = SCE_GXM_COLOR_FORMAT_F32F32_GR;
                                        surface.outputRegisterSize = SCE_GXM_OUTPUT_REGISTER_SIZE_64BIT;
                                        surface.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
                                        target.multisample_mode = SCE_GXM_MULTISAMPLE_NONE;
                                        surface.data = output.cast<void>();
                                        cube_desc.lod_min0 = mip;
                                        if (sample_2d) {
                                            auto alias = base_desc;
                                            alias.type =
                                                (crop ? SCE_GXM_TEXTURE_LINEAR_STRIDED : SCE_GXM_TEXTURE_LINEAR) >> 29;
                                            alias.width = alias.height = view_size - 1;
                                            if (tiled)
                                                alias.type = SCE_GXM_TEXTURE_TILED >> 29;
                                            if (swizzled) {
                                                alias.type = SCE_GXM_TEXTURE_SWIZZLED >> 29;
                                                alias.width_base2 = alias.height_base2 = 4 - mip;
                                            }
                                            alias.swizzle_format = swizzle;
                                            alias.gamma_mode = texture_gamma;
                                            alias.data_addr = (cube.address() + face * face_stride + offsets[mip] +
                                                               (crop_y * size + crop_x) * 4) >>
                                                              2;
                                            if (mip_chain) {
                                                alias.width = base_width - 1;
                                                alias.height = base_height - 1;
                                                if (swizzled) {
                                                    alias.width_base2 = std::bit_width(base_width) - 1;
                                                    alias.height_base2 = std::bit_width(base_height) - 1;
                                                }
                                                alias.data_addr = (cube.address() + face * face_stride) >> 2;
                                                alias.mip_count = mipmaps ? levels - 1 : 15;
                                                alias.lod_min0 = mip & 3;
                                                alias.lod_min1 = mip >> 2;
                                            }
                                            if (gradient) {
                                                alias.uaddr_mode = alias.vaddr_mode = address_mode;
                                                alias.min_filter = linear ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                                                          : SCE_GXM_TEXTURE_FILTER_POINT;
                                                alias.mag_filter = linear ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                                                          : SCE_GXM_TEXTURE_FILTER_POINT;
                                            }
                                            if (crop) {
                                                const unsigned stride = size - 1;
                                                alias.mip_filter = stride & 1;
                                                alias.min_filter = (stride >> 1) & 3;
                                                alias.mip_count = (stride >> 3) & 15;
                                                alias.lod_bias = stride >> 7;
                                            }
                                            execute(renderer::GXMState::Texture, uint32_t(0), alias);
                                        } else
                                            execute(renderer::GXMState::Texture, uint32_t(1), cube_desc);
                                        execute(renderer::GXMState::Program, reflection_addr.cast<void>(), true);
                                        execute(renderer::GXMState::UniformBuffer, reflection_uniform.cast<uint8_t>(),
                                                false, 2, uint32_t(16));
                                        execute(renderer::GXMState::Viewport, false, 4.f, 4.f, 0.f, 4.f, 4.f, 1.f);
                                        state.set_context(ctx, mem);
                                        state.draw(ctx, mem, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                                                   indices.get(mem), 3, 1);
                                        check(state.sync_surface(mem, surface), "Cannot read reflection output");
                                        const bool stripe_expected = scale == 2 && sample == 1 && !crop_x &&
                                                                     rendered(face, mip) &&
                                                                     !(revision && face == 0 && mip == levels - 1);
                                        const auto *expected_color = stripe_expected ? stripe : colors[face][mip];
                                        uint8_t memory_color[4];
                                        for (unsigned c = 0; c < 4; ++c) {
                                            const bool rendered_face = rendered(face, mip);
                                            const unsigned logical =
                                                rendered_face ? memory_channels[surface_swizzle][c] : c;
                                            double value = expected_color[logical] / 255.0;
                                            if (rendered_face && surface_gamma && logical < 3)
                                                value = value <= .0031308 ? 12.92 * value
                                                                          : 1.055 * std::pow(value, 1 / 2.4) - .055;
                                            memory_color[c] = uint8_t(std::round(value * 255));
                                        }
                                        if (float_source && rendered(face, mip)) {
                                            __fp16 half;
                                            std::memcpy(&half, expected_color, sizeof(half));
                                            const float stored = float(half);
                                            std::memcpy(memory_color, &stored, sizeof(stored));
                                        }
                                        uint8_t sampled[4];
                                        for (unsigned c = 0; c < 4; ++c) {
                                            const unsigned channel = channels[swizzle][c];
                                            double value = channel == 4 ? 1.0 : memory_color[channel] / 255.0;
                                            if (texture_gamma && channel < (texture_gamma == 3 ? 2u : 3u))
                                                value = value <= .04045 ? value / 12.92
                                                                        : std::pow((value + .055) / 1.055, 2.4);
                                            sampled[c] = uint8_t(std::round(value * 255));
                                        }
                                        if (gradient) {
                                            const bool gpu = rendered(face, mip);
                                            const unsigned w = size * (gpu ? scale : 1),
                                                           h = (base_height >> mip) * (gpu ? scale : 1);
                                            const float px = u * w - (linear ? .5f : 0.f),
                                                        py = .5f * h - (linear ? .5f : 0.f);
                                            const int ix = int(std::floor(px)), iy = int(std::floor(py));
                                            const double fx = linear ? px - ix : 0, fy = linear ? py - iy : 0;
                                            const auto pixel_at = [&](int x, int y) {
                                                x = address_coordinate(x, w);
                                                y = address_coordinate(y, h);
                                                if (!gpu)
                                                    return ram_pixel(x, y, w, h);
                                                std::array<uint8_t, 4> pixel;
                                                const bool is_stripe = scale == 2 && x == 0 &&
                                                                       !(revision && face == 0 && mip == levels - 1);
                                                std::memcpy(pixel.data(), is_stripe ? stripe : colors[face][mip], 4);
                                                return pixel;
                                            };
                                            const auto a = pixel_at(ix, iy), b = pixel_at(ix + 1, iy),
                                                       c = pixel_at(ix, iy + 1), d = pixel_at(ix + 1, iy + 1);
                                            for (unsigned channel = 0; channel < 4; ++channel)
                                                sampled[channel] = uint8_t(
                                                    std::round((a[channel] * (1 - fx) + b[channel] * fx) * (1 - fy) +
                                                               (c[channel] * (1 - fx) + d[channel] * fx) * fy));
                                        }
                                        // A swizzle can move a gamma-decoded color into alpha.
                                        // Hardware sRGB error can cross a half-rounding boundary
                                        // (captured .445177 vs ideal .445201 -> U8 113 vs 114).
                                        // Give that color the same tolerance regardless of its
                                        // destination channel; raw alpha and constants stay exact.
                                        const unsigned ac = channels[swizzle][3];
                                        const bool gamma_alpha =
                                            sample_2d && ac < 4 &&
                                            ((texture_gamma && ac < (texture_gamma == 3 ? 2u : 3u)) ||
                                             (surface_gamma && rendered(face, mip) &&
                                              memory_channels[surface_swizzle][ac] < 3));
                                        auto *actual = output.get(mem);
                                        for (unsigned pixel = 0; pixel < 64; ++pixel)
                                            for (unsigned c = 0; c < 8; ++c) {
                                                const unsigned expected = c < (sample_2d ? 4u : 3u) ? sampled[c]
                                                                          : c == 3                  ? 255
                                                                          : c == 6                  ? 127
                                                                                                    : 0;
                                                check(std::abs(int(actual[pixel * 8 + c]) - int(expected)) <=
                                                          (c < 3 || (c == 3 && gamma_alpha) ? 1 : 0),
                                                      "Alias mismatch shape=" + shape + " mode=" + texture_mode +
                                                          " source=" + source_kind +
                                                          " swizzle=" + std::to_string(swizzle) +
                                                          " surface_swizzle=" + std::to_string(surface_swizzle) +
                                                          " source_gamma=" + std::to_string(surface_gamma) +
                                                          " texture_gamma=" + std::to_string(texture_gamma) +
                                                          " scale=" + std::to_string(scale) +
                                                          " mask=" + std::to_string(render_mask) + " revision=" +
                                                          std::to_string(revision) + " face=" + std::to_string(face) +
                                                          " mip=" + std::to_string(mip) + " sample=" +
                                                          std::to_string(sample) + " byte=" + std::to_string(c) +
                                                          " actual=" + std::to_string(actual[pixel * 8 + c]) +
                                                          " expected=" + std::to_string(expected));
                                                ++compared;
                                            }
                                        ++draws;
                                    }
                        }
                    }
        state.context = nullptr;
        vp->~SceGxmVertexProgram();
        fp->~SceGxmFragmentProgram();
        reflection->~SceGxmFragmentProgram();
        std::cout << "PASS alias mode=" << texture_mode << " shape=" << shape << " source=" << source_kind
                  << " anisotropy=" << anisotropy << " addressing=" << addressing << " filtering=" << filtering
                  << " swizzle=" << swizzle << " surface_swizzle=" << surface_swizzle
                  << " source_gamma=" << surface_gamma << " texture_gamma=" << texture_gamma << " draws=" << draws
                  << " material_bytes=" << compared
                  << " stale CPU storage, mixed/rendered faces,mips,refresh,native-pixel stripes,1x/2x,1/2/4-sample "
                     "producers\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
