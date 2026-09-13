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
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
static Ptr<const SceGxmProgram> load(MemState &mem, const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(bool(file), "Cannot open GXP fixture");
    std::streamsize size = file.tellg();
    check(size >= sizeof(SceGxmProgram) && size < 64 * 1024 * 1024, "Invalid GXP fixture length");
    Ptr<SceGxmProgram> result(alloc(mem, size, "Metal GXP fixture"));
    check(bool(result), "Cannot allocate guest GXP");
    file.seekg(0); check(bool(file.read(reinterpret_cast<char *>(result.get(mem)), size)), "Cannot read GXP");
    check(result.get(mem)->magic == 0x00505847 && result.get(mem)->size <= size, "Invalid GXP fixture");
    return result.cast<const SceGxmProgram>();
}

int main(int argc, char **argv) {
    if (argc != 3) { std::cerr << "Usage: metal-gxm-validation <Sly 25188... vertex.gxp> <Sly 028582... fragment.gxp>\n"; return 2; }
    try {
        MemState mem;
        check(init(mem, false), "Cannot initialize guest memory");
        std::unique_ptr<renderer::Context> owner = std::make_unique<renderer::metal::MetalContext>();
        renderer::metal::MetalState state;
        check(state.init(), "Cannot initialize native Metal");
        auto &ctx = static_cast<renderer::metal::MetalContext &>(*owner);
        state.context = &ctx;
        {
            renderer::metal::MetalContext upload_ctx;
            upload_ctx.alloc_func = [] { return new renderer::Command{}; };
            upload_ctx.free_func = [](renderer::Command *command) { delete command; };
            const Ptr<const void> buffer(0x10000);
            // The public command used to truncate large buffers to 16 bits.
            renderer::set_uniform_buffer(state, &upload_ctx, false, 4, 522240, buffer);
            std::unique_ptr<renderer::Command> command(upload_ctx.command_list.first);
            check(bool(command) && command->opcode == renderer::CommandOpcode::SetState, "Missing uniform command");
            renderer::CommandHelper reader(command.get());
            check(reader.pop<renderer::GXMState>() == renderer::GXMState::UniformBuffer, "Wrong uniform state opcode");
            check(reader.pop<Ptr<const void>>() == buffer && !reader.pop<bool>() && reader.pop<int>() == 4,
                "Uniform command lost its binding");
            check(reader.pop<uint32_t>() == 522240, "Native uniform range above 64 KiB was truncated");
            upload_ctx.command_list = {};
            std::cout << "PASS public GXM uniform command preserves the complete 522240-byte buffer range\n";
        }
        Config config;
        // Separate-process integration coverage: the production set_app path
        // must honor the existing shader-cache setting and per-title deletion.
        if (const char *cache = std::getenv("VITA3K_METAL_TEST_CACHE")) {
            state.cache_path = cache;
            config.shader_cache = std::getenv("VITA3K_METAL_TEST_CACHE_DISABLED") == nullptr;
            state.late_init(config, "CACHE_TEST", mem);
            static_cast<renderer::State &>(state).set_app("CACHE_TEST", "eboot.bin");
        }
        auto execute = [&](renderer::GXMState command_state, auto... args) {
            renderer::Command command{};
            renderer::CommandHelper writer(&command);
            check(writer.push(command_state), "Cannot write GXM state opcode");
            check((writer.push(args) && ...), "GXM test command payload overflow");
            renderer::CommandHelper reader(&command);
            renderer::cmd_handle_set_state(state, mem, config, reader, state.features, &ctx);
        };
        Ptr<SceGxmVertexProgram> vp_addr(alloc(mem, sizeof(SceGxmVertexProgram), "Metal vertex patcher"));
        Ptr<SceGxmFragmentProgram> fp_addr(alloc(mem, sizeof(SceGxmFragmentProgram), "Metal fragment patcher"));
        auto *vp = new (vp_addr.get(mem)) SceGxmVertexProgram{};
        auto *fp = new (fp_addr.get(mem)) SceGxmFragmentProgram{};
        vp->program = load(mem, argv[1]); fp->program = load(mem, argv[2]);
        SceGxmVertexAttribute attr{};
        attr.streamIndex = 0; attr.offset = 0; attr.format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
        attr.componentCount = 4; attr.regIndex = 0;
        vp->attributes.push_back(attr);
        vp->streams.push_back({16, SCE_GXM_INDEX_SOURCE_EACH_VERTEX_16BIT});
        check(renderer::create(vp->renderer_data, state, *vp->program.get(mem), state.gxp_ptr_map, vp->attributes), "Cannot create vertex program");
        check(renderer::create(fp->renderer_data, state, *fp->program.get(mem), nullptr, state.gxp_ptr_map), "Cannot create fragment program");
        ctx.shader_hints = {};
        renderer::metal::MetalRenderTarget target;
        target.width = target.height = 128; target.multisample_mode = SCE_GXM_MULTISAMPLE_NONE;
        ctx.current_render_target = &target;
        ctx.record.depth_stencil_surface.background_depth = 1;
        ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_ALWAYS;
        const float vertices[] = {-1,-1,1,1, 3,-1,1,1, -1,3,1,1};
        Ptr<float> vertex_data(alloc(mem, sizeof(vertices), "Metal vertices"));
        std::memcpy(vertex_data.get(mem), vertices, sizeof(vertices));
        ctx.record.vertex_streams[0] = {vertex_data.cast<const uint8_t>(), sizeof(vertices)};
        execute(renderer::GXMState::Program, vp_addr.cast<void>(), false);
        execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
        execute(renderer::GXMState::Viewport, false, 64.0f, 64.0f, 0.0f, 64.0f, 64.0f, 1.0f);
        Ptr<float> color(alloc(mem, 16, "Metal uniform color"));
        const float values[] = {0.25f, 0.5f, 0.75f, 1};
        std::memcpy(color.get(mem), values, sizeof(values));
        execute(renderer::GXMState::UniformBuffer, color.cast<uint8_t>(), false, 0, uint32_t(16));
        Ptr<uint16_t> indices(alloc(mem, 6, "Metal indices"));
        indices.get(mem)[0] = 0; indices.get(mem)[1] = 1; indices.get(mem)[2] = 2;
        auto draw_primitive = [&](SceGxmPrimitiveType primitive, SceGxmIndexFormat index_format, Ptr<const void> index_ptr, uint32_t count) {
            renderer::Command draw{};
            renderer::CommandHelper writer(&draw);
            uint32_t instances = 1;
            writer.push(primitive); writer.push(index_format); writer.push(index_ptr); writer.push(count); writer.push(instances);
            renderer::CommandHelper reader(&draw);
            renderer::cmd_handle_draw(state, mem, config, reader, state.features, &ctx);
        };
        auto draw_triangle = [&] { draw_primitive(SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, indices.cast<const void>(), 3); };
        for (auto mode : {SCE_GXM_REGION_CLIP_NONE, SCE_GXM_REGION_CLIP_INSIDE, SCE_GXM_REGION_CLIP_OUTSIDE, SCE_GXM_REGION_CLIP_ALL}) {
            Ptr<uint32_t> color_data(alloc(mem, 128*128*4, "Metal framebuffer"));
            std::memset(color_data.get(mem), 0, 128*128*4);
            ctx.record.color_surface = {};
            ctx.record.color_surface.data = color_data.cast<void>();
            ctx.record.color_surface.width = ctx.record.color_surface.height = ctx.record.color_surface.strideInPixels = 128;
            ctx.record.color_surface.colorFormat = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
            ctx.record.color_surface.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
            state.set_context(ctx, mem);
            execute(renderer::GXMState::RegionClip, mode, uint32_t(32), uint32_t(96), uint32_t(32), uint32_t(96));
            draw_triangle();
            state.finish(ctx);
            const auto *pixels = reinterpret_cast<const uint8_t *>(color_data.get(mem));
            const int expected[] = {64,128,191,255};
            for (uint32_t y = 0; y < 128; ++y) for (uint32_t x = 0; x < 128; ++x) {
                bool inside = x >= 32 && x < 96 && y >= 32 && y < 96;
                bool drawn = mode == SCE_GXM_REGION_CLIP_NONE || (mode == SCE_GXM_REGION_CLIP_INSIDE && !inside) || (mode == SCE_GXM_REGION_CLIP_OUTSIDE && inside);
                for (int c = 0; c < 4; ++c)
                    check(std::abs(int(pixels[(y*128+x)*4+c]) - (drawn ? expected[c] : 0)) <= 1,
                        "GXM pixel mismatch mode=" + std::to_string(mode) + " x=" + std::to_string(x) + " y=" + std::to_string(y) + " channel=" + std::to_string(c) + " actual=" + std::to_string(pixels[(y*128+x)*4+c]));
            }
            std::cout << "PASS GXM dispatch -> native indexed draw -> guest readback, clip mode=" << uint32_t(mode) << '\n';
        }
        {
            // Exercise the actual GXP mask shader and attachment transitions at
            // an edge through a pixel center, including unresolved sample storage.
            const auto saved_surface=ctx.record.color_surface;
            const auto saved_stream=ctx.record.vertex_streams[0];
            const auto saved_stencil=ctx.record.front_stencil_state_op;
            const auto saved_stencil_values=ctx.record.front_stencil_state_values;
            const float quad[]={-1,-1,1,1, 1,-1,1,1, 1,1,1,1, -1,1,1,1};
            Ptr<float> quad_data(alloc(mem,sizeof(quad),"MSAA coverage quad"));
            std::memcpy(quad_data.get(mem),quad,sizeof(quad));
            Ptr<uint16_t> quad_indices(alloc(mem,8,"MSAA quad indices"));
            for(unsigned i=0;i<4;++i) quad_indices.get(mem)[i]=i;
            ctx.record.vertex_streams[0]={quad_data.cast<const uint8_t>(),sizeof(quad)};
            auto draw_quad=[&] { draw_primitive(SCE_GXM_PRIMITIVE_TRIANGLE_FAN,SCE_GXM_INDEX_FORMAT_U16,quad_indices.cast<const void>(),4); };
            auto encode=[](float linear) { return linear<=0.0031308f ? linear*12.92f : 1.055f*std::pow(linear,1.0f/2.4f)-0.055f; };
            for(auto mode : {SCE_GXM_MULTISAMPLE_2X,SCE_GXM_MULTISAMPLE_4X}) for(bool expanded : {false,true}) for(bool gamma : {false,true}) for(bool copy : {false,true}) {
                const uint32_t samples=mode==SCE_GXM_MULTISAMPLE_2X ? 2 : 4;
                const uint32_t sx=expanded ? samples/2 : 1, sy=expanded ? 2 : 1;
                const uint32_t w=128*sx,h=128*sy;
                target.multisample_mode=mode;
                Ptr<void> pixels(alloc(mem,size_t(w)*h*4,"MSAA guest color"));
                std::memset(pixels.get(mem),0,size_t(w)*h*4);
                ctx.record.color_surface={};
                auto &surface=ctx.record.color_surface;
                surface.data=pixels; surface.width=surface.strideInPixels=w; surface.height=h;
                surface.colorFormat=SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
                surface.surfaceType=SCE_GXM_COLOR_SURFACE_LINEAR;
                surface.downscale=!expanded; surface.gamma=gamma;
                state.set_context(ctx,mem);
                execute(renderer::GXMState::RegionClip,SCE_GXM_REGION_CLIP_NONE,uint32_t(0),uint32_t(128),uint32_t(0),uint32_t(128));
                // Each new scene explicitly resets mask storage before carving
                // its fractional edge, since same-size attachments can persist.
                fp->is_maskupdate=true;
                execute(renderer::GXMState::Program,fp_addr.cast<void>(),true);
                execute(renderer::GXMState::StencilFunc,true,SCE_GXM_STENCIL_FUNC_ALWAYS,
                    SCE_GXM_STENCIL_OP_KEEP,SCE_GXM_STENCIL_OP_KEEP,SCE_GXM_STENCIL_OP_KEEP,uint8_t(255),uint8_t(255));
                execute(renderer::GXMState::Viewport,false,64.0f,64.0f,0.0f,64.0f,64.0f,1.0f);
                draw_quad();
                execute(renderer::GXMState::StencilFunc,true,SCE_GXM_STENCIL_FUNC_NEVER,
                    SCE_GXM_STENCIL_OP_KEEP,SCE_GXM_STENCIL_OP_KEEP,SCE_GXM_STENCIL_OP_KEEP,uint8_t(255),uint8_t(255));
                execute(renderer::GXMState::Viewport,false,31.75f,64.0f,0.0f,31.75f,64.0f,1.0f);
                draw_quad();
                // Force a command-buffer boundary with the partial mask stored.
                state.finish(ctx);
                fp->is_maskupdate=false;
                execute(renderer::GXMState::Program,fp_addr.cast<void>(),true);
                execute(renderer::GXMState::Viewport,false,64.0f,64.0f,0.0f,64.0f,64.0f,1.0f);
                draw_quad(); state.finish(ctx);
                const auto *bytes=static_cast<const uint8_t *>(pixels.get(mem));
                for(uint32_t y=0;y<128;++y) for(uint32_t x=0;x<128;++x) {
                    unsigned covered=0;
                    for(uint32_t by=0;by<sy;++by) for(uint32_t bx=0;bx<sx;++bx) {
                        const auto *pixel=bytes+((y*sy+by)*w+x*sx+bx)*4;
                        const float coverage=x<63 ? 0.0f : x>63 ? 1.0f : expanded ? (pixel[3]>127 ? 1.0f : 0.0f) : 0.5f;
                        covered+=pixel[3]>127;
                        for(unsigned c=0;c<4;++c) {
                            const float linear=values[c]*coverage;
                            const int expected=int(std::lround(255*(gamma && c<3 ? encode(linear) : linear)));
                            check(std::abs(int(pixel[c])-expected)<=2,"MSAA GXM mask/resolve mismatch samples="+std::to_string(samples)
                                +" expanded="+std::to_string(expanded)+" gamma="+std::to_string(gamma)+" x="+std::to_string(x)+" y="+std::to_string(y));
                        }
                    }
                    if(expanded && x==63) check(covered==samples/2,"MSAA edge lost individual sample coverage samples="+std::to_string(samples)+" covered="+std::to_string(covered)+" y="+std::to_string(y)+" alpha0="+std::to_string(bytes[(y*sy*w+x*sx)*4+3])+" alpha1="+std::to_string(bytes[((y*sy+1)*w+x*sx)*4+3]));
                }
                check(state.sync_surface(mem,surface),"MSAA explicit surface synchronization rejected");
                const std::vector<uint8_t> before_transfer(bytes,bytes+size_t(w)*h*4);
                SceGxmTransferImage byte_fill{};
                const size_t modified=(size_t(100*sy)*w+100*sx)*4+2;
                byte_fill.address=surface.data.cast<uint8_t>()+modified;
                byte_fill.format=SCE_GXM_TRANSFER_FORMAT_U8_R;
                byte_fill.width=byte_fill.height=1; byte_fill.stride=w*4;
                if(copy) {
                    Ptr<uint8_t> source_byte(alloc(mem,1,"MSAA byte copy source"));
                    *source_byte.get(mem)=0x5a;
                    auto source=byte_fill;
                    source.address=source_byte; source.stride=1;
                    check(state.transfer_copy(mem,source,byte_fill,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
                        SCE_GXM_TRANSFER_COLORKEY_NONE,0,0),"MSAA byte copy rejected");
                } else {
                    check(state.transfer_fill(mem,byte_fill,0x5a),"MSAA byte fill rejected");
                }
                // A later masked draw exposes collapsed samples at a different
                // pixel, even when their first resolved average was unchanged.
                const float after_color[]={0.5f,0.25f,0.125f,1};
                std::memcpy(color.get(mem),after_color,sizeof(after_color));
                execute(renderer::GXMState::RegionClip,SCE_GXM_REGION_CLIP_OUTSIDE,uint32_t(63),uint32_t(64),uint32_t(0),uint32_t(128));
                draw_quad(); state.finish(ctx);
                check(state.sync_surface(mem,surface),"MSAA transfer result could not be synchronized");
                for(uint32_t y=0;y<h;++y) for(uint32_t x=0;x<w;++x) for(unsigned c=0;c<4;++c) {
                    const size_t offset=(size_t(y)*w+x)*4+c;
                    int expected=offset==modified ? 0x5a : before_transfer[offset];
                    if(x/sx==63) {
                        const float coverage=expanded ? (before_transfer[(size_t(y)*w+x)*4+3]>127 ? 1.0f : 0.0f) : 0.5f;
                        const float linear=after_color[c]*coverage;
                        expected=int(std::lround(255*(gamma && c<3 ? encode(linear) : linear)));
                    }
                    check(std::abs(int(bytes[offset])-expected)<=2,"MSAA transfer lost unrelated samples/components offset="+std::to_string(offset)
                        +" actual="+std::to_string(bytes[offset])+" expected="+std::to_string(expected));
                }
                std::memcpy(color.get(mem),values,sizeof(values));
                // An empty scene must retain its published surface.
                const std::vector<uint8_t> before_empty(bytes,bytes+size_t(w)*h*4);
                state.set_context(ctx,mem); state.finish(ctx);
                check(state.sync_surface(mem,surface),"MSAA empty scene synchronization rejected");
                check(std::equal(before_empty.begin(),before_empty.end(),bytes),"MSAA empty scene changed published bytes");
                std::cout << "PASS GXM MSAA fractional mask edge, pass split, byte transfer, masked redraw and empty scene samples=" << samples
                    << " expanded=" << expanded << " gamma=" << gamma << " copy=" << copy << '\n';
            }
            target.multisample_mode=SCE_GXM_MULTISAMPLE_NONE;
            ctx.record.color_surface=saved_surface;
            ctx.record.vertex_streams[0]=saved_stream;
            ctx.record.front_stencil_state_op=saved_stencil;
            ctx.record.front_stencil_state_values=saved_stencil_values;
            std::memcpy(color.get(mem),values,sizeof(values));
            state.set_context(ctx,mem); state.finish(ctx);
        }
        {
            // GL/Vulkan use CCW as their API front face after the GXM viewport
            // transform. Cull, per-side fragment disable and stencil must agree
            // on that same classification, including mirrored viewports.
            const auto saved_surface=ctx.record.color_surface;
            const auto saved_depth_surface=ctx.record.depth_stencil_surface;
            ctx.record.depth_stencil_surface.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_DF32);
            ctx.record.depth_stencil_surface.set_type(SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR);
            const auto saved_back_depth_func=ctx.record.back_depth_func;
            const auto saved_back_depth_write=ctx.record.back_depth_write_mode;
            ctx.record.back_depth_func=ctx.record.front_depth_func;
            ctx.record.back_depth_write_mode=ctx.record.front_depth_write_mode;
            const auto saved_front_op=ctx.record.front_stencil_state_op;
            const auto saved_back_op=ctx.record.back_stencil_state_op;
            const auto saved_front_values=ctx.record.front_stencil_state_values;
            const auto saved_back_values=ctx.record.back_stencil_state_values;
            execute(renderer::GXMState::RegionClip,SCE_GXM_REGION_CLIP_NONE,uint32_t(0),uint32_t(128),uint32_t(0),uint32_t(128));
            execute(renderer::GXMState::TwoSided,SCE_GXM_TWO_SIDED_ENABLED);
            auto render_face=[&](SceGxmCullMode cull, bool disable_front, bool disable_back, bool stencil) {
                execute(renderer::GXMState::CullMode,cull);
                execute(renderer::GXMState::FragmentProgramEnable,true,disable_front && !stencil?SCE_GXM_FRAGMENT_PROGRAM_DISABLED:SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
                execute(renderer::GXMState::FragmentProgramEnable,false,disable_back && !stencil?SCE_GXM_FRAGMENT_PROGRAM_DISABLED:SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
                for(bool front:{true,false}) execute(renderer::GXMState::StencilFunc,front,
                    stencil && (front?disable_front:disable_back)?SCE_GXM_STENCIL_FUNC_NEVER:SCE_GXM_STENCIL_FUNC_ALWAYS,
                    SCE_GXM_STENCIL_OP_KEEP,SCE_GXM_STENCIL_OP_KEEP,SCE_GXM_STENCIL_OP_KEEP,uint8_t(255),uint8_t(255));
                Ptr<uint8_t> pixels(alloc(mem,128*128*4,"Metal front/back fixture"));
                std::memset(pixels.get(mem),0,128*128*4);
                ctx.record.color_surface.data=pixels.cast<void>();
                state.set_context(ctx,mem);draw_triangle();state.finish(ctx);
                return std::vector<uint8_t>(pixels.get(mem),pixels.get(mem)+128*128*4);
            };
            for(float sx:{64.0f,-64.0f}) for(float sy:{64.0f,-64.0f}) for(bool reverse:{false,true}) {
                execute(renderer::GXMState::Viewport,false,64.0f,64.0f,0.0f,sx,sy,1.0f);
                indices.get(mem)[1]=reverse?2:1;indices.get(mem)[2]=reverse?1:2;
                const auto without_front=render_face(SCE_GXM_CULL_CCW,false,false,false);
                const auto without_back=render_face(SCE_GXM_CULL_CW,false,false,false);
                // Every full-target pixel must survive exactly one cull mode.
                for(size_t i=0;i<without_front.size();++i) {
                    const int expected=std::lround(values[i%4]*255);
                    check(std::abs(int(without_front[i])+int(without_back[i])-expected)<=1,"Complementary face culling lost or doubled geometry");
                }
                check(render_face(SCE_GXM_CULL_NONE,true,false,false)==without_front,"Front fragment disable disagrees with GXM CCW culling");
                check(render_face(SCE_GXM_CULL_NONE,false,true,false)==without_back,"Back fragment disable disagrees with GXM CW culling");
                check(render_face(SCE_GXM_CULL_NONE,true,false,true)==without_front,"Front stencil selection disagrees with GXM CCW culling");
                check(render_face(SCE_GXM_CULL_NONE,false,true,true)==without_back,"Back stencil selection disagrees with GXM CW culling");
            }
            indices.get(mem)[1]=1;indices.get(mem)[2]=2;
            execute(renderer::GXMState::Viewport,false,64.0f,64.0f,0.0f,64.0f,64.0f,1.0f);
            execute(renderer::GXMState::CullMode,SCE_GXM_CULL_NONE);
            execute(renderer::GXMState::TwoSided,SCE_GXM_TWO_SIDED_DISABLED);
            execute(renderer::GXMState::FragmentProgramEnable,true,SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
            execute(renderer::GXMState::FragmentProgramEnable,false,SCE_GXM_FRAGMENT_PROGRAM_ENABLED);
            ctx.record.depth_stencil_surface=saved_depth_surface;
            ctx.record.back_depth_func=saved_back_depth_func;
            ctx.record.back_depth_write_mode=saved_back_depth_write;
            ctx.record.front_stencil_state_op=saved_front_op;ctx.record.back_stencil_state_op=saved_back_op;
            ctx.record.front_stencil_state_values=saved_front_values;ctx.record.back_stencil_state_values=saved_back_values;
            ctx.record.color_surface=saved_surface;
            std::cout<<"PASS GXM front/back classification: cull, fragment disable and stencil agree for both windings and four viewport reflections\n";
        }
        {
            const auto saved_surface = ctx.record.color_surface;
            const auto saved_scale = state.res_multiplier;
            Ptr<void> gamma_pixels(alloc(mem,128*128*4,"Metal gamma target"));
            const float gamma_color[] = {0.003f,0.18f,0.75f,0.25f};
            std::memcpy(color.get(mem),gamma_color,sizeof(gamma_color));
            execute(renderer::GXMState::RegionClip,SCE_GXM_REGION_CLIP_NONE,uint32_t(0),uint32_t(128),uint32_t(0),uint32_t(128));
            ctx.record.color_surface.data = gamma_pixels;
            const auto encoded = [](float linear) {
                return linear <= 0.0031308f ? linear*12.92f : 1.055f*std::pow(linear,1.0f/2.4f)-0.055f;
            };
            for (float scale : {1.0f,2.0f}) for (uint32_t gamma : {0u,1u,0u,1u}) {
                state.res_multiplier = scale;
                ctx.record.color_surface.gamma = gamma;
                std::memset(gamma_pixels.get(mem),0,128*128*4);
                state.set_context(ctx,mem); draw_triangle(); state.finish(ctx);
                check(state.sync_surface(mem,ctx.record.color_surface),"Cannot read encoded gamma surface bytes");
                const auto *pixels = static_cast<const uint8_t *>(gamma_pixels.get(mem));
                for (size_t i=0;i<128*128*4;++i) {
                    const float value = gamma && i%4<3 ? encoded(gamma_color[i%4]) : gamma_color[i%4];
                    check(std::abs(int(pixels[i])-int(std::lround(value*255)))<=1,"GXM gamma target encoding or alpha mismatch");
                }
                DisplayState gamma_display;
                gamma_display.next_rendered_frame.base = gamma_pixels;
                uint32_t w=0,h=0; const auto raw=state.dump_frame(gamma_display,w,h);
                check(w==uint32_t(128*scale) && h==uint32_t(128*scale) && raw.size()==w*h,"Gamma surface missing from frame dump");
                const auto *native=reinterpret_cast<const uint8_t *>(raw.data());
                for(size_t i=0;i<raw.size()*4;++i) check(native[i]==pixels[i%4],"Native gamma bytes differ from guest encoded bytes");
            }
            std::array<uint8_t,4> encoded_pixel;
            std::memcpy(encoded_pixel.data(),gamma_pixels.get(mem),4);
            // At2x, guest RAM is not implicitly re-uploaded by SetContext.
            // Reinterpreting the same allocation must retain the encoded data.
            ctx.record.color_surface.gamma=0;
            state.set_context(ctx,mem); state.finish(ctx);
            check(state.sync_surface(mem,ctx.record.color_surface),"Cannot read gamma-reinterpreted surface");
            for(size_t i=0;i<128*128*4;++i)
                check(static_cast<const uint8_t *>(gamma_pixels.get(mem))[i]==encoded_pixel[i%4],"Gamma view change discarded stored color bytes");
            state.res_multiplier=saved_scale; ctx.record.color_surface=saved_surface;
            std::memcpy(color.get(mem),values,sizeof(values));
            std::cout << "PASS GXM gamma off/on/off/on at1x/2x, encoded guest/native RGB and unchanged alpha\n";
        }
        {
            const auto saved_surface=ctx.record.color_surface;
            const float saved_scale=state.res_multiplier;
            auto &native=static_cast<renderer::metal::MetalFragmentProgram &>(*fp->renderer_data);
            const auto saved_blend=native.blend;
            // A test-only passthrough retains the fixture's output type and
            // secondary program, but reads native color and executes only PHAS.
            auto fetch_gxp=load(mem,argv[2]).cast<SceGxmProgram>();
            fetch_gxp.get(mem)->program_flags |= SCE_GXM_PROGRAM_FLAG_NATIVECOLOR_USED | SCE_GXM_PROGRAM_FLAG_FRAGCOLOR_USED;
            fetch_gxp.get(mem)->primary_program_instr_count=1;
            Ptr<SceGxmFragmentProgram> fetch_addr(alloc(mem,sizeof(SceGxmFragmentProgram),"Alpha fetch probe"));
            auto *fetch=new(fetch_addr.get(mem)) SceGxmFragmentProgram{};
            fetch->program=fetch_gxp.cast<const SceGxmProgram>();
            check(renderer::create(fetch->renderer_data,state,*fetch->program.get(mem),nullptr,state.gxp_ptr_map),"Cannot create alpha fetch probe");
            execute(renderer::GXMState::RegionClip,SCE_GXM_REGION_CLIP_NONE,uint32_t(0),uint32_t(128),uint32_t(0),uint32_t(128));
            struct AlphaCase { SceGxmBlendFunc op; SceGxmBlendFactor src,dst; SceGxmColorMask mask; float expected; };
            const AlphaCase cases[]={
                {SCE_GXM_BLEND_FUNC_NONE,SCE_GXM_BLEND_FACTOR_ONE,SCE_GXM_BLEND_FACTOR_ZERO,SCE_GXM_COLOR_MASK_A,0.4f},
                {SCE_GXM_BLEND_FUNC_ADD,SCE_GXM_BLEND_FACTOR_ONE,SCE_GXM_BLEND_FACTOR_ONE,SCE_GXM_COLOR_MASK_A,0.6f},
                {SCE_GXM_BLEND_FUNC_ADD,SCE_GXM_BLEND_FACTOR_DST_ALPHA,SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,SCE_GXM_COLOR_MASK_A,0.24f},
                {SCE_GXM_BLEND_FUNC_ADD,SCE_GXM_BLEND_FACTOR_SRC_ALPHA,SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,SCE_GXM_COLOR_MASK_A,0.28f},
                {SCE_GXM_BLEND_FUNC_SUBTRACT,SCE_GXM_BLEND_FACTOR_ONE,SCE_GXM_BLEND_FACTOR_ONE,SCE_GXM_COLOR_MASK_A,0.2f},
                {SCE_GXM_BLEND_FUNC_MAX,SCE_GXM_BLEND_FACTOR_ONE,SCE_GXM_BLEND_FACTOR_ONE,SCE_GXM_COLOR_MASK_A,0.4f},
                {SCE_GXM_BLEND_FUNC_NONE,SCE_GXM_BLEND_FACTOR_ONE,SCE_GXM_BLEND_FACTOR_ZERO,SCE_GXM_COLOR_MASK_R,0.2f}
            };
            unsigned checks=0;
            for(float scale:{1.f,2.f}) for(const auto &test:cases) {
                state.res_multiplier=scale;
                Ptr<uint8_t> data(alloc(mem,128*128+32,"Alpha-only framebuffer"));
                std::memset(data.get(mem),0xa5,128*128+32);
                ctx.record.color_surface={};
                ctx.record.color_surface.data=Ptr<void>(data.address()+16);
                ctx.record.color_surface.width=ctx.record.color_surface.height=ctx.record.color_surface.strideInPixels=128;
                ctx.record.color_surface.colorFormat=SCE_GXM_COLOR_FORMAT_U8_A;
                ctx.record.color_surface.surfaceType=SCE_GXM_COLOR_SURFACE_LINEAR;
                native.blend=saved_blend; native.blend.colorMask=SCE_GXM_COLOR_MASK_ALL;
                native.blend.colorFunc=native.blend.alphaFunc=SCE_GXM_BLEND_FUNC_NONE;
                const float base[]={0.9f,0.7f,0.6f,0.2f}; std::memcpy(color.get(mem),base,sizeof(base));
                execute(renderer::GXMState::Program,fp_addr.cast<void>(),true);
                state.set_context(ctx,mem); draw_triangle(); state.finish(ctx);
                execute(renderer::GXMState::Program,fetch_addr.cast<void>(),true);
                state.set_context(ctx,mem); draw_triangle(); state.finish(ctx);
                check(state.sync_surface(mem,ctx.record.color_surface),"Cannot read alpha-only fetch result");
                for(size_t i=0;i<128*128;++i) check(std::abs(int(data.get(mem)[16+i])-51)<=1,"Alpha-only write/fetch lost A");
                execute(renderer::GXMState::Program,fp_addr.cast<void>(),true);
                native.blend.colorMask=test.mask;
                // Deliberately different RGB state detects using color blending
                // for an alpha-only attachment backed by physical red storage.
                native.blend.colorFunc=SCE_GXM_BLEND_FUNC_REVERSE_SUBTRACT;
                native.blend.colorSrc=SCE_GXM_BLEND_FACTOR_ZERO; native.blend.colorDst=SCE_GXM_BLEND_FACTOR_ONE;
                native.blend.alphaFunc=test.op; native.blend.alphaSrc=test.src; native.blend.alphaDst=test.dst;
                const float source[]={0.8f,0.7f,0.6f,0.4f}; std::memcpy(color.get(mem),source,sizeof(source));
                state.set_context(ctx,mem); draw_triangle(); state.finish(ctx);
                check(state.sync_surface(mem,ctx.record.color_surface),"Cannot read alpha-only blend result");
                for(size_t i=0;i<128*128;++i) {
                    check(std::abs(int(data.get(mem)[16+i])-int(std::lround(test.expected*255)))<=1,"Alpha-only blend or mask mismatch case="+std::to_string(checks));
                }
                for(size_t i=0;i<16;++i) check(data.get(mem)[i]==0xa5 && data.get(mem)[128*128+16+i]==0xa5,"Alpha-only readback overwrote padding");
                ++checks;
            }
            native.blend=saved_blend;
            execute(renderer::GXMState::Program,fp_addr.cast<void>(),true);
            fetch->~SceGxmFragmentProgram();
            ctx.record.color_surface=saved_surface; state.res_multiplier=saved_scale;
            std::memcpy(color.get(mem),values,sizeof(values));
            std::cout<<"PASS "<<checks<<" U8_A GXM write/fetch/blend/mask cases at1x/2x with guest-byte sentinels\n";
        }
        // Color -> mask -> color transitions must preserve the color attachment and
        // make the mask visible to subsequent fragments without a CPU fence.
        state.set_context(ctx, mem);
        fp->is_maskupdate = true;
        execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
        execute(renderer::GXMState::StencilFunc, true, SCE_GXM_STENCIL_FUNC_NEVER,
            SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, uint8_t(255), uint8_t(255));
        execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_OUTSIDE, uint32_t(32), uint32_t(96), uint32_t(32), uint32_t(96));
        draw_triangle();
        fp->is_maskupdate = false;
        execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
        execute(renderer::GXMState::RegionClip, SCE_GXM_REGION_CLIP_NONE, uint32_t(0), uint32_t(128), uint32_t(0), uint32_t(128));
        draw_triangle();
        state.finish(ctx);
        const auto *mask_pixels = reinterpret_cast<const uint8_t *>(ctx.record.color_surface.data.get(mem));
        for (uint32_t y = 0; y < 128; ++y) for (uint32_t x = 0; x < 128; ++x) {
            const bool masked = x >= 32 && x < 96 && y >= 32 && y < 96;
            const int expected[] = {64,128,191,255};
            for (int c = 0; c < 4; ++c)
                check(std::abs(int(mask_pixels[(y*128+x)*4+c]) - (masked ? 0 : expected[c])) <= 1,
                    "Mask update/read mismatch x=" + std::to_string(x) + " y=" + std::to_string(y));
        }
        std::cout << "PASS GXM mask update -> color pass -> masked fragment discard -> guest readback\n";
        const auto first_surface = ctx.record.color_surface;
        // Clear the mask again, draw into A, then change the recorded surface to B
        // before the backend finishes A (the production SetContext order).
        state.set_context(ctx, mem);
        fp->is_maskupdate = true;
        execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
        execute(renderer::GXMState::StencilFunc, true, SCE_GXM_STENCIL_FUNC_ALWAYS,
            SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, uint8_t(255), uint8_t(255));
        draw_triangle();
        fp->is_maskupdate = false;
        execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
        draw_triangle();
        Ptr<uint32_t> second_surface(alloc(mem, 128*128*4, "Metal scene transition"));
        std::memset(second_surface.get(mem), 37, 128*128*4);
        ctx.record.color_surface.data = second_surface.cast<void>();
        state.set_context(ctx, mem);
        state.finish(ctx);
        const auto *first_bytes = static_cast<const uint8_t *>(first_surface.data.get(mem));
        const auto *second_bytes = reinterpret_cast<const uint8_t *>(second_surface.get(mem));
        const int scene_color[] = {64,128,191,255};
        for (size_t i = 0; i < 128*128*4; ++i) {
            check(std::abs(int(first_bytes[i]) - scene_color[i % 4]) <= 1, "Scene A readback was lost");
            check(second_bytes[i] == 37, "Scene A corrupted scene B memory");
        }
        // A CPU write at a previously cached surface address must be restored.
        std::memset(second_surface.get(mem), 73, 128*128*4);
        state.set_context(ctx, mem);
        state.finish(ctx);
        for (size_t i = 0; i < 128*128*4; ++i)
            check(second_bytes[i] == 73, "Cached color surface ignored a CPU write");
        std::cout << "PASS pending scene readback isolation and CPU surface-write restoration\n";
        // Reuse and overwrite one guest uniform buffer across queued draws.
        // Each stripe must retain the values visible when it was encoded.
        std::memset(second_surface.get(mem),0,128*128*4);
        state.set_context(ctx,mem);
        for(uint32_t stripe=0;stripe<8;++stripe) {
            const float stripe_color[]={float(stripe+1)/8,0.25f,0.5f,1};
            std::memcpy(color.get(mem),stripe_color,16);
            execute(renderer::GXMState::RegionClip,SCE_GXM_REGION_CLIP_OUTSIDE,(stripe%4)*32,(stripe%4+1)*32,(stripe/4)*64,(stripe/4+1)*64);
            draw_triangle();
        }
        std::memset(color.get(mem),0,16);
        for(size_t i=0;i<128*128*4;++i) check(second_bytes[i]==0,"Read-only draws were not retained for batching");
        ctx.free_func=[](renderer::Command *command){delete command;};
        auto *nop=new renderer::Command{};nop->opcode=renderer::CommandOpcode::Nop;
        int status=-1,success=7;nop->status=&status;
        renderer::CommandHelper boundary(nop);boundary.push(success);
        state.command_buffer_queue.push(renderer::CommandList{nop,nop,&ctx});
        renderer::process_batches(state,state.features,mem,config,0);
        check(status==7,"CPU-visible NOP completion was not signalled");
        for(uint32_t y=0;y<128;++y) for(uint32_t x=0;x<128;++x) {
            const int expected[]={int(std::lround(float((y/64)*4+x/32+1)/8*255)),64,128,255};
            for(uint32_t c=0;c<4;++c) check(std::abs(int(second_bytes[(y*128+x)*4+c])-expected[c])<=1,
                "Batched uniform snapshot or NOP GPU completion lost a stripe: x="+std::to_string(x)+" y="+std::to_string(y)+" channel="+std::to_string(c)+" actual="+std::to_string(second_bytes[(y*128+x)*4+c])+" expected="+std::to_string(expected[c]));
        }
        std::memcpy(color.get(mem),values,16);
        execute(renderer::GXMState::RegionClip,SCE_GXM_REGION_CLIP_NONE,uint32_t(0),uint32_t(128),uint32_t(0),uint32_t(128));
        std::cout << "PASS eight queued draws preserve overwritten guest uniforms; NOP publishes all GPU pixels before CPU completion\n";
        {
            const auto previous = ctx.record.color_surface;
            state.res_multiplier = 2;
            std::array<SceGxmColorSurface,2> surfaces;
            const float colors[2][4] = {{0.25f,0.5f,0.75f,1},{1,0.25f,0.5f,0.75f}};
            for (unsigned i=0;i<2;++i) {
                surfaces[i] = previous;
                surfaces[i].strideInPixels = 160;
                surfaces[i].data = Ptr<void>(alloc(mem,160*128*4,"Metal explicit sync surface"));
                surfaces[i].colorFormat = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB;
                std::memset(surfaces[i].data.get(mem),0xa7,160*128*4);
                ctx.record.color_surface = surfaces[i];
                state.set_context(ctx,mem);
                std::memcpy(color.get(mem),colors[i],16);
                draw_triangle();
            }
            Ptr<uint32_t> notification(alloc(mem,4,"Metal surface sync notification"));
            auto synchronize = [&](SceGxmColorSurface *surface) {
                renderer::Command command{};
                int status=-1; command.status=&status;
                renderer::CommandHelper writer(&command);
                SceGxmNotification vertex_notification{notification,0x76543210},fragment_notification{};
                writer.push(vertex_notification);
                writer.push(fragment_notification); writer.push(surface);
                renderer::CommandHelper reader(&command);
                // Explicit requests can arrive without a current command context.
                renderer::cmd_handle_sync_surface_data(state,mem,config,reader,state.features,nullptr);
                return status;
            };
            *notification.get(mem)=0;
            check(synchronize(&surfaces[0])==0,"Explicit older surface request failed");
            check(*notification.get(mem)==0x76543210,"Surface sync notification was not published");
            const auto *untouched=static_cast<uint8_t *>(surfaces[1].data.get(mem));
            for(size_t i=0;i<160*128*4;++i) check(untouched[i]==0xa7,"Sync request read back the active surface instead of the requested surface");
            check(synchronize(&surfaces[1])==0,"Explicit active surface request failed");
            // EndScene uses the same opcode without a return status or surface
            // pointer. Its notifications must still follow guest readback.
            std::memset(surfaces[1].data.get(mem),0xa7,160*128*4);
            *notification.get(mem)=0;
            renderer::Command end_scene{};
            renderer::CommandHelper end_writer(&end_scene);
            SceGxmNotification vertex_done{},fragment_done{notification,0x12345678};
            end_writer.push(vertex_done); end_writer.push(fragment_done);
            renderer::CommandHelper end_reader(&end_scene);
            renderer::cmd_handle_sync_surface_data(state,mem,config,end_reader,state.features,&ctx);
            check(*notification.get(mem)==0x12345678,"EndScene notification was not signalled");
            for(unsigned i=0;i<2;++i) {
                const auto *bytes=static_cast<uint8_t *>(surfaces[i].data.get(mem));
                for(unsigned y=0;y<128;++y) for(unsigned x=0;x<160;++x) for(unsigned c=0;c<4;++c) {
                    const unsigned channel[]={2,1,0,3};
                    const int expected=x<128?int(std::lround(colors[i][channel[c]]*255)):0xa7;
                    check(std::abs(int(bytes[(y*160+x)*4+c])-expected)<= (x<128?1:0),"Explicit 2x surface readback lost channels or row padding");
                }
            }
            check(synchronize(nullptr)==1,"Null explicit sync was reported as successful");
            auto missing=surfaces[0]; missing.data=Ptr<void>(alloc(mem,160*128*4,"Metal uncached sync surface"));
            check(synchronize(&missing)==1,"Uncached surface was reported as synchronized");
            {
                // A transfer can address a subrange of an older GPU surface.
                auto *image=new SceGxmTransferImage{};
                image->address=surfaces[0].data.cast<uint8_t>()+4;
                image->format=SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR;
                image->x=17; image->y=9; image->width=3; image->height=4; image->stride=160*4;
                auto *fill=new renderer::Command{}; fill->opcode=renderer::CommandOpcode::TransferFill;
                renderer::CommandHelper fill_writer(fill);
                uint32_t packed=0xaabbccdd; fill_writer.push(packed); fill_writer.push(image);
                auto *notify=new renderer::Command{}; notify->opcode=renderer::CommandOpcode::SignalNotification;
                renderer::CommandHelper notify_writer(notify);
                SceGxmNotification complete{notification,0xabcdef01}; notify_writer.push(complete);
                fill->next=notify;
                *notification.get(mem)=0;
                state.command_buffer_queue.push(renderer::CommandList{fill,notify,&ctx});
                renderer::process_batches(state,state.features,mem,config,0);
                check(*notification.get(mem)==0xabcdef01,"Fill completion notification missing");
                DisplayState display; display.next_rendered_frame.base=surfaces[0].data;
                uint32_t width=0,height=0;
                const auto frame=state.dump_frame(display,width,height);
                check(width==256 && height==256,"Fill lost the 2x cached surface");
                const auto *pixels=reinterpret_cast<const uint8_t *>(frame.data());
                const int filled[]={0xbb,0xcc,0xdd,0xaa};
                for(uint32_t y=0;y<256;++y) for(uint32_t x=0;x<256;++x) for(unsigned c=0;c<4;++c) {
                    const bool touched=x>=36 && x<42 && y>=18 && y<26;
                    const int expected=touched?filled[c]:int(std::lround(colors[0][c]*255));
                    check(std::abs(int(pixels[(y*256+x)*4+c])-expected)<=(touched?0:1),"Transfer fill changed the wrong cached native pixels");
                }
                std::cout<<"PASS production transfer fill updates an offset into an older ARGB surface at 2x before notification; other GPU pixels remain intact\n";
            }
            {
                Ptr<uint8_t> allocation(alloc(mem,512,"Metal signed-stride fill"));
                std::memset(allocation.get(mem),0x71,512);
                std::array<uint8_t,512> expected; expected.fill(0x71);
                SceGxmTransferImage image{};
                image.address=allocation+200; image.format=SCE_GXM_TRANSFER_FORMAT_U8U8U8_BGR;
                image.x=2; image.y=1; image.width=5; image.height=4; image.stride=-19;
                check(state.transfer_fill(mem,image,0x11223344),"Valid negative-stride RGB24 fill failed");
                const uint8_t channels[]={0x44,0x33,0x22};
                for(int row=0;row<4;++row) for(int x=0;x<5;++x) for(int c=0;c<3;++c)
                    expected[200-(row+1)*19+6+x*3+c]=channels[c];
                check(std::memcmp(allocation.get(mem),expected.data(),512)==0,"Signed stride, RGB24 row phase or fill sentinels were corrupted");
                image.format=SCE_GXM_TRANSFER_FORMAT_RAW64;
                check(!state.transfer_fill(mem,image,0),"Wide fill read past its four-byte value");
                image.format=SCE_GXM_TRANSFER_FORMAT_U8_R; image.stride=INT32_MIN;
                check(!state.transfer_fill(mem,image,0),"Negative fill address underflow accepted");
                image.stride=INT32_MAX; image.y=UINT32_MAX;
                check(!state.transfer_fill(mem,image,0),"Fill address arithmetic overflow accepted");
                check(std::memcmp(allocation.get(mem),expected.data(),512)==0,"Rejected fill modified guest memory");
                std::cout<<"PASS negative-stride RGB24 fill and every sentinel; wide-format and signed-address rejection without writes\n";
            }
            {
                // RAM is deliberately stale in BOTH surfaces. Only the selected
                // pixels may change, and the copy must read the older GPU target.
                std::memset(surfaces[0].data.get(mem),0x19,160*128*4);
                std::memset(surfaces[1].data.get(mem),0x71,160*128*4);
                auto *images=new SceGxmTransferImage[2]{};
                images[0].address=surfaces[0].data.cast<uint8_t>()+4;
                images[1].address=surfaces[1].data.cast<uint8_t>()+8;
                for(unsigned i=0;i<2;++i) {
                    images[i].format=SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR;
                    images[i].width=5; images[i].height=6; images[i].stride=160*4;
                }
                images[0].x=16; images[0].y=8; images[1].x=38; images[1].y=50;
                auto *copy=new renderer::Command{}; copy->opcode=renderer::CommandOpcode::TransferCopy;
                renderer::CommandHelper writer(copy);
                uint32_t key=0xaa000000, mask=0xff000000;
                auto mode=SCE_GXM_TRANSFER_COLORKEY_PASS; auto layout=SCE_GXM_TRANSFER_LINEAR;
                writer.push(key); writer.push(mask); writer.push(mode); writer.push(images);
                writer.push(layout); writer.push(layout);
                auto *notify=new renderer::Command{}; notify->opcode=renderer::CommandOpcode::SignalNotification;
                renderer::CommandHelper notify_writer(notify);
                SceGxmNotification complete{notification,0x10203040}; notify_writer.push(complete);
                copy->next=notify; *notification.get(mem)=0;
                state.command_buffer_queue.push(renderer::CommandList{copy,notify,&ctx});
                renderer::process_batches(state,state.features,mem,config,0);
                check(*notification.get(mem)==0x10203040,"Copy completion notification missing");
                const uint8_t guest_color[]={0xdd,0xcc,0xbb,0xaa}, native_color[]={0xbb,0xcc,0xdd,0xaa};
                const auto *source=static_cast<const uint8_t *>(surfaces[0].data.get(mem));
                const auto *dest=static_cast<const uint8_t *>(surfaces[1].data.get(mem));
                for(uint32_t y=0;y<128;++y) for(uint32_t x=0;x<160;++x) for(unsigned c=0;c<4;++c) {
                    const bool touched=x>=41 && x<44 && y>=51 && y<55;
                    check(source[(y*160+x)*4+c]==0x19,"Copy unexpectedly rewrote source RAM");
                    check(dest[(y*160+x)*4+c]==(touched?guest_color[c]:0x71),"Copy source readback, color-key mask or destination sentinels failed");
                }
                DisplayState display; display.next_rendered_frame.base=surfaces[1].data;
                uint32_t width=0,height=0; const auto frame=state.dump_frame(display,width,height);
                check(width==256 && height==256,"Copy lost the upscaled destination");
                const auto *native=reinterpret_cast<const uint8_t *>(frame.data());
                for(uint32_t y=0;y<256;++y) for(uint32_t x=0;x<256;++x) for(unsigned c=0;c<4;++c) {
                    const bool touched=x>=82 && x<88 && y>=102 && y<110;
                    const int expected=touched?native_color[c]:int(std::lround(colors[1][c]*255));
                    check(std::abs(int(native[(y*256+x)*4+c])-expected)<=(touched?0:1),"Copy failed to preserve untouched native destination pixels");
                }
                std::cout<<"PASS queued copy reads older GPU surface with stale RAM, honors color-key pass and updates only selected 2x ARGB target pixels before notification\n";
                Ptr<uint8_t> cpu(alloc(mem,64,"Metal partial-component transfer source"));
                std::memset(cpu.get(mem),0x39,64);
                const uint8_t green[]={0x2a,0x55,0x88};
                for(unsigned y=0;y<3;++y) cpu.get(mem)[20-y*3]=green[y];
                SceGxmTransferImage byte_source{},byte_destination{};
                byte_source.address=cpu+20; byte_source.format=SCE_GXM_TRANSFER_FORMAT_U8_R;
                byte_source.width=1; byte_source.height=3; byte_source.stride=-3;
                byte_destination=byte_source;
                byte_destination.address=surfaces[1].data.cast<uint8_t>()+((70*160+70)*4+1);
                byte_destination.stride=160*4;
                check(state.transfer_copy(mem,byte_source,byte_destination,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
                    SCE_GXM_TRANSFER_COLORKEY_NONE,0,0),"CPU-to-GPU partial-component copy failed");
                const auto updated=state.dump_frame(display,width,height);
                const auto *updated_bytes=reinterpret_cast<const uint8_t *>(updated.data());
                for(unsigned y=0;y<256;++y) for(unsigned x=0;x<256;++x) for(unsigned c=0;c<4;++c) {
                    const bool changed=x>=140 && x<142 && y>=140 && y<146 && c==1;
                    const unsigned offset=(y*256+x)*4+c;
                    check(updated_bytes[offset]==(changed?green[(y-140)/2]:native[offset]),"Byte copy changed untouched native components or pixels");
                }
                // Prove the reverse path uses the GPU after poisoning these RAM
                // bytes. The result lives in an uncached, guarded CPU buffer.
                for(unsigned y=0;y<3;++y) static_cast<uint8_t *>(surfaces[1].data.get(mem))[((70+y)*160+70)*4+1]=0xee;
                Ptr<uint8_t> output(alloc(mem,64,"Metal transfer readback destination"));
                std::memset(output.get(mem),0x73,64);
                auto readback=byte_source; readback.address=output+11; readback.stride=5;
                check(state.transfer_copy(mem,byte_destination,readback,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
                    SCE_GXM_TRANSFER_COLORKEY_NONE,0,0),"GPU-to-CPU partial-component copy failed");
                for(unsigned i=0;i<64;++i) {
                    const int row=(int(i)-11)/5;
                    const bool changed=i>=11 && i<=21 && (i-11)%5==0;
                    check(output.get(mem)[i]==(changed?green[row]:0x73),"GPU readback copy or guarded destination mismatch");
                }
                std::cout<<"PASS CPU-to-GPU and GPU-to-CPU byte-component copies preserve every other native byte and read through deliberately stale RAM\n";

            }
            {
                // CPU-only sources still pass through the same native transfer
                // method. Compare complete buffers with an independent snapshot.
                const SceGxmTransferFormat formats[]={SCE_GXM_TRANSFER_FORMAT_U8_R,SCE_GXM_TRANSFER_FORMAT_RAW16,
                    SCE_GXM_TRANSFER_FORMAT_U8U8U8_BGR,SCE_GXM_TRANSFER_FORMAT_RAW32,
                    SCE_GXM_TRANSFER_FORMAT_RAW64,SCE_GXM_TRANSFER_FORMAT_RAW128};
                const unsigned sizes[]={1,2,3,4,8,16};
                for(unsigned f=0;f<6;++f) for(bool negative:{false,true}) {
                    Ptr<uint8_t> allocation(alloc(mem,4096,"Metal overlapping transfer copy"));
                    auto *data=allocation.get(mem);
                    for(unsigned i=0;i<4096;++i) data[i]=uint8_t((i*37+i/7)%251);
                    std::vector<uint8_t> before(data,data+4096), expected=before;
                    SceGxmTransferImage src{},dst{};
                    src.address=allocation+1024; src.format=formats[f];
                    src.width=9; src.height=4; src.x=1; src.y=1; src.stride=negative?-157:157;
                    dst=src; dst.address=allocation+1024+sizes[f];
                    for(int y=0;y<4;++y) for(int x=0;x<9;++x) for(unsigned c=0;c<sizes[f];++c) {
                        const int offset=1024+(y+1)*src.stride+(x+1)*sizes[f]+c;
                        expected[offset+sizes[f]]=before[offset];
                    }
                    check(state.transfer_copy(mem,src,dst,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
                        SCE_GXM_TRANSFER_COLORKEY_NONE,0,0),"Linear overlapping copy rejected");
                    check(std::memcmp(data,expected.data(),4096)==0,"Copy overlap, signed byte stride or whole-buffer sentinels failed");
                    src.stride=INT32_MIN; src.y=UINT32_MAX;
                    check(!state.transfer_copy(mem,src,dst,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
                        SCE_GXM_TRANSFER_COLORKEY_NONE,0,0),"Overflowed copy address accepted");
                    check(std::memcmp(data,expected.data(),4096)==0,"Rejected copy modified destination");
                }
                std::cout<<"PASS12 overlapping copy cases from8 to128bits, positive/negative odd byte strides and rejection without writes\n";
                unsigned layouts_checked=0;
                for(unsigned f=0;f<6;++f) for(bool tiled:{false,true}) for(bool reverse:{false,true}) for(bool negative:{false,true}) {
                    if (!tiled && negative) continue;
                    constexpr size_t allocation_size=524288, origin=262144;
                    Ptr<uint8_t> source(alloc(mem,allocation_size,"Metal transfer layout source"));
                    Ptr<uint8_t> destination(alloc(mem,allocation_size,"Metal transfer layout destination"));
                    auto *source_bytes=source.get(mem), *dest_bytes=destination.get(mem);
                    std::memset(source_bytes,0x39,allocation_size); std::memset(dest_bytes,0xa6,allocation_size);
                    SceGxmTransferImage linear{},nonlinear{};
                    linear.format=formats[f]; linear.width=tiled?37:32; linear.height=tiled?35:16;
                    linear.x=2; linear.y=3; linear.stride=int32_t(43*sizes[f]+1);
                    nonlinear=linear; nonlinear.x=tiled?29:0; nonlinear.y=tiled?31:0;
                    nonlinear.stride=int32_t(96*sizes[f])*(negative?-1:1);
                    auto src=reverse?nonlinear:linear, dst=reverse?linear:nonlinear;
                    src.address=source+origin; dst.address=destination+origin;
                    const auto special=tiled?SCE_GXM_TRANSFER_TILED:SCE_GXM_TRANSFER_SWIZZLED;
                    const auto src_type=reverse?special:SCE_GXM_TRANSFER_LINEAR, dst_type=reverse?SCE_GXM_TRANSFER_LINEAR:special;
                    // Independent address oracle: split Morton coordinates bit
                    // by bit, then append the long dimension's remaining bits.
                    auto offset=[&](const SceGxmTransferImage &image,SceGxmTransferType type,unsigned x,unsigned y) -> int64_t {
                        x+=image.x; y+=image.y;
                        if(type==SCE_GXM_TRANSFER_LINEAR) return int64_t(y)*image.stride+int64_t(x)*sizes[f];
                        if(type==SCE_GXM_TRANSFER_TILED) return int64_t(y/32)*image.stride*32+int64_t((x/32)*1024+(y%32)*32+x%32)*sizes[f];
                        uint64_t pixel=0; unsigned bit=0, position=0;
                        while((1u<<bit)<std::min(image.width,image.height)) {
                            pixel|=uint64_t((y>>bit)&1)<<position++;
                            pixel|=uint64_t((x>>bit)&1)<<position++; ++bit;
                        }
                        const unsigned longer=image.width>image.height?x:y;
                        pixel|=uint64_t(longer>>bit)<<position;
                        return int64_t(pixel*sizes[f]);
                    };
                    std::vector<uint8_t> expected(allocation_size,0xa6);
                    for(unsigned y=0;y<src.height;++y) for(unsigned x=0;x<src.width;++x) for(unsigned c=0;c<sizes[f];++c) {
                        const auto value=uint8_t((x*13+y*37+c*59)%251);
                        source_bytes[int64_t(origin)+offset(src,src_type,x,y)+c]=value;
                        expected[int64_t(origin)+offset(dst,dst_type,x,y)+c]=value;
                    }
                    const std::vector<uint8_t> unchanged(source_bytes,source_bytes+allocation_size);
                    check(state.transfer_copy(mem,src,dst,src_type,dst_type,SCE_GXM_TRANSFER_COLORKEY_NONE,0,0),"Tiled/Morton copy rejected: bytes="+std::to_string(sizes[f])+" tiled="+std::to_string(tiled)+" reverse="+std::to_string(reverse)+" negative="+std::to_string(negative));
                    check(std::memcmp(dest_bytes,expected.data(),allocation_size)==0,"Tiled/Morton copy or padding mismatch");
                    check(std::memcmp(source_bytes,unchanged.data(),allocation_size)==0,"Tiled/Morton copy changed its source");
                    ++layouts_checked;
                }
                check(layouts_checked==36,"Missing transfer layout cases");
                std::cout<<"PASS36 linear/tiled/Morton conversions, six pixel sizes, rectangular Morton and signed tile-row strides with full-buffer guards\n";
                {
                    Ptr<uint32_t> source(alloc(mem,64,"Metal color-key source")),destination(alloc(mem,64,"Metal color-key destination"));
                    for(unsigned i=0;i<16;++i) source.get(mem)[i]=0x11000000+i;
                    std::memset(destination.get(mem),0x71,64);
                    SceGxmTransferImage src{},dst{}; src.address=source; src.format=SCE_GXM_TRANSFER_FORMAT_RAW32;
                    src.width=4; src.height=4; src.stride=16; dst=src; dst.address=destination;
                    check(state.transfer_copy(mem,src,dst,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
                        SCE_GXM_TRANSFER_COLORKEY_REJECT,1,1),"Color-key reject copy failed");
                    for(unsigned i=0;i<16;++i) check(destination.get(mem)[i]==(i%2?0x71717171:source.get(mem)[i]),"Color-key reject mask mismatch");
                    const std::vector<uint32_t> before(destination.get(mem),destination.get(mem)+16);
                    dst.format=SCE_GXM_TRANSFER_FORMAT_RAW64;
                    check(!state.transfer_copy(mem,src,dst,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
                        SCE_GXM_TRANSFER_COLORKEY_NONE,0,0),"Unsupported format conversion accepted");
                    dst.format=src.format; src.address=Ptr<void>(0xfffff000);
                    check(!state.transfer_copy(mem,src,dst,SCE_GXM_TRANSFER_LINEAR,SCE_GXM_TRANSFER_LINEAR,
                        SCE_GXM_TRANSFER_COLORKEY_NONE,0,0),"Unmapped copy source accepted");
                    check(std::memcmp(destination.get(mem),before.data(),64)==0,"Rejected copy changed its destination");
                    std::cout<<"PASS color-key reject, format mismatch and unmapped-source rejection without writes\n";
                }

            }
            {
                // Two different 2x2 GPU blocks; poison guest RAM afterwards so
                // the downscale cannot pass by reading its stale CPU copy.
                const uint32_t source_words[8]={0x00000000,0xffffffff,0xf0102030,0xf0102030,
                    0x80402010,0x01020304,0xf0102030,0xf0102030};
                for(unsigned y=0;y<2;++y) for(unsigned x=0;x<4;++x) {
                    SceGxmTransferImage pixel{}; pixel.address=surfaces[0].data;
                    pixel.format=SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR;
                    pixel.width=1; pixel.height=1; pixel.x=60+x; pixel.y=80+y; pixel.stride=640;
                    check(state.transfer_fill(mem,pixel,source_words[y*4+x]),"Downscale source GPU setup failed");
                }
                DisplayState display; display.next_rendered_frame.base=surfaces[1].data;
                uint32_t width=0,height=0; const auto before=state.dump_frame(display,width,height);
                std::memset(surfaces[0].data.get(mem),0x19,160*128*4);
                std::memset(surfaces[1].data.get(mem),0x71,160*128*4);
                auto *src=new SceGxmTransferImage{}, *dst=new SceGxmTransferImage{};
                src->address=surfaces[0].data.cast<uint8_t>()+4;
                src->format=SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR;
                src->x=59; src->y=80; src->width=4; src->height=2; src->stride=640;
                *dst=*src; dst->address=surfaces[1].data.cast<uint8_t>()+8;
                dst->x=68; dst->y=90; dst->width=2; dst->height=1;
                auto *downscale=new renderer::Command{}; downscale->opcode=renderer::CommandOpcode::TransferDownscale;
                renderer::CommandHelper writer(downscale); writer.push(src); writer.push(dst);
                auto *notify=new renderer::Command{}; notify->opcode=renderer::CommandOpcode::SignalNotification;
                renderer::CommandHelper notify_writer(notify); SceGxmNotification complete{notification,0x50403020};
                notify_writer.push(complete); downscale->next=notify; *notification.get(mem)=0;
                state.command_buffer_queue.push(renderer::CommandList{downscale,notify,&ctx});
                renderer::process_batches(state,state.features,mem,config,0);
                check(*notification.get(mem)==0x50403020,"Downscale notification missing");
                const uint8_t guest[2][4]={{69,73,80,96},{48,32,16,240}};
                const auto *source=static_cast<const uint8_t *>(surfaces[0].data.get(mem));
                const auto *dest=static_cast<const uint8_t *>(surfaces[1].data.get(mem));
                for(unsigned y=0;y<128;++y) for(unsigned x=0;x<160;++x) for(unsigned c=0;c<4;++c) {
                    const bool changed=y==90 && x>=70 && x<72;
                    check(dest[(y*160+x)*4+c]==(changed?guest[x-70][c]:0x71),"Downscale average or CPU destination guards failed");
                    check(source[(y*160+x)*4+c]==0x19,"Downscale rewrote source guest RAM");
                }
                const auto after=state.dump_frame(display,width,height);
                const auto *old_bytes=reinterpret_cast<const uint8_t *>(before.data());
                const auto *new_bytes=reinterpret_cast<const uint8_t *>(after.data());
                const unsigned mapping[]={2,1,0,3};
                for(unsigned y=0;y<256;++y) for(unsigned x=0;x<256;++x) for(unsigned c=0;c<4;++c) {
                    const bool changed=y>=180 && y<182 && x>=140 && x<144;
                    const unsigned offset=(y*256+x)*4+c;
                    check(new_bytes[offset]==(changed?guest[(x-140)/2][mapping[c]]:old_bytes[offset]),"Downscale changed the wrong native byte or lost ARGB mapping");
                }
                std::cout<<"PASS queued downscale averages older GPU source despite stale RAM, updates two distinct 2x ARGB pixels and preserves every other CPU/GPU byte before notification\n";
            }
            {
                const SceGxmTransferFormat formats[]={SCE_GXM_TRANSFER_FORMAT_U8_R,SCE_GXM_TRANSFER_FORMAT_U8U8_GR,
                    SCE_GXM_TRANSFER_FORMAT_U4U4U4U4_ABGR,SCE_GXM_TRANSFER_FORMAT_U1U5U5U5_ABGR,
                    SCE_GXM_TRANSFER_FORMAT_U5U6U5_BGR,SCE_GXM_TRANSFER_FORMAT_U8U8U8_BGR,
                    SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR,SCE_GXM_TRANSFER_FORMAT_U2U10U10U10_ABGR};
                const unsigned sizes[]={1,2,2,2,2,3,4,4};
                const unsigned fields[8][4]={{8,0,0,0},{8,8,0,0},{4,4,4,4},{5,5,5,1},
                    {5,6,5,0},{8,8,8,0},{8,8,8,8},{10,10,10,2}};
                unsigned checked=0;
                for(unsigned f=0;f<8;++f) for(bool negative_source:{false,true})
                    for(bool negative_dest:{false,true}) for(bool overlap:{false,true}) {
                    Ptr<uint8_t> allocation(alloc(mem,4096,"Metal downscale signed overlap"));
                    auto *data=allocation.get(mem); std::memset(data,0x73,4096);
                    const int source_base=1024, destination_base=overlap?1024+int(sizes[f]):3072;
                    SceGxmTransferImage src{},dst{}; src.address=allocation+source_base; src.format=formats[f];
                    src.width=6; src.height=4; src.x=1; src.y=2; src.stride=negative_source?-39:39;
                    dst=src; dst.address=allocation+destination_base; dst.width=3; dst.height=2; dst.x=2; dst.y=1; dst.stride=negative_dest?-23:23;
                    uint32_t inputs[4][6]{};
                    for(unsigned y=0;y<4;++y) for(unsigned x=0;x<6;++x) {
                        unsigned shift=0;
                        for(unsigned c=0;c<4 && fields[f][c];++c) {
                            const unsigned maximum=(1u<<fields[f][c])-1;
                            const unsigned value=(x*17+y*29+c*43+(x+y)%3)% (maximum+1);
                            inputs[y][x]|=value<<shift; shift+=fields[f][c];
                        }
                        const int offset=source_base+(int(y)+2)*src.stride+(x+1)*sizes[f];
                        std::memcpy(data+offset,&inputs[y][x],sizes[f]);
                    }
                    std::vector<uint8_t> expected(data,data+4096);
                    for(unsigned y=0;y<2;++y) for(unsigned x=0;x<3;++x) {
                        uint32_t output=0; unsigned shift=0;
                        for(unsigned c=0;c<4 && fields[f][c];++c) {
                            const unsigned maximum=(1u<<fields[f][c])-1;
                            double mean=0;
                            for(unsigned dy=0;dy<2;++dy) for(unsigned dx=0;dx<2;++dx)
                                mean+=double((inputs[y*2+dy][x*2+dx]>>shift)&maximum)*0.25;
                            output|=uint32_t(std::floor(mean+0.5))<<shift; shift+=fields[f][c];
                        }
                        const int offset=destination_base+(int(y)+1)*dst.stride+(x+2)*sizes[f];
                        std::memcpy(expected.data()+offset,&output,sizes[f]);
                    }
                    check(state.transfer_downscale(mem,src,dst),"Valid downscale rejected");
                    check(std::memcmp(data,expected.data(),4096)==0,"Downscale packed components, signed stride, overlap snapshot or guard mismatch");
                    auto invalid=src; invalid.width=5;
                    check(!state.transfer_downscale(mem,invalid,dst),"Odd downscale source width accepted without defined edge handling");
                    invalid=src; invalid.y=UINT32_MAX; invalid.stride=INT32_MIN;
                    check(!state.transfer_downscale(mem,invalid,dst),"Overflowed downscale source accepted");
                    invalid=src; invalid.format=SCE_GXM_TRANSFER_FORMAT_RAW32;
                    auto invalid_dest=dst; invalid_dest.format=invalid.format;
                    check(!state.transfer_downscale(mem,invalid,invalid_dest),"Raw downscale format silently interpreted as color");
                    invalid=src; invalid_dest=dst; invalid_dest.address=Ptr<void>(0xfffff000);
                    check(!state.transfer_downscale(mem,invalid,invalid_dest),"Unmapped downscale destination accepted");
                    check(std::memcmp(data,expected.data(),4096)==0,"Rejected downscale modified guest memory");
                    ++checked;
                }
                check(checked==64,"Missing downscale cases");
                std::cout<<"PASS64 packed/unpacked downscale cases, independent component means, both stride signs, overlap snapshots and rejection guards\n";
            }
            state.res_multiplier=1;
            ctx.record.color_surface=previous;
            std::memcpy(color.get(mem),values,16);
            state.set_context(ctx,mem);
            std::cout<<"PASS explicit/EndScene surface synchronization selects older/active targets at 2x, preserves ARGB/padding, signals notifications and rejects missing requests\n";
        }
        {
            const auto previous=ctx.record.color_surface;
            for(float scale:{1.0f,2.0f}) {
                state.res_multiplier=scale;
                auto surface=previous; surface.colorFormat=SCE_GXM_COLOR_FORMAT_U16_R;
                surface.strideInPixels=160; surface.data=Ptr<void>(alloc(mem,160*128*2,"Metal U16 native color target"));
                ctx.record.color_surface=surface;
                for(float red:{-0.25f,0.0f,0.125f,0.5f,1.0f,1.25f}) {
                    std::memset(surface.data.get(mem),0xa7,160*128*2);
                    state.set_context(ctx,mem);
                    const float input[]={red,0.375f,0.75f,1.0f}; std::memcpy(color.get(mem),input,16);
                    draw_triangle();
                    check(state.sync_surface(mem,surface),"U16 native color readback rejected");
                    const auto *words=static_cast<const uint16_t *>(surface.data.get(mem));
                    const int expected=int(std::lround(std::clamp(red,0.0f,1.0f)*65535));
                    for(unsigned y=0;y<128;++y) for(unsigned x=0;x<160;++x)
                        check(x<128?std::abs(int(words[y*160+x])-expected)<=1:words[y*160+x]==0xa7a7,
                            "U16 draw normalization, scale readback or padding mismatch");
                }
            }
            state.res_multiplier=1; ctx.record.color_surface=previous;
            std::memcpy(color.get(mem),values,16); state.set_context(ctx,mem);
            std::cout<<"PASS12 native U16 target draws at1x/2x: normalization, clamping, every guest pixel and row padding\n";
        }
        {
            const auto previous=ctx.record.color_surface;
            for(float scale:{1.0f,2.0f}) for(bool reverse:{false,true}) for(uint32_t gamma:{0u,1u}) {
                state.res_multiplier=scale;
                auto surface=previous; surface.colorFormat=reverse?SCE_GXM_COLOR_FORMAT_U8U8U8_RGB:SCE_GXM_COLOR_FORMAT_U8U8U8_BGR;
                surface.gamma=gamma;
                surface.strideInPixels=160; surface.data=Ptr<void>(alloc(mem,160*128*3,"Metal RGB native color target"));
                auto *guest=static_cast<uint8_t *>(surface.data.get(mem)); std::memset(guest,0xa7,160*128*3);
                const uint8_t initial[]={0x13,0x57,0x9b};
                for(unsigned y=0;y<128;++y) for(unsigned x=0;x<128;++x) std::memcpy(guest+(y*160+x)*3,initial,3);
                ctx.record.color_surface=surface; state.set_context(ctx,mem);
                DisplayState display; display.next_rendered_frame.base=surface.data;
                uint32_t width=0,height=0; const auto restored=state.dump_frame(display,width,height);
                check(width==uint32_t(128*scale) && height==width,"RGB target dimensions changed");
                const auto *restored_bytes=reinterpret_cast<const uint8_t *>(restored.data());
                for(size_t i=0;i<size_t(width)*height;++i) for(unsigned c=0;c<4;++c)
                    check(restored_bytes[i*4+c]==(c==3?255:initial[reverse?2-c:c]),"RGB initial restore or implicit alpha failed");
                const float input[]={0.25f,0.5f,0.75f,0.125f}; std::memcpy(color.get(mem),input,16);
                const auto encode=[&](float v) { return gamma ? (v<=0.0031308f?v*12.92f:1.055f*std::pow(v,1.0f/2.4f)-0.055f) : v; };
                draw_triangle(); check(state.sync_surface(mem,surface),"RGB render readback rejected");
                for(unsigned y=0;y<128;++y) for(unsigned x=0;x<160;++x) for(unsigned c=0;c<3;++c) {
                    const int expected=x<128?int(std::lround(encode(input[reverse?2-c:c])*255)):0xa7;
                    check(std::abs(int(guest[(y*160+x)*3+c])-expected)<=(x<128?1:0),"RGB draw three-byte readback or padding mismatch");
                }
                const auto frame=state.dump_frame(display,width,height);
                const auto *native=reinterpret_cast<const uint8_t *>(frame.data());
                for(size_t i=0;i<size_t(width)*height;++i) for(unsigned c=0;c<4;++c)
                    check(std::abs(int(native[i*4+c])-(c==3?255:int(std::lround(encode(input[c])*255))))<=(c==3?0:1),"RGB draw changed backing alpha or color");
            }
            state.res_multiplier=1; ctx.record.color_surface=previous;
            std::memcpy(color.get(mem),values,16); state.set_context(ctx,mem);
            std::cout<<"PASS RGB8 native draws/restoration: gamma off/on, two byte orders at1x/2x, three-byte readback, every native pixel, constant alpha1 and padding\n";
        }
        ctx.record.depth_stencil_surface.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_DF32);
        ctx.record.front_depth_func = SCE_GXM_DEPTH_FUNC_NEVER;
        renderer::Command scene{};
        scene.opcode = renderer::CommandOpcode::SetContext;
        renderer::CommandHelper scene_writer(&scene);
        auto *scene_surface = new SceGxmColorSurface(ctx.record.color_surface);
        auto *scene_target = static_cast<renderer::RenderTarget *>(&target);
        SceGxmDepthStencilSurface *disabled_depth = nullptr;
        scene_writer.push(scene_target);
        scene_writer.push(scene_surface); scene_writer.push(disabled_depth);
        renderer::CommandHelper scene_reader(&scene);
        renderer::cmd_handle_set_context(state, mem, config, scene_reader, state.features, &ctx);
        draw_triangle();
        state.finish(ctx);
        for (size_t i = 0; i < 128*128*4; ++i)
            check(std::abs(int(second_bytes[i]) - scene_color[i % 4]) <= 1, "Disabled depth surface retained the previous scene's depth test");
        std::cout << "PASS GXM SetContext disables stale depth/stencil state\n";
        const float quad[] = {-1,-1,1,1, 1,-1,1,1, 1,1,1,1, -1,1,1,1};
        // 32-bit indices above 65535 must retain their original vertex IDs.
        constexpr uint32_t high_index = 65536;
        Ptr<float> fan_vertices(alloc(mem, (high_index + 4)*16, "Metal fan vertices"));
        std::memcpy(fan_vertices.get(mem), quad, sizeof(quad));
        std::memcpy(fan_vertices.get(mem) + high_index*4, quad, sizeof(quad));
        ctx.record.vertex_streams[0] = {fan_vertices.cast<const uint8_t>(), (high_index + 4)*16};
        Ptr<uint16_t> fan16(alloc(mem, 8, "Metal fan16"));
        Ptr<uint32_t> fan32(alloc(mem, 16, "Metal fan32"));
        for (uint32_t i = 0; i < 4; ++i) { fan16.get(mem)[i] = i; fan32.get(mem)[i] = high_index + i; }
        for (auto format : {SCE_GXM_INDEX_FORMAT_U16, SCE_GXM_INDEX_FORMAT_U32}) {
            auto fan = format == SCE_GXM_INDEX_FORMAT_U16 ? fan16.cast<const void>() : fan32.cast<const void>();
            for (uint32_t count : {0u, 1u, 2u, 4u}) {
                std::memset(second_surface.get(mem), 0, 128*128*4);
                state.set_context(ctx, mem);
                draw_primitive(SCE_GXM_PRIMITIVE_TRIANGLE_FAN, format, fan, count);
                state.finish(ctx);
                for (size_t i = 0; i < 128*128*4; ++i)
                    check(std::abs(int(second_bytes[i]) - (count == 4 ? scene_color[i % 4] : 0)) <= 1,
                        "Triangle fan topology, short draw or 32-bit index mismatch");
            }
        }
        std::cout << "PASS GXM triangle fans, 16/32-bit indices, high vertex IDs and short-draw handling\n";

        // Uncharted uses this packed float target. Verify actual GXM pipeline
        // creation and every expanded GPU pixel, independently of guest readback
        // (packing expanded color surfaces back into guest RAM is still pending).
        const auto saved_surface = ctx.record.color_surface;
        ctx.record.color_surface.data = Ptr<void>(alloc(mem, 128*128*4, "Metal packed float target"));
        ctx.record.color_surface.colorFormat = SCE_GXM_COLOR_FORMAT_U2F10F10F10_ABGR;
        state.set_context(ctx, mem);
        draw_primitive(SCE_GXM_PRIMITIVE_TRIANGLE_FAN, SCE_GXM_INDEX_FORMAT_U16, fan16.cast<const void>(), 4);
        state.finish(ctx);
        DisplayState capture;
        capture.next_rendered_frame.base = ctx.record.color_surface.data;
        state.preclose_action();
        uint32_t captured_width = 0, captured_height = 0;
        const auto captured = state.dump_frame(capture, captured_width, captured_height);
        check(captured_width == 128 && captured_height == 128 && captured.size() == 128*128, "Missing expanded float surface capture");
        const auto *captured_bytes = reinterpret_cast<const uint8_t *>(captured.data());
        for (size_t i = 0; i < captured.size()*4; ++i)
            check(std::abs(int(captured_bytes[i]) - scene_color[i%4]) <= 1, "Packed float target GPU color mismatch");
        std::cout << "PASS GXM U2F10F10F10 expanded float target, all 16384 GPU pixels\n";
        std::cout << "PASS preclose notification preserves renderer resources until worker shutdown\n";
        ctx.record.color_surface = saved_surface;

        // Interleaved A/B scenes must load their own guest-addressed depth,
        // not whichever attachment the same GXM context used most recently.
        const auto saved_depth=ctx.record.depth_stencil_surface;
        const auto saved_depth_func=ctx.record.front_depth_func;
        const auto saved_depth_write=ctx.record.front_depth_write_mode;
        Ptr<void> depth_a(alloc(mem,128*128*4,"Metal depth A"));
        Ptr<void> depth_b(alloc(mem,128*128*4,"Metal depth B"));
        ctx.record.depth_stencil_surface={};
        auto &ds=ctx.record.depth_stencil_surface;
        ds.set_format(SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24);
        ds.set_stride(128);
        ds.force_store=1;
        ds.depth_data=depth_a; ds.background_depth=0.25f;
        state.set_context(ctx,mem); state.finish(ctx);
        ds.depth_data=depth_b; ds.background_depth=0.75f;
        state.set_context(ctx,mem); state.finish(ctx);
        ds.force_load=1;
        ctx.record.front_depth_func=SCE_GXM_DEPTH_FUNC_LESS;
        ctx.record.front_depth_write_mode=SCE_GXM_DEPTH_WRITE_DISABLED;
        // Fix screen depth independently of this game's vertex shader output.
        execute(renderer::GXMState::Viewport,false,64.0f,64.0f,0.5f,64.0f,64.0f,0.0f);
        for (bool use_b : {false,true,false}) {
            ds.depth_data=use_b?depth_b:depth_a;
            std::memset(second_surface.get(mem),0,128*128*4);
            state.set_context(ctx,mem);
            draw_primitive(SCE_GXM_PRIMITIVE_TRIANGLE_FAN,SCE_GXM_INDEX_FORMAT_U16,fan16.cast<const void>(),4);
            state.finish(ctx);
            for (size_t i=0;i<128*128*4;++i)
                check(std::abs(int(second_bytes[i])-(use_b?scene_color[i%4]:0))<=1,
                    "Interleaved depth surface mismatch B="+std::to_string(use_b)+" byte="+std::to_string(i)+" actual="+std::to_string(second_bytes[i]));
        }
        std::cout << "PASS GXM A/B/A depth surface switching, force-load, every depth-tested output pixel\n";
        ctx.record.depth_stencil_surface=saved_depth;
        ctx.record.front_depth_func=saved_depth_func;
        ctx.record.front_depth_write_mode=saved_depth_write;
        execute(renderer::GXMState::Viewport,false,64.0f,64.0f,0.0f,64.0f,64.0f,1.0f);

        // A shader with undefined color output must execute without altering
        // the target, even if its instructions happen to write a color value.
        const_cast<SceGxmProgram *>(fp->program.get(mem))->program_flags |= SCE_GXM_PROGRAM_FLAG_OUTPUT_UNDEFINED;
        check(renderer::create(fp->renderer_data, state, *fp->program.get(mem), nullptr, state.gxp_ptr_map), "Cannot create no-color fragment program");
        execute(renderer::GXMState::Program, fp_addr.cast<void>(), true);
        std::memset(second_surface.get(mem), 73, 128*128*4);
        state.set_context(ctx, mem);
        draw_primitive(SCE_GXM_PRIMITIVE_TRIANGLE_FAN, SCE_GXM_INDEX_FORMAT_U16, fan16.cast<const void>(), 4);
        state.finish(ctx);
        for (size_t i = 0; i < 128*128*4; ++i)
            check(second_bytes[i] == 73, "Undefined fragment color overwrote the target");
        std::cout << "PASS native output-undefined fragment preserves every target pixel\n";

        state.context = nullptr;
        owner.reset();
        vp->~SceGxmVertexProgram(); fp->~SceGxmFragmentProgram();
        return 0;
    } catch (const std::exception &error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
