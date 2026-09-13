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

int main(int argc,char **argv) {
    if(argc!=3) { std::cerr<<"Usage: metal-depth-alias-validation <Sly 3a2fc0 vertex.gxp> <Sly 05e3d4 fragment.gxp>\n";return 2; }
    try {
        MemState mem;check(init(mem,false),"Cannot initialize guest memory");
        renderer::metal::MetalContext ctx;
        renderer::metal::MetalState state;check(state.init(),"Cannot initialize Metal");state.context=&ctx;
        Config config;state.late_init(config,"metal-depth-alias-validation",mem);
        auto execute=[&](renderer::GXMState code,auto...args) {
            renderer::Command command{};renderer::CommandHelper writer(&command);
            check(writer.push(code) && (writer.push(args)&&...),"Cannot encode state");
            renderer::CommandHelper reader(&command);
            renderer::cmd_handle_set_state(state,mem,config,reader,state.features,&ctx);
        };
        Ptr<SceGxmVertexProgram> vp_addr(alloc(mem,sizeof(SceGxmVertexProgram),"Depth alias vertex"));
        Ptr<SceGxmFragmentProgram> fp_addr(alloc(mem,sizeof(SceGxmFragmentProgram),"Depth alias fragment"));
        auto *vp=new(vp_addr.get(mem)) SceGxmVertexProgram{};
        auto *fp=new(fp_addr.get(mem)) SceGxmFragmentProgram{};
        vp->program=load(mem,argv[1]);fp->program=load(mem,argv[2]);
        const auto &gxp=*vp->program.get(mem);
        for(uint32_t i=0;i<gxp.parameter_count;++i) {
            const auto &param=gxp.program_parameters()[i];
            if(param.category!=SCE_GXM_PARAMETER_CATEGORY_ATTRIBUTE) continue;
            const std::string name=param.name();
            const unsigned offset=name=="IN.pos"?0:name=="IN.color"?16:name=="IN.uv0"?32:999;
            check(offset!=999,"Unexpected vertex input: "+name);
            SceGxmVertexAttribute a{};a.offset=offset;a.format=SCE_GXM_ATTRIBUTE_FORMAT_F32;
            a.componentCount=param.component_count;a.regIndex=param.resource_index;vp->attributes.push_back(a);
        }
        check(vp->attributes.size()==3,"Expected position/color/UV attributes");
        vp->streams.push_back({48,SCE_GXM_INDEX_SOURCE_EACH_VERTEX_16BIT});
        check(renderer::create(vp->renderer_data,state,gxp,state.gxp_ptr_map,vp->attributes),"Cannot create vertex program");
        check(renderer::create(fp->renderer_data,state,*fp->program.get(mem),nullptr,state.gxp_ptr_map),"Cannot create fragment program");
        execute(renderer::GXMState::Program,vp_addr.cast<void>(),false);
        execute(renderer::GXMState::Program,fp_addr.cast<void>(),true);
        const float vertices[]={-1,-1,0,1, 1,1,1,1, .5,.5,0,0,
            3,-1,0,1, 1,1,1,1, .5,.5,0,0, -1,3,0,1, 1,1,1,1, .5,.5,0,0};
        Ptr<float> stream(alloc(mem,sizeof(vertices),"Depth alias vertices"));std::memcpy(stream.get(mem),vertices,sizeof(vertices));
        ctx.record.vertex_streams[0]={stream.cast<const uint8_t>(),sizeof(vertices)};
        const float matrix[]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        Ptr<float> uniform(alloc(mem,sizeof(matrix),"Depth alias identity"));std::memcpy(uniform.get(mem),matrix,sizeof(matrix));
        execute(renderer::GXMState::UniformBuffer,uniform.cast<uint8_t>(),true,0,uint32_t(sizeof(matrix)));
        Ptr<uint16_t> indices(alloc(mem,6,"Depth alias indices"));for(unsigned i=0;i<3;++i) indices.get(mem)[i]=i;
        renderer::metal::MetalRenderTarget target;target.width=target.height=128;ctx.current_render_target=&target;
        ctx.record.front_depth_func=SCE_GXM_DEPTH_FUNC_ALWAYS;
        ctx.record.front_depth_write_mode=SCE_GXM_DEPTH_WRITE_DISABLED;
        ctx.record.front_stencil_state_op.func=SCE_GXM_STENCIL_FUNC_ALWAYS;
        unsigned passed=0;
        for(unsigned scale:{1u,2u}) for(auto mode:{SCE_GXM_MULTISAMPLE_NONE,SCE_GXM_MULTISAMPLE_2X,SCE_GXM_MULTISAMPLE_4X}) for(bool d16:{false,true}) {
            state.res_multiplier=scale;target.multisample_mode=mode;
            const unsigned sx=mode==SCE_GXM_MULTISAMPLE_4X?2:1,sy=mode==SCE_GXM_MULTISAMPLE_NONE?1:2,bytes=d16?2:4;
            const unsigned stride=128*sx;
            auto &surface=ctx.record.color_surface;surface={};
            surface.width=surface.height=surface.strideInPixels=128;
            surface.colorFormat=SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;surface.surfaceType=SCE_GXM_COLOR_SURFACE_LINEAR;
            surface.downscale=mode!=SCE_GXM_MULTISAMPLE_NONE;
            surface.data=Ptr<void>(alloc(mem,128*128*4,"Depth alias color"));std::memset(surface.data.get(mem),0,128*128*4);
            auto &ds=ctx.record.depth_stencil_surface;ds={};
            ds.set_format(d16?SCE_GXM_DEPTH_STENCIL_FORMAT_D16:SCE_GXM_DEPTH_STENCIL_FORMAT_DF32);
            ds.set_type(SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR);ds.set_stride(stride);ds.force_store=1;
            ds.depth_data=Ptr<void>(alloc(mem,size_t(stride)*128*sy*bytes,"Depth alias stale RAM"));
            std::memset(ds.depth_data.get(mem),0,size_t(stride)*128*sy*bytes);
            SceGxmTexture texture{};texture.type=SCE_GXM_TEXTURE_LINEAR_STRIDED>>29;texture.width=texture.height=31;
            const auto format=d16?SCE_GXM_TEXTURE_FORMAT_U16_111R:SCE_GXM_TEXTURE_FORMAT_F32_111R;
            texture.base_format=uint32_t(format)>>24;texture.swizzle_format=(uint32_t(format)>>12)&7;
            texture.data_addr=(ds.depth_data.address()+(16*stride+16)*bytes)>>2;
            const uint32_t pitch=stride*bytes/4-1;
            texture.mip_filter=pitch&1;texture.min_filter=(pitch>>1)&3;texture.mip_count=(pitch>>3)&15;texture.lod_bias=(pitch>>7)&63;
            execute(renderer::GXMState::Texture,uint32_t(0),texture);
            execute(renderer::GXMState::Viewport,false,64.f,64.f,0.f,64.f,64.f,1.f);
            execute(renderer::GXMState::RegionClip,SCE_GXM_REGION_CLIP_NONE,uint32_t(0),uint32_t(128),uint32_t(0),uint32_t(128));
            for(float clear:{.25f,.75f,.5f}) {
                ds.background_depth=clear;
                state.set_context(ctx,mem);
                // No explicit finish: sampling a partial view of this active
                // attachment must publish the clear and invalidate prior crops.
                state.draw(ctx,mem,SCE_GXM_PRIMITIVE_TRIANGLES,SCE_GXM_INDEX_FORMAT_U16,indices.get(mem),3,1);
                check(state.sync_surface(mem,surface),"Depth alias output sync rejected");
                const auto *actual=static_cast<const uint8_t *>(surface.data.get(mem));
                for(size_t i=0;i<128*128*4;++i) {
                    const int expected=i%4==0?int(std::lround(clear*255)):255;
                    check(std::abs(int(actual[i])-expected)<=1,"Depth alias sampling mismatch scale="+std::to_string(scale)
                        +" mode="+std::to_string(mode)+" d16="+std::to_string(d16)+" clear="+std::to_string(clear)
                        +" byte="+std::to_string(i)+" actual="+std::to_string(actual[i])+" expected="+std::to_string(expected));
                }
                ++passed;
            }
        }
        state.context=nullptr;vp->~SceGxmVertexProgram();fp->~SceGxmFragmentProgram();
        std::cout<<"PASS "<<passed<<" unchanged Sly GXP depth crop draws: D16/F32,1/2/4samples,1x/2x,stale RAM,active clear and crop invalidation,every output pixel\n";
        return 0;
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
}
