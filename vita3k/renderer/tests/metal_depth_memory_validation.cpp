// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/state.h>
#include <renderer/driver_functions.h>
#include <renderer/functions.h>
#include <mem/state.h>
#include <config/state.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
static void check(bool ok,const std::string &why) {if(!ok) throw std::runtime_error(why);}
int main() {
    try {
        MemState mem;check(init(mem,false),"Cannot initialize guest RAM");
        renderer::metal::MetalContext ctx;renderer::metal::MetalState state;
        check(state.init(),"Cannot initialize Metal");state.context=&ctx;
        renderer::metal::MetalRenderTarget target;target.width=32;target.height=16;ctx.current_render_target=&target;
        ctx.record.color_surface.downscale=1;
        Config config;
        Ptr<uint32_t> notification(alloc(mem,4,"Depth store notification"));
        auto end=[&] {
            *notification.get(mem)=0;
            renderer::Command command{};renderer::CommandHelper writer(&command);
            SceGxmNotification vertex{},fragment{notification,0x51a7u};
            writer.push(vertex);writer.push(fragment);
            renderer::CommandHelper reader(&command);
            renderer::cmd_handle_sync_surface_data(state,mem,config,reader,state.features,&ctx);
            check(*notification.get(mem)==0x51a7u,"Depth store notification not delivered");
        };
        unsigned cases=0;
        for(auto format:{SCE_GXM_DEPTH_STENCIL_FORMAT_D16,SCE_GXM_DEPTH_STENCIL_FORMAT_DF32,
            SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8,SCE_GXM_DEPTH_STENCIL_FORMAT_S8,SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24})
        for(auto mode:{SCE_GXM_MULTISAMPLE_NONE,SCE_GXM_MULTISAMPLE_2X,SCE_GXM_MULTISAMPLE_4X})
        for(bool tiled:{false,true}) for(unsigned scale:{1u,2u}) {
            state.res_multiplier=scale;target.multisample_mode=mode;target.width=32*scale;target.height=16*scale;
            const uint32_t width=32*(mode==SCE_GXM_MULTISAMPLE_4X?2:1),height=16*(mode==SCE_GXM_MULTISAMPLE_NONE?1:2),stride=width+32;
            const size_t rows=tiled?32:height,count=stride*rows;
            const unsigned bytes=format==SCE_GXM_DEPTH_STENCIL_FORMAT_D16?2:format==SCE_GXM_DEPTH_STENCIL_FORMAT_S8?0:4;
            const bool separate=format==SCE_GXM_DEPTH_STENCIL_FORMAT_S8 || format==SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8;
            const bool packed=format==SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24;
            auto &ds=ctx.record.depth_stencil_surface;ds={};ds.set_format(format);
            ds.set_type(tiled?SCE_GXM_DEPTH_STENCIL_SURFACE_TILED:SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR);ds.set_stride(stride);
            if(bytes) ds.depth_data=Ptr<void>(alloc(mem,count*bytes,"Guest depth plane"));
            if(separate) ds.stencil_data=Ptr<void>(alloc(mem,count,"Guest stencil plane"));
            auto offset=[&](unsigned x,unsigned y) -> size_t {return tiled?((y/32)*(stride/32)+x/32)*1024+(y%32)*32+x%32:y*stride+x;};
            for(unsigned iteration=0;iteration<3;++iteration) {
                ds.force_load=iteration!=2;ds.force_store=1;ds.background_depth=.5f;ds.stencil=0xb9;
                std::vector<uint8_t> depths(count*bytes,0xc3),stencils(separate?count:0,0xc3);
                for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
                    const size_t at=offset(x,y);const uint8_t stencil=(x+17*y+iteration*71)&255;
                    if(bytes==2) {const uint16_t v=uint16_t((x*997+y*251+iteration*137)&65535);std::memcpy(depths.data()+at*2,&v,2);}
                    else if(packed) {const uint32_t v=((x*71237+y*1237+iteration*5171)&0xffffff)|(uint32_t(stencil)<<24);std::memcpy(depths.data()+at*4,&v,4);}
                    else if(bytes) {const float v=float((x+3*y+iteration*5)%17)/16.f;std::memcpy(depths.data()+at*4,&v,4);}
                    if(separate) stencils[at]=stencil;
                }
                if(bytes) std::memcpy(ds.depth_data.get(mem),depths.data(),depths.size());
                if(separate) std::memcpy(ds.stencil_data.get(mem),stencils.data(),stencils.size());
                state.set_context(ctx,mem);
                // Destroy CPU values after import. A no-op store or a CPU copy
                // cannot satisfy the following native-depth/stencil checks.
                if(bytes) std::memset(ds.depth_data.get(mem),0x6b,depths.size());
                if(separate) std::memset(ds.stencil_data.get(mem),0x6b,stencils.size());
                std::vector<uint8_t> expected_depth(depths.size(),0x6b),expected_stencil(stencils.size(),0x6b);
                for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
                    const size_t at=offset(x,y);
                    if(bytes) {
                        if(iteration!=2) std::memcpy(expected_depth.data()+at*bytes,depths.data()+at*bytes,bytes);
                        else if(bytes==2) {const uint16_t v=32768;std::memcpy(expected_depth.data()+at*2,&v,2);}
                        else if(packed) {const uint32_t v=0xb9800000;std::memcpy(expected_depth.data()+at*4,&v,4);}
                        else {const float v=.5f;std::memcpy(expected_depth.data()+at*4,&v,4);}
                    }
                    if(separate) expected_stencil[at]=iteration==2?0xb9:stencils[at];
                }
                end();
                const std::string context=" format="+std::to_string(format)+" mode="+std::to_string(mode)+" tiled="+std::to_string(tiled)+" scale="+std::to_string(scale)+" iteration="+std::to_string(iteration);
                if(bytes) for(size_t i=0;i<depths.size();++i)
                    check(static_cast<uint8_t *>(ds.depth_data.get(mem))[i]==expected_depth[i],"Depth store byte mismatch"+context+" offset="+std::to_string(i)+" actual="+std::to_string(static_cast<uint8_t *>(ds.depth_data.get(mem))[i])+" expected="+std::to_string(expected_depth[i]));
                if(separate) for(size_t i=0;i<stencils.size();++i)
                    check(static_cast<uint8_t *>(ds.stencil_data.get(mem))[i]==expected_stencil[i],"Stencil store byte mismatch"+context+" offset="+std::to_string(i));
                ++cases;
            }
            ds.force_load=0;ds.force_store=0;
            if(bytes) std::memset(ds.depth_data.get(mem),0x47,count*bytes);
            if(separate) std::memset(ds.stencil_data.get(mem),0x47,count);
            state.set_context(ctx,mem);end();
            if(bytes) for(size_t i=0;i<count*bytes;++i) check(static_cast<uint8_t *>(ds.depth_data.get(mem))[i]==0x47,"Disabled depth store modified guest RAM");
            if(separate) for(size_t i=0;i<count;++i) check(static_cast<uint8_t *>(ds.stencil_data.get(mem))[i]==0x47,"Disabled stencil store modified guest RAM");
        }
        state.context=nullptr;
        std::cout<<"PASS "<<cases<<" depth/stencil memory cases:5formats,1/2/4samples,linear/tiled,1x/2x,initial/repeated RAM loads,clear stores,poisoned RAM,padding,notifications and disabled-store guards\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
