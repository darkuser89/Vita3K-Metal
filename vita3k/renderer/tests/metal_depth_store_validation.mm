// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <renderer/gxm_types.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
static void check(bool value, const std::string &error) { if (!value) throw std::runtime_error(error); }
int main() {
    @autoreleasepool { try {
        std::string error;
        auto device=renderer::metal::Device::create(error);check(bool(device),error);
        renderer::metal::SurfaceCaster caster(*device);
        NSError *native_error=nil;
        auto library=[device->native_device() newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;
vertex float4 vertex_main(uint i [[vertex_id]]) {
    const float2 p[]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    return float4(p[i],0,1);
}
struct Output { float depth [[depth(any)]]; uint stencil [[stencil]]; };
fragment Output fragment_main(float4 p [[position]], uint sample [[sample_id]]) {
    uint2 q=uint2(p.xy);
    return {float((q.x*13+q.y*29+sample*7)%65536)/65536.f,(q.x*3+q.y*11+sample*17)&255};
})" options:nil error:&native_error];
        check(library!=nil,native_error.localizedDescription.UTF8String ?: "Cannot compile depth fixture");
        auto render=[&](unsigned w,unsigned h,unsigned samples) {
            auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:w height:h mipmapped:NO];
            td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
            td.sampleCount=samples;if(samples>1) td.textureType=MTLTextureType2DMultisample;
            auto texture=[device->native_device() newTextureWithDescriptor:td];check(texture!=nil,"Cannot allocate depth fixture");
            auto pd=[MTLRenderPipelineDescriptor new];pd.vertexFunction=[library newFunctionWithName:@"vertex_main"];pd.fragmentFunction=[library newFunctionWithName:@"fragment_main"];
            pd.depthAttachmentPixelFormat=pd.stencilAttachmentPixelFormat=td.pixelFormat;pd.rasterSampleCount=samples;
            auto pipeline=device->create_pipeline(pd,error);check(pipeline!=nil,error);
            auto ds=[MTLDepthStencilDescriptor new];ds.depthCompareFunction=MTLCompareFunctionAlways;ds.depthWriteEnabled=YES;
            auto ss=[MTLStencilDescriptor new];ss.stencilCompareFunction=MTLCompareFunctionAlways;ss.depthStencilPassOperation=MTLStencilOperationReplace;ds.frontFaceStencil=ds.backFaceStencil=ss;
            auto pass=[MTLRenderPassDescriptor renderPassDescriptor];pass.depthAttachment.texture=pass.stencilAttachment.texture=texture;
            pass.depthAttachment.loadAction=pass.stencilAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=pass.stencilAttachment.storeAction=MTLStoreActionStore;
            auto cmd=[device->command_queue() commandBuffer];auto enc=[cmd renderCommandEncoderWithDescriptor:pass];
            [enc setRenderPipelineState:pipeline];[enc setDepthStencilState:[device->native_device() newDepthStencilStateWithDescriptor:ds]];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[enc endEncoding];check(device->submit_and_wait(cmd,error),error);return texture;
        };
        unsigned cases=0;size_t checked=0;
        for(unsigned scale:{1u,2u,3u}) for(unsigned samples:{1u,2u,4u}) for(bool tiled:{false,true}) {
            const unsigned sx=samples==4?2:1,sy=samples>1?2:1;
            const unsigned w=32*sx,h=16*sy,stride=w+32,rows=tiled?32:h;
            auto texture=render(32*scale,16*scale,samples);
            for(unsigned format=0;format<5;++format) {
                const unsigned bytes=format==0?2:format==3?0:4;
                const bool packed=format==4,separate=format==2 || format==3;
                const size_t size=size_t(stride)*rows;
                renderer::metal::DepthMemoryLayout layout{w,h,stride,bytes,size*bytes,separate?size:0,tiled,packed};
                SceGxmDepthStencilSurface surface{};
                std::vector<uint8_t> depth(layout.depth_size+32,0xa5),stencil(layout.stencil_size+32,0x79),expected=depth,expected_stencil=stencil;
                check(caster.store_depth_memory(texture,surface,layout,scale,{depth.data(),layout.depth_size},{stencil.data(),layout.stencil_size}),"Store rejected valid native image");
                for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) {
                    const unsigned nx=(x/sx)*scale,ny=(y/sy)*scale,sample=(y%sy)*sx+x%sx;
                    const float value=float((nx*13+ny*29+sample*7)%65536)/65536.f;
                    const uint8_t st=(nx*3+ny*11+sample*17)&255;
                    const size_t at=tiled?((y/32)*(stride/32)+x/32)*1024+(y%32)*32+x%32:y*stride+x;
                    if(bytes==2) {const uint16_t v=uint16_t(std::lround(double(value)*65535));std::memcpy(expected.data()+at*2,&v,2);}
                    else if(packed) {const uint32_t v=uint32_t(std::llround(double(value)*16777215))|(uint32_t(st)<<24);std::memcpy(expected.data()+at*4,&v,4);}
                    else if(bytes) std::memcpy(expected.data()+at*4,&value,4);
                    if(separate) expected_stencil[at]=st;
                }
                check(depth==expected && stencil==expected_stencil,"Native subpixel/sample selection or packed value/padding mismatch: scale="+std::to_string(scale)+" samples="+std::to_string(samples)+" format="+std::to_string(format));
                ++cases;checked+=depth.size()+stencil.size();
            }
        }
        std::cout<<"PASS native depth/stencil readback cases="<<cases<<" bytes="<<checked<<" 5formats,1x/2x/3x,1/2/4samples,tiled/linear,per-native-pixel variation and padding guards\n";
        if(std::getenv("VITA3K_METAL_BENCH_DEPTH_STORE")) {
            constexpr unsigned w=1440,h=816,scale=2;
            auto texture=render(w*scale,h*scale,1);
            renderer::metal::DepthMemoryLayout layout{w,h,w,4,size_t(w)*h*4,0,false,true};
            SceGxmDepthStencilSurface surface{};
            std::vector<uint8_t> depth(layout.depth_size);
            std::vector<double> times;
            for(unsigned i=0;i<24;++i) {
                auto start=std::chrono::steady_clock::now();
                check(caster.store_depth_memory(texture,surface,layout,scale,depth,{}),"Benchmark readback rejected");
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                if(i>=4) times.push_back(ms);
            }
            std::sort(times.begin(),times.end());
            std::cout<<"BENCH S8D24 1440x816 scale2 store median_ms="<<(times[9]+times[10])/2<<" min_ms="<<times.front()<<" max_ms="<<times.back()<<" timed_runs=20\n";
        }
        return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;} }
}
