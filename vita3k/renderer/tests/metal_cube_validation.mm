// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <renderer/metal/state.h>
#include <renderer/metal/textures.h>
#include <mem/state.h>
#include <gxm/functions.h>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
// Sample every encoded byte value and an interpolation between adjacent
// texels. Rows have padding and distinct contents; gamma changes reuse the
// same guest allocation so cache-key collisions cannot hide behind addresses.
static void check_gamma_uploads(renderer::metal::MetalState &state, MemState &mem) {
    for(unsigned components:{1u,2u,4u}) {
        const unsigned width=256, height=2, pitch=(width+8)*components;
        Ptr<uint8_t> storage(alloc(mem,pitch*height+64,"Metal gamma upload fixture"));
        auto *bytes=storage.get(mem);
        std::memset(bytes,0xcd,pitch*height+64);
        auto code=[](unsigned x,unsigned y,unsigned c) -> uint8_t {
            switch(c) {
            case 0:return uint8_t(x+y*13);
            case 1:return uint8_t(255-x+y*31);
            case 2:return uint8_t(x*73+y*11);
            default:return uint8_t(x*29+17+y*7);
            }
        };
        for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) for(unsigned c=0;c<components;++c)
            bytes[32+y*pitch+x*components+c]=code(x,y,c);
        const std::vector<uint8_t> before(bytes,bytes+pitch*height+64);
        SceGxmTexture guest{};
        guest.type=SCE_GXM_TEXTURE_LINEAR_STRIDED>>29;
        guest.width=width-1;guest.height=height-1;
        const unsigned stride_field=pitch/4-1;
        guest.mip_filter=stride_field&1;guest.min_filter=(stride_field>>1)&3;
        guest.mip_count=(stride_field>>3)&15;guest.lod_bias=stride_field>>7;
        const auto format=components==1?SCE_GXM_TEXTURE_BASE_FORMAT_U8:components==2?SCE_GXM_TEXTURE_BASE_FORMAT_U8U8:SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8;
        guest.base_format=(uint32_t(format)>>24)&31;guest.format0=uint32_t(format)>>31;
        guest.data_addr=(storage.address()+32)>>2;
        check(gxm::get_stride_in_bytes(guest)==pitch,"Incorrect gamma fixture stride");
        std::array<id<MTLTexture>,4> images{};
        for(unsigned gamma:{0u,1u,3u,1u,0u}) {
            guest.gamma_mode=gamma;
            state.texture_cache.cache_and_bind_texture(guest,mem);
            auto texture=renderer::metal::current_texture(state.texture_cache);
            check(texture!=nil,"Gamma upload has no texture");
            if(images[gamma]) check(images[gamma]==texture,"Returning to a gamma mode missed its existing cache entry");
            for(unsigned other:{0u,1u,3u}) if(other!=gamma && images[other])
                check(texture!=images[other],"Different gamma modes shared one GPU image");
            images[gamma]=texture;
            auto device=texture.device;
            NSError *error=nil;
            auto library=[device newLibraryWithSource:@R"(#include <metal_stdlib>
using namespace metal;
kernel void probe(texture2d<float> image [[texture(0)]], device float4 *output [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    constexpr sampler point(coord::pixel, address::clamp_to_edge, filter::nearest);
    constexpr sampler linear(coord::pixel, address::clamp_to_edge, filter::linear);
    uint index=p.y*image.get_width()+p.x;
    output[index*2]=image.sample(point,float2(p)+0.5);
    output[index*2+1]=image.sample(linear,float2(p)+float2(1.0,0.5));
})" options:nil error:&error];
            check(library!=nil,error.localizedDescription.UTF8String?:"Gamma probe compilation failed");
            auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
            check(pipeline!=nil,"Gamma probe pipeline failed");
            auto output=[device newBufferWithLength:width*height*32 options:MTLResourceStorageModeShared];
            auto queue=[device newCommandQueue];auto commands=[queue commandBuffer];auto encoder=[commands computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];[encoder setTexture:texture atIndex:0];[encoder setBuffer:output offset:0 atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(width,height,1) threadsPerThreadgroup:MTLSizeMake(16,1,1)];
            [encoder endEncoding];[commands commit];[commands waitUntilCompleted];
            check(commands.status==MTLCommandBufferStatusCompleted,"Gamma upload GPU probe failed");
            const auto *actual=static_cast<const float *>(output.contents);
            auto expected=[&](unsigned x,unsigned y,unsigned c) -> double {
                if(c>=components) return c==3?1:0;
                const double value=code(x,y,c)/255.0;
                const unsigned count=gamma==3?2:components==4?3:1;
                return gamma && c<count ? (value<=0.04045?value/12.92:std::pow((value+0.055)/1.055,2.4)):value;
            };
            double max_error=0;
            for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) for(unsigned c=0;c<4;++c) {
                const double point=expected(x,y,c);
                const double linear=(point+expected(std::min(x+1,width-1),y,c))*0.5;
                const unsigned i=(y*width+x)*8+c;
                max_error=std::max(max_error,std::max(std::abs(actual[i]-point),std::abs(actual[i+4]-linear)));
            }
            check(max_error<0.0006,"Incorrect per-channel gamma or interpolation occurred after decoding");
            check(std::memcmp(bytes,before.data(),before.size())==0,"Gamma upload changed guest bytes or row padding");
            std::cout<<"PASS production strided gamma upload: components="<<components<<", gamma="<<gamma
                     <<", 512 point/linear samples, max_error="<<max_error<<'\n';
        }
    }
}
int main() {
    @autoreleasepool {
        try {
            MemState mem;
            check(init(mem,false),"Cannot initialize guest memory");
            renderer::metal::MetalState state;
            check(state.init(),"Cannot initialize Metal");
            state.texture_cache.backend=renderer::Backend::Metal;
            check(state.texture_cache.init(false,(std::filesystem::temp_directory_path()/"vita3k-metal-cube-tests").string(),"fixture"),"Cannot initialize texture cache");
            for(bool cube_mode:{true,false}) for(bool bc1:{true,false}) for(bool mipmaps:{false,true}) for(unsigned gamma:{0u,1u,3u}) {
                const unsigned faces=cube_mode?6:1;
                SceGxmTexture original{};
                original.type=(bc1?SCE_GXM_TEXTURE_SWIZZLED:SCE_GXM_TEXTURE_SWIZZLED_ARBITRARY)>>29;
                if(bc1) original.width_base2=original.height_base2=7;
                else original.width=original.height=9;
                const auto format=bc1?SCE_GXM_TEXTURE_BASE_FORMAT_UBC1:SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8;
                original.base_format=(uint32_t(format)>>24)&31;
                original.format0=uint32_t(format)>>31;
                original.mip_count=mipmaps?1:15;
                original.gamma_mode=gamma;
                const size_t face_stride=cube_mode ? (bc1?(mipmaps?12288:8192):(mipmaps?1364:1024))
                    : (bc1?(mipmaps?10240:8192):(mipmaps?1280:1024));
                const size_t total=face_stride*faces;
                Ptr<uint8_t> allocation(alloc(mem,total+64,"Metal six-face fixture"));
                std::memset(allocation.get(mem),0xcd,total+64);
                original.data_addr=(allocation.address()+32)>>2;
                const auto preserved=original;
                auto cube=cube_mode?renderer::metal::cube_texture_descriptor(original):original;
                check(std::memcmp(&original,&preserved,sizeof(original))==0,"Cube interpretation modified the guest descriptor");
                check(!cube_mode || cube.texture_type()==(bc1?SCE_GXM_TEXTURE_CUBE:SCE_GXM_TEXTURE_CUBE_ARBITRARY),"Wrong effective cube layout");
                check(renderer::metal::texture_storage_size(cube)==total,"Cube face alignment/storage extent mismatch");
                std::array<std::array<float,4>,12> expected{};
                const unsigned mips=mipmaps?2:1;
                const uint16_t colors[]={0xf800,0x07e0,0x001f,0xffe0,0xf81f,0x07ff};
                for(unsigned face=0;face<faces;++face) for(unsigned mip=0;mip<mips;++mip) {
                    const size_t mip_offset=mip?(bc1?8192:1024):0;
                    auto *destination=allocation.get(mem)+32+face*face_stride+mip_offset;
                    if(bc1) {
                        const uint16_t color=colors[(face+mip*3)%6];
                        const size_t bytes=mip?2048:8192;
                        for(size_t block=0;block<bytes;block+=8) {
                            std::memcpy(destination+block,&color,2); std::memcpy(destination+block+2,&color,2);
                            std::memset(destination+block+4,0,4);
                        }
                        expected[face*mips+mip]={float((color>>11)&31)/31,float((color>>5)&63)/63,float(color&31)/31,1};
                    } else {
                        const uint8_t color[]={uint8_t(30+face*30),uint8_t(40+mip*120),uint8_t(200-face*20),137};
                        const size_t bytes=mip?256:1024;
                        for(size_t pixel=0;pixel<bytes;pixel+=4) std::memcpy(destination+pixel,color,4);
                        expected[face*mips+mip]={color[0]/255.0f,color[1]/255.0f,color[2]/255.0f,color[3]/255.0f};
                    }
                }
                if(gamma) for(auto &pixel:expected) for(unsigned c=0;c<(gamma==3?2u:3u);++c) {
                    const double encoded=pixel[c];
                    pixel[c]=encoded<=0.04045?encoded/12.92:std::pow((encoded+0.055)/1.055,2.4);
                }
                const float tolerance=gamma?0.0006f:1e-5f;
                std::vector<uint8_t> guest_before(allocation.get(mem),allocation.get(mem)+total+64);
                auto &cache=state.texture_cache;
                cache.cache_and_bind_texture(cube,mem);
                auto texture=renderer::metal::current_texture(cache);
                check(texture && texture.textureType==(cube_mode?MTLTextureTypeCube:MTLTextureType2D) && texture.mipmapLevelCount==mips,"Production cache did not create the required cube/mip texture");
                auto device=texture.device;
                NSString *source=cube_mode?@R"(#include <metal_stdlib>
using namespace metal;
kernel void probe(texturecube<float> image [[texture(0)]], device float4 *output [[buffer(0)]],
    constant uint &mips [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(coord::normalized,filter::nearest,mip_filter::nearest);
    const float3 directions[]={float3(1,0,0),float3(-1,0,0),float3(0,1,0),float3(0,-1,0),float3(0,0,1),float3(0,0,-1)};
    output[i]=image.sample(s,directions[i/mips],level(i%mips));
})":@R"(#include <metal_stdlib>
using namespace metal;
kernel void probe(texture2d<float> image [[texture(0)]], device float4 *output [[buffer(0)]],
    constant uint &mips [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(coord::normalized,filter::nearest,mip_filter::nearest);
    output[i]=image.sample(s,float2(0.5),level(i));
})";
                NSError *error=nil;
                auto library=[device newLibraryWithSource:source options:nil error:&error];
                check(library!=nil,error.localizedDescription.UTF8String?:"Cannot compile cube probe");
                auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
                auto output=[device newBufferWithLength:faces*mips*16 options:MTLResourceStorageModeShared];
                auto queue=[device newCommandQueue]; auto commands=[queue commandBuffer]; auto encoder=[commands computeCommandEncoder];
                [encoder setComputePipelineState:pipeline]; [encoder setTexture:texture atIndex:0];
                [encoder setBuffer:output offset:0 atIndex:0]; [encoder setBytes:&mips length:4 atIndex:1];
                [encoder dispatchThreads:MTLSizeMake(faces*mips,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [encoder endEncoding]; [commands commit]; [commands waitUntilCompleted];
                check(commands.status==MTLCommandBufferStatusCompleted,"Cube sampling GPU command failed");
                const auto *actual=static_cast<const float *>(output.contents);
                for(unsigned i=0;i<faces*mips;++i) for(unsigned c=0;c<4;++c)
                    check(std::abs(actual[i*4+c]-expected[i][c])<tolerance,"Wrong cube face, mip address, channel or sampled color");
                check(std::memcmp(allocation.get(mem),guest_before.data(),total+64)==0,"Cube upload changed guest storage or guards");
                cache.cache_and_bind_texture(cube,mem);
                check(renderer::metal::current_texture(cache)==texture,"Unchanged cube unnecessarily recreated its GPU storage");
                // Alter only the last face's last uploaded mip. Neither the
                // first face nor its base mip changes, so a base-only hash misses it.
                const size_t changed_offset=32+(faces-1)*face_stride+(mipmaps?(bc1?8192:1024):0);
                const size_t changed_size=bc1?(mipmaps?2048:8192):(mipmaps?256:1024);
                if(bc1) {
                    const uint16_t white=0xffff;
                    for(size_t block=0;block<changed_size;block+=8) {
                        auto *pixel=allocation.get(mem)+changed_offset+block;
                        std::memcpy(pixel,&white,2);std::memcpy(pixel+2,&white,2);std::memset(pixel+4,0,4);
                    }
                } else std::memset(allocation.get(mem)+changed_offset,255,changed_size);
                cache.cache_and_bind_texture(cube,mem);
                auto changed=renderer::metal::current_texture(cache);
                check(changed!=texture,"Later cube face/mip update was missed by the texture cache");
                auto updated_commands=[queue commandBuffer];auto updated_encoder=[updated_commands computeCommandEncoder];
                [updated_encoder setComputePipelineState:pipeline];[updated_encoder setTexture:changed atIndex:0];
                [updated_encoder setBuffer:output offset:0 atIndex:0];[updated_encoder setBytes:&mips length:4 atIndex:1];
                [updated_encoder dispatchThreads:MTLSizeMake(faces*mips,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [updated_encoder endEncoding];[updated_commands commit];[updated_commands waitUntilCompleted];
                check(updated_commands.status==MTLCommandBufferStatusCompleted,"Updated cube sampling failed");
                for(unsigned i=0;i<faces*mips;++i) for(unsigned c=0;c<4;++c)
                    check(std::abs(actual[i*4+c]-(i==faces*mips-1?1.0f:expected[i][c]))<tolerance,"Updated cube changed the wrong face/mip or retained stale pixels");
                std::cout<<"PASS production "<<(cube_mode?"cube":"2D")<<" upload and GPU samples: "<<(bc1?"BC1 128x128":"RGBA8 arbitrary 10x10")<<", gamma="<<gamma<<", mips="<<mips<<", face_stride="<<face_stride<<'\n';
                std::cout<<"PASS unchanged-cache reuse and isolated last-face/mip refresh\n";
                if(cube_mode || mipmaps) {
                    free(mem,allocation.address());
                    bool rejected=false;
                    try { cache.cache_and_bind_texture(cube,mem); }
                    catch(const std::runtime_error &) { rejected=true; }
                    check(rejected,"Freed mip/face storage reached the texture hash reader");
                    check(renderer::metal::current_texture(cache)==changed,"Rejected cache lookup disturbed the current GPU texture");
                    std::cout<<"PASS unmapped mip/face storage rejects before cache selection or memory reads\n";
                }
            }
            check_gamma_uploads(state,mem);
            return 0;
        } catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
