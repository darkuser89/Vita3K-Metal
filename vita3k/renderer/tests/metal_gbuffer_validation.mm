// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <shader/uniform_block.h>
#include "../../shader/tests/metal_shader_fixture.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
static void check(bool c, const std::string &s) { if (!c) throw std::runtime_error(s); }
int main(int argc,char **argv) {
    if (argc!=2) { std::cerr<<"Usage: metal-gbuffer-validation <Uncharted 9845df fragment.gxp>\n"; return 2; }
    @autoreleasepool {
        try {
            std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
            check(bool(file),"Cannot open unchanged GXP");
            const auto size=file.tellg();
            check(size>0 && size<1024*1024,"Invalid GXP length");
            std::vector<uint32_t> words((size_t(size)+3)/4);
            file.seekg(0); check(bool(file.read(reinterpret_cast<char *>(words.data()),size)),"Cannot read GXP");
            auto fragment=compile_gxp_fixture(words,size,"uncharted-gbuffer-original","gxp-mapped-gbuffer");
            check(!fragment.writes_guest_memory,"Uncharted read-only G-buffer shader was classified as a guest memory writer");
            std::string error;
            auto device=renderer::metal::Device::create(error);check(bool(device),error);
            auto fs=device->compile(fragment,false,error);check(bool(fs),error);
            shader::metal::Program vertex{.source=R"(#include <metal_stdlib>
using namespace metal;
struct Out { float4 position [[position]]; float4 uv [[user(locn4)]];
float4 n [[user(locn5)]]; float4 t [[user(locn6)]]; float4 b [[user(locn7)]]; };
vertex Out fullscreen(uint i [[vertex_id]]) {
 const float2 p[]={float2(-1,-1),float2(3,-1),float2(-1,3)};
 Out o; o.position=float4(p[i],0,1); o.uv=float4(0.5,0.5,0,0);
 o.n=float4(0,0,1,0);o.t=float4(1,0,0,0);o.b=float4(0,1,0,0);return o;
})",.entry_point="fullscreen",.stage=shader::metal::Stage::Vertex};
            auto vs=device->compile(vertex,false,error);check(bool(vs),error);
            auto pipeline_desc=[MTLRenderPipelineDescriptor new];
            pipeline_desc.vertexFunction=vs->function;pipeline_desc.fragmentFunction=fs->function;
            pipeline_desc.colorAttachments[0].pixelFormat=MTLPixelFormatRG32Float;
            auto pipeline=device->create_pipeline(pipeline_desc,error);check(pipeline!=nil,error);
            auto texture=[&](MTLPixelFormat format) {
                auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:8 height:8 mipmapped:NO];
                desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
                return [device->native_device() newTextureWithDescriptor:desc];
            };
            auto output=texture(MTLPixelFormatRG32Float),color=texture(MTLPixelFormatRGBA8Unorm),normal=texture(MTLPixelFormatRGBA16Float),mask=texture(MTLPixelFormatRGBA8Unorm);
            std::array<uint32_t,64> white;white.fill(0xffffffff);
            [mask replaceRegion:MTLRegionMake2D(0,0,8,8) mipmapLevel:0 withBytes:white.data() bytesPerRow:32];
            std::array<std::array<__fp16,4>,64> normals;normals.fill({0.5,0.5,1,1});
            [normal replaceRegion:MTLRegionMake2D(0,0,8,8) mipmapLevel:0 withBytes:normals.data() bytesPerRow:64];
            const uint16_t uniforms[8]={0x3c00,0,0,0,0,0,0,0}; // log(1)=0 -> packed normal alpha zero.
            auto buffer=[device->native_device() newBufferWithBytes:uniforms length:sizeof(uniforms) options:MTLResourceStorageModeShared];
            shader::RenderFragUniformBlockExtended info{};info.base_block.res_multiplier=1;
            info.set_buffer_count(3);info.set_buffer_address(2,buffer.gpuAddress);
            std::vector<uint8_t> render_info((info.get_size()+15)&~size_t(15));info.copy_to(render_info.data());
            auto sampler_desc=[MTLSamplerDescriptor new];sampler_desc.minFilter=sampler_desc.magFilter=MTLSamplerMinMagFilterNearest;
            auto sampler=[device->native_device() newSamplerStateWithDescriptor:sampler_desc];
            for (uint32_t value : {0u,0xffffffffu,0xffbf8040u,0x40332211u,0x01020304u}) {
                std::array<uint32_t,64> colors;colors.fill(value);
                [color replaceRegion:MTLRegionMake2D(0,0,8,8) mipmapLevel:0 withBytes:colors.data() bytesPerRow:32];
                auto commands=[device->command_queue() commandBuffer];
                auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture=output;pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
                auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
                [encoder setRenderPipelineState:pipeline];[encoder setViewport:MTLViewport{0,0,8,8,0,1}];
                [encoder setFragmentBytes:render_info.data() length:render_info.size() atIndex:0];
                [encoder useResource:buffer usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
                [encoder setFragmentTexture:color atIndex:0];[encoder setFragmentTexture:normal atIndex:1];[encoder setFragmentTexture:mask atIndex:17];
                [encoder setFragmentSamplerState:sampler atIndex:0];[encoder setFragmentSamplerState:sampler atIndex:1];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[encoder endEncoding];
                check(device->submit_and_wait(commands,error),error);
                std::array<uint32_t,128> pixels;
                [output getBytes:pixels.data() bytesPerRow:64 fromRegion:MTLRegionMake2D(0,0,8,8) mipmapLevel:0];
                for (size_t i=0;i<64;++i) {
                    if (pixels[i*2]!=value || pixels[i*2+1]!=0x007f0000u) {
                        std::cerr<<std::hex<<"value="<<value<<" actual color="<<pixels[i*2]<<" normal="<<pixels[i*2+1]<<std::dec<<" pixel="<<i<<'\n';
                        throw std::runtime_error("Unchanged GXP packed color/normal differs from controlled inputs");
                    }
                }
                std::cout<<"PASS unchanged Uncharted G-buffer GXP, color=0x"<<std::hex<<value<<std::dec<<", all packed color/normal words\n";
            }
            return 0;
        } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
    }
}
