// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <shader/uniform_block.h>
#include "../../shader/tests/metal_shader_fixture.h"
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void check(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}

// Supply the user's captured 9746ee... GXP locally. No game shader is bundled.
// Keep its instructions, including both loops, fallthrough branches, scratch
// prologue and all nine spills. Extra cases change only loop-bound literals.
int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "Usage: metal-thread-buffer-validation <DOA 9746ee...gxp>\n"; return 2; }
    @autoreleasepool {
        try {
            std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
            check(bool(input), "Cannot open GXP");
            const size_t size = input.tellg();
            check(size >= sizeof(SceGxmProgram) && size < 1024*1024, "Invalid GXP size");
            std::vector<uint32_t> original((size + 3) / 4);
            input.seekg(0); check(bool(input.read(reinterpret_cast<char *>(original.data()), size)), "Cannot read GXP");
            const auto &header = *reinterpret_cast<const SceGxmProgram *>(original.data());
            check(header.magic == 0x00505847 && header.size == size && header.thread_buffer_count == 9216
                && header.primary_program_instr_count == 62, "Unexpected fixture");
            std::string error;
            auto device = renderer::metal::Device::create(error); check(bool(device), error);
            shader::metal::Program vertex{.source = R"(#include <metal_stdlib>
using namespace metal;
struct Out {
 float4 position [[position]];
 float4 weights [[user(locn4)]]; float4 coord0 [[user(locn5)]];
 float4 coord1 [[user(locn6)]]; float4 coord2 [[user(locn7)]];
};
vertex Out fullscreen(uint id [[vertex_id]]) {
 const float2 points[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
 float2 uv=points[id]*0.5+0.5;
 Out o; o.position=float4(points[id],0,1);
 o.weights=float4(0.125+uv.x*0.25,0.125+(1-uv.y)*0.25,0.0625,0);
 o.coord0=float4(0.125,0,0.375,0);
 o.coord1=float4(0.625,0,0.875,0);
 o.coord2=float4(0.875,0.125,0,0); return o;
})", .entry_point="fullscreen", .stage=shader::metal::Stage::Vertex};
            auto vs=device->compile(vertex,false,error); check(bool(vs),error);
            constexpr size_t width=64, height=48;
            auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:width height:height mipmapped:NO];
            td.storageMode=MTLStorageModeShared; td.usage=MTLTextureUsageRenderTarget;
            auto output=[device->native_device() newTextureWithDescriptor:td]; check(output!=nil,"Output texture");
            auto maskDesc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
            maskDesc.storageMode=MTLStorageModeShared;
            auto mask=[device->native_device() newTextureWithDescriptor:maskDesc];
            std::vector<uint32_t> white(width*height,0xffffffff);
            [mask replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0 withBytes:white.data() bytesPerRow:width*4];
            std::array<id<MTLTexture>,2> textures;
            std::array<std::array<float,64>,2> pixels;
            for (size_t t=0;t<2;++t) {
                auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];
                desc.storageMode=MTLStorageModeShared;
                textures[t]=[device->native_device() newTextureWithDescriptor:desc];
                for(size_t y=0;y<4;++y) for(size_t x=0;x<4;++x) {
                    auto *p=&pixels[t][(y*4+x)*4];
                    p[0]=float(x+1)/8; p[1]=float(y+1)/8; p[2]=float(t+1)/4; p[3]=1;
                }
                [textures[t] replaceRegion:MTLRegionMake2D(0,0,4,4) mipmapLevel:0 withBytes:pixels[t].data() bytesPerRow:4*16];
            }
            auto sd=[MTLSamplerDescriptor new]; sd.minFilter=MTLSamplerMinMagFilterNearest; sd.magFilter=MTLSamplerMinMagFilterNearest;
            auto sampler=[device->native_device() newSamplerStateWithDescriptor:sd];
            shader::RenderFragUniformBlockExtended info{}; info.base_block.res_multiplier=1;
            std::vector<uint8_t> infoBytes(align(info.get_size(),16)); info.copy_to(infoBytes.data());
            const std::array<std::array<unsigned,2>,6> cases={{{2,1},{0,1},{1,1},{3,1},{2,3},{0,0}}};
            size_t checked=0;
            for (const auto &bounds: cases) {
                auto words=original;
                auto &gxp=*reinterpret_cast<SceGxmProgram *>(words.data());
                const auto literalByteOffset=reinterpret_cast<const uint8_t *>(gxp.literals())-reinterpret_cast<const uint8_t *>(&gxp);
                check(literalByteOffset + gxp.literals_count*sizeof(SceGxmProgramLiteral)<=size,"Invalid literals");
                bool foundSplit=false,foundStep=false;
                auto *literals=reinterpret_cast<SceGxmProgramLiteral *>(reinterpret_cast<uint8_t *>(words.data())+literalByteOffset);
                for (size_t i=0;i<gxp.literals_count;++i) {
                    if(literals[i].offset==1) { literals[i].data=std::bit_cast<float>(bounds[0]); foundSplit=true; }
                    if(literals[i].offset==10) { literals[i].data=std::bit_cast<float>(bounds[1]); foundStep=true; }
                }
                check(foundSplit && foundStep,"Missing loop literals");
                if(bounds==cases[0]) check(words==original,"Original case was modified");
                auto fragment=compile_gxp_fixture(words,size,"doa-private-scratch","gxp-mapped");
                check(!fragment.writes_guest_memory,"Private spills classified as guest stores");
                auto fs=device->compile(fragment,false,error); check(bool(fs),error);
                auto pd=[MTLRenderPipelineDescriptor new]; pd.vertexFunction=vs->function; pd.fragmentFunction=fs->function;
                pd.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA32Float;
                auto pipeline=device->create_pipeline(pd,error); check(pipeline!=nil,error);
                auto command=[device->command_queue() commandBuffer];
                auto pass=[MTLRenderPassDescriptor renderPassDescriptor]; pass.colorAttachments[0].texture=output;
                pass.colorAttachments[0].loadAction=MTLLoadActionClear; pass.colorAttachments[0].clearColor=MTLClearColorMake(-1,-1,-1,-1);
                pass.colorAttachments[0].storeAction=MTLStoreActionStore;
                auto enc=[command renderCommandEncoderWithDescriptor:pass]; [enc setRenderPipelineState:pipeline];
                [enc setFragmentBytes:infoBytes.data() length:infoBytes.size() atIndex:shader::metal::RENDER_INFO_BUFFER];
                [enc setFragmentTexture:mask atIndex:shader::metal::MASK_TEXTURE];
                for(size_t t=0;t<2;++t) { [enc setFragmentTexture:textures[t] atIndex:t]; [enc setFragmentSamplerState:sampler atIndex:t]; }
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [enc endEncoding];
                check(device->submit_and_wait(command,error),error);
                std::vector<float> actual(width*height*4);
                [output getBytes:actual.data() bytesPerRow:width*16 fromRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0];
                constexpr size_t samplePixel[]={4,14,3};
                for(size_t y=0;y<height;++y) for(size_t x=0;x<width;++x) {
                    const float weights[]={0.125f+float(x+0.5)/width*0.25f,0.125f+float(y+0.5)/height*0.25f,0.0625f};
                    for(size_t c=0;c<4;++c) {
                        float expected=c==3 ? 1 : 0;
                        if(c<3) for(unsigned t=0;t<2;++t) {
                            const auto end=t==0 ? bounds[0] : bounds[1];
                            for(unsigned i=t==0 ? 0 : bounds[0];i<end;i=i*bounds[1]+1) {
                                check(i<3,"Reference exceeded spill slots");
                                expected+=weights[i]*pixels[t][samplePixel[i]*4+c];
                            }
                        }
                        const auto got=actual[(y*width+x)*4+c];
                        check(std::isfinite(got) && std::abs(got-expected)<0.002,
                            "Mismatch split="+std::to_string(bounds[0])+" step="+std::to_string(bounds[1])+" x="+std::to_string(x)+" y="+std::to_string(y)+" c="+std::to_string(c)+" got="+std::to_string(got)+" expected="+std::to_string(expected));
                        ++checked;
                    }
                }
                std::cout<<"PASS scratch + loop result split="<<bounds[0]<<" step="<<bounds[1]<<" across "<<width*height<<" fragments\n";
            }
            std::cout<<"RESULT passed, checked "<<checked<<" channels\n"; return 0;
        } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
    }
}
