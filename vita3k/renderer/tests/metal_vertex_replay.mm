// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <shader/uniform_block.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
void check(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
std::vector<uint8_t> read(const std::filesystem::path &path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    check(bool(file),"Cannot open "+path.string());
    const auto length=file.tellg();
    check(length>0 && length<256*1024*1024,"Invalid replay input size");
    std::vector<uint8_t> data(static_cast<size_t>(length));
    file.seekg(0);check(bool(file.read(reinterpret_cast<char *>(data.data()),length)),"Cannot read replay input");
    return data;
}
}

// Diagnostic replay, not a game compatibility test: run the captured native
// vertex shader and export its interpolated TEXCOORD0 at a requested scale.
int main(int argc,char **argv) {
    if (argc!=4) { std::cerr<<"Usage: metal-vertex-replay <draw directory> <scale> <output RG32F file>\n";return 2; }
    @autoreleasepool {
        try {
            const std::filesystem::path root=argv[1];
            const double requested_scale=std::stod(argv[2]);
            check(std::isfinite(requested_scale) && requested_scale>=1 && requested_scale<=4,"Invalid replay scale");
            std::ifstream metadata(root/"draw.txt");check(bool(metadata),"Cannot open draw metadata");
            double source_scale=0;
            uint32_t width=0,height=0,index_bytes=0;
            NSUInteger count=0,instances=0;
            MTLPrimitiveType primitive=MTLPrimitiveTypeTriangle;
            MTLViewport viewport{};
            auto layout=[MTLVertexDescriptor vertexDescriptor];
            shader::RenderVertUniformBlockExtended info{};
            std::map<uint32_t,size_t> uniforms,streams;
            std::string entry,line;
            bool version_seen=false;
            while(std::getline(metadata,line)) {
                std::istringstream fields(line);std::string key;fields>>key;
                if(key=="version") { int version;fields>>version;check(version==1,"Unsupported capture version");version_seen=true; }
                else if(key=="scale") fields>>source_scale;
                else if(key=="size") fields>>width>>height;
                else if(key=="entry") fields>>entry;
                else if(key=="fragment" || key=="vertex") { std::string ignored;fields>>ignored; }
                else if(key=="output_register_size") {
                    // Captured fragment metadata does not affect this vertex-only replay.
                    uint32_t size;fields>>size;
                    check(size==SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT || size==SCE_GXM_OUTPUT_REGISTER_SIZE_64BIT,"Invalid output register size");
                }
                else if(key=="vertex_texture") throw std::runtime_error("Vertex-textured draws require texture capture; refusing an incomplete replay");
                else if(key=="buffer_count") { uint32_t n;fields>>n;check(n<=SCE_GXM_REAL_MAX_UNIFORM_BUFFER,"Invalid uniform count");info.set_buffer_count(n); }
                else if(key=="uniform" || key=="stream") {
                    uint32_t n;size_t size;fields>>n>>size;
                    check(n<(key=="uniform"?SCE_GXM_REAL_MAX_UNIFORM_BUFFER:31) && size>0 && size<256*1024*1024,"Invalid replay buffer");
                    (key=="uniform"?uniforms:streams)[n]=size;
                } else if(key=="viewport") fields>>viewport.originX>>viewport.originY>>viewport.width>>viewport.height>>viewport.znear>>viewport.zfar;
                else if(key=="attribute") {
                    uint32_t location,format,offset,buffer;fields>>location>>format>>offset>>buffer;
                    check(location<31 && buffer<31,"Invalid vertex attribute");
                    layout.attributes[location].format=static_cast<MTLVertexFormat>(format);
                    layout.attributes[location].offset=offset;layout.attributes[location].bufferIndex=buffer;
                } else if(key=="layout") {
                    uint32_t slot,stride,step,rate;fields>>slot>>stride>>step>>rate;check(slot<31,"Invalid stream slot");
                    layout.layouts[slot].stride=stride;layout.layouts[slot].stepFunction=static_cast<MTLVertexStepFunction>(step);layout.layouts[slot].stepRate=rate;
                } else if(key=="draw") {
                    uint32_t type;fields>>type>>index_bytes>>count>>instances;primitive=static_cast<MTLPrimitiveType>(type);
                } else throw std::runtime_error("Unknown draw field: "+key);
                check(!fields.fail(),"Malformed draw field: "+key);
            }
            check(version_seen && source_scale>=1 && source_scale<=4 && width && height && width<=16384 && height<=16384
                && !entry.empty() && count && instances && (index_bytes==2 || index_bytes==4),"Incomplete draw metadata");
            const double ratio=requested_scale/source_scale;
            width=uint32_t(width*ratio);height=uint32_t(height*ratio);
            check(width && height && width<=16384 && height<=16384,"Invalid scaled dimensions");
            viewport.originX*=ratio;viewport.originY*=ratio;viewport.width*=ratio;viewport.height*=ratio;
            auto base=read(root/"vertex-info.bin");check(base.size()==sizeof(info.base_block),"Render-info ABI mismatch");
            std::memcpy(&info.base_block,base.data(),base.size());
            std::string error;
            auto device=renderer::metal::Device::create(error);check(bool(device),error);
            auto source=read(root/"vertex.metal");
            shader::metal::Program vertex{.source=std::string(source.begin(),source.end()),.entry_point=entry,.stage=shader::metal::Stage::Vertex};
            auto vs=device->compile(vertex,false,error);check(bool(vs),error);
            shader::metal::Program fragment{.source=R"(#include <metal_stdlib>
using namespace metal;
struct In { float4 uv [[user(locn4)]]; };
fragment float4 coords(In in [[stage_in]]) { return float4(in.uv.xy,0,1); }
)",.entry_point="coords",.stage=shader::metal::Stage::Fragment};
            auto fs=device->compile(fragment,false,error);check(bool(fs),error);
            auto desc=[MTLRenderPipelineDescriptor new];desc.vertexFunction=vs->function;desc.fragmentFunction=fs->function;
            desc.vertexDescriptor=layout;desc.colorAttachments[0].pixelFormat=MTLPixelFormatRG32Float;
            auto pipeline=device->create_pipeline(desc,error);check(pipeline!=nil,error);
            auto td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float width:width height:height mipmapped:NO];
            td.storageMode=MTLStorageModeShared;td.usage=MTLTextureUsageRenderTarget;
            auto output=[device->native_device() newTextureWithDescriptor:td];check(output!=nil,"Cannot allocate replay output");
            auto commands=[device->command_queue() commandBuffer];
            auto pass=[MTLRenderPassDescriptor renderPassDescriptor];pass.colorAttachments[0].texture=output;
            pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
            // Uncovered pixels remain distinguishable from valid zero UV.
            pass.colorAttachments[0].clearColor=MTLClearColorMake(-1000,-1000,0,0);
            auto encoder=[commands renderCommandEncoderWithDescriptor:pass];[encoder setRenderPipelineState:pipeline];[encoder setViewport:viewport];
            for(const auto &[block,size]:uniforms) {
                check(block<info.buffer_count,"Uniform outside captured render-info layout");
                auto bytes=read(root/("uniform-"+std::to_string(block)+".bin"));check(bytes.size()==size,"Uniform size mismatch");
                auto buffer=[device->native_device() newBufferWithBytes:bytes.data() length:bytes.size() options:MTLResourceStorageModeShared];
                check(buffer!=nil,"Cannot allocate replay uniform");info.set_buffer_address(block,buffer.gpuAddress);
                [encoder useResource:buffer usage:MTLResourceUsageRead|MTLResourceUsageWrite stages:MTLRenderStageVertex];
            }
            std::vector<uint8_t> render_info((info.get_size()+15)&~size_t(15));info.copy_to(render_info.data());
            [encoder setVertexBytes:render_info.data() length:render_info.size() atIndex:0];
            for(const auto &[slot,size]:streams) {
                auto bytes=read(root/("stream-"+std::to_string(slot)+".bin"));check(bytes.size()==size,"Stream size mismatch");
                auto buffer=[device->native_device() newBufferWithBytes:bytes.data() length:bytes.size() options:MTLResourceStorageModeShared];
                check(buffer!=nil,"Cannot allocate replay stream");[encoder setVertexBuffer:buffer offset:0 atIndex:slot];
            }
            auto indices=read(root/"indices.bin");check(indices.size()/index_bytes==count && indices.size()%index_bytes==0,"Index size mismatch");
            auto index=[device->native_device() newBufferWithBytes:indices.data() length:indices.size() options:MTLResourceStorageModeShared];
            check(index!=nil,"Cannot allocate replay indices");
            [encoder drawIndexedPrimitives:primitive indexCount:count indexType:index_bytes==2?MTLIndexTypeUInt16:MTLIndexTypeUInt32 indexBuffer:index indexBufferOffset:0 instanceCount:instances];
            [encoder endEncoding];check(device->submit_and_wait(commands,error),error);
            std::vector<float> pixels(size_t(width)*height*2);
            [output getBytes:pixels.data() bytesPerRow:width*8 fromRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0];
            std::ofstream result(argv[3],std::ios::binary);result.write(reinterpret_cast<const char *>(pixels.data()),pixels.size()*4);check(bool(result),"Cannot write replay result");
            std::cout<<"Wrote interpolated TEXCOORD0: "<<width<<'x'<<height<<" RG32F, scale="<<requested_scale<<'\n';
            return 0;
        } catch(const std::exception &e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
    }
}
