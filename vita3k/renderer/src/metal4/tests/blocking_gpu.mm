// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Explicit headless GPU integration test. Not part of cpu_regressions.py.
// Links actual renderer objects: no emulator, window, game or user cache.
// A caller-provided temporary directory isolates the persistence checks.
#include <renderer/metal4/textures.h>
#include <renderer/metal4/device.h>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <vector>
using namespace renderer::metal;
static void check(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
static id<MTLTexture> texture(Device &device,MTLPixelFormat format,uint32_t width,uint32_t height,
    MTLTextureUsage usage,uint32_t samples=1) {
    auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    descriptor.usage=usage;descriptor.storageMode=samples>1?MTLStorageModePrivate:MTLStorageModeShared;
    descriptor.textureType=samples>1?MTLTextureType2DMultisample:MTLTextureType2D;
    descriptor.sampleCount=samples;
    auto result=[device.native_device() newTextureWithDescriptor:descriptor];check(bool(result),"texture allocation");return result;
}
static void complete(Device &device,V3KMetal4Commands *commands) {
    std::string error;check(device.submit_and_wait(commands,error),error);
}
static void seeds(Device &device) {
    SurfaceCaster caster(device);
    for(uint32_t samples:{2u,4u}) {
        check([device.native_device() supportsTextureSampleCount:samples],"MSAA count unsupported");
        auto source=texture(device,MTLPixelFormatRGBA8Unorm,8,4,MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView);
        std::array<uint32_t,32> input{};
        for(uint32_t i=0;i<input.size();++i)input[i]=0xff000000u|((i*103u)&0xffffffu);
        [source replaceRegion:MTLRegionMake2D(0,0,8,4) mipmapLevel:0 withBytes:input.data() bytesPerRow:32];
        auto multisample=texture(device,source.pixelFormat,8,4,
            MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView,samples);
        auto output=texture(device,source.pixelFormat,8*(samples/2),8,
            MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView);
        auto commands=[device.command_queue() commandBuffer];
        caster.seed_multisample(source,multisample,1,false,0,0,commands);
        caster.expand_multisample(multisample,output,1,0,0,commands);
        check(commands.status==MTLCommandBufferStatusNotEnqueued,"seed/expand submitted outside consumer scene");
        complete(device,commands);
        std::vector<uint32_t> pixels(output.width*output.height);
        [output getBytes:pixels.data() bytesPerRow:output.width*4 fromRegion:MTLRegionMake2D(0,0,output.width,output.height) mipmapLevel:0];
        for(uint32_t y=0;y<output.height;++y)for(uint32_t x=0;x<output.width;++x)
            check(pixels[y*output.width+x]==input[(y/2)*8+x/(samples/2)],"MSAA sample seed/expand bytes changed");
    }
    std::cout<<"PASS native MSAA: actual SurfaceCaster lazy library, 2x/4x seed -> expand in one unsubmitted scene; all expanded RGBA8 samples byte-exact\n";
}
static void raw_cube(Device &device) {
    SurfaceCaster caster(device);
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float width:7 height:7 mipmapped:YES];
    desc.mipmapLevelCount=3;desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
    auto source=[device.native_device() newTextureWithDescriptor:desc];check(bool(source),"raw source allocation");
    std::array<std::vector<uint32_t>,3> inputs;
    for(unsigned mip=0;mip<3;++mip){
        const uint32_t size=std::max(1u,7u>>mip);auto &words=inputs[mip];words.resize(size*size*2);
        for(size_t i=0;i<words.size();++i)words[i]=std::array<uint32_t,4>{0x7fc12345u,0xffc54321u,0x80000000u,0x3f800000u}[i%4]^uint32_t(mip<<8);
        [source replaceRegion:MTLRegionMake2D(0,0,size,size) mipmapLevel:mip withBytes:words.data() bytesPerRow:size*8];
    }
    auto cube_desc=[MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:source.pixelFormat size:7 mipmapped:YES];
    cube_desc.mipmapLevelCount=3;cube_desc.storageMode=MTLStorageModePrivate;cube_desc.usage=desc.usage;
    auto cube=[device.native_device() newTextureWithDescriptor:cube_desc];check(bool(cube),"private cube allocation");
    auto commands=[device.command_queue() commandBuffer];auto copy=[commands blitCommandEncoder];check(bool(copy),"cube blit");
    for(unsigned face=0;face<6;++face)for(unsigned mip=0;mip<3;++mip){
        const uint32_t size=std::max(1u,7u>>mip);
        [copy copyFromTexture:source sourceSlice:0 sourceLevel:mip sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(size,size,1)
            toTexture:cube destinationSlice:face destinationLevel:mip destinationOrigin:MTLOriginMake(0,0,0)];
    }
    [copy endEncoding];auto raw=caster.raw_texture_snapshot(cube,commands);check(bool(raw),"raw cube carrier");
    check(raw.pixelFormat==MTLPixelFormatRGBA16Unorm,"raw carrier format");
    auto output=[device.native_device() newBufferWithLength:6*3*256*8 options:MTLResourceStorageModeShared];check(bool(output),"raw readback allocation");
    auto readback=[commands blitCommandEncoder];
    for(unsigned face=0;face<6;++face)for(unsigned mip=0;mip<3;++mip){
        const uint32_t size=std::max(1u,7u>>mip);
        [readback copyFromTexture:raw sourceSlice:face sourceLevel:mip sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(size,size,1)
            toBuffer:output destinationOffset:(face*3+mip)*256*8 destinationBytesPerRow:256 destinationBytesPerImage:256*size];
    }
    [readback endEncoding];check(commands.status==MTLCommandBufferStatusNotEnqueued,"raw conversion privately submitted");
    complete(device,commands);
    auto bytes=static_cast<const uint8_t *>(output.contents);
    for(unsigned face=0;face<6;++face)for(unsigned mip=0;mip<3;++mip){
        const uint32_t size=std::max(1u,7u>>mip);
        for(unsigned y=0;y<size;++y)check(!std::memcmp(bytes+(face*3+mip)*256*8+y*256,inputs[mip].data()+y*size*2,size*8),"raw NaN/sign bits or cube mip changed");
    }
    std::cout<<"PASS native raw cube: actual copy -> SurfaceCaster carrier -> buffer in one submission; six faces/three NPOT mips, RG32 NaN payloads and signed zero byte-exact\n";
}
static void depth(Device &device) {
    SurfaceCaster caster(device);
    for(uint32_t samples:{1u,2u,4u})for(bool normalized:{false,true}) {
        auto source=texture(device,MTLPixelFormatDepth32Float_Stencil8,4,2,
            MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView,samples);
        auto producer=[device.command_queue() commandBuffer];auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
        pass.defaultRasterSampleCount=samples;pass.depthAttachment.texture=source;
        pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionStore;pass.depthAttachment.clearDepth=.25;
        auto clear=[producer renderCommandEncoderWithDescriptor:pass];check(bool(clear),"depth clear encoder");[clear endEncoding];[producer commit];
        auto snapshot=caster.depth_snapshot(source,normalized,1,0,0,false);
        auto consumer=[device.command_queue() commandBuffer];auto output=[device.native_device() newBufferWithLength:256*snapshot.height options:MTLResourceStorageModeShared];
        auto blit=[consumer blitCommandEncoder];
        [blit copyFromTexture:snapshot sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(snapshot.width,snapshot.height,1)
            toBuffer:output destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:256*snapshot.height];
        [blit endEncoding];complete(device,consumer);[producer waitUntilCompleted];
        check(producer.status==MTLCommandBufferStatusCompleted,"depth producer failed");
        for(unsigned y=0;y<snapshot.height;++y)for(unsigned x=0;x<snapshot.width;++x){
            const auto row=static_cast<const uint8_t *>(output.contents)+y*256;
            if(normalized){uint16_t value;std::memcpy(&value,row+x*2,2);check(std::abs(int(value)-16384)<=1,"normalized depth value");}
            else{float value;std::memcpy(&value,row+x*4,4);check(value==.25f,"float depth value");}
        }
    }
    std::cout<<"PASS native depth snapshots: actual queued producer -> asynchronous 1x/2x/4x snapshot -> consumer; R16/R32 expanded sample values correct\n";
}
static void cache(const std::filesystem::path &directory) {
    const std::string vertex_hash(64,'a'),fragment_hash(64,'b');
    const auto vertex_key=vertex_hash+";native-v",fragment_key=fragment_hash+";native-f";
    const std::string source="#include <metal_stdlib>\nusing namespace metal;\n"
        "vertex float4 cached_vs(uint id [[vertex_id]]) {return float4(id==1?3:-1,id==2?3:-1,0,1);}\n"
        "fragment float4 cached_fs() {return float4(.25,.5,.75,1);}\n";
    shader::metal::Program vertex{source,"cached_vs",shader::metal::Stage::Vertex};
    shader::metal::Program fragment{source,"cached_fs",shader::metal::Stage::Fragment};
    std::string error;
    {
        auto device=Device::create(error);check(bool(device),error);device->configure_cache(directory);
        device->store_cached_program(vertex_key,vertex,vertex_hash);
        device->store_cached_program(fragment_key,fragment,fragment_hash);
        device->flush_cache();std::latch ready(2);
        auto compile=[&](const std::string &key){return std::async(std::launch::async,[&,key]{@autoreleasepool{
            ready.count_down();ready.wait();auto program=device->load_cached_program(key);check(bool(program),"cached MSL missing");
            std::string failure;auto compiled=device->compile(*program,false,failure);check(bool(compiled),failure);return compiled;
        }});};
        auto v=compile(vertex_key),f=compile(fragment_key);auto vs=v.get(),fs=f.get();
        struct PipelineResult{id<MTLRenderPipelineState> pipeline;std::string error;};
        std::array<std::future<PipelineResult>,2> jobs;std::latch pipelines_ready(2);
        for(unsigned i=0;i<jobs.size();++i)jobs[i]=std::async(std::launch::async,[&,i]{@autoreleasepool{
            auto descriptor=[MTLRenderPipelineDescriptor new];descriptor.vertexFunction=vs->function;descriptor.fragmentFunction=fs->function;
            descriptor.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
            descriptor.colorAttachments[0].writeMask=i?MTLColorWriteMaskRed:MTLColorWriteMaskAll;
            pipelines_ready.count_down();pipelines_ready.wait();PipelineResult result;std::string descriptor_key;
            result.pipeline=device->create_pipeline(descriptor,result.error,&descriptor_key);
            check(bool(result.pipeline),result.error);
            device->store_cached_pipeline_template(fragment_hash,vertex_hash,vertex_key+fragment_key+std::to_string(i),vertex_key,fragment_key,descriptor,descriptor_key);
            return result;
        }});
        for(auto &job:jobs)check(bool(job.get().pipeline),"parallel native pipeline");
        device->flush_cache();const auto stats=device->cache_stats();
        check(stats.program_writes==2&&stats.archive_writes==2&&!stats.io_errors,"cache publication incomplete");
        check(device->cached_variants(vertex_hash).size()==1&&device->cached_variants(fragment_hash).size()==1,"variant index mismatch");
        check(device->cached_pipeline_templates(fragment_hash,vertex_hash).size()==2,"pipeline index mismatch");
    }
    {
        auto device=Device::create(error);check(bool(device),error);device->configure_cache(directory);
        auto vertex=device->load_cached_program(vertex_key),fragment=device->load_cached_program(fragment_key);
        check(vertex&&fragment,"persisted MSL not restored");
        auto vs=device->compile(*vertex,false,error);check(bool(vs),error);
        auto fs=device->compile(*fragment,false,error);check(bool(fs),error);
        auto saved=device->cached_pipeline_templates(fragment_hash,vertex_hash);check(saved.size()==2,"persisted templates not restored");
        for(auto &pipeline:saved){
            check(pipeline.vertex_function==vs->function.name.UTF8String&&pipeline.fragment_function==fs->function.name.UTF8String,"persisted function identity");
            pipeline.descriptor.vertexFunction=vs->function;pipeline.descriptor.fragmentFunction=fs->function;
            check(bool(device->create_pipeline(pipeline.descriptor,error)),error);
        }
        const auto stats=device->cache_stats();
        check(stats.program_disk_hits==2&&stats.pipeline_disk_hits==2&&stats.archive_loaded&&!stats.rejected_files&&!stats.io_errors,"native disk cache not reused");
    }
    std::cout<<"PASS native cache: two concurrent real shader/pipeline jobs; two MSL records, variants, templates and archives published; new Device restores both shaders/templates and gets two native archive disk hits; temporary cache only\n";
}
int main(int argc,char **argv){@autoreleasepool{
    try{
        check(argc==2,"usage: blocking_gpu <temporary-cache-directory>");
        std::string error;auto device=Device::create(error);check(bool(device),error);
        seeds(*device);raw_cube(*device);depth(*device);
        cache(argv[1]);
        std::cout<<"PASS headless native blocker integration: production renderer/library objects, real Metal 4; no emulator, window, game or user cache\n";
    }catch(const std::exception &error){std::cerr<<"FAIL native blocker integration: "<<error.what()<<'\n';return 1;}
}}
