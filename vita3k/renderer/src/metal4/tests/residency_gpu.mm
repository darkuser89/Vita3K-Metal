// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Explicit GPU test; never part of the CPU-only regression entrypoint.
#include <renderer/metal4/metal4_commands.h>
#include <renderer/metal4/metal4_pipeline.h>
#include <array>
#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>

int main() { @autoreleasepool {
    using namespace renderer::metal;
    auto device=MTLCreateSystemDefaultDevice();assert(device);
    NSError *error=nil;
    auto compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];assert(compiler);
    auto library_desc=[MTL4LibraryDescriptor new];
    library_desc.source=@"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void sample_view(texture2d<float,access::read> image [[texture(0)]],"
        "device float4 *result [[buffer(0)]]){ result[0]=image.read(uint2(0)); }";
    library_desc.options=[MTLCompileOptions new];
    library_desc.options.languageVersion=MTLLanguageVersion3_0;
    auto library=[compiler newLibraryWithDescriptor:library_desc error:&error];assert(library);
    auto pipeline=metal4_compute_pipeline(compiler,metal4_function(library,@"sample_view"),nil,&error);assert(pipeline);
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;
    desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
    auto texture=[device newTextureWithDescriptor:desc];assert(texture);
    std::array<uint8_t,64> pixels;
    for(size_t i=0;i<pixels.size();++i)pixels[i]=std::array<uint8_t,4>{64,128,192,255}[i%4];
    [texture replaceRegion:MTLRegionMake2D(0,0,4,4) mipmapLevel:0 withBytes:pixels.data() bytesPerRow:16];
    const MTLTextureSwizzleChannels swizzle{MTLTextureSwizzleBlue,MTLTextureSwizzleGreen,MTLTextureSwizzleRed,MTLTextureSwizzleAlpha};
    auto view=[texture newTextureViewWithPixelFormat:texture.pixelFormat textureType:MTLTextureType2D
        levels:NSMakeRange(0,1) slices:NSMakeRange(0,1) swizzle:swizzle];assert(view);
    auto queue=[[V3KMetal4Queue alloc] initWithDevice:device];assert(queue);
    struct Pending { V3KMetal4Commands *commands;id<MTLBuffer> output;bool view; };
    std::deque<Pending> pending;
    const auto retire=[&] {
        auto p=pending.front();pending.pop_front();[p.commands waitUntilCompleted];
        assert(p.commands.status==MTLCommandBufferStatusCompleted && !p.commands.error);
        const auto *actual=static_cast<const float *>(p.output.contents);
        const std::array<float,4> expected=p.view ? std::array<float,4>{192/255.f,128/255.f,64/255.f,1}
            : std::array<float,4>{64/255.f,128/255.f,192/255.f,1};
        for(size_t i=0;i<4;++i)assert(std::abs(actual[i]-expected[i])<0.00001f);
    };
    for(size_t i=0;i<100;++i) @autoreleasepool {
        auto commands=[queue commandBuffer];
        auto output=[device newBufferWithLength:16 options:MTLResourceStorageModeShared];assert(output);
        auto encoder=[commands computeCommandEncoder];assert(encoder);
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:i%2 ? view : texture atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];[commands commit];
        pending.push_back({commands,output,bool(i%2)});
        if(pending.size()==3)retire();
    }
    while(!pending.empty())retire();
    std::cout<<"PASS native Metal 4 texture residency: 100 completed submissions, three pending users, root/view swizzle reads and exact CPU results; no emulator/app window\n";
} }
