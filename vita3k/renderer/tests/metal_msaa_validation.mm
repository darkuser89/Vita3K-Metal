// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

// Verify the native storage contract needed by GXM MSAA before enabling it in
// the renderer. Native sample IDs here do not assert Vita sample memory order.
int main() {
    @autoreleasepool {
        try {
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error.c_str());
            NSError *native_error = nil;
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion3_0;
            auto library = [device->native_device() newLibraryWithSource:@R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 vs(uint id [[vertex_id]]) {
    const float2 p[] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
    return float4(p[id],0.5,1);
}
struct Output { float4 color [[color(0)]]; float depth [[depth(any)]]; };
fragment Output seed(float4 p [[position]], uint s [[sample_id]]) {
    return {float4(float(s+1)/8, float(uint(p.x)%4)/16, float(uint(p.y)%4)/8, 1), float(s+1)/8};
}
// No sample_id: framebuffer fetch itself must preserve the sample identity.
fragment float4 fetch(float4 previous [[color(0)]]) {
    return previous + float4(0.125,0,0.125,0);
}
struct MaskOutput { float4 color [[color(0)]]; float depth [[depth(any)]]; uint mask [[sample_mask]]; };
fragment MaskOutput masked(float4 p [[position]]) {
    return {float4(0.75,0.25,0.125,1), 0.375, (uint(p.x)+uint(p.y))%2 ? 0x5u : 0xau};
}
fragment Output depth_test() { return {float4(0.25,0.5,0.75,1), 0.4375}; }
fragment float4 stencil_test() { return float4(0.125,0.75,0.25,1); }
kernel void read_samples(texture2d_ms<float,access::read> color [[texture(0)]],
    depth2d_ms<float,access::read> depth [[texture(1)]], device float4 *out [[buffer(0)]],
    uint3 p [[thread_position_in_grid]]) {
    if (p.x>=color.get_width() || p.y>=color.get_height() || p.z>=color.get_num_samples()) return;
    const uint index = 16 + ((p.y*color.get_width()+p.x)*color.get_num_samples()+p.z)*2;
    out[index] = color.read(p.xy,p.z);
    out[index+1] = float4(depth.read(p.xy,p.z),0,0,0);
}
kernel void read_resolve(texture2d<float,access::read> color [[texture(0)]],
    device float4 *out [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x<color.get_width() && p.y<color.get_height()) out[16+p.y*color.get_width()+p.x]=color.read(p);
}
)" options:options error:&native_error];
            check(library != nil, native_error.localizedDescription.UTF8String ?: "Cannot compile MSAA probe");
            auto read_samples = [device->native_device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"read_samples"] error:&native_error];
            check(read_samples != nil, native_error.localizedDescription.UTF8String ?: "Cannot create sample reader");
            auto read_resolve = [device->native_device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"read_resolve"] error:&native_error];
            check(read_resolve != nil, native_error.localizedDescription.UTF8String ?: "Cannot create resolve reader");
            constexpr uint32_t width = 12, height = 6;
            renderer::metal::SurfaceCaster caster(*device);
            unsigned depth_copies = 0, color_copies = 0;
            using Pixel = std::array<float,4>;
            const Pixel guard = {-123,-123,-123,-123};
            unsigned passed = 0;
            for (uint32_t samples : {2u,4u}) {
                check([device->native_device() supportsTextureSampleCount:samples], "Required MSAA count is unavailable");
                for (bool srgb : {false,true}) for (bool resolve : {false,true}) {
                    const auto format = srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA16Float;
                    auto allocate = [&](MTLPixelFormat pixel_format, uint32_t count) {
                        auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format width:width height:height mipmapped:NO];
                        desc.storageMode = MTLStorageModePrivate;
                        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
                        desc.sampleCount = count;
                        if (count > 1) desc.textureType = MTLTextureType2DMultisample;
                        auto result = [device->native_device() newTextureWithDescriptor:desc];
                        check(result != nil, "Cannot allocate MSAA fixture texture");
                        return result;
                    };
                    auto color = allocate(format,samples);
                    auto depth = allocate(MTLPixelFormatDepth32Float_Stencil8,samples);
                    auto resolved = resolve ? allocate(format,1) : nil;
                    std::array<id<MTLRenderPipelineState>,5> pipelines;
                    NSArray<NSString *> *names = @[@"seed",@"fetch",@"masked",@"depth_test",@"stencil_test"];
                    for (size_t i=0;i<pipelines.size();++i) {
                        auto desc = [MTLRenderPipelineDescriptor new];
                        desc.vertexFunction = [library newFunctionWithName:@"vs"];
                        desc.fragmentFunction = [library newFunctionWithName:names[i]];
                        desc.rasterSampleCount = samples;
                        desc.colorAttachments[0].pixelFormat = format;
                        desc.depthAttachmentPixelFormat = desc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
                        pipelines[i] = [device->native_device() newRenderPipelineStateWithDescriptor:desc error:&native_error];
                        check(pipelines[i] != nil, native_error.localizedDescription.UTF8String ?: "Cannot create MSAA render pipeline");
                    }
                    std::vector<Pixel> expected(size_t(width)*height*samples);
                    std::vector<float> expected_depth(expected.size(),1);
                    std::vector<uint32_t> expected_stencil(expected.size(),0x5c);
                    for (unsigned stage=0;stage<6;++stage) {
                        auto commands = [device->command_queue() commandBuffer];
                        auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                        pass.colorAttachments[0].texture = color;
                        pass.colorAttachments[0].loadAction = stage ? MTLLoadActionLoad : MTLLoadActionClear;
                        pass.colorAttachments[0].storeAction = resolve ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore;
                        pass.colorAttachments[0].resolveTexture = resolved;
                        pass.depthAttachment.texture = pass.stencilAttachment.texture = depth;
                        pass.depthAttachment.loadAction = pass.stencilAttachment.loadAction = stage ? MTLLoadActionLoad : MTLLoadActionClear;
                        pass.depthAttachment.storeAction = pass.stencilAttachment.storeAction = MTLStoreActionStore;
                        pass.depthAttachment.clearDepth = 1;
                        pass.stencilAttachment.clearStencil = 0x5c;
                        auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
                        check(encoder != nil, "Cannot create MSAA render pass");
                        if (stage != 1) { // Also exercise an empty load/store pass.
                            const unsigned pipeline_index = stage ? stage-1 : 0;
                            [encoder setRenderPipelineState:pipelines[pipeline_index]];
                            auto ds = [MTLDepthStencilDescriptor new];
                            ds.depthCompareFunction = stage == 4 ? MTLCompareFunctionLess : MTLCompareFunctionAlways;
                            ds.depthWriteEnabled = stage == 0 || stage == 3 || stage == 4;
                            auto stencil = [MTLStencilDescriptor new];
                            stencil.stencilCompareFunction = stage == 3 ? MTLCompareFunctionAlways : MTLCompareFunctionEqual;
                            stencil.depthStencilPassOperation = stage == 3 ? MTLStencilOperationReplace : MTLStencilOperationKeep;
                            ds.frontFaceStencil = ds.backFaceStencil = stencil;
                            // Stage 4 tests depth independently of the prior sample mask.
                            if (stage == 4) stencil.stencilCompareFunction = MTLCompareFunctionAlways;
                            [encoder setDepthStencilState:[device->native_device() newDepthStencilStateWithDescriptor:ds]];
                            [encoder setStencilReferenceValue:stage == 3 || stage == 5 ? 0x37 : 0x5c];
                            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        }
                        [encoder endEncoding];
                        std::vector<Pixel> storage(expected.size()*2+32,guard);
                        auto output = [device->native_device() newBufferWithBytes:storage.data() length:storage.size()*sizeof(Pixel) options:MTLResourceStorageModeShared];
                        check(output != nil, "Cannot allocate sample readback");
                        auto compute = [commands computeCommandEncoder];
                        [compute setComputePipelineState:read_samples];
                        [compute setTexture:color atIndex:0]; [compute setTexture:depth atIndex:1];
                        [compute setBuffer:output offset:0 atIndex:0];
                        [compute dispatchThreads:MTLSizeMake(width,height,samples) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
                        [compute endEncoding];
                        std::vector<Pixel> resolve_storage(size_t(width)*height+32,guard);
                        id<MTLBuffer> resolve_output = nil;
                        if (resolve) {
                            resolve_output = [device->native_device() newBufferWithBytes:resolve_storage.data() length:resolve_storage.size()*sizeof(Pixel) options:MTLResourceStorageModeShared];
                            check(resolve_output != nil, "Cannot allocate resolve readback");
                            compute = [commands computeCommandEncoder];
                            [compute setComputePipelineState:read_resolve]; [compute setTexture:resolved atIndex:0];
                            [compute setBuffer:resolve_output offset:0 atIndex:0];
                            [compute dispatchThreads:MTLSizeMake(width,height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
                            [compute endEncoding];
                        }
                        check(device->submit_and_wait(commands,error),error.c_str());
                        const auto *actual = static_cast<const Pixel *>(output.contents);
                        const auto *resolved_pixels = static_cast<const Pixel *>(resolve_output.contents);
                        auto close = [&](float value, float wanted, float tolerance, const char *what) {
                            if (!std::isfinite(value) || std::abs(value-wanted)>tolerance)
                                throw std::runtime_error(std::string(what)+" stage="+std::to_string(stage)+" samples="+std::to_string(samples)
                                    +" srgb="+std::to_string(srgb)+" resolve="+std::to_string(resolve)
                                    +" actual="+std::to_string(value)+" expected="+std::to_string(wanted));
                        };
                        for (uint32_t y=0;y<height;++y) for (uint32_t x=0;x<width;++x) {
                            Pixel average{};
                            for (uint32_t s=0;s<samples;++s) {
                                const size_t i=(size_t(y)*width+x)*samples+s;
                                if (stage==0) {
                                    expected[i]={float(s+1)/8,float(x%4)/16,float(y%4)/8,1};
                                    expected_depth[i]=float(s+1)/8;
                                } else if (stage==2) {
                                    expected[i][0]+=0.125f; expected[i][2]+=0.125f;
                                } else if (stage==3 && (((x+y)%2 ? 0x5u : 0xau)&(1u<<s))) {
                                    expected[i]={0.75,0.25,0.125,1}; expected_depth[i]=0.375;
                                    expected_stencil[i]=0x37;
                                } else if (stage==4 && 0.4375f<expected_depth[i]) {
                                    expected[i]={0.25,0.5,0.75,1}; expected_depth[i]=0.4375;
                                } else if (stage==5 && expected_stencil[i]==0x37) {
                                    expected[i]={0.125,0.75,0.25,1};
                                }
                                for (size_t c=0;c<4;++c) {
                                    close(actual[16+i*2][c],expected[i][c],srgb ? 0.009f : 0.00001f,"Sample color mismatch");
                                    average[c]+=actual[16+i*2][c]/samples;
                                }
                                close(actual[17+i*2][0],expected_depth[i],0.000001f,"Sample depth mismatch");
                                for (size_t c=1;c<4;++c) close(actual[17+i*2][c],0,0,"Readback padding mismatch");
                            }
                            if (resolve) for (size_t c=0;c<4;++c)
                                close(resolved_pixels[16+y*width+x][c],average[c],srgb ? 0.006f : 0.00001f,"Resolve average mismatch");
                        }
                        for (size_t i=0;i<16;++i) {
                            check(actual[i]==guard && actual[storage.size()-1-i]==guard,"Sample readback guard changed");
                            if (resolve) check(resolved_pixels[i]==guard && resolved_pixels[resolve_storage.size()-1-i]==guard,"Resolve readback guard changed");
                        }
                        // Validate production packing against independently known
                        // native sample values, including every higher-resolution subpixel.
                        for (uint32_t scale : {1u,2u,3u}) for (bool normalized : {false,true}) {
                            auto snapshot=caster.depth_snapshot(depth,normalized,scale);
                            const uint32_t packed_width=width*(samples/2), packed_height=height*2;
                            std::vector<float> floats(size_t(packed_width)*packed_height);
                            std::vector<uint16_t> unorm(floats.size());
                            [snapshot getBytes:normalized ? static_cast<void *>(unorm.data()) : floats.data()
                                bytesPerRow:packed_width*(normalized ? 2 : 4) fromRegion:MTLRegionMake2D(0,0,packed_width,packed_height) mipmapLevel:0];
                            for(uint32_t y=0;y<height;++y) for(uint32_t x=0;x<width;++x) for(uint32_t sample=0;sample<samples;++sample) {
                                const uint32_t out_x=(x/scale*(samples/2)+sample%(samples/2))*scale+x%scale;
                                const uint32_t out_y=(y/scale*2+sample/(samples/2))*scale+y%scale;
                                const size_t i=(size_t(y)*width+x)*samples+sample, out=size_t(out_y)*packed_width+out_x;
                                if(normalized) check(std::abs(int(unorm[out])-int(std::lround(expected_depth[i]*65535)))<=1,"Production D16 sample packing mismatch");
                                else close(floats[out],expected_depth[i],0.000001f,"Production F32 sample packing mismatch");
                            }
                            // Crop in guest sample coordinates, deliberately starting
                            // inside a sample row/column rather than a logical pixel.
                            const uint32_t guest_w=packed_width/scale,guest_h=packed_height/scale;
                            const renderer::metal::SurfaceRect rect{1,1,guest_w-2,guest_h-2};
                            auto crop=caster.snapshot_subrectangle(snapshot,guest_w,guest_h,rect);
                            check(crop.width==rect.width*scale && crop.height==rect.height*scale,"Depth crop lost resolution scale");
                            const size_t stride=crop.width*(normalized?2:4);
                            std::vector<uint8_t> cropped(stride*crop.height);
                            [crop getBytes:cropped.data() bytesPerRow:stride fromRegion:MTLRegionMake2D(0,0,crop.width,crop.height) mipmapLevel:0];
                            const auto *source_bytes=normalized?reinterpret_cast<const uint8_t *>(unorm.data()):reinterpret_cast<const uint8_t *>(floats.data());
                            for(uint32_t y=0;y<crop.height;++y) {
                                const auto *row=source_bytes+((size_t(y)+rect.y*scale)*packed_width+rect.x*scale)*(normalized?2:4);
                                check(std::equal(row,row+stride,cropped.data()+size_t(y)*stride),"Depth crop changed individual sample bytes");
                            }
                            ++depth_copies;
                        }
                        if(stage==2) for(uint32_t scale : {1u,2u,3u}) {
                            auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width*(samples/2) height:height*2 mipmapped:NO];
                            desc.storageMode=MTLStorageModeShared;
                            desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;
                            auto packed=[device->native_device() newTextureWithDescriptor:desc];
                            check(packed!=nil,"Cannot allocate production packed samples");
                            caster.expand_multisample(color,packed,scale);
                            auto sampled=caster.sampling_snapshot(packed);
                            check(sampled!=nil,"Cannot sample expanded production color");
                            std::vector<Pixel> values(sampled.width*sampled.height);
                            [sampled getBytes:values.data() bytesPerRow:sampled.width*sizeof(Pixel) fromRegion:MTLRegionMake2D(0,0,sampled.width,sampled.height) mipmapLevel:0];
                            for(uint32_t y=0;y<height;++y) for(uint32_t x=0;x<width;++x) for(uint32_t sample=0;sample<samples;++sample) {
                                const uint32_t out_x=(x/scale*(samples/2)+sample%(samples/2))*scale+x%scale;
                                const uint32_t out_y=(y/scale*2+sample/(samples/2))*scale+y%scale;
                                const size_t i=(size_t(y)*width+x)*samples+sample;
                                for(unsigned c=0;c<4;++c) close(values[out_y*sampled.width+out_x][c],actual[16+i*2][c],srgb ? 0.009f : 0.00001f,"Production color sample packing mismatch");
                            }
                            auto restored=allocate(format,samples);
                            caster.seed_multisample(packed,restored,scale,true);
                            auto read_commands=[device->command_queue() commandBuffer]; auto reader=[read_commands computeCommandEncoder];
                            auto buffer=[device->native_device() newBufferWithBytes:storage.data() length:storage.size()*sizeof(Pixel) options:MTLResourceStorageModeShared];
                            [reader setComputePipelineState:read_samples]; [reader setTexture:restored atIndex:0]; [reader setTexture:depth atIndex:1];
                            [reader setBuffer:buffer offset:0 atIndex:0];
                            [reader dispatchThreads:MTLSizeMake(width,height,samples) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
                            [reader endEncoding]; check(device->submit_and_wait(read_commands,error),error.c_str());
                            const auto *restored_values=static_cast<const Pixel *>(buffer.contents);
                            for(size_t i=0;i<expected.size();++i) for(unsigned c=0;c<4;++c)
                                close(restored_values[16+i*2][c],actual[16+i*2][c],srgb ? 0.009f : 0.00001f,"Production sample reseed mismatch");
                            ++color_copies;
                        }
                        ++passed;
                        std::cout << "PASS stage=" << stage << " samples=" << samples << " srgb=" << srgb << " resolve=" << resolve
                            << " all sample colors/depths, resolved pixels and guards\n";
                    }
                }
            }
            std::cout << "PASS " << passed << " native MSAA cases, " << depth_copies << " production depth copies and exact sample crops, " << color_copies << " color packing/reseed cases; Vita sample ordering still needs a hardware reference\n";
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
