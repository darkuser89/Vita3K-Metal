// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <renderer/metal/textures.h>
#include <array>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error.c_str());
            NSString *source = @R"(#include <metal_stdlib>
using namespace metal;
kernel void plane(texture2d<float> input [[texture(0)]], device float4 *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = input.read(uint2(0), i);
}
kernel void pixels(texture2d<float> input [[texture(0)]], device float4 *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = input.read(uint2(i%input.get_width(),i/input.get_width()));
}
kernel void filtered(texture2d<float> input [[texture(0)]], sampler s [[sampler(0)]], device float4 *out [[buffer(0)]]) {
    out[0] = input.sample(s,float2(0.5),gradient2d(float2(0.5,0),float2(0,0.5)));
}
kernel void captured(texture2d<float> input [[texture(0)]], sampler s [[sampler(0)]], device float4 *out [[buffer(0)]],
    device const float2 *uv [[buffer(1)]], constant float &offset [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i] = input.sample(s,uv[i]+float2(offset,0),level(0));
}
kernel void cube(texturecube<float> input [[texture(0)]], device float4 *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(coord::normalized, filter::nearest, mip_filter::nearest);
    const float3 directions[] = {float3(1,0,0),float3(-1,0,0),float3(0,1,0),float3(0,-1,0),float3(0,0,1),float3(0,0,-1)};
    out[i] = input.sample(s, directions[i/2], level(i%2));
})";
            NSError *native_error = nil;
            auto library = [device->native_device() newLibraryWithSource:source options:nil error:&native_error];
            check(library != nil, native_error.localizedDescription.UTF8String ?: "Cannot compile probe");
            auto plane_pipeline = [device->native_device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"plane"] error:&native_error];
            auto cube_pipeline = [device->native_device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"cube"] error:&native_error];
            auto pixels_pipeline = [device->native_device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"pixels"] error:&native_error];
            check(plane_pipeline && cube_pipeline && pixels_pipeline, "Cannot create probe pipelines");
            {
                auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:256 height:1 mipmapped:NO];
                desc.storageMode=MTLStorageModeShared; desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
                auto alpha=[device->native_device() newTextureWithDescriptor:desc];
                std::array<uint8_t,256> bytes;
                for(unsigned i=0;i<256;++i) bytes[i]=i;
                [alpha replaceRegion:MTLRegionMake2D(0,0,256,1) mipmapLevel:0 withBytes:bytes.data() bytesPerRow:256];
                const SceGxmColorFormat rendered=SCE_GXM_COLOR_FORMAT_U8_A;
                // Independently specified logical RGBA mappings for all eight
                // single-channel texture swizzles; -1 and -2 denote zero/one.
                const int channels[][4]={{0,-1,-1,-2},{0,-1,-1,-1},{0,-2,-2,-2},{0,0,0,0},
                    {0,0,0,-1},{0,0,0,-2},{-1,-1,-1,0},{-2,-2,-2,0}};
                auto result=[device->native_device() newBufferWithLength:256*16 options:MTLResourceStorageModeShared];
                for(unsigned swizzle=0;swizzle<8;++swizzle) {
                    const auto format=static_cast<SceGxmTextureFormat>(uint32_t(SCE_GXM_TEXTURE_BASE_FORMAT_U8)|(swizzle<<12));
                    auto view=renderer::metal::sampling_view(alpha,format,&rendered);
                    auto command=[device->command_queue() commandBuffer]; auto enc=[command computeCommandEncoder];
                    [enc setComputePipelineState:pixels_pipeline]; [enc setTexture:view atIndex:0];
                    [enc setBuffer:result offset:0 atIndex:0];
                    [enc dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [enc endEncoding];
                    check(device->submit_and_wait(command,error),error.c_str());
                    const auto *values=static_cast<const float *>(result.contents);
                    for(unsigned i=0;i<256;++i) for(unsigned c=0;c<4;++c) {
                        const float expected=channels[swizzle][c]==-1?0:channels[swizzle][c]==-2?1:float(i)/255;
                        check(std::abs(values[i*4+c]-expected)<1e-6f,"U8_A surface sampling swizzle mismatch");
                    }
                }
                std::cout<<"PASS U8_A rendered surface sampling, all256 codes and8 swizzles (8192 channels)\n";
            }
            auto filtered_pipeline = [device->native_device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"filtered"] error:&native_error];
            check(filtered_pipeline != nil,"Cannot create filter probe");
            auto filter_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:2 height:2 mipmapped:NO];
            filter_desc.storageMode = MTLStorageModeShared;
            filter_desc.usage = MTLTextureUsageShaderRead;
            auto filter_texture = [device->native_device() newTextureWithDescriptor:filter_desc];
            const uint32_t pattern[] = {0xffffffff,0,0,0xffffffff};
            [filter_texture replaceRegion:MTLRegionMake2D(0,0,2,2) mipmapLevel:0 withBytes:pattern bytesPerRow:8];
            auto filter_output = [device->native_device() newBufferWithLength:16 options:MTLResourceStorageModeShared];
            for (bool strided : {false,true}) for (uint32_t mode=0;mode<4;++mode) {
                SceGxmTexture guest{};
                guest.type = (strided ? SCE_GXM_TEXTURE_LINEAR_STRIDED : SCE_GXM_TEXTURE_LINEAR)>>29;
                guest.min_filter = mode;
                guest.mag_filter = strided ? SCE_GXM_TEXTURE_FILTER_POINT : mode;
                auto sampler = renderer::metal::make_sampler(*device,guest,16);
                auto commands = [device->command_queue() commandBuffer];
                auto encoder = [commands computeCommandEncoder];
                [encoder setComputePipelineState:filtered_pipeline];
                [encoder setTexture:filter_texture atIndex:0];
                [encoder setSamplerState:sampler atIndex:0];
                [encoder setBuffer:filter_output offset:0 atIndex:0];
                [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [encoder endEncoding];
                check(device->submit_and_wait(commands,error),error.c_str());
                const bool linear = !strided && (mode==1 || mode==2);
                const auto *actual = static_cast<const float *>(filter_output.contents);
                for (size_t c=0;c<4;++c) check(std::abs(actual[c]-(linear?0.5f:1.0f))<1e-6f,"Guest texture filter mode/stride was misinterpreted");
            }
            std::cout << "PASS GPU point/linear/mipmap filter encodings and strided row bits with global anisotropy enabled\n";
            renderer::metal::SurfaceCaster caster(*device);
            if (argc==2) {
                const std::filesystem::path root=argv[1];
                std::ifstream metadata(root/"draw.txt");check(bool(metadata),"Cannot open captured draw");
                uint32_t guest_width=0,guest_height=0;double source_scale=0;
                std::string line;
                while(std::getline(metadata,line)) {
                    std::istringstream fields(line);std::string key;fields>>key;
                    if(key=="size") fields>>guest_width>>guest_height;
                    if(key=="scale") fields>>source_scale;
                }
                check(source_scale==1 && guest_width && guest_height && guest_width<=2048 && guest_height<=2048,"Capture must be a bounded 1x draw");
                auto capture_pipeline=[device->native_device() newComputePipelineStateWithFunction:[library newFunctionWithName:@"captured"] error:&native_error];
                check(capture_pipeline!=nil,"Cannot create captured-coordinate sampling probe");
                SceGxmTexture nearest{};nearest.type=SCE_GXM_TEXTURE_LINEAR>>29;
                auto sampler=renderer::metal::make_sampler(*device,nearest,16);
                for(uint32_t scale:{1u,2u,3u}) {
                    const uint32_t width=guest_width*scale,height=guest_height*scale;
                    const size_t count=size_t(width)*height;
                    std::vector<float> uv(count*2);
                    std::ifstream coordinates(root/("uv-"+std::to_string(scale)+".rg32"),std::ios::binary);
                    check(bool(coordinates.read(reinterpret_cast<char *>(uv.data()),uv.size()*4)),"Missing captured UV data");
                    auto uv_buffer=[device->native_device() newBufferWithBytes:uv.data() length:uv.size()*4 options:MTLResourceStorageModeShared];
                    auto output=[device->native_device() newBufferWithLength:count*16 options:MTLResourceStorageModeShared];
                    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float width:width height:height mipmapped:NO];
                    desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
                    auto packed=[device->native_device() newTextureWithDescriptor:desc];
                    std::vector<uint32_t> words(count*2);
                    for(uint32_t y=0;y<height;++y) for(uint32_t x=0;x<width;++x) {
                        words[(size_t(y)*width+x)*2]=(x&255)|((y&255)<<8)|0xff7f0000u;
                        words[(size_t(y)*width+x)*2+1]=((x*3)%255)|(((y*7)%255)<<8)|0x7f800000u;
                    }
                    [packed replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0 withBytes:words.data() bytesPerRow:width*8];
                    for(uint32_t word=0;word<2;++word) {
                        auto alias=caster.rgba8_from_rg32(packed,scale,false,word,word==1);
                        const float offset=renderer::metal::packed_alias_x_offset(scale,uint32_t(alias.width));
                        auto commands=[device->command_queue() commandBuffer];auto encoder=[commands computeCommandEncoder];
                        [encoder setComputePipelineState:capture_pipeline];[encoder setTexture:alias atIndex:0];[encoder setSamplerState:sampler atIndex:0];
                        [encoder setBuffer:output offset:0 atIndex:0];[encoder setBuffer:uv_buffer offset:0 atIndex:1];[encoder setBytes:&offset length:4 atIndex:2];
                        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
                        check(device->submit_and_wait(commands,error),error.c_str());
                        auto *values=static_cast<const float *>(output.contents);
                        for(size_t i=0;i<count;++i) for(uint32_t c=0;c<4;++c) {
                            const int byte=(words[i*2+word]>>(c*8))&255;
                            const float expected=word ? std::max(-1.0f,float(byte>=128?byte-256:byte)/127.0f) : float(byte)/255.0f;
                            if(std::abs(values[i*4+c]-expected)>1e-6f)
                                throw std::runtime_error("Captured UV selected wrong native pixel/word: scale="+std::to_string(scale)+" word="+std::to_string(word)+" pixel="+std::to_string(i));
                        }
                    }
                    std::cout<<"PASS actual captured vertex UVs, scale="<<scale<<", both packed words and every native pixel/channel\n";
                }
            }
            {
                const SceGxmColorFormat formats[] = {SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,
                    SCE_GXM_COLOR_FORMAT_U8U8U8U8_ARGB, SCE_GXM_COLOR_FORMAT_U8U8U8U8_RGBA,
                    SCE_GXM_COLOR_FORMAT_U8U8U8U8_BGRA};
                const unsigned memory_channels[][4] = {{0,1,2,3},{2,1,0,3},{3,2,1,0},{3,0,1,2}};
                const unsigned texture_channels[][4] = {{0,1,2,3},{2,1,0,3},{3,2,1,0},{1,2,3,0},
                    {0,1,2,4},{2,1,0,4},{3,2,1,4},{1,2,3,4}};
                std::array<uint8_t,1024> bytes;
                for(unsigned pixel=0;pixel<256;++pixel) {
                    bytes[pixel*4]=pixel; bytes[pixel*4+1]=(pixel+71)%256;
                    bytes[pixel*4+2]=255-pixel; bytes[pixel*4+3]=(pixel*37)%256;
                }
                SceGxmTexture linear{};
                linear.type=SCE_GXM_TEXTURE_LINEAR>>29;linear.mip_count=15;
                linear.min_filter=linear.mag_filter=SCE_GXM_TEXTURE_FILTER_LINEAR;
                auto sampler=renderer::metal::make_sampler(*device,linear,1);
                unsigned raw_count=0,sampled_count=0,filtered_count=0;
                for(bool target_gamma:{false,true}) for(unsigned order=0;order<4;++order) {
                    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:target_gamma?MTLPixelFormatRGBA8Unorm_sRGB:MTLPixelFormatRGBA8Unorm
                        width:16 height:16 mipmapped:NO];
                    desc.storageMode=MTLStorageModeShared; desc.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
                    auto target=[device->native_device() newTextureWithDescriptor:desc];
                    [target replaceRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0 withBytes:bytes.data() bytesPerRow:64];
                    auto memory=caster.rgba8_memory_snapshot(target,formats[order]);
                    std::array<uint8_t,1024> raw;
                    [memory getBytes:raw.data() bytesPerRow:64 fromRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0];
                    for(size_t i=0;i<raw.size();++i) {
                        check(raw[i]==bytes[(i/4)*4+memory_channels[order][i%4]],"Gamma memory snapshot changed channel bytes");
                        ++raw_count;
                    }
                    for(unsigned sample_gamma:{0u,1u,3u}) {
                        auto decoded=caster.rgba8_surface_sampling(target,formats[order],sample_gamma);
                        for(unsigned swizzle=0;swizzle<8;++swizzle) {
                            auto format=static_cast<SceGxmTextureFormat>(SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR|(swizzle<<12));
                            auto view=renderer::metal::sampling_view(decoded,format);
                            const auto expected_at=[&](unsigned pixel,unsigned c) {
                                const unsigned channel=texture_channels[swizzle][c];
                                if(channel==4) return 1.0;
                                double expected=bytes[pixel*4+memory_channels[order][channel]]/255.0;
                                if(sample_gamma && channel<(sample_gamma==3?2u:3u))
                                    expected=expected<=.04045?expected/12.92:std::pow((expected+.055)/1.055,2.4);
                                return expected;
                            };
                            auto output=caster.sampling_snapshot(view);
                            std::array<float,1024> sampled;
                            [output getBytes:sampled.data() bytesPerRow:256 fromRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0];
                            for(unsigned i=0;i<sampled.size();++i) {
                                const double expected=expected_at(i/4,i%4);
                                if(!(std::abs(sampled[i]-expected)<.0003))
                                    throw std::runtime_error("Gamma decode mismatch source="+std::to_string(target_gamma)+
                                        " order="+std::to_string(order)+" gamma="+std::to_string(sample_gamma)+
                                        " swizzle="+std::to_string(swizzle)+" component="+std::to_string(i)+
                                        " actual="+std::to_string(sampled[i])+" expected="+std::to_string(expected));
                                ++sampled_count;
                            }
                            auto commands=[device->command_queue() commandBuffer];
                            auto encoder=[commands computeCommandEncoder];
                            [encoder setComputePipelineState:filtered_pipeline];[encoder setTexture:view atIndex:0];
                            [encoder setSamplerState:sampler atIndex:0];[encoder setBuffer:filter_output offset:0 atIndex:0];
                            [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                            [encoder endEncoding];check(device->submit_and_wait(commands,error),error.c_str());
                            auto actual=static_cast<const float *>(filter_output.contents);
                            for(unsigned c=0;c<4;++c) {
                                const double expected=(expected_at(119,c)+expected_at(120,c)+expected_at(135,c)+expected_at(136,c))/4;
                                check(std::abs(actual[c]-expected)<.0003,"Gamma conversion must precede bilinear filtering");
                                ++filtered_count;
                            }
                        }
                    }
                }
                std::cout<<"PASS gamma source/sample/swizzle matrix raw_bytes="<<raw_count<<" sampled_components="<<sampled_count
                    <<" filtered_components="<<filtered_count<<" all256 codes,4 surface orders,8 texture orders,gamma0/1/3,linear/sRGB sources\n";
            }
            auto depth_desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:13 height:7 mipmapped:NO];
            depth_desc.storageMode=MTLStorageModePrivate;
            depth_desc.usage=MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            auto depth_texture=[device->native_device() newTextureWithDescriptor:depth_desc];
            for (double value : {0.0,0.125,0.75,1.0,0.123456,0.000021,0.99998}) {
                auto commands=[device->command_queue() commandBuffer];
                auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
                pass.depthAttachment.texture=depth_texture;
                pass.depthAttachment.loadAction=MTLLoadActionClear;
                pass.depthAttachment.storeAction=MTLStoreActionStore;
                pass.depthAttachment.clearDepth=value;
                auto encoder=[commands renderCommandEncoderWithDescriptor:pass];
                [encoder endEncoding];
                check(device->submit_and_wait(commands,error),error.c_str());
                auto snapshot=caster.depth_snapshot(depth_texture);
                std::array<float,13*7> pixels;
                [snapshot getBytes:pixels.data() bytesPerRow:13*4 fromRegion:MTLRegionMake2D(0,0,13,7) mipmapLevel:0];
                for (float actual:pixels) check(actual==float(value),"Depth snapshot differs from GPU depth attachment");
                auto normalized=caster.depth_snapshot(depth_texture,true);
                check(normalized.pixelFormat==MTLPixelFormatR16Unorm,"D16 snapshot has wrong storage format");
                std::array<uint16_t,13*7> words;
                [normalized getBytes:words.data() bytesPerRow:13*2 fromRegion:MTLRegionMake2D(0,0,13,7) mipmapLevel:0];
                const auto expected=uint16_t(std::lround(float(value)*65535.0f));
                for (uint16_t actual:words) check(actual==expected,"D16 snapshot did not quantize normalized GPU depth");
                // GXM names channels in ABGR order: 111R samples as (R,1,1,1).
                auto view=renderer::metal::sampling_view(normalized,SCE_GXM_TEXTURE_FORMAT_U16_111R);
                auto output=[device->native_device() newBufferWithLength:13*7*16 options:MTLResourceStorageModeShared];
                auto sample_commands=[device->command_queue() commandBuffer];
                auto sample_encoder=[sample_commands computeCommandEncoder];
                [sample_encoder setComputePipelineState:pixels_pipeline]; [sample_encoder setTexture:view atIndex:0];
                [sample_encoder setBuffer:output offset:0 atIndex:0];
                [sample_encoder dispatchThreads:MTLSizeMake(13*7,1,1) threadsPerThreadgroup:MTLSizeMake(13,1,1)];
                [sample_encoder endEncoding]; check(device->submit_and_wait(sample_commands,error),error.c_str());
                const auto *values=static_cast<const float *>(output.contents);
                auto diagnostic=caster.sampling_snapshot(view);
                check(diagnostic && diagnostic.pixelFormat==MTLPixelFormatRGBA32Float,"Missing canonical sampling diagnostic");
                std::array<float,13*7*4> diagnostic_pixels;
                [diagnostic getBytes:diagnostic_pixels.data() bytesPerRow:13*16 fromRegion:MTLRegionMake2D(0,0,13,7) mipmapLevel:0];
                for(size_t i=0;i<diagnostic_pixels.size();++i)
                    check(diagnostic_pixels[i]==values[i],"Diagnostic snapshot differs from shader-visible values");
                for(size_t i=0;i<13*7;++i) for(unsigned c=0;c<4;++c) {
                    const float wanted=c?1.0f:float(expected)/65535.0f;
                    if (std::abs(values[i*4+c]-wanted)>=1e-6f)
                        throw std::runtime_error("D16 sample mismatch depth="+std::to_string(value)+" channel="+std::to_string(c)
                            +" actual="+std::to_string(values[i*4+c])+" expected="+std::to_string(wanted));
                }
            }
            std::cout << "PASS private GPU depth attachment -> R32F and quantized/sampleable R16Unorm, seven depths, every pixel/channel\n";
            for (uint32_t scale : {1u,2u,3u}) for (bool swap : {false,true})
                for (uint32_t offset : {0u,1u}) for (bool snorm : {false,true}) {
                const uint32_t width=7*scale, height=3*scale;
                auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float width:width height:height mipmapped:NO];
                desc.storageMode=MTLStorageModeShared;
                desc.usage=MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
                auto packed=[device->native_device() newTextureWithDescriptor:desc];
                std::vector<uint32_t> words(size_t(width)*height*2);
                const uint32_t patterns[]={0x7fc12345,0xff800000,0x12345678,0x01020304,0xffffffff,0,0x99887766};
                for (size_t i=0;i<words.size();++i) words[i]=patterns[i%7] ^ uint32_t(i/7);
                [packed replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0 withBytes:words.data() bytesPerRow:width*8];
                auto unpacked=caster.rgba8_from_rg32(packed,scale,swap,offset,snorm);
                std::vector<uint32_t> pixels(size_t(width)*height*2);
                [unpacked getBytes:pixels.data() bytesPerRow:width*8 fromRegion:MTLRegionMake2D(0,0,width*2,height) mipmapLevel:0];
                auto sampled=[device->native_device() newBufferWithLength:pixels.size()*16 options:MTLResourceStorageModeShared];
                auto commands=[device->command_queue() commandBuffer];
                auto encoder=[commands computeCommandEncoder];
                [encoder setComputePipelineState:pixels_pipeline];
                [encoder setTexture:unpacked atIndex:0];
                [encoder setBuffer:sampled offset:0 atIndex:0];
                [encoder dispatchThreads:MTLSizeMake(pixels.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [encoder endEncoding];
                check(device->submit_and_wait(commands,error),error.c_str());
                const auto *values=static_cast<const float *>(sampled.contents);
                // Treat the rendered native RG32 image as a row-major stream
                // of adjacent 32-bit words. NaN payloads remain raw bytes.
                for (uint32_t gy=0;gy<3;++gy) for (uint32_t gx=0;gx<7;++gx)
                    for (uint32_t sy=0;sy<scale;++sy) for (uint32_t sx=0;sx<scale;++sx)
                        for (uint32_t word=0;word<2;++word) {
                            const uint32_t y=gy*scale+sy, x=(gx*2+word)*scale+sx;
                            const uint32_t next=y*width*2+x+offset;
                            const uint32_t expected=next < words.size() ? words[(next&~1u)+(swap?1-next%2:next%2)] : 0;
                            check(pixels[y*width*2+x]==expected,"RG32/RGBA8 cast lost raw bits or mixed scaled guest pixels");
                            for (uint32_t c=0;c<4;++c) {
                                const int byte=(expected>>(c*8))&255;
                                const float normalized=snorm ? std::max(-1.0f,float(byte>=128?byte-256:byte)/127.0f) : float(byte)/255.0f;
                                check(std::abs(values[(y*width*2+x)*4+c]-normalized)<1e-6f,"RG32 alias has incorrect signed/unsigned sampled values");
                            }
                        }
                std::cout << "PASS RG32 -> twice-width RGBA8, scale=" << scale << " swap=" << swap << " word-offset=" << offset << " snorm=" << snorm << ", every byte and sampled channel, including NaN payloads and row crossings\n";
            }
            // Four-channel names describe memory from most to least significant.
            // One/two-channel swizzle names instead describe output replication.
            struct Case { SceGxmTextureBaseFormat format; const char *name; uint32_t mode; };
            std::vector<Case> cases;
            const char *one[] = {"100R","000R","111R","RRRR","0RRR","1RRR","R000","R111"};
            const char *two[] = {"10GR","00GR","GRRR","RGGG","GRGR","00RG"};
            const char *three[] = {"1BGR","1RGB"};
            const char *four[] = {"ABGR","ARGB","RGBA","BGRA","1BGR","1RGB","RGB1","BGR1"};
            for (uint32_t i=0;i<8;++i) cases.push_back({SCE_GXM_TEXTURE_BASE_FORMAT_U8,one[i],i});
            for (uint32_t i=0;i<6;++i) cases.push_back({SCE_GXM_TEXTURE_BASE_FORMAT_U8U8,two[i],i});
            for (uint32_t i=0;i<2;++i) cases.push_back({SCE_GXM_TEXTURE_BASE_FORMAT_F11F11F10,three[i],i});
            for (uint32_t i=0;i<8;++i) cases.push_back({SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8,four[i],i});
            auto channel_index = [](char c) -> int { return c=='R'?0:c=='G'?1:c=='B'?2:3; };
            for (bool cube : {false,true}) {
                auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:2 height:2 mipmapped:YES];
                desc.textureType = cube ? MTLTextureTypeCube : MTLTextureType2D;
                desc.storageMode = MTLStorageModeShared;
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
                auto texture = [device->native_device() newTextureWithDescriptor:desc];
                check(texture != nil, "Cannot create probe texture");
                const size_t slices = cube ? 6 : 1;
                std::vector<std::array<float,4>> colors(slices*2);
                for (size_t slice=0;slice<slices;++slice) for (size_t mip=0;mip<2;++mip) {
                    const size_t i=slice*2+mip;
                    colors[i]={float(i+1)/64,float(i+17)/64,float(i+33)/64,float(i+49)/64};
                    std::array<std::array<float,4>,4> pixels;
                    pixels.fill(colors[i]);
                    const size_t dim=2>>mip;
                    [texture replaceRegion:MTLRegionMake2D(0,0,dim,dim) mipmapLevel:mip slice:slice withBytes:pixels.data() bytesPerRow:dim*16 bytesPerImage:dim*dim*16];
                }
                auto output = [device->native_device() newBufferWithLength:colors.size()*16 options:MTLResourceStorageModeShared];
                auto probe = [&](SceGxmTextureFormat format, const char *name, const SceGxmColorFormat *surface, const char *surface_name) {
                    auto view = renderer::metal::sampling_view(texture, format, surface);
                    check(view.mipmapLevelCount==2 && view.textureType==texture.textureType, "View lost mip/cube shape");
                    auto commands = [device->command_queue() commandBuffer];
                    auto encoder = [commands computeCommandEncoder];
                    [encoder setComputePipelineState:cube?cube_pipeline:plane_pipeline];
                    [encoder setTexture:view atIndex:0];
                    [encoder setBuffer:output offset:0 atIndex:0];
                    [encoder dispatchThreads:MTLSizeMake(colors.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                    [encoder endEncoding];
                    check(device->submit_and_wait(commands,error),error.c_str());
                    auto actual = static_cast<const float *>(output.contents);
                    for (size_t i=0;i<colors.size();++i) {
                        std::array<float,4> memory=colors[i];
                        if (surface_name) for (size_t j=0;j<4;++j) {
                            const char c=surface_name[3-j]; memory[j]=c=='1'?1:colors[i][channel_index(c)];
                        }
                        for (size_t c=0;c<4;++c) {
                            char value=name[3-c];
                            float expected=value=='0'?0:value=='1'?1:memory[channel_index(value)];
                            if ((uint32_t(format) & SCE_GXM_TEXTURE_BASE_FORMAT_MASK) == SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8) {
                                expected=1;
                                for (size_t j=0;j<4;++j)
                                    if (name[3-j]=="RGBA"[c]) expected=memory[j];
                            }
                            if (std::abs(actual[i*4+c]-expected)>=1e-6f)
                                throw std::runtime_error(std::string("GPU mismatch ") + name + " surface=" + (surface_name ?: "none")
                                    + " cube=" + std::to_string(cube) + " sample=" + std::to_string(i) + " channel=" + std::to_string(c)
                                    + " actual=" + std::to_string(actual[i*4+c]) + " expected=" + std::to_string(expected));
                        }
                    }
                };
                for (const auto &c:cases) probe(static_cast<SceGxmTextureFormat>(c.format | (c.mode<<12)),c.name,nullptr,nullptr);
                const char *surface_names[]={"ABGR","ARGB","RGBA","BGRA"};
                for (uint32_t color_mode=0;color_mode<4;++color_mode) for (uint32_t texture_mode=0;texture_mode<8;++texture_mode) {
                    const auto surface=static_cast<SceGxmColorFormat>(SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8 | (color_mode<<20));
                    probe(static_cast<SceGxmTextureFormat>(SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 | (texture_mode<<12)),four[texture_mode],&surface,surface_names[color_mode]);
                }
                for (uint32_t color_mode=0;color_mode<2;++color_mode) for (uint32_t texture_mode=0;texture_mode<8;++texture_mode) {
                    const auto surface=static_cast<SceGxmColorFormat>(SCE_GXM_COLOR_BASE_FORMAT_U8U8U8 | (color_mode<<20));
                    probe(static_cast<SceGxmTextureFormat>(SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 | (texture_mode<<12)),four[texture_mode],&surface,color_mode?"1RGB":"1BGR");
                }
                std::cout<<"PASS RGB8 rendered-surface sampling: both byte orders, eight texture mappings, implicit alpha1 despite nonconstant backing alpha\n";
                std::cout << "PASS " << (cube?"six cube faces":"2D") << ": 24 guest channel mappings, 32 surface/texture compositions, two mips, all GPU values\n";
            }
            return 0;
        } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
    }
}
