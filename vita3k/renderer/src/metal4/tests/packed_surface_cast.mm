// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Actual CPU readback/packers and actual new encoder method; Metal is replaced
// only at its driver boundary. No device or app is created.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <gxm/types.h>
#include <util/float_to_half.h>
#include <renderer/metal4/packed_surface_cast.h>
#include <renderer/metal4/linear_surface_patch.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <arm_neon.h>
using namespace renderer::metal;
namespace gxm {
// GXM_BASE_HOOK
}
namespace texture { size_t encode_morton(uint32_t,uint32_t,uint32_t,uint32_t) { throw std::logic_error("general cast is always linear"); } }
namespace renderer::texture {
// X8_DECODE_HOOK
}
struct SceGxmColorSurface { SceGxmColorFormat colorFormat; uint32_t width,height,strideInPixels; SceGxmColorSurfaceType surfaceType; };
struct SurfaceMemoryRange { size_t offset,size; };
struct SurfaceRect { uint32_t x,y,width,height; };
using Map=std::array<MTLTextureSwizzle,4>;
constexpr auto R=MTLTextureSwizzleRed,G=MTLTextureSwizzleGreen,B=MTLTextureSwizzleBlue,A=MTLTextureSwizzleAlpha,Z=MTLTextureSwizzleZero,O=MTLTextureSwizzleOne;
constexpr Map identity{R,G,B,A};
constexpr Map four[]={{R,G,B,A},{B,G,R,A},{A,B,G,R},{G,B,A,R},{R,G,B,O},{B,G,R,O},{A,B,G,O},{G,B,A,O}};
constexpr Map packed_four[]={{A,B,G,R},{G,B,A,R},{R,G,B,A},{B,G,R,A}};
struct SurfaceComponents { uint32_t count=0,bytes=0; MTLPixelFormat native=MTLPixelFormatInvalid; };
static constexpr std::array<std::array<uint8_t,4>,4> mixed_signed_guest_bytes{{{{0,1,2,3}},{{2,1,0,3}},{{3,2,1,0}},{{1,2,3,0}}}};
static std::optional<std::array<uint32_t,17>> packed_surface_cast_config(SceGxmColorFormat,SceGxmTextureBaseFormat,uint32_t,MTLPixelFormat,uint64_t,uint64_t);
// CPU_REFERENCE_FUNCTIONS
namespace gpu {
using uint=uint32_t;
using std::min; using std::max; using std::clamp; using std::abs;
#define thread
#define constant const
// GPU_CORE_HOOK
#undef thread
#undef constant
}
// CONFIG_HOOK
static size_t native_bytes(MTLPixelFormat f) {
    switch(f) {
    case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm: return 1;
    case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Float:
    case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatB5G6R5Unorm: case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm: case MTLPixelFormatA1BGR5Unorm: return 2;
    case MTLPixelFormatRGBA16Float: case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm: case MTLPixelFormatRG32Float: return 8;
    default: return 4;
    }
}
@interface CastTexture : NSObject {
@public std::vector<uint8_t> pixels; size_t cpu_reads,cpu_writes;
}
@property NSUInteger width,height;
@property MTLPixelFormat pixelFormat;
@property MTLStorageMode storageMode;
@property(getter=isFramebufferOnly) BOOL framebufferOnly;
@property NSUInteger sampleCount;
- (MTLTextureType)textureType;
- (id<MTLBuffer>)buffer;
- (NSUInteger)bufferBytesPerRow;
- (NSUInteger)bufferOffset;
- (void)getBytes:(void *)bytes bytesPerRow:(NSUInteger)pitch fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level;
- (void)replaceRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level withBytes:(const void *)bytes bytesPerRow:(NSUInteger)pitch;
@end
@implementation CastTexture
- (instancetype)init { if((self=[super init])) { _storageMode=MTLStorageModeShared; _sampleCount=1; } return self; }
- (MTLTextureType)textureType { return MTLTextureType2D; }
- (id<MTLBuffer>)buffer { return nil; }
- (NSUInteger)bufferBytesPerRow { return 0; }
- (NSUInteger)bufferOffset { return 0; }
- (void)getBytes:(void *)bytes bytesPerRow:(NSUInteger)pitch fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level {
    assert(!level && region.size.depth==1 && region.origin.x+region.size.width<=_width && region.origin.y+region.size.height<=_height);
    const auto b=native_bytes(_pixelFormat); assert(pitch>=region.size.width*b);
    for(size_t y=0;y<region.size.height;++y) std::memcpy(static_cast<uint8_t *>(bytes)+y*pitch,pixels.data()+((y+region.origin.y)*_width+region.origin.x)*b,region.size.width*b);
    ++cpu_reads;
}
- (void)replaceRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level withBytes:(const void *)bytes bytesPerRow:(NSUInteger)pitch { ++cpu_writes; assert(false && "new path must not CPU-upload pixels"); }
@end
@interface CastBuffer : NSObject { @public std::vector<uint8_t> bytes; }
@end
@implementation CastBuffer
- (NSUInteger)length { return bytes.size(); }
@end
@interface CastEncoder : NSObject {
@public NSMutableArray<void (^)(void)> *work;
    CastBuffer *input_buffer,*output_buffer;
    std::array<uint32_t,17> config;
}
- (void)copyFromTexture:(CastTexture *)texture sourceSlice:(NSUInteger)s sourceLevel:(NSUInteger)l sourceOrigin:(MTLOrigin)o sourceSize:(MTLSize)size toBuffer:(CastBuffer *)buffer destinationOffset:(NSUInteger)offset destinationBytesPerRow:(NSUInteger)pitch destinationBytesPerImage:(NSUInteger)image;
- (void)copyFromBuffer:(CastBuffer *)buffer sourceOffset:(NSUInteger)offset sourceBytesPerRow:(NSUInteger)pitch sourceBytesPerImage:(NSUInteger)image sourceSize:(MTLSize)size toTexture:(CastTexture *)texture destinationSlice:(NSUInteger)s destinationLevel:(NSUInteger)l destinationOrigin:(MTLOrigin)o;
- (void)setComputePipelineState:(id)state;
- (void)setTexture:(id)texture atIndex:(NSUInteger)index;
- (void)setBuffer:(CastBuffer *)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)dispatchThreads:(MTLSize)size threadsPerThreadgroup:(MTLSize)group;
- (void)endEncoding;
@end
@implementation CastEncoder
- (void)copyFromTexture:(CastTexture *)t sourceSlice:(NSUInteger)s sourceLevel:(NSUInteger)l sourceOrigin:(MTLOrigin)o sourceSize:(MTLSize)size toBuffer:(CastBuffer *)b destinationOffset:(NSUInteger)offset destinationBytesPerRow:(NSUInteger)pitch destinationBytesPerImage:(NSUInteger)image {
    assert(!s&&!l&&!o.x&&!o.y&&!o.z&&!offset&&!image&&size.depth==1&&size.width==t.width&&size.height==t.height);
    assert(pitch==t.width*native_bytes(t.pixelFormat)&&b->bytes.size()>=pitch*t.height);
    [work addObject:^{ std::memcpy(b->bytes.data(),t->pixels.data(),t->pixels.size()); }];
}
- (void)copyFromBuffer:(CastBuffer *)b sourceOffset:(NSUInteger)offset sourceBytesPerRow:(NSUInteger)pitch sourceBytesPerImage:(NSUInteger)image sourceSize:(MTLSize)size toTexture:(CastTexture *)t destinationSlice:(NSUInteger)s destinationLevel:(NSUInteger)l destinationOrigin:(MTLOrigin)o {
    assert(!s&&!l&&!o.x&&!o.y&&!o.z&&!offset&&!image&&size.depth==1&&size.width==t.width&&size.height==t.height);
    assert(pitch==t.width*native_bytes(t.pixelFormat)&&b->bytes.size()>=pitch*t.height);
    [work addObject:^{ std::memcpy(t->pixels.data(),b->bytes.data(),t->pixels.size()); }];
}
- (void)setComputePipelineState:(id)state { assert(state); }
- (void)setTexture:(id)texture atIndex:(NSUInteger)index { assert(false); }
- (void)setBuffer:(CastBuffer *)b offset:(NSUInteger)offset atIndex:(NSUInteger)index { assert(!offset&&index<2); if(index) output_buffer=b; else input_buffer=b; }
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index { assert(index==2&&length==sizeof(config)); std::memcpy(config.data(),bytes,length); }
- (void)dispatchThreads:(MTLSize)size threadsPerThreadgroup:(MTLSize)group {
    assert(size.width==config[0]&&size.height==config[1]&&size.depth==1&&group.width==8&&group.height==8&&group.depth==1);
    auto in=input_buffer,out=output_buffer; auto c=config;
    [work addObject:^{
        for(uint32_t y=0;y<c[1];++y) for(uint32_t x=0;x<c[0];++x) {
            uint32_t native[8]{},guest[8]{};
            for(unsigned i=0;i<c[4];++i) native[i]=in->bytes.at(y*c[2]+x*c[4]+i);
            gpu::pc_pixel(native,guest,c.data());
            for(unsigned i=0;i<c[5];++i) out->bytes.at(y*c[3]+x*c[5]+i)=uint8_t(guest[i]);
        }
    }];
}
- (void)endEncoding { input_buffer=nil; output_buffer=nil; }
@end
@interface V3KMetal4Commands : NSObject
@property MTLCommandBufferStatus status;
@property(strong) CastEncoder *encoder;
- (CastEncoder *)computeCommandEncoder;
- (void)runGPU;
@end
@implementation V3KMetal4Commands
- (instancetype)init { if((self=[super init])) { _status=MTLCommandBufferStatusNotEnqueued; _encoder=[CastEncoder new]; _encoder->work=[NSMutableArray new]; } return self; }
- (CastEncoder *)computeCommandEncoder { return _encoder; }
- (void)runGPU { for(void (^block)(void) in _encoder->work) block(); [_encoder->work removeAllObjects]; _status=MTLCommandBufferStatusCompleted; }
@end
static bool fail_library,fail_pipeline,fail_texture,fail_buffer,fail_submit;
static size_t libraries,pipelines,allocations,submissions;
static uint64_t max_buffer=UINT64_MAX;
@interface CastDevice : NSObject
- (NSUInteger)maxBufferLength;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)descriptor;
@end
@implementation CastDevice
- (NSUInteger)maxBufferLength { return max_buffer; }
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {
    assert(options==MTLResourceStorageModePrivate); ++allocations; if(fail_buffer)return nil;
    auto b=[CastBuffer new]; b->bytes.resize(length,0xa5); return (id<MTLBuffer>)b;
}
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d {
    ++allocations; if(fail_texture)return nil;
    assert(d.storageMode==MTLStorageModeShared&&d.usage==(MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView));
    auto t=[CastTexture new];t.width=d.width;t.height=d.height;t.pixelFormat=d.pixelFormat;
    t->pixels.resize(t.width*t.height*native_bytes(t.pixelFormat),0x5a); return (id<MTLTexture>)t;
}
@end
struct Device {
    CastDevice *native=[CastDevice new];
    id<MTLDevice> native_device() { return (id<MTLDevice>)native; }
    id<MTLLibrary> compile_library(NSString *source,id,NSError **error) {
        ++libraries; if(fail_library) { *error=[NSError errorWithDomain:@"test" code:1 userInfo:nil];return nil; }
        assert([source containsString:@"kernel void repack_packed_surface"]);return (id<MTLLibrary>)[NSObject new];
    }
    bool submit_and_wait(V3KMetal4Commands *c,std::string &error) { ++submissions; if(fail_submit) {error="test submit failure";return false;} [c runGPU];return true; }
};
struct SurfaceCaster {
    Device &device; id<MTLLibrary> packed_cast_library=nil,multisample_library=nil;
    id<MTLComputePipelineState> packed_cast_pipeline=nil,component_cast_pipeline=nil;
    id<MTLBuffer> packed_cast_input=nil,packed_cast_output=nil;
    id<MTLTexture> surface_format_cast(id<MTLTexture>,SceGxmColorFormat,SceGxmTextureBaseFormat,uint32_t,V3KMetal4Commands *,bool=false);
};
static V3KMetal4Commands *surface_command_buffer(Device &,NSString *) { return [V3KMetal4Commands new]; }
static id<MTLComputePipelineState> cached_compute(Device &,id<MTLLibrary>,NSString *name) { ++pipelines; if(fail_pipeline)throw std::runtime_error("pipeline failure");return (id<MTLComputePipelineState>)name; }
static MTLPixelFormat sample_bits_format(MTLPixelFormat) { throw std::logic_error("test exercises packed branch only"); }
// BUFFER_HOOK
// CAST_HOOK
struct BindingTexture { SceGxmTextureFormat format; };
namespace gxm { SceGxmTextureFormat get_format(const BindingTexture &t) { return t.format; } }
// REPACKED_COLOR_HOOK
static std::optional<SceGxmColorFormat> binding_metadata(SceGxmTextureFormat format,bool gamma) {
    const size_t index=0;
    BindingTexture texture{format};
    std::array<id<MTLTexture>,1> inline_cast_images{(id<MTLTexture>)[CastTexture new]};
    std::array<id<MTLTexture>,1> inline_gamma_images{gamma ? (id<MTLTexture>)[CastTexture new] : nil};
    id<MTLTexture> native=nil;
    SceGxmColorFormat original=SCE_GXM_COLOR_FORMAT_U5U6U5_BGR;
    const SceGxmColorFormat *rendered_format=&original;
    std::optional<SceGxmColorFormat> repacked_color;
    // BIND_CAST_HOOK
    // BIND_GAMMA_HOOK
    assert(native==(gamma ? inline_gamma_images[0] : inline_cast_images[0]));
    return rendered_format ? std::optional(*rendered_format) : std::nullopt;
}
// TESTS_HOOK
