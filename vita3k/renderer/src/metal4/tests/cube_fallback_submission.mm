// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Extracted production fallback block, delayed copy/raw/driver boundaries.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
static unsigned waits,submits,copies,raw_encodes,allocations;
static bool fail_allocation,fail_encoder;
static bool allow_native_views;
#define LOG_INFO(...) ((void)0)
static void require(bool b,const std::string &error){if(!b)throw std::runtime_error(error);}
struct PixelStore {
    using Rows=std::vector<std::vector<uint32_t>>;
    std::shared_ptr<Rows> data=std::make_shared<Rows>();
    auto &operator[](size_t i){return (*data)[i];}
    size_t size()const{return data->size();}void resize(size_t n){data->resize(n);}
    auto begin(){return data->begin();}auto end(){return data->end();}
    PixelStore &operator=(const PixelStore &other){*data=*other.data;return *this;}
};
@interface CubeTexture : NSObject {
@public PixelStore pixels;
}
@property NSUInteger width,height,mipmapLevelCount,sampleCount;
@property MTLTextureType textureType;
@property MTLPixelFormat pixelFormat;
- (id)newTextureViewWithPixelFormat:(MTLPixelFormat)format;
@end
@implementation CubeTexture
- (id)newTextureViewWithPixelFormat:(MTLPixelFormat)format {
    if(!allow_native_views)return nil;
    assert(format==MTLPixelFormatRGBA16Unorm);
    auto view=[CubeTexture new];view.width=_width;view.height=_height;view.mipmapLevelCount=_mipmapLevelCount;
    view.sampleCount=_sampleCount;view.textureType=_textureType;view.pixelFormat=format;view->pixels.data=pixels.data;return view;
}
@end
static CubeTexture *image(unsigned width,unsigned mips,MTLPixelFormat format,bool cube) {
    auto t=[CubeTexture new];t.width=t.height=width;t.mipmapLevelCount=mips;t.pixelFormat=format;
    t.sampleCount=1;t.textureType=cube?MTLTextureTypeCube:MTLTextureType2D;t->pixels.resize(mips*(cube?6:1));
    for(unsigned face=0;face<(cube?6u:1u);++face)for(unsigned mip=0;mip<mips;++mip){
        const unsigned w=std::max(1u,width>>mip);
        t->pixels[face*mips+mip].resize(w*w*(format==MTLPixelFormatRG32Float?2:1));
    }
    return t;
}
@class V3KMetal4Commands;
@interface CubeBlit : NSObject
@property(strong) V3KMetal4Commands *commands;
- (void)copyFromTexture:(id)src sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sm sourceOrigin:(MTLOrigin)so
    sourceSize:(MTLSize)size toTexture:(id)dst destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dm destinationOrigin:(MTLOrigin)d;
- (void)endEncoding;
@end
@interface V3KMetal4Commands : NSObject {
@public NSMutableArray<void (^)(void)> *work;
}
@property MTLCommandBufferStatus status;
- (CubeBlit *)blitCommandEncoder;
- (void)complete;
@end
@implementation V3KMetal4Commands
- (instancetype)init {if((self=[super init])){_status=MTLCommandBufferStatusNotEnqueued;work=[NSMutableArray new];}return self;}
- (CubeBlit *)blitCommandEncoder {assert(_status==MTLCommandBufferStatusNotEnqueued);if(fail_encoder)return nil;auto b=[CubeBlit new];b.commands=self;return b;}
- (void)complete {
    assert(_status==MTLCommandBufferStatusNotEnqueued);_status=MTLCommandBufferStatusCommitted;++submits;
    for(void (^block)(void) in work)block();[work removeAllObjects];_status=MTLCommandBufferStatusCompleted;
}
@end
@implementation CubeBlit
- (void)copyFromTexture:(id)source sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sm sourceOrigin:(MTLOrigin)so
    sourceSize:(MTLSize)size toTexture:(id)destination destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dm destinationOrigin:(MTLOrigin)d {
    CubeTexture *src=source,*dst=destination;assert(ss==0&&ds<6&&sm==dm&&sm<src.mipmapLevelCount);
    assert(so.x==0&&so.y==0&&so.z==0&&d.x==0&&d.y==0&&d.z==0&&size.depth==1);
    assert(size.width==std::max(NSUInteger(1),src.width>>sm)&&size.height==size.width);++copies;
    [_commands->work addObject:^{dst->pixels[ds*dst.mipmapLevelCount+dm]=src->pixels[sm];}];
}
- (void)endEncoding {}
@end
@interface CubeDevice : NSObject
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d;
- (V3KMetal4Commands *)commandBuffer;
@end
@implementation CubeDevice
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d {
    ++allocations;if(fail_allocation)return nil;
    assert(d.textureType==MTLTextureTypeCube&&d.width==d.height&&d.storageMode==MTLStorageModePrivate);
    assert(d.usage==(MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView));
    return (id<MTLTexture>)image(unsigned(d.width),unsigned(d.mipmapLevelCount),d.pixelFormat,true);
}
- (V3KMetal4Commands *)commandBuffer {return [V3KMetal4Commands new];}
@end
struct Device {
    CubeDevice *native=[CubeDevice new];id<MTLDevice> native_device(){return (id<MTLDevice>)native;}
    CubeDevice *command_queue(){return native;}
    bool submit_and_wait(V3KMetal4Commands *c,std::string &){++waits;[c complete];return true;}
};
static V3KMetal4Commands *scene_command_buffer(Device &){return [V3KMetal4Commands new];}
// RAW_SUPPORT_HOOK
// RAW_VIEW_HOOK
static id<MTLTexture> sampling_view(id<MTLTexture> t,int,void *){return t;}
namespace gxm {static int get_format(int){return 0;}}
struct SurfaceCaster {
    Device &device;explicit SurfaceCaster(Device &d):device(d){}
    id<MTLTexture> raw_texture_snapshot(id<MTLTexture> source,V3KMetal4Commands *commands=nil){
        if(auto view=raw_texture_snapshot_view(source))return view;
        CubeTexture *src=(CubeTexture *)source;auto dst=image(unsigned(src.width),unsigned(src.mipmapLevelCount),src.pixelFormat,src.textureType==MTLTextureTypeCube);
        auto c=commands?:scene_command_buffer(device);assert(c.status==MTLCommandBufferStatusNotEnqueued);++raw_encodes;
        // Actual raw helper encoding is covered separately. This boundary
        // detects whether its input copy has executed before conversion.
        [c->work addObject:^{dst->pixels=src->pixels;for(auto &level:dst->pixels)for(auto &word:level)word^=0x80000000u;}];
        if(!commands){std::string error;device.submit_and_wait(c,error);}return (id<MTLTexture>)dst;
    }
};
struct MetalContext {struct State{V3KMetal4Commands *commands=nil;bool encoder=true;};std::unique_ptr<State> impl=std::make_unique<State>();};
struct MetalState {
    struct Impl {
        struct CubeAlias{id<MTLTexture> source,cube;uint64_t last_used_frame=0;};
        std::unique_ptr<Device> device=std::make_unique<Device>();std::unique_ptr<SurfaceCaster> caster;
        std::map<uintptr_t,CubeAlias> cube_aliases;bool sync_draws=false;uint64_t frame_timestamp=9;
    };
    std::unique_ptr<Impl> impl=std::make_unique<Impl>();
    void mid_scene_flush(MetalContext &ctx,bool wait){assert(!wait);ctx.impl->encoder=false;}
    void finish(MetalContext &ctx){ctx.impl->encoder=false;if(ctx.impl->commands){std::string error;impl->device->submit_and_wait(ctx.impl->commands,error);ctx.impl->commands=nil;}}
    id<MTLTexture> prepare(MetalContext &ctx,id<MTLTexture> source,bool capture_draw,bool raw_cast_requested){
        const unsigned index=0,address=0x1000;const int texture=0;std::array<id<MTLTexture>,1> prepared_images{};
        struct Info{std::array<unsigned,4> control{};};std::array<Info,1> texture_mip_info{};
        // FALLBACK_HOOK
        assert(texture_mip_info[0].control[3]==unsigned(raw_cast_requested&&raw_texture_snapshot_supported(source)));return prepared_images[0];
    }
    id<MTLTexture> prepare_uploaded(MetalContext &ctx,id<MTLTexture> base_image,bool capture_draw,bool raw_cast){
        const unsigned index=0;id<MTLTexture> uploaded=base_image;
        std::vector<int> rendered;std::array<id<MTLTexture>,1> prepared_images{};
        for(unsigned once=0;once<1;++once){
            // UPLOADED_HOOK
        }
        return prepared_images[0];
    }
};
static void fill(CubeTexture *src,uint32_t tag){
    for(size_t mip=0;mip<src->pixels.size();++mip)for(size_t i=0;i<src->pixels[mip].size();++i)
        src->pixels[mip][i]=tag+uint32_t(mip*65536+i);
}
static void compare(CubeTexture *out,CubeTexture *src,bool raw){
    assert(out.textureType==MTLTextureTypeCube&&out.mipmapLevelCount==src.mipmapLevelCount);
    for(unsigned face=0;face<6;++face)for(unsigned mip=0;mip<src.mipmapLevelCount;++mip)
    for(unsigned i=0;i<src->pixels[mip].size();++i)
        assert(out->pixels[face*src.mipmapLevelCount+mip][i]==(src->pixels[mip][i]^(raw?0x80000000u:0u)));
}
int main(){@autoreleasepool{
    size_t cases=0;
    if(!std::getenv("VITA3K_UPLOADED_RAW_ONLY")){
    for(unsigned size:{1u,2u,7u,8u,32u})for(auto format:{MTLPixelFormatRGBA8Unorm,MTLPixelFormatRG32Float})
    for(bool raw:{false,true})for(bool capture:{false,true})for(bool sync:{false,true})for(bool existing:{false,true})for(bool views:{false,true}) {
        allow_native_views=views;
        MetalState state;MetalContext ctx;state.impl->sync_draws=sync;auto src=image(size,std::bit_width(size),format,false);fill(src,17);
        if(existing){ctx.impl->commands=scene_command_buffer(*state.impl->device);[ctx.impl->commands->work addObject:^{fill(src,73);}];}
        auto prior=ctx.impl->commands;waits=submits=copies=raw_encodes=allocations=0;
        auto output=(CubeTexture *)state.prepare(ctx,(id<MTLTexture>)src,capture,raw);const bool queued=!capture&&!sync;
        const bool converted=raw&&raw_texture_snapshot_supported((id<MTLTexture>)src)&&!views;
        if(queued){
            assert(waits==0&&submits==0&&"GPU-only cube fallback must remain in its consumer scene");
            assert(ctx.impl->commands&&(!prior||prior==ctx.impl->commands));
            [ctx.impl->commands->work addObject:^{compare(output,src,converted);}];
            [ctx.impl->commands complete];ctx.impl->commands=nil;
        }else{assert(waits>0);compare(output,src,converted);}
        assert(copies==6*src.mipmapLevelCount&&allocations==1&&raw_encodes==unsigned(converted));
        const auto copied=copies;
        auto cached=(CubeTexture *)state.prepare(ctx,(id<MTLTexture>)src,capture,raw);
        assert(copies==copied&&state.impl->cube_aliases.size()==1);
        if(queued&&ctx.impl->commands){[ctx.impl->commands complete];ctx.impl->commands=nil;}compare(cached,src,converted);
        // Cache eviction cannot release the textures pinned by an outstanding
        // submission. Delayed consumer blocks keep no source/cube cache entry.
        if(queued){
            auto again=(CubeTexture *)state.prepare(ctx,(id<MTLTexture>)image(size,src.mipmapLevelCount,format,false),false,raw);
            state.impl->cube_aliases.clear();assert(again);[ctx.impl->commands complete];ctx.impl->commands=nil;
        }
        ++cases;
    }
    allow_native_views=false;
    for(bool encoder_failure:{false,true}){
        MetalState state;MetalContext ctx;auto src=image(4,3,MTLPixelFormatRGBA8Unorm,false);
        fail_encoder=encoder_failure;fail_allocation=!encoder_failure;waits=0;bool rejected=false;
        try{state.prepare(ctx,(id<MTLTexture>)src,false,false);}catch(const std::runtime_error &){rejected=true;}
        assert(rejected&&waits==0&&state.impl->cube_aliases.empty());fail_encoder=fail_allocation=false;
        state.prepare(ctx,(id<MTLTexture>)src,false,false);[ctx.impl->commands complete];
    }
    {
        MetalState state;MetalContext ctx;NSMutableArray<CubeTexture *> *sources=[NSMutableArray new],*outputs=[NSMutableArray new];
        for(unsigned i=0;i<12;++i){
            auto src=image(7,3,MTLPixelFormatRG32Float,false);fill(src,1000*i);[sources addObject:src];
            [outputs addObject:(CubeTexture *)state.prepare(ctx,(id<MTLTexture>)src,false,false)];
            assert(state.impl->cube_aliases.size()<=8);
        }
        assert(state.impl->cube_aliases.size()==8);[ctx.impl->commands complete];
        for(unsigned i=0;i<12;++i)compare(outputs[i],sources[i],false);
    }
        std::cout<<"PASS 2D cube fallback: "<<cases<<" production-block cases; all six faces/mips byte-exact, existing producer-copy-raw-consumer order, cache reuse/eviction, capture/sync waits, allocation/encoder retry; ASAN/UBSAN, CPU doubles only, no GPU/app\n";
    }
    size_t upload_cases=0;
    for(bool cube:{false,true})for(bool raw:{false,true})for(bool capture:{false,true})for(bool sync:{false,true})for(bool existing:{false,true})for(bool views:{false,true}){
        allow_native_views=views;
        MetalState state;MetalContext ctx;state.impl->sync_draws=sync;auto src=image(7,3,MTLPixelFormatRG32Float,cube);fill(src,71);
        const bool queued=!capture&&!sync;__block unsigned prior_seen=0;
        if(existing){ctx.impl->commands=scene_command_buffer(*state.impl->device);[ctx.impl->commands->work addObject:^{++prior_seen;}];}
        auto prior=ctx.impl->commands;waits=submits=0;
        auto result=(CubeTexture *)state.prepare_uploaded(ctx,(id<MTLTexture>)src,capture,raw);
        assert(result&&result.textureType==src.textureType&&result.mipmapLevelCount==src.mipmapLevelCount);
        if(queued){
            assert(waits==0&&submits==0&&"uploaded raw carrier is only consumed by the scene GPU");
            if(raw&&!views)assert(ctx.impl->commands&&(!prior||prior==ctx.impl->commands));
        }else if(raw&&!views)assert(waits==1);
        if(raw&&views)assert(ctx.impl->encoder&&waits==0);
        if(ctx.impl->commands){[ctx.impl->commands complete];ctx.impl->commands=nil;}
        assert(prior_seen==unsigned(existing));
        for(size_t mip=0;mip<src->pixels.size();++mip)for(size_t i=0;i<src->pixels[mip].size();++i)
            assert(result->pixels[mip][i]==(src->pixels[mip][i]^(raw&&!views?0x80000000u:0u)));
        ++upload_cases;
    }
    std::cout<<"PASS uploaded raw carriers: "<<upload_cases<<" production-block 2D/cube, raw/ordinary, capture/sync and existing-command cases; same scene, complete mip/face transport and synchronous diagnostics preserved; CPU doubles, no GPU/app\n";
    size_t view_cases=0;
    for(auto format:{MTLPixelFormatRGBA16Float,MTLPixelFormatRGBA16Unorm,MTLPixelFormatRGBA16Snorm,MTLPixelFormatRG32Float,MTLPixelFormatRG32Uint,MTLPixelFormatRGBA8Unorm})
    for(bool cube:{false,true})for(unsigned samples:{1u,4u})for(bool views:{false,true}){
        allow_native_views=views;auto src=image(2,2,format,cube);src.sampleCount=samples;
        auto result=(CubeTexture *)raw_texture_snapshot_view((id<MTLTexture>)src);
        const bool expected=samples==1&&format!=MTLPixelFormatRGBA8Unorm&&(views||format==MTLPixelFormatRGBA16Unorm);
        assert(bool(result)==expected);
        if(result){assert(result->pixels.data==src->pixels.data&&result.pixelFormat==MTLPixelFormatRGBA16Unorm);}
        ++view_cases;
    }
    assert(!raw_texture_snapshot_view(nil));
    std::cout<<"PASS raw view probe: "<<view_cases<<" complete production support/view cases; identity UNORM, supported/unsupported formats, MSAA rejection, all faces/mips share storage, no encoder/flush\n";
}}
