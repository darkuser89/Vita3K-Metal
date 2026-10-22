// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Production publication planning/ownership, with simulated GPU and pixel conversion.
#include <renderer/metal4/apple.h>
#include <renderer/metal4/color_surface.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <vector>
using namespace renderer::metal;
static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
struct SurfaceRect {uint32_t x,y,width,height;};
struct PublicationClip {SurfaceRect guest,source;};
struct SurfaceMemoryRange {size_t offset,size;};
struct SurfaceWriteStamp {bool dirty=false;bool changed()const{return dirty;}};
static uint64_t XXH3_64bits(const void *bytes,size_t size){uint64_t h=0;for(auto b:std::span((const uint8_t *)bytes,size))h=h*131+b;return h;}
namespace gxm {static SceGxmColorBaseFormat get_base_format(SceGxmColorFormat f){return SceGxmColorBaseFormat(uint32_t(f)&SCE_GXM_COLOR_BASE_FORMAT_MASK);}}
@interface PublicationTexture : NSObject
@property NSUInteger width,height,sampleCount;
@property MTLPixelFormat pixelFormat;
@property MTLTextureType textureType;
@property(strong) NSMutableData *bytes;
@end
@implementation PublicationTexture
@end
static id<MTLTexture> image(uint32_t width,uint32_t height,uint8_t marker=0) {
    auto t=[PublicationTexture new];t.width=width;t.height=height;t.sampleCount=1;t.textureType=MTLTextureType2D;
    t.pixelFormat=MTLPixelFormatRGBA8Unorm;t.bytes=[NSMutableData dataWithLength:size_t(width)*height];
    std::memset(t.bytes.mutableBytes,marker,t.bytes.length);return (id<MTLTexture>)t;
}
static uint8_t marker(id<MTLTexture> texture){return ((const uint8_t *)((PublicationTexture *)texture).bytes.bytes)[0];}
@interface V3KMetal4Commands : NSObject
@property MTLCommandBufferStatus status;
@property(strong) NSError *error;
@property bool fail;
@property unsigned waits;
- (void)add:(void (^)(void))work;
- (void)commit;
- (void)waitUntilCompleted;
@end
@implementation V3KMetal4Commands {NSMutableArray<void (^)(void)> *_work;}
- (instancetype)init{if((self=[super init]))_work=[NSMutableArray new];return self;}
- (void)add:(void (^)(void))work{assert(_status==MTLCommandBufferStatusNotEnqueued);[_work addObject:work];}
- (void)commit{assert(_status==MTLCommandBufferStatusNotEnqueued);_status=MTLCommandBufferStatusCommitted;}
- (void)waitUntilCompleted{++_waits;assert(_status!=MTLCommandBufferStatusNotEnqueued);if(_fail){_status=MTLCommandBufferStatusError;return;}
    for(void (^work)(void) in _work)work();[_work removeAllObjects];_status=MTLCommandBufferStatusCompleted;}
@end
struct Device {};
struct PreparedImageCall {SurfaceRect source{},destination{};bool raw=false;std::optional<PublicationClip> clip;};
static unsigned copies=0,resamples=0;
static PreparedImageCall last_call;
struct SurfaceCaster {
    explicit SurfaceCaster(Device &){}
    id<MTLTexture> enqueue_subrectangle(id<MTLTexture> source,uint32_t width,uint32_t height,SurfaceRect rect,V3KMetal4Commands *commands){
        assert(commands.status==MTLCommandBufferStatusNotEnqueued&&rect.x==0&&rect.y==0&&rect.width==width&&rect.height==height);
        ++copies;auto result=image(width,height);
        [commands add:^{auto *dst=(PublicationTexture *)result;std::memset(dst.bytes.mutableBytes,marker(source),dst.bytes.length);}];return result;
    }
    id<MTLTexture> resample_publication(id<MTLTexture> source,uint32_t width,uint32_t height,SurfaceRect src,SurfaceRect dest,
        bool raw,V3KMetal4Commands *commands,const PublicationClip *clip){
        ++resamples;last_call={src,dest,raw,clip?std::optional(*clip):std::nullopt};auto result=image(width,height);
        void (^work)(void)=^{auto *dst=(PublicationTexture *)result;std::memset(dst.bytes.mutableBytes,marker(source),dst.bytes.length);};
        if(commands)[commands add:work];else work();return result;
    }
};
// STRUCTS_HOOK
static bool fail_ranges=false,region_supported=true,read_supported=true;
static unsigned reads=0;
// Deliberately one byte per logical pixel: tests cover range selection and
// untouched storage, not format packing, tiled/Morton addressing or GPU math.
static bool surface_memory_ranges(const SceGxmColorSurface &s,SurfaceRect rect,std::vector<SurfaceMemoryRange> &ranges){
    ranges.clear();if(fail_ranges){ranges.push_back({7,9});return false;}
    if(!rect.width||!rect.height||uint64_t(rect.x)+rect.width>s.width||uint64_t(rect.y)+rect.height>s.height)return false;
    for(uint32_t y=rect.y;y<rect.y+rect.height;++y)ranges.push_back({size_t(y)*s.strideInPixels+rect.x,rect.width});return true;
}
static bool read_surface_memory(id<MTLTexture> texture,const SceGxmColorSurface &s,std::span<uint8_t> output,const SurfaceRect *region=nullptr){
    ++reads;if(!read_supported||!texture||output.size()<size_t(s.strideInPixels)*s.height||(region&&!region_supported))return false;
    const auto rect=region?*region:SurfaceRect{0,0,s.width,s.height};
    for(uint32_t y=rect.y;y<rect.y+rect.height;++y)std::memset(output.data()+size_t(y)*s.strideInPixels+rect.x,marker(texture),rect.width);
    return true;
}
// FUNCTIONS_HOOK
static bool same_rect(SurfaceRect a,SurfaceRect b){return a.x==b.x&&a.y==b.y&&a.width==b.width&&a.height==b.height;}
static SceGxmColorSurface guest(uint32_t w=8,uint32_t h=6){SceGxmColorSurface s{};s.width=w;s.height=h;s.strideInPixels=w+3;s.colorFormat=SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;s.surfaceType=SCE_GXM_COLOR_SURFACE_LINEAR;return s;}
static Surface surface(SceGxmColorSurface s){Surface e;e.guest=s;e.color=image(s.width,s.height,43);e.written_tiles={0,0,s.width,s.height};e.scene_writes=e.written_tiles;e.cpu_snapshot.assign(size_t(s.strideInPixels)*s.height,7);return e;}
static void policy_cases(){
    uint64_t seed=0x193562;auto random=[&](uint32_t limit){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return uint32_t(seed%limit);};
    const auto color=image(1,1),raw=image(1,1);
    for(unsigned n=0;n<40000;++n){
        auto s=guest(n%11?1+random(24):513,n%13?1+random(17):513);s.strideInPixels=s.width+random(5);
        s.surfaceType=static_cast<SceGxmColorSurfaceType>(random(3));
        const unsigned kind=random(5);s.colorFormat=kind==1?SCE_GXM_COLOR_FORMAT_U8U8U8_BGR:kind==2?SCE_GXM_COLOR_FORMAT_U2F10F10F10_ABGR:kind==3?SCE_GXM_COLOR_FORMAT_SE5M9M9M9_BGR:SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
        Surface e;e.color=color;((PublicationTexture *)color).pixelFormat=kind==3?MTLPixelFormatRGBA16Float:MTLPixelFormatRGBA8Unorm;
        const unsigned raw_state=random(3);e.raw_color=raw_state?raw:nil;e.raw_color_invalidated=raw_state==2;
        auto rect=[&]{const auto x=random(s.width+2),y=random(s.height+2);return SurfaceWriteBounds{x,y,x+random(s.width+2),y+random(s.height+2)};};
        e.written_tiles=n%7?rect():SurfaceWriteBounds{0,0,s.width,s.height};e.scene_writes=rect();
        e.scene_macroblock_bounds=rect();if(n%3)e.scene_render_area={0,0,3,4};
        if(n%4){e.rendered_width=1+random(s.width);e.rendered_height=1+random(s.height);}
        const size_t output_size=size_t(s.strideInPixels)*s.height-(n%19==0?1:0);
        const auto plan=plan_surface_publication(e,s,output_size);
        const bool small=s.width<=512&&s.height<=512,full_tiles=e.written_tiles.x0==0&&e.written_tiles.y0==0&&e.written_tiles.x1>=s.width&&e.written_tiles.y1>=s.height;
        if(small&&s.surfaceType==SCE_GXM_COLOR_SURFACE_TILED&&!full_tiles){assert(plan.outcome==SurfacePublication::Skipped);continue;}
        const auto inside=[](uint32_t x,uint32_t y,const SurfaceWriteBounds &r){return x>=r.x0&&x<r.x1&&y>=r.y0&&y<r.y1;};
        std::vector<SurfaceMemoryRange> expected;uint32_t left=s.width,top=s.height,right=0,bottom=0;
        for(uint32_t y=0;y<s.height;++y){uint32_t first=s.width,last=0;
            for(uint32_t x=0;x<s.width;++x){bool own=true;
                if(raw_state!=1&&kind!=1&&kind!=2&&kind!=3){
                    own=inside(x,y,e.written_tiles);if(n%3)own&=inside(x,y,e.scene_macroblock_bounds);
                    if(e.rendered_width&&e.rendered_height)own&=x<e.rendered_width&&y<e.rendered_height;
                }
                if(small&&s.surfaceType==SCE_GXM_COLOR_SURFACE_LINEAR&&!full_tiles)own&=inside(x,y,e.scene_writes);
                if(own){first=std::min(first,x);last=x+1;}
            }
            if(last){expected.push_back({size_t(y)*s.strideInPixels+first,last-first});left=std::min(left,first);right=std::max(right,last);top=std::min(top,y);bottom=y+1;}
        }
        if(expected.empty()){assert(plan.outcome==SurfacePublication::Skipped);continue;}
        if(expected.back().offset+expected.back().size>output_size){assert(plan.outcome==SurfacePublication::Unavailable&&plan.written.empty());continue;}
        assert(plan.outcome==SurfacePublication::Published&&plan.bounds.x0==left&&plan.bounds.y0==top&&plan.bounds.x1==right&&plan.bounds.y1==bottom);
        assert(plan.written.size()==expected.size());for(size_t i=0;i<expected.size();++i)assert(plan.written[i].offset==expected[i].offset&&plan.written[i].size==expected[i].size);
    }
    std::cout<<"PASS color publication policy: 40000 actual production plans vs independent per-pixel ownership oracle; raw/repack exemptions, scene/macroblock/extent intersections, small tiled deferral, linear ownership, 512 boundary, padding and short output; layout range encoding simulated\n";
}
static void payload_cases(){
    Device device;size_t cases=0;
    for(bool pending:{false,true})for(bool scaled:{false,true})for(bool partial:{false,true})for(bool supports_region:{false,true}){
        __weak id weak_image,weak_commands;
        @autoreleasepool {
        auto s=guest();auto e=surface(s);e.render_scale=scaled?1.5f:1.f;if(scaled)e.color=image(12,9,43);
        if(partial)e.written_tiles=e.scene_writes={2,1,7,5};
        auto commands=pending?[V3KMetal4Commands new]:nil;std::unique_ptr<SurfaceCaster> caster;
        const auto before_copies=copies,before_resamples=resamples;
        auto payload=prepare_surface_publication(e,s,size_t(s.strideInPixels)*s.height,device,caster,commands);
        assert(copies-before_copies==unsigned(pending&&!scaled)&&resamples-before_resamples==unsigned(scaled));
        assert(payload.source&&(pending||scaled?payload.source!=e.color:payload.source==e.color));
        assert(payload.commands==commands&&payload.guest.width==8);
        if(scaled){assert(same_rect(last_call.destination,partial?SurfaceRect{2,1,5,4}:SurfaceRect{0,0,8,6}));assert(same_rect(last_call.source,partial?SurfaceRect{3,1,7,6}:SurfaceRect{0,0,12,9}));assert(!last_call.raw);}
        auto original=e.color;
        if(pending){[commands add:^{auto *t=(PublicationTexture *)original;std::memset(t.bytes.mutableBytes,99,t.bytes.length);}];[commands commit];}
        auto copied=payload;weak_image=payload.source;weak_commands=commands;
        e={};caster.reset();commands=nil;s=guest(1,1); // Mutable producer state is gone.
        std::vector<uint8_t> output(size_t(payload.guest.strideInPixels)*payload.guest.height,7);region_supported=supports_region;
        auto job=std::async(std::launch::async,[payload,&output]{@autoreleasepool{return finish_surface_publication(payload,output);}});
        assert(job.get()==SurfacePublication::Published);
        for(uint32_t y=0;y<payload.guest.height;++y)for(uint32_t x=0;x<payload.guest.strideInPixels;++x){const bool written=x<payload.guest.width&&(!partial||(x>=2&&x<7&&y>=1&&y<5));assert(output[size_t(y)*payload.guest.strideInPixels+x]==(written?43:7));}
        if(pending)assert(marker(original)==99&&marker(payload.source)==43&&payload.commands.waits==1);
        payload={};assert(weak_image);copied={};original=nil;job=std::future<SurfacePublication>{};
        }
        assert(!weak_image);assert(!weak_commands);++cases;
    }
    region_supported=true;
    std::cout<<"PASS color publication payload: "<<cases<<" actual prepare/finish cases; sync alias avoids extra snapshot, queued work owns a copy/resample before later writes, metadata and ARC owners outlive caster/cache/producer, completion-thread consumption, bounded/fallback conversion preserves all unowned bytes; GPU/copy/resample/pixel conversion simulated\n";
}
static void boundary_cases(){
    Device device;auto s=guest();auto e=surface(s);std::unique_ptr<SurfaceCaster> caster;std::vector<uint8_t> output(e.cpu_snapshot),written_bytes;std::vector<SurfaceMemoryRange> written;
    auto before_reads=reads;output[0]=9;assert(read_surface_publication(e,s,output,written,device,caster)==SurfacePublication::CpuNewer&&reads==before_reads&&written.empty());
    output=e.cpu_snapshot;e.cpu_writes.dirty=true;assert(read_surface_publication(e,s,output,written,device,caster)==SurfacePublication::CpuNewer&&reads==before_reads);e.cpu_writes.dirty=false;
    assert(read_surface_publication(e,s,output,written,device,caster)==SurfacePublication::Published&&reads==before_reads+1);
    // Failures must reach no guest store. A not-yet-submitted payload cannot wait.
    for(unsigned failure=0;failure<3;++failure){auto commands=[V3KMetal4Commands new];auto payload=prepare_surface_publication(e,s,output.size(),device,caster,commands);
        output.assign(output.size(),7);bool failed=false;
        if(failure==1){commands.fail=true;[commands commit];}else if(failure==2){[commands commit];output.resize(1);}
        try{const auto result=finish_surface_publication(payload,output);assert(failure==2&&result==SurfacePublication::Unavailable);failed=true;}catch(const std::runtime_error &){failed=true;assert(failure<2);}
        assert(failed&&std::all_of(output.begin(),output.end(),[](auto b){return b==7;}));assert(commands.waits==(failure==1));
    }
    fail_ranges=true;auto invalid=prepare_surface_publication(e,s,999,device,caster);assert(invalid.outcome==SurfacePublication::Unavailable&&invalid.written.size()==1&&!invalid.source&&!invalid.commands);fail_ranges=false;
    // Exact original clip coordinates survive preparation; sample-grid variants omit clip filtering.
    e.render_scale=1.5f;e.color=image(12,9,43);auto bits=renderer::metal::color_surface_bits(s);bits.clip_enabled=1;bits.clip_x_min=1;bits.clip_y_min=2;bits.clip_x_max=6;bits.clip_y_max=4;
    (void)prepare_surface_publication(e,s,999,device,caster);assert(last_call.clip&&same_rect(last_call.clip->guest,{1,2,6,3})&&same_rect(last_call.clip->source,{1,3,9,4}));
    e.publication_samples_x=2;(void)prepare_surface_publication(e,s,999,device,caster);assert(!last_call.clip);
    e.publication_samples_x=1;e.raw_color=image(12,9,61);(void)prepare_surface_publication(e,s,999,device,caster);assert(last_call.raw);
    e.raw_color=nil;e.scene_render_area={1,1,8,6};e.scene_macroblock_bounds={0,0,7,5};e.publication_samples_x=2;e.publication_samples_y=2;
    (void)prepare_surface_publication(e,s,999,device,caster);assert(same_rect(last_call.source,{2,2,10,7})&&same_rect(last_call.destination,{0,0,7,5}));
    std::cout<<"PASS color publication boundaries: CPU-newer suppression and prior sync behavior, exact scaled ownership/macroblock/clip/raw parameters, unsubmitted/error/short-output rejection before stores and retained invalid-range reporting; no device/GPU/app\n";
}
int main(){@autoreleasepool{policy_cases();payload_cases();boundary_cases();}}
