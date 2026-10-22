// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Exact production enqueue/CPU conversion with a recorded GPU boundary.
#import <Foundation/Foundation.h>
#include <renderer/metal4/readback_buffers.h>
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <span>
#include <vector>
using namespace renderer::metal;
@interface Buffer : NSObject
@property(readonly) NSUInteger length;
@property(readonly) void *contents;
- (instancetype)initWithLength:(NSUInteger)length;
@end
@implementation Buffer {NSMutableData *_bytes;NSUInteger _length;}
- (instancetype)initWithLength:(NSUInteger)length {if((self=[super init])){_length=length;_bytes=[NSMutableData dataWithLength:std::min<NSUInteger>(length,16384)];}return self;}
- (NSUInteger)length{return _length;}
- (void *)contents{return _bytes.mutableBytes;}
@end
@interface Native : NSObject
@property NSUInteger maxBufferLength;
@property bool fail;
@property unsigned allocations;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
@end
@implementation Native
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {
    assert(options==MTLResourceStorageModeShared);++_allocations;
    if(_fail){_fail=false;return nil;}
    return (id<MTLBuffer>)[[Buffer alloc]initWithLength:length];
}
@end
@interface Texture : NSObject
@property NSUInteger width,height,sampleCount;
@property MTLPixelFormat pixelFormat;
@property bool failView;
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format;
@end
@implementation Texture
- (id<MTLTexture>)newTextureViewWithPixelFormat:(MTLPixelFormat)format {
    assert(format==MTLPixelFormatX32_Stencil8);return _failView?nil:(id<MTLTexture>)self;
}
@end
@interface Encoder : NSObject
@property(strong) NSString *pipeline;
@property(strong) id<MTLBuffer> buffer;
@property(strong) NSData *config;
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)dispatchThreads:(MTLSize)size threadsPerThreadgroup:(MTLSize)group;
- (void)endEncoding;
@end
@implementation Encoder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state{_pipeline=(NSString *)state;}
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index{
    assert(texture&&index<2);if(!index)assert(bool([_pipeline hasSuffix:@"_ms"])==(texture.sampleCount>1));
}
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index{assert(!offset&&!index);_buffer=buffer;}
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index{assert(index==1&&length==16);_config=[NSData dataWithBytes:bytes length:length];}
- (void)dispatchThreads:(MTLSize)size threadsPerThreadgroup:(MTLSize)group {
    const auto *c=(const uint32_t *)_config.bytes;assert(size.width==c[0]&&size.height==c[1]&&size.depth==1&&group.width==8&&group.height==8);
}
- (void)endEncoding{assert(_buffer&&_pipeline&&_config);}
@end
@interface V3KMetal4Commands : NSObject
- (Encoder *)computeCommandEncoder;
- (void)complete:(uint8_t)marker;
@end
@implementation V3KMetal4Commands {NSMutableArray<Encoder *> *_encoders;}
- (instancetype)init{if((self=[super init]))_encoders=[NSMutableArray new];return self;}
- (Encoder *)computeCommandEncoder{auto e=[Encoder new];[_encoders addObject:e];return e;}
- (void)complete:(uint8_t)marker {
    for(Encoder *e in _encoders) {
        const auto *c=(const uint32_t *)e.config.bytes;const size_t count=size_t(c[0])*c[1];
        if([e.pipeline hasPrefix:@"store_packed"])std::memset(e.buffer.contents,marker,e.buffer.length);
        else if([e.pipeline hasPrefix:@"store_mask"]) {
            auto *words=(uint32_t *)e.buffer.contents;for(size_t i=0;i<count;++i)words[i]=marker&1;
        } else {
            struct Value{float depth;uint32_t stencil;};auto *v=(Value *)e.buffer.contents;
            for(size_t i=0;i<count;++i)v[i]={float(marker)/255.f,marker};
        }
    }
}
@end
enum {SCE_GXM_DEPTH_STENCIL_FORMAT_D16,SCE_GXM_DEPTH_STENCIL_FORMAT_DF32,SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8,PACKED,MASKED};
struct SceGxmDepthStencilSurface {unsigned format;bool masked=false;unsigned get_format()const{return format;}};
static bool has_depth_mask_bit(const SceGxmDepthStencilSurface &s){return s.masked;}
// STRUCT_HOOK
// GEOMETRY_HOOK
static bool depth_memory_valid(id<MTLTexture> t,const DepthMemoryLayout &l,float scale,size_t d,size_t s) {
    return t&&scale==1&&t.width>=l.width&&t.height>=l.height&&d>=l.depth_size&&s>=l.stencil_size;
}
struct Device {Native *native;id<MTLDevice> native_device(){return (id<MTLDevice>)native;}};
struct SurfaceCaster {
    Device &device;
    ReadbackBufferPool depth_store_buffers,mask_store_buffers;
    id<MTLComputePipelineState> depth_store_pipeline,depth_store_ms_pipeline,packed_depth_store_pipeline,
        packed_depth_store_ms_pipeline,mask_store_pipeline,mask_store_ms_pipeline;
    explicit SurfaceCaster(Device &device):device(device){}
    id<MTLComputePipelineState> compute_pipeline(id<MTLComputePipelineState> __strong &,NSString *name){return (id<MTLComputePipelineState>)name;}
    bool enqueue_depth_store(id<MTLTexture>,const SceGxmDepthStencilSurface &,const DepthMemoryLayout &,float,
        size_t,size_t,id<MTLTexture>,V3KMetal4Commands *,DepthStoreReadback &);
    static void finish_depth_store(const DepthStoreReadback &,const SceGxmDepthStencilSurface &,const DepthMemoryLayout &,
        std::span<uint8_t>,std::span<uint8_t>);
};
// METHODS_HOOK
static void pool_tests(Native *native) {
    ReadbackBufferPool pool;auto a=pool.acquire((id<MTLDevice>)native,100),b=pool.acquire((id<MTLDevice>)native,200);
    auto ab=a->buffer,bb=b->buffer;auto copy=a;a.reset();
    auto c=pool.acquire((id<MTLDevice>)native,50);assert(c->buffer!=ab&&c->buffer!=bb);
    std::promise<void> release;auto gate=release.get_future().share();
    auto consumer=std::async(std::launch::async,[copy=std::move(copy),gate]()mutable{gate.wait();copy.reset();});
    native.fail=true;bool failed=false;try{(void)pool.acquire((id<MTLDevice>)native,300);}catch(const std::runtime_error &){failed=true;}
    assert(failed);release.set_value();consumer.get();
    auto d=pool.acquire((id<MTLDevice>)native,50);assert(d->buffer==ab);d.reset();b.reset();
    auto e=pool.acquire((id<MTLDevice>)native,150);assert(e->buffer==bb);
    // Large logical lengths test cache accounting without large allocations.
    ReadbackBufferPool large;auto big=large.acquire((id<MTLDevice>)native,64*1024*1024+1);auto identity=big->buffer;big.reset();
    auto again=large.acquire((id<MTLDevice>)native,64*1024*1024+1);assert(again->buffer!=identity);
    ReadbackBufferPool bounded;auto x=bounded.acquire((id<MTLDevice>)native,40*1024*1024),y=bounded.acquire((id<MTLDevice>)native,40*1024*1024);
    auto xb=x->buffer,yb=y->buffer;x.reset();y.reset();
    auto xr=bounded.acquire((id<MTLDevice>)native,40*1024*1024),yr=bounded.acquire((id<MTLDevice>)native,40*1024*1024);
    assert(xr->buffer==xb&&yr->buffer!=yb);
    std::cout<<"PASS readback pool: exclusive copied/moved leases, completion-thread release, best fit, uncached overflow, 64 MiB cached-byte budget, allocation failure preserves owners; CPU sanitizer fixture, no GPU\n";
}
int main() {
    auto native=[Native new];native.maxBufferLength=1024*1024;pool_tests(native);Device device{native};size_t cases=0;
    for(unsigned kind=0;kind<5;++kind)for(unsigned w:{2u,7u,32u})for(unsigned h:{1u,3u,32u})
    for(bool tiled:{false,true})for(bool padding:{false,true})for(unsigned samples:{1u,4u}) {
        const unsigned stride=tiled?32u+(padding?32u:0u):w+(padding?3u:0u),rows=tiled?((h+31)/32)*32:h;
        const bool packed=kind==1,masked=kind==4;const unsigned bytes=kind==0?2:4;
        DepthMemoryLayout l{w,h,stride,bytes,size_t(stride)*rows*bytes,kind==1?size_t(stride)*rows*4:kind==3?size_t(stride)*rows:0,tiled,packed};
        SceGxmDepthStencilSurface s{kind==0?SCE_GXM_DEPTH_STENCIL_FORMAT_D16:kind==1?PACKED:kind==2?SCE_GXM_DEPTH_STENCIL_FORMAT_DF32:kind==3?SCE_GXM_DEPTH_STENCIL_FORMAT_DF32_S8:MASKED,masked};
        auto texture=[Texture new];texture.width=w;texture.height=h;texture.sampleCount=samples;texture.pixelFormat=MTLPixelFormatDepth32Float_Stencil8;
        auto mask=[Texture new];mask.width=w;mask.height=h;mask.sampleCount=samples;mask.pixelFormat=MTLPixelFormatRGBA8Unorm;
        SurfaceCaster caster(device);auto ca=[V3KMetal4Commands new],cb=[V3KMetal4Commands new];DepthStoreReadback a,b;
        auto enqueue=[&](V3KMetal4Commands *commands,DepthStoreReadback &r){assert(caster.enqueue_depth_store((id<MTLTexture>)texture,s,l,1,l.depth_size,l.stencil_size,(id<MTLTexture>)mask,commands,r));};
        enqueue(ca,a);enqueue(cb,b);assert(a.depth!=b.depth&&a.depth_owner&&b.depth_owner);
        if(masked)assert(a.mask!=b.mask&&a.mask_owner&&b.mask_owner);
        [ca complete:0x11];[cb complete:0x22];
        auto check=[&](const DepthStoreReadback &r,uint8_t marker) {
            std::vector<uint8_t> depth(l.depth_size,0x7b),stencil(l.stencil_size,0x7b),expected=depth,expected_stencil=stencil;
            SurfaceCaster::finish_depth_store(r,s,l,depth,stencil);
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x) {
                const size_t address=tiled?size_t(y/32)*(stride/32)*1024+size_t(x/32)*1024+(y%32)*32+x%32:size_t(y)*stride+x;
                uint32_t word=uint32_t(marker)*0x01010101u;
                if(!r.packed_direct)word=(std::bit_cast<uint32_t>(float(marker)/255.f)&(masked?0x7fffffffu:0xffffffffu))|(masked&&(marker&1)?0x80000000u:0);
                std::memcpy(expected.data()+address*bytes,&word,bytes);
                if(l.stencil_size)expected_stencil[address*(packed?4:1)+(packed?3:0)]=marker;
            }
            assert(depth==expected&&stencil==expected_stencil);
        };
        check(a,0x11);check(b,0x22);auto copied=a;auto old_a=a.depth,old_b=b.depth;a={};
        auto cc=[V3KMetal4Commands new];DepthStoreReadback c;enqueue(cc,c);assert(c.depth!=old_a&&c.depth!=old_b);
        [cc complete:0x33];check(c,0x33);check(copied,0x11);check(b,0x22);
        copied={};b={};auto cd=[V3KMetal4Commands new];DepthStoreReadback d;enqueue(cd,d);assert(d.depth==old_a||d.depth==old_b);
        [cd complete:0x44];check(d,0x44);++cases;
    }
    assert(cases==360);
    {
        DepthMemoryLayout l{2,1,2,2,4,0,false,false};SceGxmDepthStencilSurface s{SCE_GXM_DEPTH_STENCIL_FORMAT_D16};
        auto texture=[Texture new];texture.width=2;texture.height=1;texture.sampleCount=1;
        auto commands=[V3KMetal4Commands new];DepthStoreReadback result;
        {
            SurfaceCaster temporary(device);
            assert(temporary.enqueue_depth_store((id<MTLTexture>)texture,s,l,1,4,0,nil,commands,result));
        }
        std::weak_ptr<const ReadbackBuffer> owner=result.depth_owner;
        assert(!owner.expired());[commands complete:0x5a];
        std::vector<uint8_t> bytes(4);SurfaceCaster::finish_depth_store(result,s,l,bytes,{});
        assert(bytes==std::vector<uint8_t>(4,0x5a));result={};assert(owner.expired());
    }
    std::cout<<"PASS depth readback ownership: "<<cases<<" actual enqueue/static conversion cases; two completed submissions consumed later keep distinct depth/mask bytes, copied payload blocks reuse, third busy result isolated and released owners reused, payload survives caster destruction; D16/S8D24/DF32/separate stencil/mask, linear/tiled/padding/MSAA metadata and exact guest bytes; recorded GPU boundary, no app/device/GPU\n";
}
