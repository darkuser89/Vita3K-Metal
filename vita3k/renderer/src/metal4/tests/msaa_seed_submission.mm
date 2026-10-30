// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Production MSAA encoder and renderer call sites with delayed driver doubles.
// This tests submission ordering/ownership, not GPU shader arithmetic.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
static unsigned submits, command_buffers, pipeline_builds, seeds;
static bool fail_encoder, fail_pipeline;
struct Words { uint64_t value = 0; };
@interface SeedTexture : NSObject {
@public std::shared_ptr<Words> words;
}
@property NSUInteger width, height, sampleCount;
@property MTLTextureType textureType;
@property MTLPixelFormat pixelFormat;
- (id)newTextureViewWithPixelFormat:(MTLPixelFormat)format;
@end
@implementation SeedTexture
- (id)newTextureViewWithPixelFormat:(MTLPixelFormat)format {
    auto view=[SeedTexture new];view->words=words;
    view.width=_width;view.height=_height;view.sampleCount=_sampleCount;
    view.textureType=_textureType;view.pixelFormat=format;return view;
}
@end
static id<MTLTexture> image(unsigned samples, MTLPixelFormat format, unsigned width=8, unsigned height=4) {
    auto t=[SeedTexture new];t->words=std::make_shared<Words>();
    t.sampleCount=samples;t.pixelFormat=format;t.width=width;t.height=height;
    t.textureType=samples==1?MTLTextureType2D:MTLTextureType2DMultisample;return (id<MTLTexture>)t;
}
@interface SeedPipelineDescriptor : NSObject
@property(strong) id vertexFunction, fragmentFunction;
@property(strong) MTLRenderPipelineColorAttachmentDescriptorArray *colorAttachments;
@property NSUInteger rasterSampleCount;
@end
@implementation SeedPipelineDescriptor
- (instancetype)init {
    if ((self=[super init])) _colorAttachments=[MTLRenderPipelineDescriptor new].colorAttachments;
    return self;
}
@end
@class V3KMetal4Commands;
@interface SeedEncoder : NSObject
@property(strong) V3KMetal4Commands *commands;
@property(strong) id<MTLTexture> source,destination;
- (void)setRenderPipelineState:(id)pipeline;
- (void)setFragmentTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)drawPrimitives:(MTLPrimitiveType)type vertexStart:(NSUInteger)start vertexCount:(NSUInteger)count;
- (void)endEncoding;
@end
@interface V3KMetal4Commands : NSObject {
@public NSMutableArray<void (^)(void)> *work;
}
@property MTLCommandBufferStatus status;
- (SeedEncoder *)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)pass;
- (void)complete;
@end
@implementation V3KMetal4Commands
- (instancetype)init {
    if ((self=[super init])) { _status=MTLCommandBufferStatusNotEnqueued;work=[NSMutableArray new];++command_buffers; }
    return self;
}
- (SeedEncoder *)renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor *)pass {
    assert(_status==MTLCommandBufferStatusNotEnqueued);
    if(fail_encoder)return nil;
    assert(pass.colorAttachments[0].loadAction==MTLLoadActionDontCare);
    assert(pass.colorAttachments[0].storeAction==MTLStoreActionStore);
    auto e=[SeedEncoder new];e.commands=self;e.destination=pass.colorAttachments[0].texture;return e;
}
- (void)complete {
    assert(_status==MTLCommandBufferStatusNotEnqueued);
    _status=MTLCommandBufferStatusCommitted;
    for(void (^block)(void) in work)block();
    [work removeAllObjects];_status=MTLCommandBufferStatusCompleted;
}
@end
@implementation SeedEncoder
- (void)setRenderPipelineState:(id)p { assert(p); }
- (void)setFragmentTexture:(id<MTLTexture>)t atIndex:(NSUInteger)i { assert(i==0);_source=t; }
- (void)setFragmentBytes:(const void *)bytes length:(NSUInteger)n atIndex:(NSUInteger)i {
    assert(bytes && n==(i==0?16:8));
}
- (void)drawPrimitives:(MTLPrimitiveType)t vertexStart:(NSUInteger)s vertexCount:(NSUInteger)n {
    assert(t==MTLPrimitiveTypeTriangle&&s==0&&n==3);++seeds;
    SeedTexture *src=(SeedTexture *)_source,*dst=(SeedTexture *)_destination;
    [_commands->work addObject:^{ dst->words->value=src->words->value; }];
}
- (void)endEncoding {}
@end
struct Device {
    bool submit_and_wait(V3KMetal4Commands *c,std::string &) { ++submits;[c complete];return true; }
    id<MTLRenderPipelineState> create_pipeline(SeedPipelineDescriptor *d,std::string &error) {
        ++pipeline_builds;assert(d.vertexFunction&&d.fragmentFunction);
        if(fail_pipeline){error="pipeline";return nil;}return (id<MTLRenderPipelineState>)[NSObject new];
    }
};
static V3KMetal4Commands *scene_command_buffer(Device &) {return [V3KMetal4Commands new];}
static V3KMetal4Commands *surface_command_buffer(Device &,NSString *) {return [V3KMetal4Commands new];}
static id<MTLFunction> metal4_function(id<MTLLibrary>,NSString *name) {return (id<MTLFunction>)name;}
// BITS_HOOK
struct SurfaceCaster {
    Device &device;
    std::map<std::pair<uint32_t,uint32_t>,id<MTLRenderPipelineState>> seed_pipelines;
    explicit SurfaceCaster(Device &d):device(d){}
    id<MTLLibrary> surface_library(){return nil;}
    void seed_multisample(id<MTLTexture>,id<MTLTexture>,float,bool,uint32_t=0,uint32_t=0,V3KMetal4Commands * = nil);
};
// SEED_HOOK
struct Address {unsigned value=1;unsigned address()const{return value;}explicit operator bool()const{return value;}};
struct Surface {Address data;unsigned width=8,height=4;};
struct Entry {Surface guest;id<MTLTexture> color,raw_color;bool multisample_dirty=true,raw_multisample_dirty=true;};
struct ContextState {
    bool encoder=false,expanded_color=false;
    unsigned samples=2;
    float sample_scale=1;
    Surface guest_color;
    id<MTLTexture> color,render_color,raw_render_color;
    V3KMetal4Commands *commands=nil;
};
struct MetalContext {std::unique_ptr<ContextState> impl=std::make_unique<ContextState>();};
struct MetalState {
    struct Impl {std::unique_ptr<Device> device=std::make_unique<Device>();std::unique_ptr<SurfaceCaster> caster;std::map<unsigned,Entry> surfaces;};
    std::unique_ptr<Impl> impl=std::make_unique<Impl>();
    void draw_seed(MetalContext &ctx) {
        // DRAW_HOOK
    }
    void bind_seed(MetalContext &ctx,bool seed_color_samples) {
        const auto &surface=ctx.impl->guest_color;
        // set_context allocates this scene command before the seed block.
        if(!ctx.impl->commands)ctx.impl->commands=scene_command_buffer(*impl->device);
        // BIND_HOOK
    }
};
int main() { @autoreleasepool {
    size_t cases=0;
    for(unsigned samples:{2u,4u})for(bool expanded:{false,true})for(float scale:{0.5f,1.f,2.f})
    for(auto format:{MTLPixelFormatRGBA8Unorm,MTLPixelFormatRGBA16Float})for(bool raw:{false,true})
    for(bool existing:{false,true})for(bool bind:{false,true}) {
        MetalState state;MetalContext ctx;auto &s=*ctx.impl;auto &entry=state.impl->surfaces[1];
        s.samples=samples;s.expanded_color=expanded;s.sample_scale=scale;
        const unsigned w=expanded?8/(samples/2):8,h=expanded?2:4;
        s.color=entry.color=image(1,format);entry.raw_color=image(1,format);
        s.render_color=image(samples,format,w,h);if(raw)s.raw_render_color=image(samples,format,w,h);
        // An unsubmitted producer is the critical case: a private synchronous
        // seed would execute before it and read the old value.
        if(existing)s.commands=scene_command_buffer(*state.impl->device);
        auto old=s.commands;
        SeedTexture *source=(SeedTexture *)s.color,*raw_source=(SeedTexture *)entry.raw_color;
        if(existing)[s.commands->work addObject:^{source->words->value=73;raw_source->words->value=91;}];
        else {source->words->value=73;raw_source->words->value=91;}
        submits=0;const unsigned before=command_buffers;
        if(bind)state.bind_seed(ctx,true);else state.draw_seed(ctx);
        assert(submits==0 && "MSAA seed must not wait/submit outside its consumer scene");
        assert(s.commands && (!old||old==s.commands));
        assert(command_buffers==before+(existing?0:1));
        assert(!entry.multisample_dirty);
        if(!bind&&raw)assert(!entry.raw_multisample_dirty);
        SeedTexture *output=(SeedTexture *)s.render_color,*raw_output=(SeedTexture *)s.raw_render_color;
        assert(output->words->value==0);
        const bool check_raw=!bind&&raw;
        [s.commands->work addObject:^{assert(output->words->value==73);if(check_raw)assert(raw_output->words->value==91);}];
        // Already seeded calls enqueue no duplicate work.
        const auto encoded=seeds;
        if(bind)state.bind_seed(ctx,false);else state.draw_seed(ctx);
        assert(seeds==encoded && submits==0);
        [s.commands complete];++cases;
    }
    // Dirty ownership changes only after successful encoding, so retry remains
    // possible after an encoder/pipeline failure; no partial submission occurs.
    for(bool pipeline_failure:{false,true}) {
        MetalState state;MetalContext ctx;auto &s=*ctx.impl;auto &e=state.impl->surfaces[1];
        e.color=s.color=image(1,MTLPixelFormatRGBA8Unorm);s.render_color=image(2,MTLPixelFormatRGBA8Unorm);
        fail_encoder=!pipeline_failure;fail_pipeline=pipeline_failure;bool rejected=false;
        try{state.draw_seed(ctx);}catch(const std::runtime_error &){rejected=true;}
        assert(rejected&&e.multisample_dirty&&submits==0);
        fail_encoder=fail_pipeline=false;state.draw_seed(ctx);assert(!e.multisample_dirty);[s.commands complete];
    }
    std::cout<<"PASS MSAA seed: "<<cases<<" production bind/draw encoder cases, 2x/4x, expanded/resolved, integer/F16 views, scales, delayed producer-seed-consumer order, no private submissions, PSO/encoder failure retry; ASAN/UBSAN, no GPU/app\n";
} }
