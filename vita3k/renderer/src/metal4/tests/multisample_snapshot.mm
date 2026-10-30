// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Actual depth_snapshot/expand_multisample with delayed queue boundary doubles.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
static unsigned submissions, waits, encodes, callbacks, reported_errors;
#define LOG_ERROR(...) (++reported_errors)
@interface SnapshotTexture : NSObject
@property NSUInteger width,height,sampleCount;
@property MTLTextureType textureType;
@property MTLPixelFormat pixelFormat;
@property uint64_t value;
@end
@implementation SnapshotTexture
@end
@class V3KMetal4Commands;
@interface SnapshotEncoder : NSObject
@property(strong) V3KMetal4Commands *commands;
@property(strong) SnapshotTexture *source,*output;
- (void)setComputePipelineState:(id)p;
- (void)setTexture:(id)t atIndex:(NSUInteger)i;
- (void)setBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i;
- (void)dispatchThreads:(MTLSize)s threadsPerThreadgroup:(MTLSize)g;
- (void)endEncoding;
@end
@interface V3KMetal4Commands : NSObject {
@public NSMutableArray<void (^)(void)> *work;
}
@property MTLCommandBufferStatus status;
@property BOOL fail;
@property(strong) NSError *error;
@property(strong) NSMutableArray<void (^)(V3KMetal4Commands *)> *handlers;
- (SnapshotEncoder *)computeCommandEncoder;
- (void)addCompletedHandler:(void (^)(V3KMetal4Commands *))handler;
- (void)commit;
- (void)waitUntilCompleted;
- (void)complete;
@end
static NSMutableArray<V3KMetal4Commands *> *pending;
@implementation V3KMetal4Commands
- (instancetype)init {
    if((self=[super init])){_status=MTLCommandBufferStatusNotEnqueued;work=[NSMutableArray new];_handlers=[NSMutableArray new];}return self;
}
- (SnapshotEncoder *)computeCommandEncoder {
    assert(_status==MTLCommandBufferStatusNotEnqueued);auto e=[SnapshotEncoder new];e.commands=self;return e;
}
- (void)addCompletedHandler:(void (^)(V3KMetal4Commands *))h {[_handlers addObject:[h copy]];}
- (void)commit {assert(_status==MTLCommandBufferStatusNotEnqueued);_status=MTLCommandBufferStatusCommitted;[pending addObject:self];++submissions;}
- (void)complete {
    assert(_status==MTLCommandBufferStatusCommitted);
    if(_fail){_status=MTLCommandBufferStatusError;_error=[NSError errorWithDomain:@"test" code:1 userInfo:nil];}
    else{for(void (^block)(void) in work)block();_status=MTLCommandBufferStatusCompleted;}
    [work removeAllObjects];for(void (^h)(V3KMetal4Commands *) in _handlers){++callbacks;h(self);}[_handlers removeAllObjects];
    [pending removeObject:self];
}
- (void)waitUntilCompleted {++waits;while(_status==MTLCommandBufferStatusCommitted)[pending.firstObject complete];}
@end
@implementation SnapshotEncoder
- (void)setComputePipelineState:(id)p {assert(p);}
- (void)setTexture:(id)t atIndex:(NSUInteger)i {assert(t);if(i==0)_source=t;else{assert(i==1);_output=t;}}
- (void)setBytes:(const void *)p length:(NSUInteger)n atIndex:(NSUInteger)i {assert(p&&n==8&&i==0);}
- (void)dispatchThreads:(MTLSize)s threadsPerThreadgroup:(MTLSize)g {assert(s.width==_output.width&&s.height==_output.height&&g.width==8);}
- (void)endEncoding {++encodes;auto src=_source,dst=_output;[_commands->work addObject:^{dst.value=src.value;}];}
@end
@interface SnapshotDevice : NSObject
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d;
@end
@implementation SnapshotDevice
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor *)d {
    assert(d.storageMode==MTLStorageModeShared);auto t=[SnapshotTexture new];t.width=d.width;t.height=d.height;
    t.textureType=d.textureType;t.sampleCount=d.sampleCount;t.pixelFormat=d.pixelFormat;return (id<MTLTexture>)t;
}
@end
struct Device {
    SnapshotDevice *native=[SnapshotDevice new];id<MTLDevice> native_device(){return (id<MTLDevice>)native;}
    bool submit_and_wait(V3KMetal4Commands *c,std::string &error){[c commit];[c waitUntilCompleted];error="GPU failure";return c.status==MTLCommandBufferStatusCompleted;}
};
static V3KMetal4Commands *surface_command_buffer(Device &,NSString *) {return [V3KMetal4Commands new];}
// BITS_HOOK
struct SurfaceCaster {
    Device &device;
    id<MTLComputePipelineState> depth_pipeline=nil,multisample_integer_pipeline=nil,multisample_depth_pipeline=nil,multisample_pipeline=nil;
    id<MTLComputePipelineState> compute_pipeline(id<MTLComputePipelineState> __strong &slot,NSString *name){slot=(id<MTLComputePipelineState>)name;return slot;}
    void expand_multisample(id<MTLTexture>,id<MTLTexture>,float,uint32_t,uint32_t,V3KMetal4Commands * // WAIT_PARAMETER
    );
    id<MTLTexture> depth_snapshot(id<MTLTexture>,bool,float,uint32_t,uint32_t,bool,V3KMetal4Commands *);
};
// EXPAND_HOOK
// SNAPSHOT_HOOK
int main(){@autoreleasepool{
    pending=[NSMutableArray new];size_t cases=0;
    for(unsigned samples:{1u,2u,4u})for(bool wait:{false,true})for(bool queued:{false,true})for(bool normalize:{false,true}) {
        Device device;SurfaceCaster caster{device};auto src=[SnapshotTexture new];
        src.width=7;src.height=3;src.sampleCount=samples;src.pixelFormat=MTLPixelFormatDepth32Float_Stencil8;
        src.textureType=samples==1?MTLTextureType2D:MTLTextureType2DMultisample;src.value=73;
        V3KMetal4Commands *commands=queued?[V3KMetal4Commands new]:nil;
        const unsigned initial=submissions,waited=waits;
        SnapshotTexture *dst=(SnapshotTexture *)caster.depth_snapshot((id<MTLTexture>)src,normalize,1,0,0,wait,commands);
        assert(submissions==initial+(queued?0:1));
        assert(waits==waited+(!queued&&wait?1:0)&&"MSAA must preserve the requested asynchronous snapshot policy");
        assert(dst.pixelFormat==(normalize?MTLPixelFormatR16Unorm:MTLPixelFormatR32Float));
        assert(dst.width==7*(samples==4?2:1)&&dst.height==3*(samples>1?2:1));
        if(queued){assert(commands.status==MTLCommandBufferStatusNotEnqueued&&dst.value==0);[commands commit];}
        if(queued||!wait){assert(dst.value==0);[pending.firstObject complete];}
        assert(dst.value==73&&pending.count==0);++cases;
    }
    // The submission retains input/output until feedback; an asynchronous GPU
    // failure reports once instead of silently appearing as a completed image.
    Device device;SurfaceCaster caster{device};__weak SnapshotTexture *weak_source;
    @autoreleasepool {
        auto src=[SnapshotTexture new];src.width=4;src.height=2;src.sampleCount=4;src.textureType=MTLTextureType2DMultisample;src.pixelFormat=MTLPixelFormatDepth32Float_Stencil8;
        weak_source=src;(void)caster.depth_snapshot((id<MTLTexture>)src,false,1,0,0,false,nil);
    }
    assert(weak_source&&pending.count==1);auto c=pending.firstObject;c.fail=YES;
    const auto reports=reported_errors;[c complete];assert(reported_errors==reports+1&&!weak_source);
    std::cout<<"PASS depth snapshots: "<<cases<<" complete production 1x/2x/4x, sync/async/pending, R16/R32 policies, delayed output, native ownership and asynchronous error callback; ASAN/UBSAN, no GPU/app\n";
}}
