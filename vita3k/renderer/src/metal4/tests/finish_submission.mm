// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Actual finish submission/retirement prefix, CPU driver boundary doubles.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
static std::vector<std::string> events;
static unsigned commits,waits,submits,expansions;
static bool fail_expansion;
struct UploadRecord {bool submitted=false,retired=false,recycled=false;};
@interface FinishTexture : NSObject
@property uint64_t value;
@end
@implementation FinishTexture
@end
@interface V3KMetal4Commands : NSObject {
@public NSMutableArray<void (^)(void)> *work;std::shared_ptr<UploadRecord> upload;
}
@property MTLCommandBufferStatus status;
@property BOOL fail;
@property(readonly) NSError *error;
@property(readonly) double GPUStartTime,GPUEndTime;
- (void)commit;
- (void)waitUntilCompleted;
@end
static NSMutableArray<V3KMetal4Commands *> *queue;
@implementation V3KMetal4Commands
- (instancetype)init {if((self=[super init])){_status=MTLCommandBufferStatusNotEnqueued;work=[NSMutableArray new];}return self;}
- (NSError *)error {return _status==MTLCommandBufferStatusError?[NSError errorWithDomain:@"test" code:1 userInfo:nil]:nil;}
- (double)GPUStartTime {return 10;}
- (double)GPUEndTime {return 10.001;}
- (void)commit {assert(_status==MTLCommandBufferStatusNotEnqueued);_status=MTLCommandBufferStatusCommitted;++commits;events.push_back("commit");if(upload)upload->submitted=true;[queue addObject:self];}
- (void)waitUntilCompleted {
    assert(_status>=MTLCommandBufferStatusCommitted);++waits;events.push_back("wait");
    while(_status==MTLCommandBufferStatusCommitted) {
        auto next=queue.firstObject;assert(next);[queue removeObjectAtIndex:0];
        if(!next.fail)for(void (^block)(void) in next->work)block();
        [next->work removeAllObjects];next.status=next.fail?MTLCommandBufferStatusError:MTLCommandBufferStatusCompleted;
        if(next->upload)next->upload->retired=true;
    }
}
@end
struct UploadBufferArena {
    std::shared_ptr<UploadRecord> record;
    UploadBufferArena()=default;
    UploadBufferArena(UploadBufferArena &&)=default;
    UploadBufferArena &operator=(UploadBufferArena &&)=default;
    ~UploadBufferArena(){assert(!record||!record->submitted||record->retired);}
    void reset_after_completion(){assert(!record||record->retired);if(record)record->recycled=true;record.reset();events.push_back("recycle");}
};
struct Device {
    bool submit_and_wait(V3KMetal4Commands *c,std::string &error) {
        ++submits;[c commit];[c waitUntilCompleted];error="submission failed";return c.status==MTLCommandBufferStatusCompleted;
    }
};
struct State {
    struct PendingBatch {V3KMetal4Commands *commands;UploadBufferArena uploads;};
    V3KMetal4Commands *commands=nil;
    bool guest_write_watches_pending=false,expanded_color=false,scene_active=false,direct_guest_memory_used=false;
    struct {bool force_store=false;} guest_depth;
    unsigned samples=1;
    float sample_scale=2;
    struct {unsigned width=20,height=10;} guest_color;
    id<MTLTexture> color=nil,render_color=nil,raw_color=nil,raw_render_color=nil;
    UploadBufferArena uploads;
    std::deque<PendingBatch> pending_batches;
    std::vector<unsigned> visibility_results;
};
struct MetalContext {std::unique_ptr<State> impl=std::make_unique<State>();};
struct SurfaceCaster {
    explicit SurfaceCaster(Device &) {events.push_back("caster");}
    void expand_multisample(id<MTLTexture> source,id<MTLTexture> destination,float scale,unsigned w,unsigned h,V3KMetal4Commands *commands) {
        assert(commands&&commands.status==MTLCommandBufferStatusNotEnqueued&&source&&destination&&scale==2&&w==20&&h==10);
        if(fail_expansion)throw std::runtime_error("expansion failed");
        events.push_back("expand");++expansions;
        // A delayed GPU marker proves producer -> expansion ordering and that
        // the expanded bytes are not published before command retirement.
        auto src=(FinishTexture *)source,dst=(FinishTexture *)destination;
        [commands->work addObject:^{dst.value=src.value;}];
    }
};
struct RenderState {
    Device *device;std::unique_ptr<SurfaceCaster> caster;
    bool trace_finish_timing=false;
    double pending_wait_ms=0,submit_wait_ms=0,max_submit_wait_ms=0,gpu_execution_ms=0,gpu_paired_wait_ms=0,max_gpu_execution_ms=0;
    unsigned timed_pending_waits=0,timed_gpu_submits=0,timed_submits=0;
};
static void require(bool b,const std::string &error){if(!b)throw std::runtime_error(error);}
static void prepare_pass_completion(MetalContext &,Device &,std::unique_ptr<SurfaceCaster> &) {events.push_back("prepare_pass");}
// EXPAND_HOOK
struct MetalState {
    Device device;RenderState storage{&device,{}};RenderState *impl=&storage;
    bool disable_surface_sync=false;
    size_t publications_drained=0;
    void drain_guest_publications() { ++publications_drained; }
    bool finish(MetalContext &,bool,bool);
};
// FINISH_SUBMISSION_HOOK
static UploadBufferArena bind_upload(V3KMetal4Commands *commands) {
    UploadBufferArena arena;arena.record=std::make_shared<UploadRecord>();commands->upload=arena.record;return arena;
}
static void drain() {if(queue.count)[queue.lastObject waitUntilCompleted];assert(!queue.count);}
static void matrix_tests() {
    size_t cases=0;
    for(unsigned flags=0;flags<256;++flags)for(unsigned samples:{1u,2u,4u})for(unsigned prior=0;prior<3;++prior)for(bool current:{false,true}) {
        @autoreleasepool {
            queue=[NSMutableArray new];MetalState metal;MetalContext ctx;auto &s=*ctx.impl;
            const bool publish_color=flags&1,wait_without_publication=flags&2;
            s.guest_write_watches_pending=flags&4;if(flags&8)s.visibility_results={1};
            s.expanded_color=flags&16;s.samples=samples;metal.impl->trace_finish_timing=flags&128;
            if(flags&32)s.color=(id<MTLTexture>)[FinishTexture new];s.render_color=(id<MTLTexture>)[FinishTexture new];
            if(flags&64){s.raw_color=(id<MTLTexture>)[FinishTexture new];s.raw_render_color=(id<MTLTexture>)[FinishTexture new];}
            std::vector<std::shared_ptr<UploadRecord>> records;
            for(unsigned i=0;i<prior;++i){auto c=[V3KMetal4Commands new];auto arena=bind_upload(c);records.push_back(arena.record);[c commit];s.pending_batches.push_back({c,std::move(arena)});}
            std::shared_ptr<UploadRecord> active;
            if(current) {
                s.commands=[V3KMetal4Commands new];s.uploads=bind_upload(s.commands);active=s.uploads.record;records.push_back(active);
                auto color=(FinishTexture *)s.render_color,raw=(FinishTexture *)s.raw_render_color;
                [s.commands->work addObject:^{color.value=0x12345678;raw.value=0xfedcba98;}];
            }
            const bool expand=current&&samples>1&&s.expanded_color&&s.color;
            const bool wait_all=publish_color||wait_without_publication||s.guest_write_watches_pending;
            const bool deferred=current&&!wait_all&&s.visibility_results.empty();
            commits=waits=submits=expansions=0;events.clear();
            assert(!metal.finish(ctx,publish_color,wait_without_publication));
            assert(metal.publications_drained==size_t(publish_color||wait_without_publication||(flags&4)||(flags&8)));
            assert(events.front()=="prepare_pass");
            assert(commits==unsigned(current)&&submits==unsigned(current&&!deferred));
            const unsigned expected_waits=current&&!deferred ? prior+1 : wait_all ? prior : deferred&&prior==2 ? 1 : 0;
            assert(waits==expected_waits);
            assert(expansions==(expand?(s.raw_render_color?2:1):0));
            assert(!s.commands);
            const unsigned remaining=deferred?std::min(prior+1,2u):current||wait_all?0:prior;
            assert(s.pending_batches.size()==remaining);
            if(expand) {
                assert(std::find(events.begin(),events.end(),"expand")<std::find(events.begin(),events.end(),"commit"));
                assert(((FinishTexture *)s.color).value==(deferred?0:0x12345678));
            }
            if(deferred)assert(active->submitted&&!active->retired&&!active->recycled);
            assert(s.guest_write_watches_pending==bool(flags&4)&&s.visibility_results.size()==bool(flags&8));
            if(deferred&&prior==2)assert(records.front()->retired&&records.front()->recycled);
            drain();
            if(expand){assert(((FinishTexture *)s.color).value==0x12345678);if(s.raw_color)assert(((FinishTexture *)s.raw_color).value==0xfedcba98);}
            for(auto &record:records)assert(record->retired);
            s.pending_batches.clear();s.uploads.reset_after_completion();++cases;
        }
    }
    std::cout<<"PASS finish submission: "<<cases<<" actual submission/retirement prefix cases; queued normal/raw MSAA expansion, producer ordering, bounded pending uploads, publication/wait/write-watch/query gates and no premature reuse; CPU doubles only\n";
}
static void failure_tests() {
    for(unsigned failure=0;failure<3;++failure) {
        queue=[NSMutableArray new];MetalState metal;MetalContext ctx;auto &s=*ctx.impl;
        s.commands=[V3KMetal4Commands new];s.uploads=bind_upload(s.commands);auto record=s.uploads.record;
        s.samples=4;s.expanded_color=true;s.color=(id<MTLTexture>)[FinishTexture new];s.render_color=(id<MTLTexture>)[FinishTexture new];
        if(failure==2)for(unsigned i=0;i<2;++i){auto c=[V3KMetal4Commands new];c.fail=i==0;auto arena=bind_upload(c);[c commit];s.pending_batches.push_back({c,std::move(arena)});}
        fail_expansion=failure==0;s.commands.fail=failure==1;commits=waits=submits=expansions=0;
        bool threw=false;try{metal.finish(ctx,failure==1,false);}catch(const std::runtime_error &){threw=true;}
        assert(threw);
        if(failure==0)assert(!record->submitted&&!commits&&!waits&&s.commands);
        if(failure==1)assert(record->retired&&s.commands.status==MTLCommandBufferStatusError&&!record->recycled);
        if(failure==2)assert(!record->retired&&s.pending_batches.size()==2&&!s.commands);
        fail_expansion=false;drain();s.pending_batches.clear();s.uploads.record.reset();
    }
    size_t policies=0;
    for(unsigned bits=0;bits<256;++bits) {
        MetalContext ctx;auto &s=*ctx.impl;
        const bool allow_deferred=bits&1,reused_published_depth=bits&2;
        s.scene_active=bits&4;s.guest_depth.force_store=bits&8;s.direct_guest_memory_used=bits&16;
        if(bits&32)s.commands=[V3KMetal4Commands new];if(bits&64)s.visibility_results={7};s.guest_write_watches_pending=bits&128;
        // SCENE_POLICY_HOOK
        assert(defer_scene==bool((bits&1)&&(bits&4)&&(!(bits&8)||(bits&2))&&!(bits&16)));
        assert(submits_deferred==bool(defer_scene&&(bits&32)&&!(bits&64)&&!(bits&128)));++policies;
    }
    std::cout<<"PASS finish failures/policy: expansion and submitted/current/pending GPU errors retain safe ownership; "<<policies<<" actual EndScene defer/counter gates preserve force-store/direct-memory/query/write-watch boundaries\n";
}
int main(){@autoreleasepool{matrix_tests();failure_tests();}}
