// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// CPU boundary doubles. No Metal device, native queue or emulator is created.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <shader/metal4/msl_recompiler.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include <vector>
static std::vector<std::string> events;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
@interface CaptureBuffer : NSObject { @public std::vector<uint8_t> bytes; }
@property(readonly) void *contents;
@end
@implementation CaptureBuffer
- (void *)contents {return bytes.data();}
@end
@interface CaptureDevice : NSObject
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
@end
@implementation CaptureDevice
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options {
    assert(options==MTLResourceStorageModeShared);auto b=[CaptureBuffer new];b->bytes.resize(length);return (id<MTLBuffer>)b;
}
@end
struct Device {CaptureDevice *native=[CaptureDevice new];CaptureDevice *native_device(){return native;}};
@interface CaptureCommands : NSObject
@property MTLCommandBufferStatus status;
- (void)commit;
- (void)waitUntilCompleted;
@end
@implementation CaptureCommands
- (void)commit {throw std::runtime_error("unexpected submission in GPU-only preparation");}
- (void)waitUntilCompleted {throw std::runtime_error("unexpected CPU wait in GPU-only preparation");}
@end
using V3KMetal4Commands=CaptureCommands;
@interface CaptureEncoder : NSObject {
@public
    std::vector<std::function<void()>> pending;
    uint32_t guest;
    bool writes,compact;
    size_t barriers,draws;
    id<MTLBuffer> output;
    size_t output_offset;
}
- (void)flush;
- (void)orderVertexMemoryWrites;
- (void)endEncoding;
- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)slot;
- (void)drawIndexedPrimitives:(MTLPrimitiveType)type indexCount:(NSUInteger)count indexType:(MTLIndexType)indexType indexBuffer:(id<MTLBuffer>)buffer indexBufferOffset:(NSUInteger)offset instanceCount:(NSUInteger)instances baseVertex:(NSInteger)base baseInstance:(NSUInteger)instance;
@end
@implementation CaptureEncoder
- (void)flush {
    // Simulate unordered shader memory work until a barrier. Omitting the
    // production barrier reverses dependent occurrences and fails the oracle.
    for(auto at=pending.rbegin();at!=pending.rend();++at)(*at)();pending.clear();
}
- (void)orderVertexMemoryWrites {++barriers;[self flush];}
- (void)endEncoding {events.push_back("end");[self flush];}
- (void)setVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)slot {
    assert(slot==shader::metal::VERTEX_OUTPUT_CAPTURE_BUFFER);output=buffer;output_offset=offset;
}
- (void)drawIndexedPrimitives:(MTLPrimitiveType)type indexCount:(NSUInteger)count indexType:(MTLIndexType)indexType indexBuffer:(id<MTLBuffer>)buffer indexBufferOffset:(NSUInteger)offset instanceCount:(NSUInteger)instances baseVertex:(NSInteger)base baseInstance:(NSUInteger)instance {
    assert(type==MTLPrimitiveTypePoint && indexType==MTLIndexTypeUInt32 && instances==1 && base==0);
    ++draws;
    const auto *indices=(const uint32_t *)((const uint8_t *)buffer.contents+offset);
    for(size_t i=0;i<count;++i) {
        const auto index=indices[i];const auto destination=output;
        const auto at=output_offset+(compact?0:index*26*16);
        pending.emplace_back([self,index,instance,destination,at]{
            const auto before=guest;
            if(writes)guest=guest*17+index+uint32_t(instance);
            const std::array<uint32_t,4> result{writes?before:0,writes?guest:0,index,uint32_t(instance)};
            std::memcpy((uint8_t *)destination.contents+at,result.data(),sizeof(result));
        });
    }
}
@end
struct State {
    CaptureEncoder *encoder=[CaptureEncoder new];
    CaptureCommands *commands=[CaptureCommands new];
    bool pending_color_writes=false,color_guest_dirty=false,guest_write_watches_pending=true;
    bool color_clip_samplewise=false,color_clip_restore_pending=false,expanded_color=false;
    id<MTLTexture> color=nil,render_color=nil,raw_color=nil,raw_render_color=nil;
    unsigned samples=1;
    float sample_scale=2;
    struct {unsigned width=20,height=10;} guest_color;
    std::vector<unsigned> visibility_results;
    unsigned uploads_used=1234;
};
struct MetalContext {std::unique_ptr<State> impl=std::make_unique<State>();};
struct SurfaceCaster {
    SurfaceCaster(Device &) {events.push_back("caster");}
    void resolve_raw_multisample(id<MTLTexture>,id<MTLTexture>,id<MTLTexture>,CaptureCommands *commands) {
        assert(commands && commands.status==MTLCommandBufferStatusNotEnqueued);events.push_back("raw");
    }
    void expand_multisample(id<MTLTexture> source,id<MTLTexture> destination,float scale,unsigned w,unsigned h,CaptureCommands *commands) {
        assert(source && destination && scale==2 && w==20 && h==10 && commands && commands.status==MTLCommandBufferStatusNotEnqueued);
        events.push_back("expand");
    }
};
static void fill_half_pixel_strips(MetalContext &) {events.push_back("strips");}
static void restore_color_outside_clip(MetalContext &ctx,SurfaceCaster *caster) {
    events.push_back("restore");
    if(ctx.impl->color_clip_samplewise && ctx.impl->color_clip_restore_pending)assert(caster);
    ctx.impl->color_clip_restore_pending=false;
}
// PASS_HELPERS
static void boundary_tests() {
    size_t cases=0;
    for(unsigned flags=0;flags<512;++flags) for(unsigned mode=0;mode<4;++mode) {
        MetalContext ctx;Device device;std::unique_ptr<SurfaceCaster> caster;
        auto &s=*ctx.impl;
        s.pending_color_writes=flags&1;s.color_guest_dirty=flags&2;
        if(flags&4)s.visibility_results={77};
        s.color_clip_samplewise=flags&8;s.color_clip_restore_pending=flags&16;
        s.samples=flags&32?4:1;s.expanded_color=flags&64;
        if(flags&128)s.color=(id<MTLTexture>)[NSObject new];
        s.render_color=(id<MTLTexture>)[NSObject new];
        if(flags&256){s.raw_color=(id<MTLTexture>)[NSObject new];s.raw_render_color=(id<MTLTexture>)[NSObject new];}
        s.commands.status=mode==0?MTLCommandBufferStatusNotEnqueued:mode==1?MTLCommandBufferStatusCommitted:MTLCommandBufferStatusError;
        if(mode==3)s.commands=nil;
        auto commands=s.commands;
        const bool expected=mode==0 && !(flags&7);
        const bool raw=s.commands && s.raw_render_color && s.raw_color && s.color && s.samples>1 && !s.expanded_color;
        const bool expand=s.samples>1 && s.expanded_color && s.color;
        events.clear();const bool result=prepare_face_replay(ctx,device,caster);assert(result==expected);
        assert(s.commands==commands && s.uploads_used==1234 && s.guest_write_watches_pending);
        assert(s.pending_color_writes==bool(flags&1) && s.color_guest_dirty==bool(flags&2) && s.visibility_results.size()==bool(flags&4));
        if(!expected) {assert(events.empty() && s.encoder && !caster);}
        else {
            std::vector<std::string> want{"end","strips"};bool allocated=false;
            if((flags&8)&&(flags&16)){want.push_back("caster");allocated=true;}
            want.push_back("restore");
            if(raw) {if(!allocated){want.push_back("caster");allocated=true;}want.push_back("raw");}
            if(expand) {if(!allocated)want.push_back("caster");want.push_back("expand");if(s.raw_render_color)want.push_back("expand");}
            assert(events==want && !s.encoder && !s.color_clip_restore_pending);
        }
        ++cases;
    }
    std::cout << "PASS capture pass boundary: " << cases << " actual helper cases, same unsubmitted commands, guarded guest-color/query publication, raw/MSAA/clip operation order, retained write watches/uploads and no CPU wait; CPU doubles\n";
}
static void capture_tests() {
    size_t cases=0;
    for(unsigned mode=0;mode<4;++mode) for(uint32_t instances:{1u,2u,5u}) for(unsigned n=1;n<=32;++n) {
        Device device;struct Impl{Device *device;} data{&device};auto *impl=&data;
        auto encoder=[CaptureEncoder new];encoder->guest=9;
        const bool split=mode==0,capture_per_occurrence=mode<=1,compact_point_capture=mode<=2;
        encoder->writes=capture_per_occurrence;encoder->compact=compact_point_capture;
        std::vector<uint32_t> capture_vertex_indices;
        for(unsigned i=0;i<n;++i)capture_vertex_indices.push_back((i*7)%11);
        // Dense read-only capture deduplicates upstream. Keep that contract.
        if(mode==3){std::sort(capture_vertex_indices.begin(),capture_vertex_indices.end());capture_vertex_indices.erase(std::unique(capture_vertex_indices.begin(),capture_vertex_indices.end()),capture_vertex_indices.end());}
        const auto &split_capture_source_indices=capture_vertex_indices;
        auto index_storage=[CaptureBuffer new];index_storage->bytes.resize(64+capture_vertex_indices.size()*4);
        std::memcpy(index_storage->bytes.data()+64,capture_vertex_indices.data(),capture_vertex_indices.size()*4);
        struct {id<MTLBuffer> buffer;size_t offset;} capture_indices{(id<MTLBuffer>)index_storage,64};
        constexpr size_t vertex_bytes=26*16;
        const size_t captured_vertices=compact_point_capture?capture_vertex_indices.size():11;
        std::vector<id<MTLBuffer>> captured_instances,split_captured_instances;
        if(split) {
            // SPLIT_CAPTURE_LOOP
        } else {
            // ROUTED_CAPTURE_LOOP
        }
        [encoder endEncoding];
        const auto &buffers=split?split_captured_instances:captured_instances;
        assert(buffers.size()==instances);
        assert(encoder->draws==instances*(compact_point_capture?capture_vertex_indices.size():1));
        assert(encoder->barriers==(capture_per_occurrence?instances*capture_vertex_indices.size():0));
        uint32_t guest=9;
        for(uint32_t instance=0;instance<instances;++instance) for(size_t ordinal=0;ordinal<capture_vertex_indices.size();++ordinal) {
            const auto index=capture_vertex_indices[ordinal],before=guest;
            if(capture_per_occurrence)guest=guest*17+index+instance;
            const std::array<uint32_t,4> expected{capture_per_occurrence?before:0,capture_per_occurrence?guest:0,index,instance};
            const auto *bytes=(const uint8_t *)buffers[instance].contents+(compact_point_capture?ordinal:index)*vertex_bytes;
            assert(!std::memcmp(bytes,expected.data(),16));
        }
        assert(encoder->guest==guest);++cases;
    }
    std::cout << "PASS capture occurrence ordering: " << cases << " actual dense/compact/face capture loops, duplicate indices, five instances, dependent guest stores under delayed/reversed work and no extra read-only barriers; CPU doubles\n";
}
using Address=uint32_t;
struct MemState {std::unique_ptr<uint8_t[]> memory=std::make_unique<uint8_t[]>(1024);bool protected_pages=true;};
struct WatchToken {bool gpu_write_pending=false;};
struct Watcher {
    std::shared_ptr<WatchToken> owner=std::make_shared<WatchToken>();
    std::map<unsigned,std::weak_ptr<WatchToken>> tokens{{1,owner}};
    unsigned prepares=0,rearms=0;
    void prepare_gpu_writes(MemState &mem,Address address,size_t size) {
        assert(address==128 && size==16);++prepares;owner->gpu_write_pending=true;mem.protected_pages=false;
    }
    void rearm(MemState &mem) {++rearms;mem.protected_pages=true;}
    // WATCH_COMPLETE
};
static void write_watch_tests() {
    size_t cases=0;
    for(bool vertex_stores:{false,true})for(bool fragment_stores:{false,true})for(unsigned boundary=0;boundary<4;++boundary) {
        MetalContext ctx;Device device;MemState mem;
        struct DirectCache {unsigned clears=0;void clear(){++clears;}};
        struct Impl {Device *device;std::unique_ptr<SurfaceCaster> caster;Watcher surface_writes;DirectCache direct_guest_buffers;} storage{&device};
        auto *impl=&storage;
        struct {bool writes_guest_memory;} fs{fragment_stores};
        const bool shader_guest_stores=vertex_stores || fragment_stores;
        const auto base=reinterpret_cast<uintptr_t>(mem.memory.get());
        struct Range {void *data;size_t size;};
        const std::vector<Range> guest_ranges{{mem.memory.get()+128,16},
            {reinterpret_cast<void *>(base-16),16},{reinterpret_cast<void *>(base+uint64_t(UINT32_MAX)+1),16},
            {reinterpret_cast<void *>(base+UINT32_MAX-8),16}};
        ctx.impl->guest_write_watches_pending=false;
        // WRITE_PREPARE
        assert(impl->surface_writes.prepares==shader_guest_stores);
        assert(mem.protected_pages==!shader_guest_stores);
        auto finish=[&](MetalContext &ctx) {
            if(ctx.impl->guest_write_watches_pending) {
                impl->surface_writes.complete_gpu_writes(mem);ctx.impl->guest_write_watches_pending=false;
            }
            ctx.impl->color_guest_dirty=false;ctx.impl->visibility_results.clear();ctx.impl->commands=nil;
        };
        auto scene_command_buffer=[](Device &) {return [CaptureCommands new];};
        if(boundary==3) {
            finish(ctx); // CPU geometry processing is retained on this path.
            // CPU_REPLAY_PREPARE
        } else {
            if(boundary==1)ctx.impl->color_guest_dirty=true;
            if(boundary==2)ctx.impl->visibility_results={1};
            // FACE_TRANSITION
        }
        const bool writable=boundary==0?shader_guest_stores:fragment_stores;
        assert(mem.protected_pages==!writable && ctx.impl->guest_write_watches_pending==writable);
        assert(impl->surface_writes.owner->gpu_write_pending==writable);
        const unsigned prepares=unsigned(shader_guest_stores)+unsigned(boundary!=0 && fragment_stores);
        assert(impl->surface_writes.prepares==prepares && impl->direct_guest_buffers.clears==prepares);
        finish(ctx);
        assert(mem.protected_pages && !impl->surface_writes.owner->gpu_write_pending && !ctx.impl->guest_write_watches_pending);
        ++cases;
    }
    std::cout << "PASS replay write watches: " << cases << " actual preparation/callsite/completion combinations, vertex/fragment stores, pure GPU vs color/query/CPU-clip boundaries, range rejection and final rearm; protection callbacks are CPU doubles\n";
}
int main(){@autoreleasepool{boundary_tests();capture_tests();write_watch_tests();}}
