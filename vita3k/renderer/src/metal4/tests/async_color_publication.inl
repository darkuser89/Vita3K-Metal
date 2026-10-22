// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Actual color completion/collection with a simulated GPU, real worker queue,
// MemState/protection/alias writer. Guest allocation validity is controlled.
// PENDING_HOOK
struct MetalContext {
    struct Impl {
        bool scene_active=true,direct_guest_memory_used=false,guest_write_watches_pending=false;
        struct {bool force_store=false;}guest_depth;
        std::vector<int> visibility_results;
        V3KMetal4Commands *commands=nil;
        struct Batch {V3KMetal4Commands *commands;};std::vector<Batch> pending_batches;
        SceGxmColorSurface guest_color{};id<MTLTexture> color=nil;
        uint64_t color_binding_generation=1;
        bool pending_color_writes=true,color_guest_dirty=false,color_guest_current=false,color_publication_discarded=false;
    } data;Impl *impl=&data;
};
static bool owns_color_binding(const Surface &s,const MetalContext &ctx){return s.binding_generation==ctx.impl->color_binding_generation&&s.color==ctx.impl->color;}
static size_t surface_memory_size(const SceGxmColorSurface &s){return size_t(s.strideInPixels)*s.height;}
static V3KMetal4Commands *scene_command_buffer(Device &){return [V3KMetal4Commands new];}
static void update_cpu_snapshot(Surface &s,std::span<const uint8_t> bytes,std::span<const SurfaceMemoryRange> ranges){for(const auto r:ranges)std::memcpy(s.cpu_snapshot.data()+r.offset,bytes.data()+r.offset,r.size);}
struct MetalState {
    bool disable_surface_sync=false;
    struct {bool enable_memory_mapping=false;}features;
    struct Impl {
        std::shared_ptr<GuestPublicationQueue> guest_publications=GuestPublicationQueue::create();
        MemState *publication_memory=nullptr;
        std::deque<std::shared_ptr<PendingColorPublication>> color_publications;
        SurfaceWriteTracker surface_writes;
        std::map<Address,Surface> surfaces;
        std::unique_ptr<Device> device=std::make_unique<Device>();
        std::unique_ptr<SurfaceCaster> caster;
    } data;Impl *impl=&data;MetalContext *context=nullptr;
    std::mutex notification_mutex;std::condition_variable notification_ready;
    unsigned scene_ends=0;
    bool end_scene(MetalContext &ctx,bool deferred){assert(deferred);++scene_ends;[ctx.impl->commands commit];ctx.impl->pending_batches.push_back({ctx.impl->commands});ctx.impl->commands=nil;ctx.impl->scene_active=false;ctx.impl->pending_color_writes=false;ctx.impl->color_guest_dirty=true;return false;}
    void collect_color_publications();void drain_guest_publications();void check_guest_publications();
    bool defer_color_scene_completion(MetalContext &,MemState &,SceGxmNotification,SceGxmNotification);
};
// ASYNC_METHODS_HOOK
struct AsyncFixture {
    Fixture memory;MetalState state;MetalContext ctx;
    SceGxmNotification vertex{},fragment{};
    AsyncFixture(unsigned edge=0){
        auto &mem=memory.mem;state.impl->publication_memory=&mem;register_guest_publications(&mem,state.impl->guest_publications);
        auto s=guest(64,edge==3?vm_page_size/32+1:vm_page_size/64-(edge?1:0));s.strideInPixels=64;s.data=Ptr<void>(memory.address()+((edge==1||edge==3)?4:0));ctx.impl->guest_color=s;
        auto e=surface(s);e.revision=13;e.binding_generation=1;e.cpu_writes=state.impl->surface_writes.capture(mem,s.data.address(),surface_memory_size(s));
        ctx.impl->color=e.color;state.impl->surfaces[s.data.address()]=e;state.context=&ctx;
        auto commands=[V3KMetal4Commands new];commands.gate=[NSCondition new];ctx.impl->commands=commands;
        id<MTLTexture> source=e.color;auto *object=(PublicationTexture *)source;std::memset(object.bytes.mutableBytes,11,object.bytes.length);
        [commands add:^{auto *t=(PublicationTexture *)source;std::memset(t.bytes.mutableBytes,43,t.bytes.length);}];
        const Address notes=memory.address()+(edge==3?2*vm_page_size+128:vm_page_size+16);
        vertex.address=Ptr<uint32_t>(notes);vertex.value=31;fragment.address=Ptr<uint32_t>(notes+4);fragment.value=37;
        *vertex.address.get(mem)=*fragment.address.get(mem)=0;
    }
    void release(V3KMetal4Commands *commands){[commands.gate lock];commands.released=true;[commands.gate broadcast];[commands.gate unlock];}
    ~AsyncFixture(){state.context=nullptr;unregister_guest_publications(&memory.mem,state.impl->guest_publications);}
};
static void integrated_color_cases(){
    for(unsigned kind=0;kind<9;++kind){AsyncFixture f;auto &mem=f.memory.mem;auto draw=f.ctx.impl->commands;
        const auto vertex_slot=f.vertex.address,fragment_slot=f.fragment.address;
        if(kind==8){f.vertex={};f.fragment={};}
        if(kind==3)draw.fail=true;
        if(kind==4)f.fragment.address=Ptr<uint32_t>(uint32_t(mapped_size)+4);
        assert(f.state.defer_color_scene_completion(f.ctx,mem,f.vertex,f.fragment));assert(f.state.scene_ends==1);
        assert(*vertex_slot.get(mem)==0&&f.memory.bytes()[0]==7);
        assert(!f.ctx.impl->color_guest_dirty&&f.ctx.impl->color_publication_discarded);
        if(kind==1){assert(handle_access_violation_metal3(mem,f.memory.bytes().data(),true));f.memory.bytes()[0]=99;}
        if(kind==2){assert(handle_access_violation_metal3(mem,f.memory.bytes().data(),true));f.memory.bytes()[0]=7;}
        if(kind==5)++f.state.impl->surfaces[f.memory.address()].revision;
        if(kind==6)++f.state.impl->surfaces[f.memory.address()].binding_generation;
        f.release(draw);bool error=false;try{f.state.impl->guest_publications->drain();}catch(const std::runtime_error&){error=true;}
        assert(error==(kind==3||kind==4));
        if(error){assert(f.memory.bytes()[0]==7&&*vertex_slot.get(mem)==0);
            assert(f.state.impl->color_publications.front()->readback.commands.status==MTLCommandBufferStatusCompleted);continue;}
        assert(*vertex_slot.get(mem)==(kind==8?0:31)&&*fragment_slot.get(mem)==(kind==8?0:37));
        const auto result=f.state.impl->color_publications.front();assert(result->ready);
        if(kind==1||kind==2){assert(result->outcome==SurfacePublication::CpuNewer&&f.memory.bytes()[0]==(kind==1?99:7)&&f.memory.bytes()[1]==7);}
        else{assert(result->outcome==SurfacePublication::Published&&f.memory.bytes()[0]==43&&f.memory.bytes()[1]==43);}
        if(kind==7)f.memory.bytes()[0]=99; // After publication, before render-thread adoption.
        f.state.check_guest_publications();assert(f.state.impl->color_publications.empty());
        const bool adopted=kind==0||kind==8;
        assert(f.state.impl->surfaces[f.memory.address()].cpu_snapshot[0]==(adopted?43:7));
        assert(f.ctx.impl->color_guest_current==adopted);
    }
    std::cout<<"PASS async color completion: nine actual production submit/worker/collect sequences; draw before frozen readback, real RAM alias writer, both optional notifications after publication or neither requested, same-value/newer CPU stores, GPU/invalid-notification failure before stores, stale revision/binding and post-completion CPU writes refuse cache adoption; GPU and codec simulated\n";
}
static void eligibility(){
    size_t cases=0;
    for(unsigned bit:{0u,1u,2u,3u,4u,5u,6u,8u,9u,10u,11u,12u}){AsyncFixture f;auto &c=*f.ctx.impl;auto &mem=f.memory.mem;
        if(bit==0)f.state.disable_surface_sync=true;
        if(bit==1)f.state.features.enable_memory_mapping=true;
        if(bit==2)c.scene_active=false;
        if(bit==3)c.guest_depth.force_store=true;
        if(bit==4)c.direct_guest_memory_used=true;
        if(bit==5)c.guest_write_watches_pending=true;
        if(bit==6)c.visibility_results.push_back(1);
        if(bit==8)c.commands=nil;
        if(bit==9)c.color_binding_generation=2;
        if(bit==10)f.state.impl->surfaces[f.memory.address()].cpu_snapshot.clear();
        if(bit==11)c.guest_color.data=Ptr<void>(f.memory.address()+4);
        if(bit==12){c.guest_color.height=255;}
        assert(!f.state.defer_color_scene_completion(f.ctx,mem,f.vertex,f.fragment)&&f.state.scene_ends==0&&f.state.impl->guest_publications->idle());++cases;
    }
    std::cout<<"PASS async color gates: "<<cases<<" actual mapping/depth/direct/query/watch/command/binding/baseline exclusions preserve synchronous fallback; no device/GPU/app\n";
}
static void repeated_publications(){
    AsyncFixture f;auto &mem=f.memory.mem;auto first=f.ctx.impl->commands;
    assert(f.state.defer_color_scene_completion(f.ctx,mem,f.vertex,f.fragment));f.release(first);f.state.drain_guest_publications();
    for(uint8_t value:{67,91}){
        auto &entry=f.state.impl->surfaces[f.memory.address()];++entry.revision;
        auto commands=[V3KMetal4Commands new];id<MTLTexture> target=entry.color;
        [commands add:^{auto *t=(PublicationTexture *)target;std::memset(t.bytes.mutableBytes,value,t.bytes.length);}];
        f.ctx.impl->commands=commands;f.ctx.impl->scene_active=true;f.ctx.impl->pending_color_writes=true;
        assert(f.state.defer_color_scene_completion(f.ctx,mem,f.vertex,f.fragment));f.state.drain_guest_publications();
        assert(f.memory.bytes()[0]==value&&entry.cpu_snapshot[0]==value&&f.ctx.impl->color_guest_current);
    }
    assert(f.state.impl->color_publications.empty()&&f.state.impl->guest_publications->idle());
    std::cout<<"PASS async color ordering: three successive actual publications to identical RAM adopt their own predecessor baseline; renderer write watches are not misclassified as CPU writes; result/alias owners retire after collection\n";
}
static void partial_scene_cases(){
    for(unsigned edge:{1u,2u,3u}){
        AsyncFixture f(edge);auto &mem=f.memory.mem;const auto address=f.ctx.impl->guest_color.data.address();const auto size=surface_memory_size(f.ctx.impl->guest_color);
        const auto before=address-1,after=Address(address+size);mem.memory[before]=59;mem.memory[after]=61;
        auto commands=f.ctx.impl->commands;assert(f.state.defer_color_scene_completion(f.ctx,mem,f.vertex,f.fragment));
        // While the GPU is stalled, a CPU store beside the frame does not
        // touch its ownership or need to consume a frame-page write watch.
        mem.memory[after]=73;f.release(commands);f.state.drain_guest_publications();
        for(size_t i=0;i<size;++i)assert(mem.memory[address+i]==43);
        assert(mem.memory[before]==59&&mem.memory[after]==73);
        const auto &surface=f.state.impl->surfaces[address];assert(surface.cpu_snapshot[0]==43&&f.ctx.impl->color_guest_current);
        assert(WritableGuestAlias::original_is_writable(mem.memory.get()+after));
    }
    for(unsigned edge:{1u,2u}){
        AsyncFixture f(edge);auto &mem=f.memory.mem;const auto address=f.ctx.impl->guest_color.data.address();
        auto commands=f.ctx.impl->commands;assert(f.state.defer_color_scene_completion(f.ctx,mem,f.vertex,f.fragment));
        assert(mem.protect_tree.empty()); // No persistent watch on neighboring edge data.
        mem.memory[address+1]=99;f.release(commands);f.state.drain_guest_publications();
        assert(mem.memory[address]==7&&mem.memory[address+1]==99&&!f.ctx.impl->color_guest_current);
        assert(WritableGuestAlias::original_is_writable(mem.memory.get()+address));
    }
    std::cout<<"PASS async color edge scenes: three non-host-aligned publications and two newer-CPU suppressions with/without leading edge and with interior watched pages; CPU neighbor changes during stalled GPU work survive, do not suppress the frame, and matching results adopt; real RAM/protection/worker, simulated GPU/codec\n";
}
int main(){@autoreleasepool{integrated_color_cases();eligibility();repeated_publications();partial_scene_cases();}}
