// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Actual async scene-completion method/allocator free path, CPU boundaries.
#import <Foundation/Foundation.h>
#include <renderer/metal4/guest_publication_queue.h>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <vector>
using namespace renderer::metal;
using Address=uint32_t;
constexpr uint32_t STANDARD_PAGE_SIZE=4096;
constexpr bool PAGE_NAME_TRACKING=false;
#define LOG_CRITICAL(...)
struct Slot { bool allocated=false;uint32_t size=0; };
using AllocMemPage=Slot;
constexpr uint32_t KiB(uint32_t value) {return value*1024;}
struct GenerationMutex : std::mutex {
    std::atomic<unsigned> attempts=0;
    void lock() {++attempts;attempts.notify_all();std::mutex::lock();}
};
struct MemState {
    GenerationMutex generation_mutex;
    bool use_page_table=false;
    std::unique_ptr<uint8_t[]> memory=std::make_unique<uint8_t[]>(65536);
    std::unique_ptr<Slot[]> alloc_table=std::make_unique<Slot[]>(16);
    std::array<uint8_t *,16> page_table{};
    std::map<unsigned,std::string> page_name_map;
    struct { std::atomic<unsigned> frees=0;void free(unsigned,unsigned) {++frees;} } allocator;
    std::atomic<unsigned> decommits=0;
};
static void decommit_free_host_pages(MemState &m,uint64_t,uint64_t) {++m.decommits;}
// FREE_HOOK
struct Pointer {
    Address value=0;
    explicit operator bool() const {return value!=0;}
    Address address() const {return value;}
    uint32_t *get(MemState &m) const {return reinterpret_cast<uint32_t *>(m.memory.get()+value);}
};
struct SceGxmNotification { Pointer address;uint32_t value; };
static bool is_valid_addr_range(const MemState &m,Address a,Address b) {
    return a>=4096 && b>=a && b<=65536 && m.alloc_table[a/4096].allocated;
}
enum { MTLCommandBufferStatusCommitted=2,MTLCommandBufferStatusCompleted=4,MTLCommandBufferStatusError=5 };
@interface V3KMetal4Commands : NSObject {
    NSCondition *_condition;
    int _status;
}
@property(readonly) int status;
@property(readonly) NSError *error;
- (void)waitUntilCompleted;
- (void)finish:(bool)success;
@end
@implementation V3KMetal4Commands
- (instancetype)init { if((self=[super init])) {_condition=[NSCondition new];_status=MTLCommandBufferStatusCommitted;}return self; }
- (int)status {[_condition lock];auto s=_status;[_condition unlock];return s;}
- (NSError *)error {return [NSError errorWithDomain:@"injected GPU failure" code:1 userInfo:nil];}
- (void)waitUntilCompleted {[_condition lock];while(_status<MTLCommandBufferStatusCompleted)[_condition wait];[_condition unlock];}
- (void)finish:(bool)success {[_condition lock];_status=success?MTLCommandBufferStatusCompleted:MTLCommandBufferStatusError;[_condition broadcast];[_condition unlock];}
@end
struct MetalContext {
    struct Impl {
        bool scene_active=true,direct_guest_memory_used=false,guest_write_watches_pending=false;
        struct {bool force_store=false;}guest_depth;
        std::vector<int> visibility_results;
        struct Batch {V3KMetal4Commands *commands;};
        std::vector<Batch> pending_batches;
    } data;
    Impl *impl=&data;
};
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
struct MetalState {
    bool disable_surface_sync=true;
    struct {bool enable_memory_mapping=false;}features;
    struct Impl {
        std::shared_ptr<GuestPublicationQueue> guest_publications=GuestPublicationQueue::create();
        MemState *publication_memory=nullptr;
    } data;
    Impl *impl=&data;
    std::mutex notification_mutex;
    std::condition_variable notification_ready;
    unsigned scene_ends=0;
    void collect_color_publications() {}
    bool defer_color_scene_completion(MetalContext &,MemState &,SceGxmNotification,SceGxmNotification) {return false;}
    bool end_scene(MetalContext &ctx,bool deferred) {assert(deferred);++scene_ends;ctx.impl->scene_active=false;return false;}
    bool defer_scene_completion(MetalContext &,MemState &,SceGxmNotification,SceGxmNotification);
    void drain_guest_publications();
    void check_guest_publications();
};
// METHODS_HOOK
static void register_state(MetalState &s,MemState &mem) {
    const std::lock_guard lock(mem.generation_mutex);
    s.impl->publication_memory=&mem;register_guest_publications(&mem,s.impl->guest_publications);
    mem.alloc_table[1]={true,1};mem.alloc_table[2]={true,1};
}
static uint32_t word(MemState &mem,Address address) {uint32_t v;std::memcpy(&v,mem.memory.get()+address,4);return v;}
int main() {
    {
        // Observe the real producer at its generation-lock attempt. A free
        // path may hold this lock after its final idle check; no producer may
        // reserve/enclose a scene in that interval. No scheduler sleep needed.
        MemState mem;MetalState state;register_state(state,mem);MetalContext ctx;
        std::unique_lock<std::mutex> allocator_lock(mem.generation_mutex);
        const auto previous=mem.generation_mutex.attempts.load();
        auto producer=std::async(std::launch::async,[&]{return state.defer_scene_completion(ctx,mem,{{4096},31},{});});
        mem.generation_mutex.attempts.wait(previous);
        assert(state.impl->guest_publications->idle()&&state.scene_ends==0);
        allocator_lock.unlock();assert(producer.get());state.drain_guest_publications();
        assert(word(mem,4096)==31);unregister_guest_publications(&mem,state.impl->guest_publications);
    }
    {
        MemState mem;MetalState state;register_state(state,mem);
        std::shared_ptr<int> owner=std::make_shared<int>(17);std::weak_ptr<int> weak=owner;
        auto q=state.impl->guest_publications;
        std::promise<void> started,release;auto gate=release.get_future().share();
        q->reserve().submit([owner,&started,gate]{started.set_value();gate.wait();});owner.reset();started.get_future().wait();
        assert(!weak.expired());
        {auto canceled=q->reserve();} // A canceled reservation returns capacity.
        std::promise<void> queued;
        q->reserve().submit([&queued]{queued.set_value();});
        auto third=std::async(std::launch::async,[q]{return q->reserve();});
        assert(third.wait_for(std::chrono::milliseconds(10))==std::future_status::timeout);
        release.set_value();auto lease=third.get();queued.get_future().wait();lease.submit([]{});q->drain();
        assert(weak.expired() && q->idle());
        unregister_guest_publications(&mem,q);
    }
    size_t gates=0;
    for(unsigned bits=0;bits<256;++bits) {
        MemState mem;MetalState state;register_state(state,mem);MetalContext ctx;
        state.disable_surface_sync=!(bits&1);state.features.enable_memory_mapping=bits&2;
        ctx.data.scene_active=!(bits&4);ctx.data.guest_depth.force_store=bits&8;
        ctx.data.direct_guest_memory_used=bits&16;ctx.data.guest_write_watches_pending=bits&32;
        if(bits&64)ctx.data.visibility_results.push_back(1);
        const SceGxmNotification note{{bits&128?0u:4096u},7};
        const bool handled=state.defer_scene_completion(ctx,mem,note,{});
        assert(handled==(bits==0));state.drain_guest_publications();
        assert(state.scene_ends==(bits==0?1u:0u));
        assert(word(mem,4096)==(bits==0?7u:0u));
        unregister_guest_publications(&mem,state.impl->guest_publications);++gates;
    }
    {
        MemState mem;MetalState state;register_state(state,mem);MetalContext ctx;
        auto commands=[V3KMetal4Commands new];ctx.data.pending_batches.push_back({commands});
        SceGxmNotification v{{4096},19},f{{4100},23};
        assert(state.defer_scene_completion(ctx,mem,v,f));v.value=91;f.value=92;
        ctx.data.pending_batches.clear(); // Completion owns no context references.
        __weak V3KMetal4Commands *retained=commands;
        commands=nil; // The queued closure is now the sole command owner.
        assert(retained);
        assert(!state.impl->guest_publications->idle() && !word(mem,4096));
        auto freed=std::async(std::launch::async,[&]{free_metal3(mem,4096);});
        assert(freed.wait_for(std::chrono::milliseconds(10))==std::future_status::timeout);
        assert(mem.generation_mutex.try_lock());mem.generation_mutex.unlock();
        [retained finish:true];
        assert(freed.wait_for(std::chrono::seconds(5))==std::future_status::ready);freed.get();
        assert(word(mem,4096)==19 && word(mem,4100)==23);
        assert(mem.allocator.frees==1 && mem.decommits==1 && !mem.alloc_table[1].allocated);
        {std::lock_guard lock(mem.generation_mutex);mem.alloc_table[1]={true,1};std::memset(mem.memory.get()+4096,0xa5,4096);}
        state.drain_guest_publications();assert(word(mem,4096)==0xa5a5a5a5);
        assert(!retained);
        unregister_guest_publications(&mem,state.impl->guest_publications);
    }
    {
        // Both targets must validate before the first store, so a malformed
        // fragment address cannot partially publish a valid vertex address.
        MemState mem;MetalState state;register_state(state,mem);MetalContext ctx;
        assert(state.defer_scene_completion(ctx,mem,{{4096},19},{{65535},23}));
        bool rejected=false;
        try {state.drain_guest_publications();}catch(const std::runtime_error &){rejected=true;}
        assert(rejected && !word(mem,4096) && state.impl->guest_publications->idle());
        unregister_guest_publications(&mem,state.impl->guest_publications);
    }
    {
        // A job cannot signal a later scene after an earlier GPU failure.
        MemState mem;MetalState state;register_state(state,mem);MetalContext ctx;
        auto failed=[V3KMetal4Commands new];ctx.data.pending_batches.push_back({failed});
        assert(state.defer_scene_completion(ctx,mem,{{4096},1},{}));
        MetalContext next;assert(state.defer_scene_completion(next,mem,{{4100},2},{}));
        [failed finish:false];
        state.impl->guest_publications->drain(false);
        assert(!word(mem,4096) && !word(mem,4100));
        bool reported=false;try {state.check_guest_publications();}catch(const std::runtime_error &){reported=true;}
        assert(reported);bool rejected=false;try{state.impl->guest_publications->reserve();}catch(const std::runtime_error &){rejected=true;}
        assert(rejected);unregister_guest_publications(&mem,state.impl->guest_publications);
    }
    {
        // Reservation/registration versus free: the recheck under generation
        // catches a producer that reserves between the initial drain and lock.
        MemState mem;auto q=GuestPublicationQueue::create();
        {std::lock_guard lock(mem.generation_mutex);register_guest_publications(&mem,q);mem.alloc_table[1]={true,1};}
        auto reservation=q->reserve();
        auto freed=std::async(std::launch::async,[&]{free_metal3(mem,4096);});
        assert(freed.wait_for(std::chrono::milliseconds(10))==std::future_status::timeout);
        {std::lock_guard lock(mem.generation_mutex);reservation.submit([&]{std::lock_guard lock(mem.generation_mutex);mem.memory[4096]=42;});}
        assert(freed.wait_for(std::chrono::seconds(5))==std::future_status::ready);freed.get();
        assert(mem.memory[4096]==42 && mem.allocator.frees==1);unregister_guest_publications(&mem,q);
    }
    assert(guest_publication_registry.empty());
    std::cout<<"PASS guest publication: actual async scene method, "<<gates
        <<" eligibility combinations, atomic reservation at observed allocator-lock boundary, bounded FIFO/reservations/cancel/owner release, immutable notifications after simulated GPU completion, actual private free path waits outside generation lock, reuse protection and failed-GPU cancellation; CPU sanitizer fixture, no device/GPU/app\n";
}
