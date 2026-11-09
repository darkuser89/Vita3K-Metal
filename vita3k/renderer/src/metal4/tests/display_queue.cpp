// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Real queue and production GXM exports; CPU memory/kernel/render boundaries.
#include <gxm/types.h>
#include <threads/queue.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <ranges>
#include <semaphore>
#include <thread>
#include <vector>
using namespace std::chrono_literals;
using Address=uint32_t;
struct MemState {
    std::mutex mutex;
    std::map<Address,std::vector<uint8_t>> owned;
    std::map<Address,void *> borrowed;
    Address next=0x4000;
    std::atomic<unsigned> allocated=0,freed=0;
    void *resolve(Address address) {
        std::lock_guard lock(mutex);
        if(auto it=borrowed.find(address);it!=borrowed.end())return it->second;
        return owned.at(address).data();
    }
};
template<class T> struct Ptr {
    Address value=0;
    Ptr()=default;
    Ptr(Address value):value(value){}
    Address address()const{return value;}
    explicit operator bool()const{return value!=0;}
    T *get(MemState &mem)const{return value?static_cast<T *>(mem.resolve(value)):nullptr;}
};
static Address alloc(MemState &mem,uint32_t size,const char *) {
    std::lock_guard lock(mem.mutex);auto address=mem.next;mem.next+=0x100;
    mem.owned.emplace(address,std::vector<uint8_t>(size));++mem.allocated;return address;
}
static void free(MemState &mem,Address address){std::lock_guard lock(mem.mutex);assert(mem.owned.erase(address)==1);++mem.freed;}
struct SceGxmSyncObject { std::atomic<uint32_t> timestamp_ahead{0};uint32_t last_display=99; };
struct DisplayCallback {
    Address data{};Ptr<SceGxmSyncObject> old_sync,new_sync;
    uint32_t old_sync_timestamp{},new_sync_timestamp{};bool frame_predicted{};
};
struct ObservedQueue:Queue<DisplayCallback> {
    std::binary_semaphore *space=nullptr,*draining=nullptr;
    std::atomic<bool> space_armed{true},drain_armed{true},fail_push{false};
    void observed_space(){if(space&&space_armed.exchange(false))space->release();}
    bool wait_space(){observed_space();return Queue::wait_space();}
    bool push(const DisplayCallback &entry){observed_space();if(fail_push.exchange(false))abort_synchronized();return Queue::push(entry);}
    void wait_empty(){if(draining&&drain_armed.exchange(false))draining->release();Queue::wait_empty();}
};
struct SceGxmInitializeParams {
    uint32_t flags{},displayQueueMaxPendingCount{};Ptr<void> displayQueueCallback;
    uint32_t displayQueueCallbackDataSize{},parameterBufferSize{};
};
struct ThreadState {
    std::atomic<unsigned> host_waits=0,removed=0;
    struct HostWait {
        ThreadState &thread;
        explicit HostWait(ThreadState &thread):thread(thread){++thread.host_waits;}
        ~HostWait(){--thread.host_waits;}
    };
    void remove(){++removed;}
};
namespace renderer {
enum class Backend{Metal,OpenGL};enum class CommandOpcode{NewFrame};
struct Context{};
struct State { Backend current_backend=Backend::Metal;std::atomic<bool> render_abort=false;std::atomic<unsigned> frames=0; };
template<class... Args> void send_single_command(State &state,Args...){++state.frames;}
}
using renderer::Backend;
struct SceGxmContext{std::unique_ptr<renderer::Context> renderer;};
struct DisplayFrameInfo{};
struct GxmState {
    SceGxmInitializeParams params;
    ObservedQueue display_queue;
    std::mutex display_queue_add_mutex;
    bool display_queue_closed=false;
    std::atomic<uint32_t> global_timestamp{1};uint32_t last_display_global{};
    std::map<SceGxmContext *,Address> immediate_contexts,deferred_contexts;
    std::map<void *,Address> render_targets;
    Address last_immediate_context{};int display_queue_thread=1;
    std::thread display_host_thread;
};
struct EmuEnvState {
    MemState mem;GxmState gxm;
    struct {ThreadState thread;ThreadState *get_thread(int){return &thread;}} kernel;
    struct {std::atomic<bool> abort=false;} display;
    std::unique_ptr<renderer::State> renderer=std::make_unique<renderer::State>();
    SceGxmSyncObject old_sync,new_sync;
    uint8_t input[4]{1,2,3,4};
    explicit EmuEnvState(unsigned limit=1) {
        gxm.params.displayQueueCallback=Ptr<void>(1);gxm.params.displayQueueCallbackDataSize=4;
        gxm.params.displayQueueMaxPendingCount=limit;gxm.display_queue.maxPendingCount_=limit;
        mem.borrowed={{0x1000,&old_sync},{0x2000,&new_sync},{0x3000,input}};
    }
};
static DisplayFrameInfo *predict_next_image(EmuEnvState &,Address){return nullptr;}
namespace gxm {void destroy_all_contexts_metal3(EmuEnvState &,bool){}void destroy_all_render_targets_metal3(EmuEnvState &,bool){}}
// CALL_RESULT_TYPE
struct CPUState {};
static int export_sceGxmSetYuvProfile_metal3(){return 0;}
static int export_sceGxmSetYuvProfile(){return 0;}
template<auto Fn> static auto make_bridge(const char *,decltype(Fn)) {
    return [](EmuEnvState &,CPUState &,SceUID){
        CallResult result;result.count=1;
        result.regs[0]=Fn==&export_sceGxmSetYuvProfile_metal3?uint32_t(SCE_GXM_ERROR_INVALID_POINTER):uint32_t(SCE_GXM_ERROR_INVALID_VALUE);
        return result;
    };
}
#define TRACY_FUNC(...) ((void)0)
#define RET_ERROR(code) int(code)
// PRODUCTION_EXPORTS
static int add(EmuEnvState &emu,bool alias=true){return export_sceGxmDisplayQueueAddEntry_metal3(emu,1,"test",Ptr<SceGxmSyncObject>(0x1000),Ptr<SceGxmSyncObject>(alias?0x1000:0x2000),Ptr<const void>(0x3000));}
template<class Work> static auto launch(Work work){std::packaged_task<std::invoke_result_t<Work>()> task(std::move(work));auto result=task.get_future();return std::pair{std::thread(std::move(task)),std::move(result)};}
int main() {
    if(!std::getenv("VITA3K_GXM_QUEUE_WAIT_ONLY")) {
        for(unsigned n=1;n<=16;++n){EmuEnvState e;auto p=e.gxm.params;p.displayQueueMaxPendingCount=n;assert(configure(e,&p)==0);assert(e.gxm.display_queue.maxPendingCount_==n);}
        for(unsigned n:{0u,17u}){EmuEnvState e;auto p=e.gxm.params;p.displayQueueMaxPendingCount=n;assert(configure(e,&p)==int(SCE_GXM_ERROR_INVALID_VALUE));}
        EmuEnvState e;auto p=e.gxm.params;p.displayQueueCallbackDataSize=0;assert(configure(e,&p)==int(SCE_GXM_ERROR_INVALID_VALUE));
    }
    {
        EmuEnvState e;e.gxm.display_queue.push({});std::binary_semaphore entered{0};e.gxm.display_queue.space=&entered;
        auto [thread,result]=launch([&]{return add(e);});assert(entered.try_acquire_for(5s));
        const bool unchanged=e.mem.allocated==0&&e.old_sync.timestamp_ahead==0&&e.old_sync.last_display==99;
        e.gxm.display_queue.pop();assert(result.get()==0);thread.join();
        auto entry=e.gxm.display_queue.pop();assert(entry&&entry->old_sync_timestamp==0&&entry->new_sync_timestamp==0);
        assert(std::memcmp(e.mem.resolve(entry->data),e.input,4)==0);free(e.mem,entry->data);
        assert(unchanged&&"Full display queue reserved sync objects or allocated before capacity was available");
        assert(e.old_sync.timestamp_ahead==1&&e.renderer->frames==1&&!e.kernel.thread.host_waits);
    }
    for(bool alias:{false,true}){
        EmuEnvState e;e.gxm.display_queue.fail_push=true;assert(add(e,alias)==0);
        assert(e.old_sync.last_display==99&&e.new_sync.last_display==99);
        assert(e.mem.allocated==1&&e.mem.freed==1&&e.renderer->frames==0&&e.gxm.display_queue.is_aborted());
    }
    {
        EmuEnvState e(16);constexpr unsigned count=512;
        std::vector<DisplayCallback> callbacks;auto [consumer_thread,consumer]=launch([&]{for(unsigned i=0;i<count;++i){auto c=e.gxm.display_queue.pop();assert(c);callbacks.push_back(*c);free(e.mem,c->data);}});
        std::vector<std::thread> producers;for(unsigned t=0;t<8;++t)producers.emplace_back([&]{for(unsigned i=0;i<count/8;++i)assert(add(e)==0);});
        for(auto &producer:producers)producer.join();consumer.get();consumer_thread.join();
        for(unsigned i=0;i<count;++i)assert(callbacks[i].old_sync_timestamp==i&&callbacks[i].new_sync_timestamp==i);
        assert(e.old_sync.timestamp_ahead==count&&e.renderer->frames==count&&e.mem.allocated==count&&e.mem.freed==count);
        assert(!e.kernel.thread.host_waits);
    }
    {
        EmuEnvState e;e.gxm.display_queue.push({});std::binary_semaphore entered{0};e.gxm.display_queue.draining=&entered;
        auto [thread,result]=launch([&]{return export_sceGxmTerminate_metal3(e,1,"test");});assert(entered.try_acquire_for(5s));
        assert(add(e)==int(SCE_GXM_ERROR_UNINITIALIZED));assert(e.mem.allocated==0&&e.old_sync.timestamp_ahead==0);
        e.gxm.display_queue.pop();assert(result.get()==0);thread.join();assert(e.kernel.thread.removed==1);
    }
    {
        EmuEnvState e;e.gxm.display_queue.push({});std::binary_semaphore entered{0};e.gxm.display_queue.draining=&entered;
        auto [thread,result]=launch([&]{return finish(e,1,"test");});assert(entered.try_acquire_for(5s));assert(e.kernel.thread.host_waits==1);
        e.gxm.display_queue.pop();assert(result.get()==0);thread.join();assert(!e.kernel.thread.host_waits);
        e.gxm.params.displayQueueCallback={};assert(add(e)==int(SCE_GXM_ERROR_UNINITIALIZED));assert(finish(e,1,"test")==int(SCE_GXM_ERROR_UNINITIALIZED));
    }
    for(unsigned backend=0;backend<3;++backend) {
        EmuEnvState e;CPUState cpu;
        if(backend==1)e.renderer->current_backend=Backend::OpenGL;
        if(backend==2)e.renderer.reset();
        const auto result=import_sceGxmSetYuvProfile_dispatch_metal3(e,cpu,1);
        assert(result.count==1&&!result.in_s0&&result.regs[0]==uint32_t(backend?SCE_GXM_ERROR_INVALID_VALUE:SCE_GXM_ERROR_INVALID_POINTER));
    }
    std::cout<<"PASS GXM display queue: requested limits 1..16, invalid parameters, capacity before reservation/allocation, aliased/distinct abort rollback, 512 additions from 8 producers in sync order, terminate rejects additions before drain, HostWait spans Finish; production exports/queue, no GPU/app\n";
    std::cout<<"PASS YUV import dispatch: new CallResult ABI preserves return register/count in Metal, other-backend and absent-renderer routes; no GPU/app\n";
}
