// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Real dispatch slots/queue and atomic files; native archive serialization is
// the only replaced boundary. No Metal device, app or GPU is involved.
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>
// STATS_HOOK
#define atomic_write original_atomic_write
// ATOMIC_HOOK
#undef atomic_write
struct FileGate {
    std::filesystem::path path;
    dispatch_semaphore_t entered=dispatch_semaphore_create(0),release=dispatch_semaphore_create(0);
    std::atomic<bool> paused=false;
};
static std::shared_ptr<FileGate> file_gate;
static bool atomic_write(const std::filesystem::path &path,std::string_view bytes) {
    if(file_gate && path==file_gate->path && !file_gate->paused.exchange(true)) {
        dispatch_semaphore_signal(file_gate->entered);
        dispatch_semaphore_wait(file_gate->release,DISPATCH_TIME_FOREVER);
    }
    return original_atomic_write(path,bytes);
}
struct PipelineArchiveWrite { std::string bytes; dispatch_semaphore_t gate=nullptr,entered=nullptr; bool fail=false; };
struct PipelineRetireGate {
    bool compute;
    dispatch_semaphore_t entered=dispatch_semaphore_create(0),release=dispatch_semaphore_create(0);
    std::atomic<bool> armed{true};
    explicit PipelineRetireGate(bool compute):compute(compute){}
};
static std::atomic<PipelineRetireGate *> pipeline_retire_gate{nullptr};
@interface RetiringPipeline : NSObject
@property BOOL compute;
@end
@implementation RetiringPipeline
- (void)dealloc {
    auto *gate=pipeline_retire_gate.load();
    if(gate && bool(_compute)==gate->compute && gate->armed.exchange(false)) {
        dispatch_semaphore_signal(gate->entered);
        dispatch_semaphore_wait(gate->release,DISPATCH_TIME_FOREVER);
    }
}
@end
static std::atomic<bool> full_slot_observed=false;
static thread_local bool observe_slots=false;
static long observed_wait(dispatch_semaphore_t slots,dispatch_time_t timeout) {
    const auto result=dispatch_semaphore_wait(slots,timeout);
    if(observe_slots && timeout==DISPATCH_TIME_NOW && result)full_slot_observed=true;
    return result;
}
static void wait_for_full_slot() {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!full_slot_observed && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
    assert(full_slot_observed);
}
struct Writer {
    std::mutex mutex;
    CacheStats counters;
    dispatch_queue_t writer=dispatch_queue_create("vita3k.cpu-cache-slots",DISPATCH_QUEUE_SERIAL);
    dispatch_queue_t file_writer=dispatch_queue_create("vita3k.cpu-cache-files",DISPATCH_QUEUE_SERIAL);
    dispatch_semaphore_t file_slots=dispatch_semaphore_create(8),archive_slots=dispatch_semaphore_create(8);
    std::map<std::filesystem::path,std::shared_ptr<PipelineArchiveWrite>> pending_archives;
    std::map<std::filesystem::path,id> pending_render,pending_compute;
    std::map<std::filesystem::path,std::shared_ptr<const std::string>> pending_programs;
    size_t queued_files=0;
    Writer() { counters.archive_writable=true; }
    ~Writer() { drain_writes(); }
    void io_error(const char *) { ++counters.io_errors; }
    void publish_archive(std::shared_ptr<PipelineArchiveWrite> archive,const std::filesystem::path &path) {
        if(archive->gate){dispatch_semaphore_signal(archive->entered);dispatch_semaphore_wait(archive->gate,DISPATCH_TIME_FOREVER);}
        if(archive->fail)throw std::runtime_error("controlled serializer failure");
        assert(atomic_write(path,archive->bytes));
        std::lock_guard lock(mutex);++counters.archive_writes;
    }
    CacheStats stats() { std::lock_guard lock(mutex);return counters; }
    // Observe the failed real NOW acquisition, so the blocked-producer test
    // does not infer scheduling from a sleep. The semaphore itself is real.
#define dispatch_semaphore_wait observed_wait
    // SLOT_HOOK
#undef dispatch_semaphore_wait
    // SAVE_HOOK
    // RETIRE_HOOK
    // FILE_HOOK
    // DRAIN_HOOK
    // INDEX_DRAIN_HOOK
};
static std::string read(const std::filesystem::path &path) {
    std::ifstream f(path,std::ios::binary);assert(f);return {std::istreambuf_iterator<char>(f),{}};
}
static void permits(dispatch_semaphore_t slots) {
    for(unsigned n=0;n<8;++n)assert(!dispatch_semaphore_wait(slots,DISPATCH_TIME_NOW));
    assert(dispatch_semaphore_wait(slots,DISPATCH_TIME_NOW));
    for(unsigned n=0;n<8;++n)dispatch_semaphore_signal(slots);
}
int main(int argc,char **argv) {
    assert(argc==2);const std::filesystem::path folder=argv[1];std::filesystem::create_directories(folder);
    for(bool compute:{false,true}) for(bool failed:{false,true}) {
        if(std::getenv("VITA3K_CACHE_RETIRE_COMPUTE_ONLY") && !compute)continue;
        Writer w;const auto path=folder/(compute?"retire-compute":"retire-render");
        @autoreleasepool {
            auto pipeline=[RetiringPipeline new];pipeline.compute=compute;
            (compute?w.pending_compute:w.pending_render)[path]=pipeline;
        }
        PipelineRetireGate gate(compute);pipeline_retire_gate=&gate;
        auto archive=std::make_shared<PipelineArchiveWrite>();archive->bytes="pipeline";archive->fail=failed;w.save(archive,path);
        assert(!dispatch_semaphore_wait(gate.entered,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC)));
        auto metadata=std::async(std::launch::async,[&]{
            auto stats=w.stats();assert(stats.archive_writable!=failed&&stats.io_errors==unsigned(failed));
            w.queue_file(folder/"retire-independent","shader",true);
            w.drain_index_writes();return stats;
        });
        const bool ready=metadata.wait_for(std::chrono::seconds(2))==std::future_status::ready;
        auto flush=std::async(std::launch::async,[&]{w.drain_writes();});
        const bool waited=flush.wait_for(std::chrono::milliseconds(100))==std::future_status::timeout;
        dispatch_semaphore_signal(gate.release);(void)metadata.get();flush.get();w.drain_writes();
        pipeline_retire_gate=nullptr;
        assert(ready&&"Independent shader/cache operations blocked on pipeline destruction");
        assert(waited&&read(folder/"retire-independent")=="shader");
        assert(w.pending_render.empty()&&w.pending_compute.empty()&&w.pending_archives.empty());
        permits(w.archive_slots);permits(w.file_slots);
    }
    // An older completed job must never detach a replacement's native owners.
    {
        Writer w;const auto path=folder/"replaced-archive";
        auto old=std::make_shared<PipelineArchiveWrite>();old->bytes="old";
        old->entered=dispatch_semaphore_create(0);old->gate=dispatch_semaphore_create(0);
        w.save(old,path);assert(!dispatch_semaphore_wait(old->entered,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC)));
        auto replacement=std::make_shared<PipelineArchiveWrite>();
        auto render=[NSObject new],compute=[NSObject new];
        {std::lock_guard lock(w.mutex);w.pending_archives[path]=replacement;w.pending_render[path]=render;w.pending_compute[path]=compute;}
        dispatch_semaphore_signal(old->gate);w.drain_writes();
        assert(w.pending_archives.at(path)==replacement&&w.pending_render.at(path)==render&&w.pending_compute.at(path)==compute);
        permits(w.archive_slots);
    }
    if(!std::getenv("VITA3K_CACHE_FILE_GATE_ONLY")) {
        Writer w;auto archive=std::make_shared<PipelineArchiveWrite>();archive->bytes="gated archive";
        archive->gate=dispatch_semaphore_create(0);archive->entered=dispatch_semaphore_create(0);
        w.save(archive,folder/"independent-archive");
        assert(!dispatch_semaphore_wait(archive->entered,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC)));
        auto index=std::async(std::launch::async,[&]{w.drain_index_writes();});
        const bool ready=index.wait_for(std::chrono::seconds(2))==std::future_status::ready;
        // Release the real writer before asserting, including old-code failure.
        dispatch_semaphore_signal(archive->gate);index.get();w.drain_writes();
        assert(ready&&"Warmup index lookup blocked on archive-only publication");
        assert(w.queued_files==0&&read(folder/"independent-archive")=="gated archive");
    }
    if(!std::getenv("VITA3K_CACHE_FILE_GATE_ONLY")) {
        Writer w;auto archive=std::make_shared<PipelineArchiveWrite>();archive->bytes="archive before index";
        archive->gate=dispatch_semaphore_create(0);archive->entered=dispatch_semaphore_create(0);
        w.save(archive,folder/"ordered-archive");
        assert(!dispatch_semaphore_wait(archive->entered,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC)));
        w.queue_file(folder/"ordered-index","index",false);
        auto index=std::async(std::launch::async,[&]{w.drain_index_writes();});
        const bool ready=index.wait_for(std::chrono::seconds(2))==std::future_status::ready;
        dispatch_semaphore_signal(archive->gate);index.get();w.drain_writes();
        assert(ready&&"Queued index publication must not wait for unrelated native archive serialization");
        assert(w.queued_files==0&&read(folder/"ordered-index")=="index");
    }
    {
        Writer w;
        file_gate=std::make_shared<FileGate>();file_gate->path=folder/"program";
        auto gate=file_gate->release;
        for(unsigned n=0;n<8;++n)
            w.queue_file(folder/"program",std::to_string(n),true,folder/"variant",std::to_string(n));
        assert(!dispatch_semaphore_wait(file_gate->entered,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC)));
        assert(w.stats().file_write_slots.acquisitions==8 && !w.stats().file_write_slots.contended);
        // Independent native/archive capacity remains available while all MSL
        // slots are occupied. Its file must complete while MSL is still gated.
        auto archive=std::make_shared<PipelineArchiveWrite>();archive->bytes="archive";
        w.save(archive,folder/"archive");
        assert(w.stats().archive_write_slots.acquisitions==1 && !w.stats().archive_write_slots.contended);
        auto archive_done=dispatch_semaphore_create(0);
        dispatch_async(w.writer, ^{dispatch_semaphore_signal(archive_done);});
        const bool native_ready=!dispatch_semaphore_wait(archive_done,dispatch_time(DISPATCH_TIME_NOW,2*NSEC_PER_SEC));
        if (!native_ready) {dispatch_semaphore_signal(gate);w.drain_writes();}
        assert(native_ready&&"Native archive publication must not wait behind queued shader files");
        assert(read(folder/"archive")=="archive");
        auto index=std::async(std::launch::async,[&]{w.drain_index_writes();});
        assert(index.wait_for(std::chrono::milliseconds(100))==std::future_status::timeout);
        full_slot_observed=false;
        auto blocked=std::async(std::launch::async,[&]{observe_slots=true;w.queue_file(folder/"program","last",true,folder/"variant","last");});
        wait_for_full_slot();assert(blocked.wait_for(std::chrono::seconds(0))==std::future_status::timeout);
        { std::lock_guard lock(w.mutex);assert(w.pending_programs.size()==1 && *w.pending_programs.at(folder/"program")=="7"); }
        dispatch_semaphore_signal(gate);
        index.get();
        assert(blocked.wait_for(std::chrono::seconds(5))==std::future_status::ready);blocked.get();w.drain_writes();
        auto s=w.stats();assert(s.file_write_slots.acquisitions==9 && s.file_write_slots.contended==1);
        assert(s.file_write_slots.total_wait_ns>0 && s.file_write_slots.max_wait_ns==s.file_write_slots.total_wait_ns);
        assert(s.program_writes==9 && s.archive_writes==1 && !s.io_errors);
        assert(read(folder/"program")=="last" && read(folder/"variant")=="last" && read(folder/"archive")=="archive");
        assert(w.pending_programs.empty() && w.pending_archives.empty());permits(w.file_slots);permits(w.archive_slots);
        file_gate.reset();
    }
    {
        Writer w;auto gate=dispatch_semaphore_create(0);
        dispatch_async(w.writer, ^{dispatch_semaphore_wait(gate,DISPATCH_TIME_FOREVER);});
        std::weak_ptr<PipelineArchiveWrite> owner;
        for(unsigned n=0;n<8;++n) {
            auto a=std::make_shared<PipelineArchiveWrite>();a->bytes=std::to_string(n);owner=a;
            w.pending_render[folder/"archive"]=[NSObject new];w.pending_compute[folder/"archive"]=[NSObject new];
            w.save(a,folder/"archive");
        }
        full_slot_observed=false;
        auto blocked=std::async(std::launch::async,[&]{observe_slots=true;auto a=std::make_shared<PipelineArchiveWrite>();a->bytes="last";w.save(a,folder/"archive");});
        wait_for_full_slot();assert(blocked.wait_for(std::chrono::seconds(0))==std::future_status::timeout);
        { std::lock_guard lock(w.mutex);assert(w.pending_archives.size()==1 && !owner.expired()); }
        dispatch_semaphore_signal(gate);
        assert(blocked.wait_for(std::chrono::seconds(5))==std::future_status::ready);blocked.get();w.drain_writes();
        auto s=w.stats();assert(s.archive_write_slots.acquisitions==9 && s.archive_write_slots.contended==1);
        assert(s.archive_write_slots.total_wait_ns>0 && s.archive_write_slots.max_wait_ns==s.archive_write_slots.total_wait_ns);
        assert(read(folder/"archive")=="last" && s.archive_writes==9);
        assert(owner.expired() && w.pending_archives.empty() && w.pending_render.empty() && w.pending_compute.empty());
        // Disabled native archive writes return their slot, preserve the old
        // complete file and leave file-writer slots independent.
        w.counters.archive_writable=false;
        w.save(std::make_shared<PipelineArchiveWrite>(),folder/"archive");w.drain_writes();
        assert(w.stats().archive_write_slots.acquisitions==10 && w.stats().archive_writes==9);
        assert(read(folder/"archive")=="last");permits(w.archive_slots);permits(w.file_slots);
    }
    {
        Writer w;const auto bad=folder/"program"/"cannot-be-directory";
        w.queue_file(bad,"unpublished",true,folder/"missing-variant","unpublished");w.drain_writes();
        assert(w.stats().io_errors==1 && !std::filesystem::exists(folder/"missing-variant"));
        assert(w.pending_programs.empty());permits(w.file_slots);
    }
    // Concurrent producers and snapshots stress the real counter mutex. File
    // publication has an independent last-value oracle for every producer.
    {
        Writer w;std::vector<std::thread> producers;
        for(unsigned t=0;t<8;++t)producers.emplace_back([&,t]{
            for(unsigned n=0;n<64;++n) {
                w.queue_file(folder/("thread"+std::to_string(t)),std::to_string(n),true);
                auto s=w.stats();assert(s.file_write_slots.contended<=s.file_write_slots.acquisitions);
                assert(s.file_write_slots.max_wait_ns<=s.file_write_slots.total_wait_ns);
            }
        });
        for(auto &t:producers)t.join();w.drain_writes();auto s=w.stats();
        assert(s.program_writes==512 && s.file_write_slots.acquisitions==512 && !s.io_errors);
        for(unsigned t=0;t<8;++t)assert(read(folder/("thread"+std::to_string(t)))=="63");
        assert(w.pending_programs.empty());permits(w.file_slots);
    }
    // A full flush must still await either class, despite independent workers.
    for(bool native:{false,true}) {
        Writer w;
        const auto path=folder/(native?"flush-archive":"flush-file");
        dispatch_semaphore_t entered,release;
        if(native) {
            auto a=std::make_shared<PipelineArchiveWrite>();a->bytes="flushed";
            a->entered=entered=dispatch_semaphore_create(0);
            a->gate=release=dispatch_semaphore_create(0);w.save(a,path);
        } else {
            file_gate=std::make_shared<FileGate>();file_gate->path=path;
            entered=file_gate->entered;release=file_gate->release;
            w.queue_file(path,"flushed",true);
        }
        assert(!dispatch_semaphore_wait(entered,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC)));
        auto flush=std::async(std::launch::async,[&]{w.drain_writes();});
        const bool waited=flush.wait_for(std::chrono::milliseconds(100))==std::future_status::timeout;
        dispatch_semaphore_signal(release);flush.get();
        assert(waited&&read(path)=="flushed");
        assert(w.pending_archives.empty()&&w.pending_programs.empty()&&w.queued_files==0);
        file_gate.reset();
    }
    // Destruction drains and publishes jobs before releasing ownership/slots.
    { Writer w;w.queue_file(folder/"destructor","complete",true); }
    assert(read(folder/"destructor")=="complete");
    for(auto &file:std::filesystem::directory_iterator(folder))assert(file.path().string().find(".tmp.")==std::string::npos);
    std::cout<<"PASS cache write slots: independent serial MSL/archive publication, gated archive cannot block actual index files, gated MSL cannot block archive completion, actual pending index files still wait, real 8-slot capacities, timed ninth producers without held completion mutex, latest-value atomic files/indices, ownership/disabled/error permit return, 512 concurrent writes, full flush awaits both workers and destructor drain; no device/GPU/app\n";
    std::cout<<"PASS pipeline retirement: actual last Objective-C render/compute owners can pause destruction while independent cache statistics and shader/index publication progress, including serializer errors; full flush/eight-slot ownership await retirement, newer replacement owners preserved; no GPU/app\n";
}
