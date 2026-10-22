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
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
// STATS_HOOK
// ATOMIC_HOOK
struct PipelineArchiveWrite { std::string bytes; };
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
    dispatch_semaphore_t file_slots=dispatch_semaphore_create(8),archive_slots=dispatch_semaphore_create(8);
    std::map<std::filesystem::path,std::shared_ptr<PipelineArchiveWrite>> pending_archives;
    std::map<std::filesystem::path,int> pending_render,pending_compute;
    std::map<std::filesystem::path,std::shared_ptr<const std::string>> pending_programs;
    Writer() { counters.archive_writable=true; }
    ~Writer() { drain_writes(); }
    void io_error(const char *) { ++counters.io_errors; }
    void publish_archive(std::shared_ptr<PipelineArchiveWrite> archive,const std::filesystem::path &path) {
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
    // FILE_HOOK
    // DRAIN_HOOK
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
    {
        Writer w;
        auto gate=dispatch_semaphore_create(0);
        dispatch_async(w.writer, ^{ dispatch_semaphore_wait(gate,DISPATCH_TIME_FOREVER); });
        for(unsigned n=0;n<8;++n)
            w.queue_file(folder/"program",std::to_string(n),true,folder/"variant",std::to_string(n));
        assert(w.stats().file_write_slots.acquisitions==8 && !w.stats().file_write_slots.contended);
        // Independent native/archive capacity remains available while all MSL
        // slots are occupied. Both writers still share ordered publication.
        auto archive=std::make_shared<PipelineArchiveWrite>();archive->bytes="archive";
        w.save(archive,folder/"archive");
        assert(w.stats().archive_write_slots.acquisitions==1 && !w.stats().archive_write_slots.contended);
        full_slot_observed=false;
        auto blocked=std::async(std::launch::async,[&]{observe_slots=true;w.queue_file(folder/"program","last",true,folder/"variant","last");});
        wait_for_full_slot();assert(blocked.wait_for(std::chrono::seconds(0))==std::future_status::timeout);
        { std::lock_guard lock(w.mutex);assert(w.pending_programs.size()==1 && *w.pending_programs.at(folder/"program")=="7"); }
        dispatch_semaphore_signal(gate);
        assert(blocked.wait_for(std::chrono::seconds(5))==std::future_status::ready);blocked.get();w.drain_writes();
        auto s=w.stats();assert(s.file_write_slots.acquisitions==9 && s.file_write_slots.contended==1);
        assert(s.file_write_slots.total_wait_ns>0 && s.file_write_slots.max_wait_ns==s.file_write_slots.total_wait_ns);
        assert(s.program_writes==9 && s.archive_writes==1 && !s.io_errors);
        assert(read(folder/"program")=="last" && read(folder/"variant")=="last" && read(folder/"archive")=="archive");
        assert(w.pending_programs.empty() && w.pending_archives.empty());permits(w.file_slots);permits(w.archive_slots);
    }
    {
        Writer w;auto gate=dispatch_semaphore_create(0);
        dispatch_async(w.writer, ^{dispatch_semaphore_wait(gate,DISPATCH_TIME_FOREVER);});
        std::weak_ptr<PipelineArchiveWrite> owner;
        for(unsigned n=0;n<8;++n) {
            auto a=std::make_shared<PipelineArchiveWrite>();a->bytes=std::to_string(n);owner=a;
            w.pending_render[folder/"archive"]=1;w.pending_compute[folder/"archive"]=2;
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
    // Destruction drains and publishes jobs before releasing ownership/slots.
    { Writer w;w.queue_file(folder/"destructor","complete",true); }
    assert(read(folder/"destructor")=="complete");
    for(auto &file:std::filesystem::directory_iterator(folder))assert(file.path().string().find(".tmp.")==std::string::npos);
    std::cout<<"PASS cache write slots: real 8-slot backpressure for MSL/archive queues, independent capacities, timed ninth producers without held completion mutex, latest-value atomic files/indices, ownership/disabled/error permit return, 512 concurrent writes and destructor drain; ASAN/UBSAN, no device/GPU/app\n";
}
