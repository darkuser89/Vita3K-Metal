// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/shader_index_writer.h>
#include <cassert>
#include <atomic>
#include <fstream>
#include <future>
#include <iostream>
#include <new>
#include <semaphore>

static thread_local bool observe_append=false;
static thread_local size_t largest_append_allocation=0;
struct AllocationGate { std::binary_semaphore entered{0}, release{0}; bool armed=true; };
static thread_local AllocationGate *allocation_gate=nullptr;
void *operator new(size_t size) {
    if(observe_append)largest_append_allocation=std::max(largest_append_allocation,size);
    if(allocation_gate && std::exchange(allocation_gate->armed,false)) {
        allocation_gate->entered.release();allocation_gate->release.acquire();
    }
    if(auto *p=std::malloc(size?size:1))return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept {std::free(p);}
void *operator new[](size_t n) {return ::operator new(n);}
void operator delete[](void *p) noexcept {::operator delete(p);}
void operator delete(void *p,size_t) noexcept {::operator delete(p);}
void operator delete[](void *p,size_t) noexcept {::operator delete(p);}

using Writer = renderer::metal::ShaderIndexWriter;
static Writer::Pair pair(unsigned value) {
    Writer::Pair p{};
    for (size_t i=0;i<p.size();++i) p[i]=uint8_t(value+i);
    return p;
}
static std::vector<uint8_t> read(const std::filesystem::path &path) {
    std::ifstream in(path,std::ios::binary);assert(in);
    return {std::istreambuf_iterator<char>(in),{}};
}
static void check(const std::filesystem::path &path, const std::vector<Writer::Pair> &expected, uint32_t version, uint32_t features) {
    const auto bytes=read(path);size_t count=0;uint32_t v=0,f=0;
    const size_t header=sizeof(size_t)+2*sizeof(uint32_t);
    assert(bytes.size()==header+expected.size()*64);
    std::memcpy(&count,bytes.data(),sizeof(count));
    std::memcpy(&v,bytes.data()+sizeof(count),4);std::memcpy(&f,bytes.data()+sizeof(count)+4,4);
    assert(count==expected.size() && v==version && f==features);
    for(size_t i=0;i<count;++i) assert(std::memcmp(bytes.data()+header+i*64,expected[i].data(),64)==0);
}
int main(int argc,char **argv) {
    assert(argc==2);const std::filesystem::path root=argv[1];
    // Allocator latency before accepting a new block must not prevent shutdown
    // from acquiring the shared mutex. The delayed append rechecks stop.
    {
        Writer writer(root/"stopped-before-accept.dat",63,20,{});
        AllocationGate gate;
        std::packaged_task<bool()> append([&]{allocation_gate=&gate;const bool accepted=writer.append(pair(4));allocation_gate=nullptr;return accepted;});
        auto added=append.get_future();std::thread adding(std::move(append));
        assert(gate.entered.try_acquire_for(std::chrono::seconds(5)));
        std::packaged_task<Writer::Stats()> finish([&]{return writer.finish();});
        auto done=finish.get_future();std::thread stopping(std::move(finish));
        const bool independent=done.wait_for(std::chrono::seconds(2))==std::future_status::ready;
        gate.release.release();const auto stats=done.get();const bool accepted=added.get();stopping.join();adding.join();
        assert(independent&&"Shader index shutdown blocked behind a producer allocation");
        assert(!accepted&&!stats.writes);
    }
    if(!std::getenv("VITA3K_INDEX_BACKLOG_ONLY")) {
        std::vector<Writer::Pair> initial(8192);for(size_t i=0;i<initial.size();++i)initial[i]=pair(unsigned(i));
        auto expected=initial;expected.push_back(pair(8192));
        const auto path=root/"loaded-history.dat";
        Writer writer(path,60,17,std::move(initial),Writer::atomic_write,std::chrono::seconds(60));
        largest_append_allocation=0;observe_append=true;
        assert(writer.append(expected.back()));observe_append=false;
        const auto maximum=largest_append_allocation;
        const auto stats=writer.finish();check(path,expected,60,17);
        assert(stats.pairs==8193&&stats.error.empty());
        assert(maximum<8192*sizeof(Writer::Pair)&&"Appending one pair relocated the loaded shader history on the producer");
        std::cout<<"PASS loaded history: largest append allocation="<<maximum<<" bytes for 8192 loaded records\n";
    }
    {
        std::promise<void> entered,release;auto gate=release.get_future().share();unsigned calls=0;
        const auto path=root/"pending-backlog.dat";
        Writer writer(path,61,18,{pair(0)},[&](const auto &p,auto bytes){
            if(++calls==1){entered.set_value();gate.wait();}Writer::atomic_write(p,bytes);
        },std::chrono::milliseconds(0));
        entered.get_future().wait();std::vector<Writer::Pair> expected;expected.reserve(4097);expected.push_back(pair(0));
        largest_append_allocation=0;observe_append=true;
        for(unsigned i=1;i<=4096;++i){const auto value=pair(i);assert(writer.append(value));expected.push_back(value);}
        observe_append=false;const auto maximum=largest_append_allocation;
        release.set_value();const auto stats=writer.finish();check(path,expected,61,18);
        assert(stats.pairs==4097&&stats.error.empty());
        assert(maximum<=8192&&"Blocked writer backlog relocated a bulk pair vector under the producer mutex");
        std::cout<<"PASS blocked I/O: largest append allocation="<<maximum<<" bytes for 4096 queued records\n";
    }
    {
        std::vector<uint8_t> published;
        Writer writer(root/"concurrent.dat",64,21,{},[&](const auto &,auto bytes){published.assign(bytes.begin(),bytes.end());},std::chrono::seconds(60));
        std::vector<std::thread> producers;
        for(uint32_t t=0;t<4;++t)producers.emplace_back([&,t]{for(uint32_t i=0;i<512;++i){
            Writer::Pair value{};const uint32_t identity=t*512+i;std::memcpy(value.data(),&identity,sizeof(identity));assert(writer.append(value));
        }});
        for(auto &producer:producers)producer.join();
        const auto stats=writer.finish();assert(stats.pairs==2048&&stats.error.empty());
        assert(published.size()==sizeof(size_t)+8+2048*64);
        std::vector<bool> seen(2048,false);
        for(size_t i=0;i<2048;++i){uint32_t identity=0;std::memcpy(&identity,published.data()+sizeof(size_t)+8+i*64,4);
            assert(identity<2048&&!seen[identity]);seen[identity]=true;}
    }
    const auto path=root/"hashs-metal.dat";
    std::vector<Writer::Pair> expected{pair(1),pair(2)};
    {
        Writer writer(path,49,0xa512,expected,Writer::atomic_write,std::chrono::seconds(60));
        for(unsigned i=0;i<1000;++i) {expected.push_back(pair(i));assert(writer.append(expected.back()));}
        const auto stats=writer.finish();
        assert(stats.error.empty() && stats.writes==1 && stats.pairs==expected.size());
        assert(stats.bytes==sizeof(size_t)+8+expected.size()*64);
        assert(!writer.append(pair(9)));
    }
    check(path,expected,49,0xa512);
    const auto old=read(path);
    // A deliberately blocked disk must not hold the append mutex or expose a
    // partial file. The second write includes every addition accepted meanwhile.
    std::promise<void> writing,release;
    auto gate=release.get_future().share();
    size_t calls=0;
    {
        Writer writer(path,50,7,{pair(3)},[&](const auto &p,auto bytes) {
            if (++calls==1) {writing.set_value();gate.wait();}
            Writer::atomic_write(p,bytes);
        },std::chrono::milliseconds(0));
        writing.get_future().wait();assert(read(path)==old);
        auto adding=std::async(std::launch::async,[&] {
            for(unsigned i=0;i<1000;++i) assert(writer.append(pair(i)));
        });
        const bool nonblocking=adding.wait_for(std::chrono::seconds(5))==std::future_status::ready;
        release.set_value();adding.get();assert(nonblocking);
        auto stats=writer.finish();assert(stats.error.empty() && stats.writes==2 && stats.pairs==1001);
    }
    expected={pair(3)};for(unsigned i=0;i<1000;++i)expected.push_back(pair(i));check(path,expected,50,7);
    const auto prior=read(path);
    {
        Writer writer(path,51,8,{pair(4)},[](auto &,auto){throw std::runtime_error("disk unavailable");});
        const auto stats=writer.finish();assert(stats.error=="disk unavailable" && !stats.bytes);
        assert(read(path)==prior);
    }
    // A transient I/O failure keeps the worker's complete history for recovery.
    std::promise<void> first_failure;calls=0;
    {
        Writer writer(path,52,9,{pair(5)},[&](const auto &p,auto bytes) {
            if (++calls==1) {first_failure.set_value();throw std::runtime_error("retry");}
            Writer::atomic_write(p,bytes);
        },std::chrono::milliseconds(0));
        first_failure.get_future().wait();assert(writer.append(pair(6)));
        const auto stats=writer.finish();assert(stats.error.empty() && stats.pairs==2 && stats.writes==2);
    }
    check(path,{pair(5),pair(6)},52,9);
    // Writer destruction flushes without depending on a further frame/draw.
    const auto second=root/"game-b"/"hashs-metal.dat";
    {Writer writer(second,53,10,{pair(7)},Writer::atomic_write,std::chrono::seconds(60));assert(writer.append(pair(8)));}
    check(second,{pair(7),pair(8)},53,10);check(path,{pair(5),pair(6)},52,9);
    bool rejected=false;
    try {Writer writer(path,1,1,std::vector<Writer::Pair>(Writer::max_pairs+1));}
    catch(const std::invalid_argument &){rejected=true;}
    assert(rejected);check(path,{pair(5),pair(6)},52,9);
    // A failed atomic rename must remove its temporary file and preserve target.
    const auto target=root/"existing-directory";std::filesystem::create_directories(target);
    rejected=false;try {Writer::atomic_write(target,prior);}catch(const std::system_error &){rejected=true;}
    assert(rejected && std::filesystem::is_directory(target));
    const auto empty=root/"empty.dat";
    {Writer writer(empty,1,2,{});const auto stats=writer.finish();assert(!stats.writes&&!std::filesystem::exists(empty));}
    const auto initial_only=root/"initial-only.dat";
    {Writer writer(initial_only,62,19,{pair(8)});assert(writer.finish().pairs==1);}
    check(initial_only,{pair(8)},62,19);
    // Racing producers that allocate outside the lock must recheck the reader
    // limit before accepting the final available slot.
    {
        Writer writer(root/"limit.dat",65,22,std::vector<Writer::Pair>(Writer::max_pairs-1),
            [&](const auto &,auto bytes){
                size_t count=0;std::memcpy(&count,bytes.data(),sizeof(count));
                assert(count==Writer::max_pairs&&bytes.size()==sizeof(size_t)+8+count*64);
                const auto last=bytes[sizeof(size_t)+8+(count-1)*64];assert(last>=40&&last<44);
            },std::chrono::seconds(60));
        std::atomic<unsigned> accepted=0;std::vector<std::thread> producers;
        for(unsigned t=0;t<4;++t)producers.emplace_back([&,t]{accepted+=writer.append(pair(40+t));});
        for(auto &producer:producers)producer.join();
        assert(accepted==1&&!writer.append(pair(44)));
        const auto stats=writer.finish();assert(stats.pairs==Writer::max_pairs&&stats.error.empty());
    }
    for(const auto &entry:std::filesystem::recursive_directory_iterator(root))
        assert(entry.path().filename().string().find(".tmp.")==std::string::npos);
    std::cout<<"PASS shader index: fixed-block allocation outside shared mutex, shutdown during allocator stall, 2048 concurrent unique records, 1000 updates coalesced, byte-compatible pairs/headers, initial-only/empty shutdown, atomic failure/recovery and concurrent final reader-limit slot; no GPU/app\n";
}
