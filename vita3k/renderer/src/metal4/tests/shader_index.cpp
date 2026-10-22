// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/shader_index_writer.h>
#include <cassert>
#include <fstream>
#include <future>
#include <iostream>

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
    for(const auto &entry:std::filesystem::recursive_directory_iterator(root))
        assert(entry.path().filename().string().find(".tmp.")==std::string::npos);
    std::cout<<"PASS shader index: 1000 updates coalesced, append during blocked I/O, byte-compatible headers/pairs, atomic failure/recovery, shutdown and reader bound\n";
}
