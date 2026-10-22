// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Real host page protection/remapping. Allocator validity is a controlled boundary.
#include <renderer/metal4/guest_memory_publication.h>
#include <util/align.h>
#include <sys/mman.h>
#include <csignal>
#include <thread>
#include <cassert>
#include <iostream>
using namespace renderer::metal;
static size_t mapped_size=0;
static bool allocation_valid=true;
static Address valid_begin=0,valid_end=0;
static std::atomic<unsigned> critical_logs=0;
const uint64_t TOTAL_MEM_SIZE=uint64_t(vm_page_size)*4;
constexpr unsigned STANDARD_PAGE_SIZE=4096;
constexpr bool LOG_PROTECT=false;
#define LOG_CRITICAL(...) (++critical_logs)
namespace fmt {template<class... T> void print(T&&...) {}}
static Address log_hex(Address a){return a;}
bool is_valid_addr_range(const MemState &m,Address a,Address b){return allocation_valid&&a>=m.host_page_size&&b>=a&&b<=mapped_size
    &&(!valid_begin||(a>=valid_begin&&b<=valid_end));}
bool is_valid_addr_range_metal3(const MemState &m,Address a,Address b){return is_valid_addr_range(m,a,b);}
bool is_valid_addr(const MemState &m,Address a){return is_valid_addr_range(m,a,a+1);}
void protect_inner(MemState &m,Address a,uint32_t s,MemPerm p){assert(a%m.host_page_size==0&&s%m.host_page_size==0);assert(!mprotect(m.memory.get()+a,s,p==MemPerm::None?PROT_NONE:p==MemPerm::ReadOnly?PROT_READ:PROT_READ|PROT_WRITE));}
void unprotect_inner(MemState &m,Address a,uint32_t s){
    if(m.use_page_table)for(const auto &[host,range]:m.external_mapping)if(a>=range.address&&uint64_t(a)+s<=uint64_t(range.address)+range.size){
        assert(!mprotect((void *)(host+a-range.address),s,PROT_READ|PROT_WRITE));return;
    }
    protect_inner(m,a,s,MemPerm::ReadWrite);
}
// PROTECTION_HOOK
bool add_protect(MemState &m,Address a,uint32_t s,MemPerm p,const ProtectCallback &f){return add_protect_metal3(m,a,s,p,f);}
bool handle_access_violation(MemState &m,uint8_t *p,bool write)noexcept{return handle_access_violation_metal3(m,p,write);}
struct Range{size_t offset,size;};
struct Fixture{
    MemState mem;SurfaceWriteTracker tracker;
    Fixture(){mem.host_page_size=vm_page_size;mapped_size=vm_page_size*4;allocation_valid=true;valid_begin=valid_end=0;auto *p=(uint8_t *)mmap(nullptr,mapped_size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);assert(p!=MAP_FAILED);mem.memory=Memory(p,[](auto *q){assert(!munmap(q,mapped_size));});std::memset(p,7,mapped_size);}
    Address address()const{return mem.host_page_size;}
    std::span<uint8_t> bytes(){return {mem.memory.get()+address(),mem.host_page_size};}
};
static MemState *fault_memory=nullptr;
static std::atomic<bool> fault_entered=false;
static void fault_handler(int,siginfo_t *info,void*){fault_entered.store(true);if(!fault_memory||!handle_access_violation_metal3(*fault_memory,(uint8_t *)info->si_addr,true))_Exit(91);}
static void ordinary_cases(){
    for(unsigned kind=0;kind<9;++kind){Fixture f;auto original=f.tracker.capture(f.mem,f.address(),f.bytes().size());auto guard=GuestMemoryPublication::prepare(f.mem,f.tracker,f.address()+(kind==8?0:3),f.bytes().size()-(kind==8?0:7));assert(guard);
        std::vector<uint8_t> before(f.bytes().size()-(kind==8?0:7),7),after(before.size(),43);std::array<Range,2> ranges{{{2,19},{77,43}}};
        if(kind==1){assert(handle_access_violation_metal3(f.mem,f.bytes().data(),true));f.bytes()[8]=7;} // Same-value CPU store still owns it.
        if(kind==2){assert(handle_access_violation_metal3(f.mem,f.bytes().data(),true));f.bytes()[8]=99;}
        if(kind==3)original.token->epoch.fetch_add(1);
        if(kind==4){ranges[0]={before.size(),1};}
        if(kind==5){ranges[1]={4,9};}
        if(kind==6)allocation_valid=false;
        if(kind==7)add_protect_metal3(f.mem,f.address(),f.mem.host_page_size,MemPerm::ReadOnly,[](Address,bool){return true;});
        if(kind==8)assert(!mprotect(f.bytes().data(),f.mem.host_page_size,PROT_READ|PROT_WRITE));
        bool threw=false,stored=false;try{stored=guard->publish(f.mem,original,before,after,std::span<const Range>{ranges});}catch(const std::runtime_error&){threw=true;}
        assert(threw==(kind>=4));assert(stored==(kind==0));
        if(stored){for(size_t i=0;i<f.bytes().size();++i){const bool written=(i>=5&&i<24)||(i>=80&&i<123);assert(f.bytes()[i]==(written?43:7));}
            assert(!original.changed());f.tracker.rearm(f.mem);bool adopted=false;
            std::vector<uint8_t> published(before);for(const auto r:ranges)std::fill_n(published.begin()+r.offset,r.size,43);
            assert(guard->adopt(f.mem,original,published,[&](auto bytes){adopted=true;assert(std::equal(bytes.begin(),bytes.end(),published.begin()));}));assert(adopted);
        }else{for(size_t i=0;i<f.bytes().size();++i)assert(f.bytes()[i]==(kind==2&&i==8?99:7));}
        allocation_valid=true;
    }
    {Fixture f;f.mem.use_page_table=true;assert(!GuestMemoryPublication::prepare(f.mem,f.tracker,f.address(),f.mem.host_page_size));}
    {Fixture f;allocation_valid=false;assert(!GuestMemoryPublication::prepare(f.mem,f.tracker,f.address(),f.mem.host_page_size));allocation_valid=true;}
    {Fixture f;add_protect_metal3(f.mem,f.address(),f.mem.host_page_size,MemPerm::ReadOnly,[](Address,bool){return true;});assert(!GuestMemoryPublication::prepare(f.mem,f.tracker,f.address(),f.mem.host_page_size));}
    std::cout<<"PASS guest RAM guard: real anonymous Mach alias and original read-only pages, exact partial ranges/padding, same-value and changed CPU writes, source epochs, invalid/overlapping ranges, lost allocation/foreign traps/kernel protection, page-table/invalid-allocation exclusions and protected adoption; actual private watch registration/fault bodies; allocator validity controlled\n";
}
static void concurrent_store(){
    Fixture f;auto source=f.tracker.capture(f.mem,f.address(),f.mem.host_page_size);auto guard=GuestMemoryPublication::prepare(f.mem,f.tracker,f.address(),f.mem.host_page_size);assert(guard);
    std::atomic<bool> copied=false,writer_done=false;fault_entered=false;fault_memory=&f.mem;
    assert(add_write_watch(f.mem,f.address(),f.mem.host_page_size,[&](Address,bool){copied=true;while(!fault_entered.load())std::this_thread::yield();assert(!writer_done.load());return true;}));
    struct sigaction action{},previous_bus{},previous_segv{};action.sa_sigaction=fault_handler;action.sa_flags=SA_SIGINFO;sigemptyset(&action.sa_mask);assert(!sigaction(SIGBUS,&action,&previous_bus)&&!sigaction(SIGSEGV,&action,&previous_segv));
    std::thread writer([&]{while(!copied.load())std::this_thread::yield();*reinterpret_cast<volatile uint8_t *>(f.bytes().data())=99;writer_done=true;});
    std::vector<uint8_t> before(f.mem.host_page_size,7),after(before.size(),43);const Range range{0,after.size()};
    assert(guard->publish(f.mem,source,before,after,std::span(&range,1)));writer.join();assert(fault_entered&&writer_done&&f.bytes()[0]==99&&f.bytes()[1]==43);
    assert(!sigaction(SIGBUS,&previous_bus,nullptr)&&!sigaction(SIGSEGV,&previous_segv,nullptr));fault_memory=nullptr;
    f.tracker.rearm(f.mem);bool adopted=false;assert(!guard->adopt(f.mem,source,after,[&](auto){adopted=true;}));assert(!adopted);
    std::cout<<"PASS guest RAM concurrent store: real CPU write fault observed while publisher owns protection mutex; alias stores finish before original-page unprotect; later CPU store wins and stale cache adoption is refused; isolated signal fixture, no device/GPU/app\n";
}
static void replaced_mapping(){Fixture f;auto source=f.tracker.capture(f.mem,f.address(),f.mem.host_page_size);auto guard=GuestMemoryPublication::prepare(f.mem,f.tracker,f.address(),f.mem.host_page_size);std::vector<uint8_t> before(f.mem.host_page_size,7),after(before.size(),43);const Range range{0,after.size()};assert(guard->publish(f.mem,source,before,after,std::span(&range,1)));f.tracker.rearm(f.mem);
    // Replacing the original VM object leaves the old job alias alive. Adoption
    // must compare current storage, not those still-pinned physical pages.
    auto *replacement=mmap(f.bytes().data(),f.mem.host_page_size,PROT_READ|PROT_WRITE,MAP_FIXED|MAP_PRIVATE|MAP_ANON,-1,0);assert(replacement==f.bytes().data());std::memset(replacement,61,f.mem.host_page_size);
    f.mem.protect_tree.clear();assert(add_write_watch(f.mem,f.address(),f.mem.host_page_size,[](Address,bool){return true;}));bool adopted=false;assert(!guard->adopt(f.mem,source,after,[&](auto){adopted=true;}));assert(!adopted&&f.bytes()[0]==61);
    std::cout<<"PASS guest RAM replacement: old alias pins former object, new anonymous allocation at same address remains intact; fresh-mapping adoption rejects stale completion\n";
}
static void edge_cases(){
    size_t cases=0;
    const size_t page=vm_page_size;
    for(size_t offset:{0ul,1ul,3ul,page/2,page-1})for(size_t length:{1ul,3ul,page/2,page-7,page+3,2*page+3}){
        Fixture f;const Address address=Address(f.address()+offset);if(uint64_t(address)+length>mapped_size)continue;
        valid_begin=address;valid_end=Address(address+length);
        const auto source=f.tracker.capture(f.mem,address,length);
        auto guard=GuestMemoryPublication::prepare(f.mem,f.tracker,address,length);assert(guard);
        auto *original=f.mem.memory.get()+address;
        std::vector<uint8_t> before(length,7),after(length,43);
        if(offset)f.mem.memory[address-1]=59;
        if(address+length<mapped_size)f.mem.memory[address+length]=61;
        // Non-frame bytes in edge host pages are neither allocations owned by
        // this publication nor reasons to discard its image.
        if(offset)assert(WritableGuestAlias::original_is_writable(original));
        const Range range{0,length};assert(guard->publish(f.mem,source,before,after,std::span(&range,1)));
        for(size_t i=0;i<mapped_size;++i){const auto expected=i>=address&&i<address+length?43:offset&&i==address-1?59:i==address+length?61:7;assert(f.mem.memory[i]==expected);}
        f.tracker.rearm(f.mem);bool called=false;
        assert(guard->adopt(f.mem,source,after,[&](auto){called=true;assert(WritableGuestAlias::original_is_read_only(original,length));}));assert(called);
        // Only fully owned interior pages remain watched after rearm. Edges
        // recover their pre-adoption permission, without an untracked RO page.
        if(address%page)assert(WritableGuestAlias::original_is_writable(original));
        if((address+length)%page)assert(WritableGuestAlias::original_is_writable(original+length-1));
        ++cases;valid_begin=valid_end=0;
    }
    std::cout<<"PASS guest RAM edges: "<<cases<<" actual partial-host-page spans including one-byte/tiny/cross-page and controlled unallocated neighbors; no pending edge watch, outside bytes preserved, stable protected adoption and exact edge-permission restoration; real Mach APIs\n";
}
static void concurrent_edge_adoption(){
    Fixture f;const Address address=f.address()+3;const size_t size=f.mem.host_page_size-7;
    auto source=f.tracker.capture(f.mem,address,size);assert(!source.token);
    auto guard=GuestMemoryPublication::prepare(f.mem,f.tracker,address,size);assert(guard);
    std::vector<uint8_t> before(size,7),after(size,43);const Range range{0,size};assert(guard->publish(f.mem,source,before,after,std::span(&range,1)));
    std::atomic<bool> inside=false,done=false;fault_entered=false;fault_memory=&f.mem;const auto logs=critical_logs.load();
    struct sigaction action{},previous_bus{},previous_segv{};action.sa_sigaction=fault_handler;action.sa_flags=SA_SIGINFO;sigemptyset(&action.sa_mask);assert(!sigaction(SIGBUS,&action,&previous_bus)&&!sigaction(SIGSEGV,&action,&previous_segv));
    std::thread writer([&]{while(!inside.load())std::this_thread::yield();*reinterpret_cast<volatile uint8_t *>(f.mem.memory.get()+address)=99;done=true;});
    assert(guard->adopt(f.mem,source,after,[&](auto){inside=true;while(!fault_entered.load())std::this_thread::yield();assert(!done.load());}));
    writer.join();assert(fault_entered&&done&&f.mem.memory[address]==99&&f.mem.memory[address+1]==43);
    assert(critical_logs.load()==logs&&WritableGuestAlias::original_is_writable(f.mem.memory.get()+address));
    assert(!sigaction(SIGBUS,&previous_bus,nullptr)&&!sigaction(SIGSEGV,&previous_segv,nullptr));fault_memory=nullptr;
    assert(!guard->adopt(f.mem,source,after,[](auto){assert(false);}));
    std::cout<<"PASS guest RAM edge fault: real CPU write blocks on transient edge protection, resumes on restored RW mapping without spurious missing-watch error, preserves newer bytes and prevents later stale adoption\n";
}
static void external_fault_address(){
    Fixture f;void *external=mmap(nullptr,vm_page_size,PROT_READ,MAP_PRIVATE|MAP_ANON,-1,0);assert(external!=MAP_FAILED);
    f.mem.use_page_table=true;f.mem.external_mapping.emplace(reinterpret_cast<uint64_t>(external),MemExternalMapping{f.address(),uint32_t(vm_page_size)});
    assert(WritableGuestAlias::original_is_writable(f.bytes().data())&&!WritableGuestAlias::original_is_writable(external));
    assert(handle_access_violation_metal3(f.mem,(uint8_t *)external,true));
    assert(WritableGuestAlias::original_is_writable(external));assert(!munmap(external,vm_page_size));
    std::cout<<"PASS guest RAM fault address: actual private handler checks protection of the faulting external mapping, not the unrelated writable original arena; legacy unprotect routing retained at controlled mapping boundary\n";
}
int main(){ordinary_cases();concurrent_store();replaced_mapping();edge_cases();concurrent_edge_adoption();external_fault_address();}
