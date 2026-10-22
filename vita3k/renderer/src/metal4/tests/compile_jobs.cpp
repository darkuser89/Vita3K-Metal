// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/compile_jobs.h>
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

int main() {
    using renderer::metal::CompileJobs;
    size_t admission_cases=0;
    for(bool enabled:{false,true}) for(bool mask:{false,true}) for(bool capture:{false,true})
    for(size_t count:{0u,1u,6u,7u,1024u}) for(uint32_t instances:{0u,1u,2u,64u}) {
        const auto allowed=renderer::metal::allow_async_draw_compile(enabled,mask,capture,count,instances);
        if(mask || !enabled || capture) assert(!allowed);
        else if(count<=6 && instances<=1) assert(!allowed);
        else assert(allowed);
        ++admission_cases;
    }
    assert(admission_cases==160);
    // Exercise a pending compiler job for an otherwise large mask draw.
    // Admission must make this lookup wait rather than returning no shader.
    CompileJobs<int> mask_jobs;
    std::promise<void> mask_release;
    auto mask_gate=mask_release.get_future().share();
    assert(mask_jobs.submit("mask",2,[mask_gate]{mask_gate.wait();return 73;}));
    assert(!mask_jobs.take("mask",false));
    const bool mask_async=renderer::metal::allow_async_draw_compile(true,true,false,1000,4);
    auto mask_future=mask_jobs.take("mask",!mask_async);
    assert(mask_future);
    mask_release.set_value();assert(mask_future->get()==73);
    std::cout<<"PASS mask compile admission: 160 size/instance/async/capture/mask cases; large mask draw retains and waits for its pending shader instead of skipping, ordinary async thresholds preserved; no GPU/app\n";
    CompileJobs<std::unique_ptr<int>> jobs;
    std::promise<void> release;
    auto gate = release.get_future().share();
    auto owned = std::make_shared<const std::vector<int>>(std::vector<int>{7,11});
    std::weak_ptr<const std::vector<int>> weak = owned;
    std::atomic<int> entered{0}, duplicate{0};
    const auto owner_thread = std::this_thread::get_id();
    for (int i=0;i<2;++i)
        assert(jobs.submit(std::to_string(i),2,[owned,gate,&entered,owner_thread,i] {
            assert(std::this_thread::get_id()!=owner_thread);
            ++entered;gate.wait();return std::make_unique<int>((*owned)[i]);
        }));
    owned.reset();assert(!weak.expired());
    assert(!jobs.submit("0",2,[&] {++duplicate;return std::make_unique<int>(0);}));
    assert(!jobs.submit("full",2,[&] {++duplicate;return std::make_unique<int>(0);}));
    assert(!jobs.take("0",false));
    jobs.collect(false,[](auto &,auto) {assert(false);});
    release.set_value();
    int sum=0,published=0;
    jobs.collect(true,[&](auto &,auto future) {
        assert(std::this_thread::get_id()==owner_thread);
        sum+=*future.get();++published;
    });
    assert(sum==18 && published==2 && entered==2 && duplicate==0 && jobs.size()==0 && weak.expired());
    // Exceptions are removed before get, and don't poison a reused key.
    assert(jobs.submit("failure",2,[]()->std::unique_ptr<int> {throw std::runtime_error("rejected");}));
    auto failing=jobs.take("failure",true);assert(failing && !jobs.contains("failure"));
    bool rejected=false;try {(void)failing->get();}catch(const std::runtime_error &){rejected=true;}
    assert(rejected);
    assert(jobs.submit("failure",2,[]{return std::make_unique<int>(9);}));
    auto retry=jobs.take("failure",true);assert(*retry->get()==9);
    // Draining ready work must not depend on the original draw key recurring.
    assert(jobs.submit("never-drawn-again",2,[]{return std::make_unique<int>(42);}));
    jobs.collect(true,[](auto &key,auto future){assert(key=="never-drawn-again" && *future.get()==42);});
    assert(jobs.submit("next-game",2,[]{return std::make_unique<int>(43);}));
    jobs.clear();assert(jobs.size()==0);
    // Clear must join the actual worker before its borrowed device can die.
    std::atomic<bool> device_alive{true}, completed{false};
    std::promise<void> permit;
    auto released=permit.get_future().share();
    assert(jobs.submit("shutdown",2,[&device_alive,&completed,released] {
        released.wait();assert(device_alive);completed=true;return std::make_unique<int>(1);
    }));
    auto signal=std::thread([&permit]{permit.set_value();});
    jobs.clear();assert(completed);device_alive=false;signal.join();
    assert(!jobs.contains("shutdown") && jobs.size()==0);
    std::cout<<"PASS compiler jobs: bounded admission, duplicate keys, owned inputs, render-thread publication, failures, stale keys and shutdown join\n";
}
