// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/library_reuse.h>
#include <atomic>
#include <barrier>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Program { std::string entry_point; unsigned stage; std::string source; };
static std::string native_identity(const Program &program, bool gamma_correction) {
    // IDENTITY_HOOK
    return identity;
}
int main() {
    Program p{"main", 0, "source"};
    const auto identity = native_identity(p, false);
    assert(identity == native_identity(p, false));
    assert(identity != native_identity(p, true));
    p.stage = 1; assert(identity != native_identity(p, false)); p.stage = 0;
    p.entry_point = "other"; assert(identity != native_identity(p, false)); p.entry_point = "main";
    p.source = "changed"; assert(identity != native_identity(p, false));
    renderer::metal::LibraryReuse<int> cache;
    std::atomic<unsigned> builds = 0;
    std::barrier start(17);
    std::vector<std::future<std::shared_ptr<const int>>> jobs;
    for (unsigned i = 0; i < 16; ++i)
        jobs.push_back(std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            bool reused;
            return cache.get("same source, stage and gamma", [&] {
                ++builds;
                return std::make_shared<const int>(42);
            }, reused);
        }));
    start.arrive_and_wait();
    std::vector<std::shared_ptr<const int>> owners;
    for (auto &job : jobs) owners.push_back(job.get());
    assert(builds == 1);
    for (auto &owner : owners) assert(owner == owners.front() && *owner == 42);
    for (unsigned i = 0; i < 10000; ++i) {
        bool reused = false;
        auto owner = cache.get("same source, stage and gamma", [&] {
            ++builds;
            return std::make_shared<const int>(0);
        }, reused);
        assert(reused && owner == owners.front());
    }
    // Independent identities must compile concurrently, without a cache lock
    // around driver calls. A lock held across either build deadlocks this gate.
    std::barrier independent(2);
    auto other = [&](std::string key, int value) {
        bool reused = true;
        auto owner = cache.get(key, [&] {
            independent.arrive_and_wait();
            return std::make_shared<const int>(value);
        }, reused);
        assert(!reused && *owner == value);
        return owner;
    };
    auto first = std::async(std::launch::async, other, "different stage", 1);
    auto second = std::async(std::launch::async, other, "different gamma", 2);
    assert(first.get() != second.get());
    bool reused = true;
    try {
        cache.get("retry", []() -> std::shared_ptr<const int> { throw std::runtime_error("driver error"); }, reused);
        assert(false);
    } catch (const std::runtime_error &) {}
    assert(!reused);
    auto retry = cache.get("retry", [] { return std::make_shared<const int>(9); }, reused);
    assert(!reused && *retry == 9);
    std::weak_ptr<const int> weak = owners.front();
    owners.clear(); jobs.clear();
    assert(weak.expired());
    auto rebuilt = cache.get("same source, stage and gamma", [&] {
        ++builds;
        return std::make_shared<const int>(43);
    }, reused);
    assert(!reused && builds == 2 && *rebuilt == 43);
    cache.clear();
    auto after_clear = cache.get("same source, stage and gamma", [] { return std::make_shared<const int>(44); }, reused);
    assert(!reused && after_clear != rebuilt && *rebuilt == 43);
    std::cout << "PASS native library reuse: 16 concurrent equal requests, 10000 hits, independent parallel builds, stage/gamma isolation, failure retry, weak lifetime and game reset; ASAN/UBSAN\n";
}
