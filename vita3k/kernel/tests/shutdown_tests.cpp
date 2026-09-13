// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <kernel/state.h>
#include <kernel/thread/thread_state.h>
#include <mem/state.h>
#include <mem/functions.h>
#include <SDL3/SDL.h>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace std::chrono_literals;
static void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        check(SDL_Init(0), "SDL initialization failed");
        MemState mem;
        check(init(mem, false), "Guest memory initialization failed");
        KernelState kernel;
        unsigned failures=0;
        // Hold the last host-thread-owned guest allocation in its deleter.
        // This reproduces the real shutdown race without unmapping live RAM.
        for (bool already_deleting : {false, true}) {
            check(kernel.init(mem, {}, true), "Kernel initialization failed");
            auto thread=kernel.create_thread(mem, "shutdown-barrier-fixture");
            check(bool(thread), "Cannot create fixture thread");
            std::promise<void> cleanup_entered, release_cleanup, cleanup_finished;
            auto entered=cleanup_entered.get_future();
            auto release=release_cleanup.get_future().share();
            auto finished=cleanup_finished.get_future();
            const Address allocation=alloc(mem, 4096, "Delayed thread cleanup");
            check(allocation!=0, "Cannot allocate cleanup fixture");
            thread->stack=Block(allocation, [&](Address address) {
                cleanup_entered.set_value();
                release.wait();
                free(mem, address);
                cleanup_finished.set_value();
            });
            if (already_deleting) thread->exit_delete(false);
            thread.reset();
            if (already_deleting) check(entered.wait_for(10s)==std::future_status::ready, "Thread cleanup did not start");
            auto shutdown=std::async(std::launch::async, [&] { kernel.process_exit(); });
            check(entered.wait_for(10s)==std::future_status::ready, "Shutdown did not reach thread cleanup");
            const bool returned_early=shutdown.wait_for(250ms)==std::future_status::ready;
            // Always let the host thread finish before checking the result or
            // destroying the test state, including on the unfixed implementation.
            release_cleanup.set_value();
            check(finished.wait_for(10s)==std::future_status::ready, "Cleanup did not finish");
            check(shutdown.wait_for(10s)==std::future_status::ready, "Shutdown remained blocked after cleanup");
            shutdown.get();
            if (returned_early) ++failures;
            std::cout<<(returned_early ? "FAIL" : "PASS")<<" shutdown waits for final guest allocation destructor, already_deleting="<<already_deleting<<'\n';
            kernel.deinit(mem);
        }
        for (unsigned round=0;round<16;++round) {
            check(kernel.init(mem, {}, true), "Cannot reinitialize kernel");
            for(unsigned i=0;i<8;++i) check(bool(kernel.create_thread(mem,"repeated-shutdown-fixture")),"Cannot create shutdown stress thread");
            kernel.process_exit();
            check(kernel.threads.empty(),"Threads survived process exit");
            check(!kernel.create_thread(mem,"rejected-during-shutdown"),"Shutdown admitted a new guest thread");
            kernel.deinit(mem);
            deinit_mem(mem);
            check(init(mem, false), "Cannot reset guest memory after shutdown");
        }
        deinit_mem(mem);
        SDL_Quit();
        std::cout<<"Repeated shutdown:16cycles,128dormant guest threads; barrier failures="<<failures<<'\n';
        return failures ? 1 : 0;
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
}
