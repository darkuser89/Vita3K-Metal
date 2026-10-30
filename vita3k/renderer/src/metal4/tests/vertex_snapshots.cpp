// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Production snapshot capture/read/reset methods, with a small CPU ring.
#include <renderer/metal4/vertex_snapshots.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <new>
#include <semaphore>
#include <thread>
#include <type_traits>

// Observe the real std::vector allocation/free boundary, not a simulated ring.
struct MemoryGate {
    std::binary_semaphore entered{0}, release{0};
    std::atomic<bool> armed{true};
    size_t bytes = 0;
    void *retired = nullptr;
    bool fail = false;
    std::atomic<unsigned> allocations{0};
};
static std::atomic<MemoryGate *> memory_gate{nullptr};
static thread_local bool observe_ring_allocation = false;
void *operator new(size_t size) {
    auto *gate = memory_gate.load();
    if (gate && observe_ring_allocation && size == gate->bytes) {
        ++gate->allocations;
        if (gate->armed.exchange(false)) {
            gate->entered.release(); gate->release.acquire();
            if (gate->fail) throw std::bad_alloc();
        }
    }
    if (auto *p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept {
    auto *gate = memory_gate.load();
    if (p && gate && p == gate->retired && gate->armed.exchange(false)) {
        gate->entered.release(); gate->release.acquire();
    }
    std::free(p);
}
void *operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, size_t) noexcept { ::operator delete(p); }

using namespace renderer::metal;
using Address = uint32_t;
struct MemState {
    bool use_page_table = false;
    std::vector<uint8_t> bytes = std::vector<uint8_t>(16384);
    std::array<std::vector<uint8_t>, 4> pages;
    MemState() { for (auto &p : pages) p.resize(4096); }
};
template <typename T> struct Ptr {
    Address value;
    Address address() const { return value; }
    explicit operator bool() const { return value != 0; }
    T *get(MemState &mem) const {
        return reinterpret_cast<T *>(mem.use_page_table ? mem.pages.at(value / 4096).data() + value % 4096 : mem.bytes.data() + value);
    }
};
static bool is_valid_addr_range(const MemState &mem, Address begin, Address end) {
    return begin && begin <= end && end <= mem.bytes.size();
}
static void require(bool condition, const char *) { assert(condition); }
enum class CommandOpcode { SetState, MidSceneFlush, Other };
enum class GXMState { VertexStream, Other };
struct Command {
    CommandOpcode opcode = CommandOpcode::SetState;
    GXMState state = GXMState::VertexStream;
    Ptr<const uint8_t> address{100};
    size_t size = 100;
    uint64_t handle = ~0ull;
    Command *next = nullptr;
};
struct CommandList { Command *first, *last; };
struct CommandHelper {
    Command *cmd;
    unsigned sizes = 0;
    explicit CommandHelper(Command *cmd) : cmd(cmd) {}
    template <typename T> T pop() {
        if constexpr (std::is_same_v<T, GXMState>) return cmd->state;
        else if constexpr (std::is_same_v<T, Ptr<const uint8_t>>) return cmd->address;
        else { static_assert(std::is_same_v<T, size_t>); return sizes++ ? cmd->size : 0; }
    }
    bool push(uint64_t handle) { cmd->handle = handle; return true; }
};
struct MetalState {
    struct Impl {
        // Force wraps cheaply; the production handle/slot algorithm is kept.
        static constexpr size_t STREAM_SNAPSHOT_RING_SIZE = 1024;
        static constexpr size_t STREAM_SNAPSHOT_MAX_BYTES = 256;
        struct StreamSnapshotHeader { uint64_t handle; uint32_t size; uint32_t reserved = 0; };
        std::mutex stream_snapshot_mutex;
        std::mutex stream_snapshot_allocation_mutex;
        std::vector<uint8_t> stream_snapshot_ring;
        VertexSnapshotOwners stream_snapshot_owners;
        uint64_t stream_snapshot_cursor = 0;
        bool stream_snapshots_disabled = false;
    };
    std::unique_ptr<Impl> impl = std::make_unique<Impl>();
    void snapshot_vertex_streams(MemState &, CommandList &);
    OwnedVertexSnapshot vertex_stream_snapshot(uint64_t, size_t);
    void disable_vertex_stream_snapshots();
    void reset_vertex_stream_snapshots();
};
// SNAPSHOT_METHODS_HOOK

template <typename Work> static auto joined_async(Work work) {
    std::packaged_task<std::invoke_result_t<Work>()> task(std::move(work));
    auto future = task.get_future();
    return std::pair{std::thread(std::move(task)), std::move(future)};
}
static void allocation_lock_test() {
    MetalState renderer; MemState mem; Command c; CommandList list{&c, &c};
    MemoryGate gate; gate.bytes = MetalState::Impl::STREAM_SNAPSHOT_RING_SIZE;
    memory_gate = &gate;
    auto [capture_thread, capture] = joined_async([&] {
        observe_ring_allocation = true;
        renderer.snapshot_vertex_streams(mem, list);
        observe_ring_allocation = false;
    });
    assert(gate.entered.try_acquire_for(std::chrono::seconds(5)));
    auto [read_thread, read] = joined_async([&] { return renderer.vertex_stream_snapshot(0, 100); });
    const bool ready = read.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    gate.release.release(); capture.get(); (void)read.get();
    capture_thread.join(); read_thread.join(); memory_gate = nullptr;
    assert(ready && "Snapshot lookup must not wait for ring allocation");
    assert(renderer.vertex_stream_snapshot(c.handle, c.size));
}
static void retirement_lock_test() {
    for (bool disable : {false, true}) for (bool retire_owner : {false, true}) {
        MetalState renderer; MemState mem; Command c; CommandList list{&c, &c};
        renderer.snapshot_vertex_streams(mem, list);
        auto owner = renderer.vertex_stream_snapshot(c.handle, c.size);
        assert(owner);
        const std::weak_ptr<const std::vector<uint8_t>> weak = owner;
        MemoryGate gate;
        gate.retired = retire_owner ? const_cast<uint8_t *>(owner->data()) : renderer.impl->stream_snapshot_ring.data();
        if (retire_owner) owner.reset();
        memory_gate = &gate;
        auto [retire_thread, retire] = joined_async([&] {
            if (disable) renderer.disable_vertex_stream_snapshots();
            else renderer.reset_vertex_stream_snapshots();
        });
        assert(gate.entered.try_acquire_for(std::chrono::seconds(5)));
        auto [read_thread, read] = joined_async([&] { return renderer.vertex_stream_snapshot(c.handle, c.size); });
        const bool ready = read.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        gate.release.release(); retire.get(); const auto expired = read.get();
        retire_thread.join(); read_thread.join(); memory_gate = nullptr;
        assert(ready && "Snapshot lookup must not wait for retired ring deallocation");
        assert(!expired && renderer.impl->stream_snapshots_disabled == disable);
        if (retire_owner) assert(weak.expired());
        else assert(owner->size() == c.size);
    }
}
static void initialization_races_test() {
    // A disable/reset is allowed to complete while the allocation is pending.
    for (bool disable : {false, true}) {
        MetalState renderer; MemState mem; Command c; CommandList list{&c, &c};
        MemoryGate gate; gate.bytes = MetalState::Impl::STREAM_SNAPSHOT_RING_SIZE;
        memory_gate = &gate;
        auto [capture_thread, capture] = joined_async([&] {
            observe_ring_allocation = true; renderer.snapshot_vertex_streams(mem, list); observe_ring_allocation = false;
        });
        assert(gate.entered.try_acquire_for(std::chrono::seconds(5)));
        auto [change_thread, change] = joined_async([&] {
            if (disable) renderer.disable_vertex_stream_snapshots(); else renderer.reset_vertex_stream_snapshots();
        });
        const bool ready = change.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        gate.release.release(); capture.get(); change.get(); capture_thread.join(); change_thread.join(); memory_gate = nullptr;
        assert(ready && renderer.impl->stream_snapshots_disabled == disable);
        if (disable) assert(c.handle == ~0ull && renderer.impl->stream_snapshot_ring.empty());
        else assert(renderer.vertex_stream_snapshot(c.handle, c.size));
    }
    // Two first-time producers share one allocation, preserving distinct handles.
    {
        MetalState renderer; MemState mem; Command a, b;
        CommandList first{&a, &a}, second{&b, &b};
        MemoryGate gate; gate.bytes = MetalState::Impl::STREAM_SNAPSHOT_RING_SIZE; memory_gate = &gate;
        auto [ta, fa] = joined_async([&] {
            observe_ring_allocation = true; renderer.snapshot_vertex_streams(mem, first); observe_ring_allocation = false;
        });
        assert(gate.entered.try_acquire_for(std::chrono::seconds(5)));
        std::promise<void> started;
        auto [tb, fb] = joined_async([&] {
            started.set_value(); observe_ring_allocation = true;
            renderer.snapshot_vertex_streams(mem, second); observe_ring_allocation = false;
        });
        started.get_future().wait();
        const bool waiting = fb.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
        gate.release.release(); fa.get(); fb.get(); ta.join(); tb.join(); memory_gate = nullptr;
        assert(waiting && gate.allocations == 1 && a.handle != b.handle);
        assert(renderer.vertex_stream_snapshot(a.handle, a.size) && renderer.vertex_stream_snapshot(b.handle, b.size));
    }
    // Failed allocation releases both locks and can be retried normally.
    {
        MetalState renderer; MemState mem; Command c; CommandList list{&c, &c};
        MemoryGate gate; gate.bytes = MetalState::Impl::STREAM_SNAPSHOT_RING_SIZE; gate.fail = true; memory_gate = &gate;
        auto [thread, future] = joined_async([&] {
            observe_ring_allocation = true; renderer.snapshot_vertex_streams(mem, list);
        });
        assert(gate.entered.try_acquire_for(std::chrono::seconds(5))); gate.release.release();
        bool failed = false; try { future.get(); } catch (const std::bad_alloc &) { failed = true; }
        thread.join(); memory_gate = nullptr;
        assert(failed && c.handle == ~0ull && renderer.impl->stream_snapshot_ring.empty());
        renderer.snapshot_vertex_streams(mem, list);
        assert(renderer.vertex_stream_snapshot(c.handle, c.size));
    }
}

int main() {
    if (!std::getenv("VITA3K_SNAPSHOT_RETIRE_ONLY")) allocation_lock_test();
    retirement_lock_test();
    initialization_races_test();
    MetalState renderer; MemState mem; Command command;
    CommandList list{&command, &command};
    std::fill(mem.bytes.begin(), mem.bytes.end(), 0x13);
    renderer.snapshot_vertex_streams(mem, list);
    const auto first_handle = command.handle;
    auto first = renderer.vertex_stream_snapshot(first_handle, command.size);
    assert(first && first->size() == 100 && first->front() == 0x13);
    std::fill(mem.bytes.begin(), mem.bytes.end(), 0x74);
    for (unsigned i = 0; i < 10000; ++i) {
        const auto hit = renderer.vertex_stream_snapshot(first_handle, command.size);
        assert(hit == first && hit->front() == 0x13);
    }
    auto prefix = renderer.vertex_stream_snapshot(first_handle, 99);
    assert(prefix && prefix != first && prefix->size() == 99);
    assert(!renderer.vertex_stream_snapshot(first_handle, 101));
    assert(!renderer.vertex_stream_snapshot(~0ull, 100));
    assert(!renderer.vertex_stream_snapshot(first_handle, 0));
    for (unsigned i = 0; i < 20; ++i) renderer.snapshot_vertex_streams(mem, list);
    assert(!renderer.vertex_stream_snapshot(first_handle, 100)); // No stale owner hit after wrap.
    assert(first->front() == 0x13 && prefix->front() == 0x13);
    const auto last_handle = command.handle;
    auto held = renderer.vertex_stream_snapshot(last_handle, 100);
    renderer.disable_vertex_stream_snapshots();
    assert(!renderer.vertex_stream_snapshot(last_handle, 100) && held->front() == 0x74);
    renderer.snapshot_vertex_streams(mem, list); assert(command.handle == last_handle);
    renderer.reset_vertex_stream_snapshots();
    renderer.snapshot_vertex_streams(mem, list);
    assert(command.handle > last_handle && !renderer.vertex_stream_snapshot(last_handle, 100));
    // Page-table guest ranges are not physically contiguous across pages.
    mem.use_page_table = true;
    std::fill(mem.pages[0].begin(), mem.pages[0].end(), 0x12);
    std::fill(mem.pages[1].begin(), mem.pages[1].end(), 0x34);
    command.address.value = 4090; command.size = 100;
    renderer.snapshot_vertex_streams(mem, list);
    const auto spanning = renderer.vertex_stream_snapshot(command.handle, command.size);
    for (size_t i = 0; i < 100; ++i) assert((*spanning)[i] == (i < 6 ? 0x12 : 0x34));
    // A prior MidSceneFlush separates already captured inputs from this kick.
    Command earlier, flush, later; earlier.handle = 123; flush.opcode = CommandOpcode::MidSceneFlush;
    earlier.next = &flush; flush.next = &later;
    CommandList partial{&earlier, &later};
    renderer.snapshot_vertex_streams(mem, partial); assert(earlier.handle == 123 && later.handle != ~0ull);
    Command invalid; invalid.size = 257;
    CommandList bad{&invalid, &invalid}; renderer.snapshot_vertex_streams(mem, bad);
    assert(invalid.handle == ~0ull);
    // Bounded cache retirement does not invalidate a snapshot held by a draw.
    VertexSnapshotOwners owners;
    std::array<uint8_t, 16> bytes{};
    std::vector<std::weak_ptr<const std::vector<uint8_t>>> weak;
    for (unsigned i = 0; i < 65; ++i) weak.push_back(owners.get(i, bytes));
    assert(weak.front().expired() && !weak.back().expired());
    owners.clear(); assert(weak.back().expired());

    // Delayed render consumption while another thread continually kicks and
    // wraps the ring. A valid result must contain that kick's bytes, never a
    // later record at the same physical offset. Expired results stay absent.
    renderer.reset_vertex_stream_snapshots(); mem.use_page_table = false;
    struct Expected { uint64_t handle; uint8_t value; };
    std::mutex queue_mutex; std::deque<Expected> queue; std::atomic<bool> done{false};
    std::thread writer([&] {
        Command c; CommandList one{&c, &c};
        for (unsigned i = 0; i < 10000; ++i) {
            std::fill(mem.bytes.begin(), mem.bytes.end(), uint8_t(i));
            renderer.snapshot_vertex_streams(mem, one);
            const std::lock_guard lock(queue_mutex); queue.push_back({c.handle, uint8_t(i)});
        }
        done.store(true);
    });
    size_t checked = 0;
    for (;;) {
        Expected expected{}; bool found = false;
        {
            const std::lock_guard lock(queue_mutex);
            if (!queue.empty()) { expected = queue.front(); queue.pop_front(); found = true; }
            else if (done.load()) break;
        }
        if (!found) { std::this_thread::yield(); continue; }
        const auto snapshot = renderer.vertex_stream_snapshot(expected.handle, 100);
        if (snapshot) for (auto value : *snapshot) assert(value == expected.value);
        ++checked;
    }
    writer.join(); assert(checked == 10000);
    std::cout << "PASS vertex snapshots: observed allocation/ring/owner-free gates permit independent lookups; reset/disable during allocation, single allocation for two producers, allocation failure/retry; real capture/read/reset methods, 10000 identical owner hits, ring wrap/expiry, prefixes, separated guest pages, MidSceneFlush, disable/reinit, bounded owners and 10000 concurrent kicks; no GPU/app\n";
}
