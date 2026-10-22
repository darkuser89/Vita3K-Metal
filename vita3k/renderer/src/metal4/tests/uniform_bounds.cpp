// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <gxm/metal4/uniform_bounds.h>
#include <array>
#include <cassert>
#include <future>
#include <iostream>
#include <map>
#include <random>

using Address = uint32_t;
struct MemoryMapInfo { Address offset; uint32_t size, perm; };
struct CountingMap : std::map<Address, MemoryMapInfo> {
    mutable size_t traversals = 0;
    auto begin() const { ++traversals; return std::map<Address, MemoryMapInfo>::begin(); }
    auto begin() { return std::map<Address, MemoryMapInfo>::begin(); }
};
struct GxmState {
    CountingMap memory_mapped_regions;
    struct { gxm::metal3::UniformMappingBounds uniform_mapping_bounds; } metal;
};
auto &metal_gxm_bookkeeping(GxmState &gxm) { return gxm.metal; }
// HELPER_HOOK
// REFERENCE_HOOK

// Only the export's environment/renderer boundary is mocked. Map alignment,
// successful/failed paths and index invalidation use the production exports.
#define TRACY_FUNC(...)
#define LOG_WARN_ONCE(...)
#define LOG_ERROR(...)
#define RET_ERROR(x) (x)
using SceUID = int;
constexpr int SCE_GXM_ERROR_INVALID_POINTER = -1;
constexpr uint32_t KiB(uint32_t value) { return value * 1024; }
template <typename T> T align_down(T value, T alignment) { return value & ~(alignment - 1); }
template <typename T> T align(T value, T alignment) { return (value + alignment - 1) & ~(alignment - 1); }
struct MemState { std::array<uint8_t, 65536> bytes{}; uint32_t host_page_size = 4096; };
template <typename T> struct Ptr {
    Address addr;
    Address address() const { return addr; }
    explicit operator bool() const { return addr != 0; }
    template <typename U> Ptr<U> cast() const { return {addr}; }
    T *get(MemState &mem) const { assert(addr < mem.bytes.size()); return reinterpret_cast<T *>(mem.bytes.data() + addr); }
};
namespace renderer {
enum class Backend { Metal };
enum class CommandOpcode { MemoryMap, MemoryUnmap };
constexpr int CommandErrorCodeNone = 0;
struct State {
    Backend current_backend = Backend::Metal;
    struct { bool enable_memory_mapping = false; } features;
    int result = 0;
    size_t calls = 0;
};
int send_single_command(State &state, void *, CommandOpcode, bool, Address, uint32_t = 0) {
    ++state.calls; return state.result;
}
}
struct EmuEnvState { GxmState gxm; renderer::State *renderer; MemState mem; };
// EXPORTS_HOOK

void export_paths() {
    renderer::State renderer;
    EmuEnvState env{}; env.renderer = &renderer;
    const auto map = [&](uint32_t address, uint32_t bytes) {
        return export_sceGxmMapMemory_metal3(env, 0, "test", Ptr<void>{address}, bytes, 0);
    };
    const auto unmap = [&](uint32_t address) {
        return export_sceGxmUnmapMemory_metal3(env, 0, "test", Ptr<void>{address});
    };
    const auto bound = [&](uint32_t address, uint32_t bytes) {
        return metal_mapped_uniform_size_metal3(env.gxm, address, bytes);
    };
    assert(bound(4096, 16384) == 16384); // Warm an empty index.
    assert(map(4097, 4095) == 0); // Alignment creates [4096,8192).
    assert(bound(4096, 16384) == 4096);
    assert(map(8192, 4096) == 0);
    assert(bound(4096, 16384) == 8192);
    assert(unmap(8192) == 0 && bound(4096, 16384) == 4096);
    // Replace at identical map size and address, with a different extent.
    assert(unmap(4096) == 0 && map(4096, 12288) == 0);
    assert(bound(4096, 16384) == 12288);
    assert(map(8192, 4096) == 0); // Older larger owner must still contain it.
    assert(bound(8192, 16384) == 8192);
    assert(unmap(8193) == 0 && bound(8192, 16384) == 8192);
    assert(map(20480, 0) == 0 && bound(20480, 123) == 123);
    const auto before = env.gxm.memory_mapped_regions.traversals;
    assert(map(4096, 4096) == -1); // Duplicate; authoritative map stays unchanged.
    renderer.result = -1;
    assert(map(24576, 4096) == -1);
    assert(map(0, 4096) == -1 && map(0xFFFFF000, 8192) == -1);
    assert(unmap(0) == -1 && unmap(24576) == -1);
    assert(bound(4096, 16384) == 12288);
    assert(env.gxm.memory_mapped_regions.traversals == before);
    renderer.result = 0;
    renderer.features.enable_memory_mapping = true;
    assert(map(36864, 4096) == 0);
    assert(map(32768, 12288) == -1); // Existing mapped-GPU overlap rejection.
    assert(bound(36864, 8192) == 4096);
    assert(unmap(36864) == 0 && bound(36864, 8192) == 8192);
}

static size_t cases;
void check(GxmState &gxm, Address address, uint32_t requested) {
    const auto expected = old_mapped_uniform_size(gxm, address, requested);
    assert(metal_mapped_uniform_size_metal3(gxm, address, requested) == expected);
    ++cases;
}

int main() {
    export_paths();
    // All subsets of nested, adjacent, disjoint and zero-sized ranges, with
    // every address and request length around both ends and every hole.
    const std::array<std::pair<Address, uint32_t>, 8> candidates{{
        {0, 10}, {1, 2}, {3, 4}, {7, 2}, {10, 3}, {13, 0}, {15, 2}, {18, 1}}};
    for (unsigned subset = 0; subset < 256; ++subset) {
        GxmState gxm;
        for (unsigned n = 0; n < candidates.size(); ++n)
            if (subset & (1u << n)) {
                const auto [base, size] = candidates[n];
                gxm.memory_mapped_regions.emplace(base, MemoryMapInfo{base, size, 0});
            }
        for (Address address = 0; address < 24; ++address)
            for (uint32_t size = 0; size < 28; ++size) check(gxm, address, size);
    }
    GxmState gxm;
    std::mt19937 rng(0xA008);
    for (unsigned step = 0; step < 4000; ++step) {
        auto &regions = gxm.memory_mapped_regions;
        const Address address = uint32_t(rng() % 256) * 4096;
        gxm.metal.uniform_mapping_bounds.invalidate();
        if (step % 13 == 0) regions.clear();
        else if (step % 3 == 0) regions.erase(address);
        else regions[address] = {address, uint32_t(rng() % 32) * 4096, 0};
        for (unsigned n = 0; n < 32; ++n) check(gxm, rng() % 0x120000, rng() % 0x20000);
        check(gxm, address, 0); check(gxm, address, UINT32_MAX);
    }
    // 64-bit ends must never wrap, including test-only extents beyond the
    // guest address space. The exported mapping API rejects those extents.
    gxm.memory_mapped_regions = {};
    for (Address address : {0u, 1u, 0x7FFFFFFFu, 0xFFFFFF00u, UINT32_MAX})
        gxm.memory_mapped_regions[address] = {address, UINT32_MAX, 0};
    gxm.metal.uniform_mapping_bounds.invalidate();
    for (Address address : {0u, 1u, 2u, 0x7FFFFFFEu, 0x7FFFFFFFu, 0xFFFFFFFEu, UINT32_MAX})
        for (uint32_t size : {0u, 1u, 2u, 65536u, 0x80000000u, UINT32_MAX}) check(gxm, address, size);

    // Explicit endpoint/no-map behavior; connected mappings extend a prefix,
    // while a hole truncates it only when the binding itself is mapped.
    GxmState edges;
    edges.memory_mapped_regions.emplace(100, MemoryMapInfo{100, 10, 0});
    edges.memory_mapped_regions.emplace(110, MemoryMapInfo{110, 5, 0});
    assert(metal_mapped_uniform_size_metal3(edges, 105, 20) == 10);
    assert(metal_mapped_uniform_size_metal3(edges, 115, 20) == 20);
    assert(metal_mapped_uniform_size_metal3(edges, 99, 20) == 20);
    edges.metal.uniform_mapping_bounds.invalidate();
    edges.memory_mapped_regions.erase(110);
    assert(metal_mapped_uniform_size_metal3(edges, 105, 20) == 5);
    edges.metal.uniform_mapping_bounds.invalidate();
    edges.memory_mapped_regions.erase(100);
    assert(metal_mapped_uniform_size_metal3(edges, 105, 20) == 20);

    GxmState large;
    for (Address i = 0; i < 10000; ++i)
        large.memory_mapped_regions.emplace(i * 8192, MemoryMapInfo{i * 8192, 4096, 0});
    assert(metal_mapped_uniform_size_metal3(large, 9999 * 8192 + 4090, 16) == 6);
    assert(large.memory_mapped_regions.traversals == 1);
    for (unsigned n = 0; n < 10000; ++n)
        assert(metal_mapped_uniform_size_metal3(large, (n % 10000) * 8192 + 4090, 16) == 6);
    assert(large.memory_mapped_regions.traversals == 1);
    // Concurrent readers share one already-built immutable snapshot.
    std::array<std::future<void>, 4> readers;
    for (auto &reader : readers) reader = std::async(std::launch::async, [&] {
        for (unsigned n = 0; n < 10000; ++n)
            assert(metal_mapped_uniform_size_metal3(large, (n % 10000) * 8192 + 4090, 16) == 6);
    });
    for (auto &reader : readers) reader.get();
    assert(large.memory_mapped_regions.traversals == 1);
    // A new state owns a fresh index, with no process-global stale pointers.
    GxmState fresh;
    assert(metal_mapped_uniform_size_metal3(fresh, 9999 * 8192 + 4090, 16) == 16);
    std::cout << "PASS uniform mapping bounds: " << cases
        << " original-scan comparisons, actual Map/Unmap export success/failure/alignment, overlap/adjacency/zero/hole/32-bit edges, 10000 warm queries plus 40000 concurrent reads without source-map traversal\n";
}
