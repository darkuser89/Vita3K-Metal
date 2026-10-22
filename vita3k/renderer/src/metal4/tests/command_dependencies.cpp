// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/command_dependencies.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
using namespace renderer::metal;
struct Operation {
    unsigned from, to, count, stage;
    bool opaque;
};
int main() {
    // Independent oracle uses physical byte indices, not CommandAccess::conflicts.
    // GPU addresses intentionally differ for half of CPU-aliasing resources.
    std::mt19937 rng(0x4ba771);
    size_t operations = 0, barriers = 0, skipped = 0;
    for (unsigned round = 0; round < 1000; ++round) {
        CommandDependencies tracker;
        std::vector<Operation> epoch;
        std::array<unsigned, 4096> sequential{}, reordered{};
        for (unsigned i = 0; i < sequential.size(); ++i) sequential[i] = reordered[i] = i;
        const auto execute = [](auto &memory, const Operation &op) {
            std::vector<unsigned> snapshot(memory.begin() + op.from, memory.begin() + op.from + op.count);
            for (unsigned i = 0; i < op.count; ++i) memory[op.to + i] = snapshot[i] + 1;
        };
        const auto flush = [&] {
            for (auto it = epoch.rbegin(); it != epoch.rend(); ++it) execute(reordered, *it);
            assert(sequential == reordered);
            epoch.clear();
        };
        uint8_t stages = 0;
        for (unsigned command = 0; command < 256; ++command) {
            Operation op{rng()%4000, rng()%4000, 1 + rng()%64,
                rng()%9 == 0 ? CommandDependencies::dispatch : CommandDependencies::blit, false};
            op.opaque = op.stage == CommandDependencies::dispatch;
            const auto access = [&](unsigned offset, bool write) {
                CommandAccess a{}; a.opaque = op.opaque; a.write = write;
                a.cpu_begin = 0x10000 + offset; a.cpu_end = a.cpu_begin + op.count;
                a.gpu_begin = 0x100000 + offset + (rng()%2)*0x10000;
                a.gpu_end = a.gpu_begin + op.count;
                return a;
            };
            CommandAccess accesses[]{access(op.from, false), access(op.to, true)};
            const auto barrier = tracker.prepare(op.stage, accesses);
            if (barrier) { assert(barrier == stages); flush(); stages = 0; ++barriers; }
            else {
                for (const auto &previous : epoch) {
                    assert(!op.opaque && !previous.opaque);
                    // Literal byte membership, independent of interval formula.
                    for (unsigned i = 0; i < op.count; ++i)
                        for (unsigned j = 0; j < previous.count; ++j) {
                            assert(op.to+i != previous.to+j);
                            assert(op.from+i != previous.to+j);
                            assert(op.to+i != previous.from+j);
                        }
                }
                skipped += !epoch.empty();
            }
            execute(sequential, op); epoch.push_back(op); stages |= op.stage; ++operations;
        }
        flush();
    }
    assert(skipped > 100000 && barriers > 1000);
    // Different identity, same heap; view identity; read/read; exact GPU alias.
    for (unsigned kind = 0; kind < 3; ++kind) for (bool read_only : {false, true}) {
        CommandDependencies tracker;
        CommandAccess a{}, b{}; a.opaque = b.opaque = false;
        if (kind == 0) { a.heap = b.heap = 7; a.texture = 1; b.texture = 2; }
        if (kind == 1) a.texture = b.texture = 3;
        if (kind == 2) { a.gpu_begin = 100; a.gpu_end = 200; b.gpu_begin = 150; b.gpu_end = 250; }
        a.write = false; b.write = !read_only;
        assert(!tracker.prepare(CommandDependencies::blit, {&a, 1}));
        assert(bool(tracker.prepare(CommandDependencies::blit, {&b, 1})) == !read_only);
    }
    CommandDependencies capacity;
    for (unsigned i = 0; i < 65; ++i) {
        CommandAccess a{}; a.opaque = false; a.texture = i + 1;
        assert(capacity.prepare(CommandDependencies::blit, {&a, 1}) == (i == 64 ? CommandDependencies::blit : 0));
    }
    assert(capacity.prepare(CommandDependencies::dispatch, {}) == CommandDependencies::blit);
    std::array<CommandAccess, 100> oversized{};
    assert(capacity.prepare(CommandDependencies::blit, oversized) == CommandDependencies::dispatch);
    assert(capacity.prepare(CommandDependencies::dispatch, {}) == CommandDependencies::blit);
    // Alternating stages: forgetting an old epoch is safe only with both targets.
    CommandDependencies chain;
    CommandAccess a{}; a.opaque = false; a.texture = 1;
    assert(!chain.prepare(CommandDependencies::dispatch, {&a, 1}));
    a.texture = 2; assert(!chain.prepare(CommandDependencies::blit, {&a, 1}));
    a.texture = 1; assert(chain.prepare(CommandDependencies::blit, {&a, 1}) == 3);
    std::cout << "PASS command dependency planner: " << operations << " commands, reverse-epoch byte oracle, "
        << barriers << " barriers / " << skipped << " independent commands; RAW/WAR/WAW, CPU/GPU/heap/view aliases, capacity/opaque/stage chain\n";
}
