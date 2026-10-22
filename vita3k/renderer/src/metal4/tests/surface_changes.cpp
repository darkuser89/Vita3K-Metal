// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/surface_changes.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

using Range = std::pair<size_t, size_t>;
using Ranges = std::vector<Range>;

// The previous set_context enumeration, including its full memcmp fast path.
static Ranges reference(std::span<const uint8_t> cpu, std::span<const uint8_t> snapshot) {
    Ranges changes;
    const auto bytes = cpu.size();
    if (bytes && std::memcmp(cpu.data(), snapshot.data(), bytes) != 0)
        for (size_t at = 0; at < bytes;) {
            if (cpu[at] == snapshot[at]) { ++at; continue; }
            const size_t first = at++;
            while (at < bytes && cpu[at] != snapshot[at]) ++at;
            changes.emplace_back(first, at - first);
        }
    return changes;
}

static Ranges actual(std::span<const uint8_t> cpu, std::span<const uint8_t> snapshot) {
    Ranges changes;
    renderer::metal::for_each_surface_byte_change(cpu, snapshot,
        [&](size_t offset, size_t size) { changes.emplace_back(offset, size); });
    return changes;
}

static size_t cases = 0;
static void check(std::span<const uint8_t> cpu, std::span<const uint8_t> snapshot) {
    const auto expected = reference(cpu, snapshot);
    const auto got = actual(cpu, snapshot);
    assert(got == expected);
    size_t end = 0;
    std::vector<uint8_t> patched(snapshot.begin(), snapshot.end());
    for (const auto &[offset, size] : got) {
        assert(size && offset + size <= cpu.size());
        assert(!end || offset > end); // Adjacent runs must merge across blocks.
        for (size_t i = offset; i < offset + size; ++i) assert(cpu[i] != snapshot[i]);
        std::memcpy(patched.data() + offset, cpu.data() + offset, size);
        end = offset + size;
    }
    assert(std::equal(patched.begin(), patched.end(), cpu.begin(), cpu.end()));
    ++cases;
}

int main() {
    check({}, {});
    // Exhaust all masks within a short range: unchanged bytes between edits
    // must never be included in an upload (MSAA patch ownership depends on it).
    std::vector<uint8_t> baseline(18, 0x5a), cpu = baseline;
    for (unsigned mask = 0; mask < (1u << 18); ++mask) {
        for (unsigned bit = 0; bit < 18; ++bit)
            cpu[bit] = baseline[bit] ^ ((mask >> bit) & 1);
        check(cpu, baseline);
    }

    // Exact allocation ends are ASAN redzones. Cover all four partial/full
    // block ends and deliberately unaligned input addresses independently.
    for (size_t size : {1, 2, 31, 4095, 4096, 4097, 8191, 8192, 8193, 16391})
        for (size_t left = 0; left < 8; ++left)
            for (size_t right = 0; right < 8; ++right) {
                auto a = std::make_unique<uint8_t[]>(size + left);
                auto b = std::make_unique<uint8_t[]>(size + right);
                std::span<uint8_t> current{a.get() + left, size};
                std::span<uint8_t> previous{b.get() + right, size};
                std::fill(current.begin(), current.end(), 0x17);
                std::fill(previous.begin(), previous.end(), 0x17);
                check(current, previous);
                current.front() ^= 1; check(current, previous);
                current.back() ^= 2; check(current, previous);
                for (size_t i = 4094; i < std::min(size, size_t(8195)); ++i) current[i] ^= 4;
                check(current, previous);
                std::fill(current.begin(), current.end(), 0xee); check(current, previous);
                for (size_t i = 0; i < size; i += 2) current[i] = previous[i];
                check(current, previous);
            }

    // Random sparse/dense/multi-block changes, including long runs that span
    // several blocks and runs ending exactly at a block boundary.
    std::mt19937 random(0x4d345355);
    for (unsigned trial = 0; trial < 3000; ++trial) {
        const size_t size = random() % 65537;
        baseline.resize(size);
        for (auto &byte : baseline) byte = uint8_t(random());
        cpu = baseline;
        for (unsigned edit = 0; size && edit < trial % 19; ++edit) {
            const size_t offset = random() % size;
            const size_t count = std::min(size - offset, size_t(random() % 16385));
            for (size_t i = offset; i < offset + count; ++i) cpu[i] ^= 0x81;
        }
        check(cpu, baseline);
    }

    // Same-value and write-restore events produce no byte patch; the caller's
    // independent epoch branch must retain its full-surface import policy.
    baseline.assign(32768, 19); cpu = baseline;
    cpu[4100] = 19; check(cpu, baseline);
    cpu[4100] = 20; cpu[4100] = 19; check(cpu, baseline);
    assert(actual(cpu, baseline).empty());
    for (size_t offset : {size_t(0), size_t(4095), size_t(4096), size_t(8192), cpu.size() - 1}) {
        cpu[offset] ^= 1; check(cpu, baseline); cpu[offset] ^= 1;
    }
    // No shared/static enumeration state may survive an allocation failure
    // in the consumer. A retry enumerates the exact same complete result.
    cpu[1] ^= 1; cpu[9000] ^= 1;
    bool threw = false;
    try {
        renderer::metal::for_each_surface_byte_change(cpu, baseline,
            [](size_t, size_t) { throw std::runtime_error("consumer"); });
    } catch (const std::runtime_error &) { threw = true; }
    assert(threw); check(cpu, baseline);
    threw = false;
    try { actual(std::span(cpu).first(1), baseline); }
    catch (const std::invalid_argument &) { threw = true; }
    assert(threw);
    std::cout << "PASS surface changes: " << cases
              << " exact-range comparisons against original enumeration; sparse/dense/block-edge/unaligned/empty spans and failed-consumer retry; ASAN/UBSAN\n";
}
