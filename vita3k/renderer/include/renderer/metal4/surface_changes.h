// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

namespace renderer::metal {
// Enumerate maximal runs of changed bytes, including partial edge blocks.
// These are comparison blocks, not memory-protection pages: no byte is
// assumed clean from a write epoch (renderer/GPU writes can suppress it).
template <typename Emit>
void for_each_surface_byte_change(std::span<const uint8_t> bytes,
    std::span<const uint8_t> snapshot, Emit emit) {
    if (bytes.size() != snapshot.size())
        throw std::invalid_argument("Metal: mismatched surface comparison spans");
    constexpr size_t block_size = 4096;
    size_t first = 0;
    bool in_change = false;
    for (size_t block = 0; block < bytes.size();) {
        const size_t end = block + std::min(block_size, bytes.size() - block);
        // Most CPU patches touch a small part of the surface. Do not scan
        // every unchanged byte individually after finding the first patch.
        if (std::memcmp(bytes.data() + block, snapshot.data() + block, end - block) == 0) {
            if (in_change) {
                emit(first, block - first);
                in_change = false;
            }
        } else {
            for (size_t at = block; at < end; ++at) {
                if (bytes[at] != snapshot[at]) {
                    if (!in_change) first = at;
                    in_change = true;
                } else if (in_change) {
                    emit(first, at - first);
                    in_change = false;
                }
            }
        }
        block = end;
    }
    if (in_change) emit(first, bytes.size() - first);
}
}
