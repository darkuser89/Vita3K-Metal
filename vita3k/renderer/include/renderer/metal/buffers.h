// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal/device.h>
#include <span>
#include <vector>

namespace renderer::metal {
// A zero guest stride broadcasts one record, regardless of vertex/instance ID.
void configure_vertex_stream(MTLVertexBufferLayoutDescriptor *, size_t stride, size_t extent, bool per_instance);
struct GuestBufferRange {
    uint8_t *data = nullptr;
    size_t size = 0;
};
struct UploadBufferSlice {
    id<MTLBuffer> buffer;
    size_t offset = 0;
};
// Each allocation owns a distinct slice until reset_after_completion(). The
// caller must wait for every command using these slices before resetting.
class UploadBufferArena {
    struct Block {
        id<MTLBuffer> buffer;
        size_t used = 0;
    };
    std::vector<Block> blocks;
    size_t current = 0;
    size_t block_size;
public:
    explicit UploadBufferArena(size_t block_size = 1024 * 1024) : block_size(block_size) {}
    UploadBufferArena(const UploadBufferArena &) = delete;
    UploadBufferArena &operator=(const UploadBufferArena &) = delete;
    UploadBufferSlice allocate(Device &, size_t length, size_t alignment = 256);
    void reset_after_completion();
};
// Borrowed mappings require the guest pages until GPU completion. Snapshot
// mappings instead own copied input bytes; make_resident lets the command buffer
// retain them after this wrapper and the original guest pages are released.
class GuestBufferBindings {
    struct Mapping {
        uintptr_t begin, end;
        id<MTLBuffer> buffer;
        size_t offset = 0;
    };
    std::vector<Mapping> mappings;
    bool snapshots;
public:
    GuestBufferBindings(Device &, std::span<const GuestBufferRange>, size_t host_page_size, bool snapshot = false,
        UploadBufferArena *arena = nullptr);
    uint64_t address(const GuestBufferRange &) const;
    size_t allocated_bytes() const;
    void make_resident(id<MTLRenderCommandEncoder>) const;
};
}
