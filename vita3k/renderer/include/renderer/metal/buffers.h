// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal/device.h>
#include <map>
#include <span>
#include <vector>

namespace renderer::metal {
// GXM's default uniform buffer is guest slot 14 but renderer slot 0.
// Explicit guest slots 0..13 occupy renderer slots 1..14.
bool memory_backed_uniform_slot(uint32_t buffer_flags, uint32_t renderer_slot);
// A zero guest stride broadcasts one record, regardless of vertex/instance ID.
void configure_vertex_stream(MTLVertexBufferLayoutDescriptor *, size_t stride, size_t extent, bool per_instance);
struct GuestBufferRange {
    uint8_t *data = nullptr;
    size_t size = 0;
    bool direct = false;
};
struct MappedGuestRange {
    uint32_t address;
    size_t size;
    bool mapped = false;
};
// Preserve GXM map extents for shaders that use memory-backed uniform slots.
// An overlapping later map must not hide a larger earlier region.
class MappedGuestRegions {
    struct Region {
        uint32_t size;
        uint64_t prefix_end;
    };
    std::map<uint32_t, Region> regions;
    void update_prefixes(std::map<uint32_t, Region>::iterator first);
public:
    void map(uint32_t address, uint32_t size);
    void unmap(uint32_t address);
    MappedGuestRange range(uint32_t address, size_t bound_size) const;
    size_t extent(uint32_t address, size_t bound_size) const;
};
struct UploadBufferSlice {
    id<MTLBuffer> buffer = nil;
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
    UploadBufferArena(UploadBufferArena &&) = default;
    UploadBufferArena &operator=(UploadBufferArena &&) = default;
    UploadBufferSlice allocate(Device &, size_t length, size_t alignment = 256);
    void reset_after_completion();
};
// A GXM-mapped guest region stays live until its ordered Unmap command.
// Reuse its no-copy Metal resource across draws and drop it after the GPU
// has finished, before guest pages can be released or reused.
class DirectGuestBufferCache {
    std::map<std::pair<uintptr_t, uintptr_t>, id<MTLBuffer>> buffers;
public:
    UploadBufferSlice get(Device &, uintptr_t begin, uintptr_t end);
    void clear() { buffers.clear(); }
    size_t size() const { return buffers.size(); }
};
// Borrowed mappings require the guest pages until GPU completion. Snapshot
// mappings instead own copied input bytes; make_resident lets the command buffer
// retain them after this wrapper and the original guest pages are released.
class GuestBufferBindings {
    struct Mapping {
        uintptr_t begin, end;
        id<MTLBuffer> buffer;
        size_t offset = 0;
        bool direct = false;
        bool cacheable = false;
    };
    std::vector<Mapping> mappings;
    bool snapshots;
public:
    GuestBufferBindings(Device &, std::span<const GuestBufferRange>, size_t host_page_size, bool snapshot = false,
        UploadBufferArena *arena = nullptr, DirectGuestBufferCache *direct_cache = nullptr);
    uint64_t address(const GuestBufferRange &) const;
    size_t allocated_bytes() const;
    void make_resident(id<MTLRenderCommandEncoder>) const;
};
}
