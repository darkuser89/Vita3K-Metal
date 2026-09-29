// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/buffers.h>
#include <gxm/types.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstring>

namespace renderer::metal {
bool memory_backed_uniform_slot(uint32_t buffer_flags, uint32_t renderer_slot) {
    if (renderer_slot >= SCE_GXM_REAL_MAX_UNIFORM_BUFFER) return false;
    const uint32_t guest_slot = renderer_slot == SCE_GXM_DEFAULT_UNIFORM_BUFFER_CONTAINER_INDEX
        ? SCE_GXM_MAX_UNIFORM_BUFFERS : renderer_slot - SCE_GXM_UNIFORM_BUFFER_OFFSET;
    return (buffer_flags & (2u << (2 * guest_slot))) != 0;
}
void MappedGuestRegions::map(uint32_t address, uint32_t size) {
    if (size) regions[address] = size;
}
void MappedGuestRegions::unmap(uint32_t address) {
    regions.erase(address);
}
size_t MappedGuestRegions::extent(uint32_t address, size_t bound_size) const {
    const uint64_t bound_end = uint64_t(address) + bound_size;
    size_t result = bound_size;
    for (auto region = regions.upper_bound(address); region != regions.begin();) {
        --region;
        const uint64_t end = uint64_t(region->first) + region->second;
        if (bound_end <= end) result = std::max(result, size_t(end - address));
    }
    return result;
}
void configure_vertex_stream(MTLVertexBufferLayoutDescriptor *binding, size_t stride, size_t extent, bool per_instance) {
    binding.stride = ((stride ? stride : extent) + 3) & ~size_t(3);
    binding.stepFunction = !stride ? MTLVertexStepFunctionConstant
        : per_instance ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
    binding.stepRate = stride ? 1 : 0;
}
UploadBufferSlice UploadBufferArena::allocate(Device &device, size_t length, size_t alignment) {
    if (!length || !alignment || (alignment & (alignment - 1))
        || length > device.native_device().maxBufferLength)
        throw std::invalid_argument("Metal: invalid upload allocation");
    for (; current < blocks.size(); ++current) {
        auto &block = blocks[current];
        if (block.used > std::numeric_limits<size_t>::max() - (alignment - 1)) continue;
        const size_t offset = (block.used + alignment - 1) & ~(alignment - 1);
        if (offset <= block.buffer.length && length <= block.buffer.length - offset) {
            block.used = offset + length;
            return {block.buffer, offset};
        }
    }
    const size_t capacity = std::min(size_t(device.native_device().maxBufferLength), std::max(block_size, length));
    id<MTLBuffer> buffer = [device.native_device() newBufferWithLength:capacity options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("Metal: upload buffer allocation failed");
    blocks.push_back({buffer, length});
    return {buffer, 0};
}
void UploadBufferArena::reset_after_completion() {
    // Avoid retaining the largest exceptional draw for the entire game.
    size_t retained = 0, keep = 0;
    for (auto &block : blocks) {
        if (block.buffer.length > 32 * 1024 * 1024 - retained) break;
        retained += block.buffer.length;
        block.used = 0;
        ++keep;
    }
    blocks.resize(keep);
    current = 0;
}
id<MTLBuffer> DirectGuestBufferCache::get(Device &device, uintptr_t begin, uintptr_t end) {
    if (begin >= end || end - begin > device.native_device().maxBufferLength)
        throw std::invalid_argument("Metal: invalid direct guest buffer range");
    const auto key = std::make_pair(begin, end);
    if (const auto found = buffers.find(key); found != buffers.end())
        return found->second;
    id<MTLBuffer> buffer = [device.native_device() newBufferWithBytesNoCopy:reinterpret_cast<void *>(begin)
        length:end - begin options:MTLResourceStorageModeShared deallocator:nil];
    if (!buffer || !buffer.gpuAddress)
        throw std::runtime_error("Metal: cannot map direct guest buffer");
    buffers.emplace(key, buffer);
    return buffer;
}
GuestBufferBindings::GuestBufferBindings(Device &device, std::span<const GuestBufferRange> ranges, size_t page_size, bool snapshot,
    UploadBufferArena *arena, DirectGuestBufferCache *direct_cache)
    : snapshots(snapshot) {
    if (!page_size || (page_size & (page_size - 1)))
        throw std::invalid_argument("Metal: invalid host page size");
    struct Pages { uintptr_t begin, end; bool direct; };
    std::vector<Pages> pages;
    for (const auto &range : ranges) {
        if (!range.size) continue;
        const auto begin = reinterpret_cast<uintptr_t>(range.data);
        if (!begin || range.size > std::numeric_limits<uintptr_t>::max() - begin
            || begin + range.size > std::numeric_limits<uintptr_t>::max() - (page_size - 1))
            throw std::invalid_argument("Metal: invalid guest buffer range");
        pages.push_back({begin & ~(page_size - 1), (begin + range.size + page_size - 1) & ~(page_size - 1), range.direct});
    }
    std::sort(pages.begin(), pages.end(), [](const Pages &a, const Pages &b) { return a.begin < b.begin; });
    for (const auto &[begin, end, direct] : pages) {
        // Aliases and adjacent guest pages share one Metal resource so GXP
        // address arithmetic can cross a boundary between bound ranges.
        if (!mappings.empty() && begin <= mappings.back().end) {
            mappings.back().end = std::max(mappings.back().end, end);
            mappings.back().direct |= direct;
            mappings.back().cacheable &= direct;
        } else mappings.push_back({begin, end, nil, 0, direct, direct});
    }
    for (auto &mapping : mappings) {
        const auto length = mapping.end - mapping.begin;
        if (length > device.native_device().maxBufferLength)
            throw std::runtime_error("Metal: guest buffer exceeds maximum native buffer length");
        if (snapshots && !mapping.direct) {
            if (arena) {
                const auto slice = arena->allocate(device, length, page_size);
                mapping.buffer = slice.buffer;
                mapping.offset = slice.offset;
            } else mapping.buffer = [device.native_device() newBufferWithLength:length options:MTLResourceStorageModeShared];
            // GXP LD/ST addresses can move beyond the uniform's declared
            // register span. The GPU binding covers whole guest pages, so its
            // asynchronous snapshot must preserve those pages as well.
            if (mapping.buffer)
                std::memcpy(static_cast<uint8_t *>(mapping.buffer.contents)+mapping.offset,
                    reinterpret_cast<const void *>(mapping.begin),length);
        } else if (mapping.cacheable && direct_cache)
            mapping.buffer = direct_cache->get(device, mapping.begin, mapping.end);
        else mapping.buffer = [device.native_device() newBufferWithBytesNoCopy:reinterpret_cast<void *>(mapping.begin)
            length:length options:MTLResourceStorageModeShared deallocator:nil];
        if (!mapping.buffer || !mapping.buffer.gpuAddress)
            throw std::runtime_error("Metal: cannot map guest uniform buffer");
    }
}
size_t GuestBufferBindings::allocated_bytes() const {
    size_t bytes=0;
    for(const auto &mapping:mappings) if (snapshots && !mapping.direct) bytes+=mapping.end-mapping.begin;
    return bytes;
}
uint64_t GuestBufferBindings::address(const GuestBufferRange &range) const {
    if (!range.size) return 0;
    const auto begin = reinterpret_cast<uintptr_t>(range.data);
    for (const auto &mapping : mappings)
        if (begin >= mapping.begin && begin < mapping.end && range.size <= mapping.end - begin)
            return mapping.buffer.gpuAddress + mapping.offset + (begin - mapping.begin);
    throw std::out_of_range("Metal: uniform buffer is not mapped");
}
void GuestBufferBindings::make_resident(id<MTLRenderCommandEncoder> encoder) const {
    for (const auto &mapping : mappings)
        [encoder useResource:mapping.buffer usage:snapshots && !mapping.direct ? MTLResourceUsageRead : MTLResourceUsageRead | MTLResourceUsageWrite stages:MTLRenderStageVertex | MTLRenderStageFragment];
}
}
