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
void MappedGuestRegions::update_prefixes(std::map<uint32_t, Region>::iterator first) {
    uint64_t prefix_end = first == regions.begin() ? 0 : std::prev(first)->second.prefix_end;
    for (auto region = first; region != regions.end(); ++region) {
        prefix_end = std::max(prefix_end, uint64_t(region->first) + region->second.size);
        if (region->second.prefix_end == prefix_end)
            break;
        region->second.prefix_end = prefix_end;
    }
}
void MappedGuestRegions::map(uint32_t address, uint32_t size) {
    if (!size) return;
    update_prefixes(regions.insert_or_assign(address, Region{size, 0}).first);
}
void MappedGuestRegions::unmap(uint32_t address) {
    if (auto region = regions.find(address); region != regions.end())
        update_prefixes(regions.erase(region));
}
MappedGuestRange MappedGuestRegions::range(uint32_t address, size_t bound_size) const {
    const uint64_t bound_end = uint64_t(address) + bound_size;
    uint32_t begin = address;
    uint64_t end = bound_end;
    bool mapped = false;
    for (auto region = regions.upper_bound(address); region != regions.begin();) {
        --region;
        // If this prefix cannot reach the binding's end, no earlier mapping
        // can contain it either. Most draws inspect only their nearest map.
        if (region->second.prefix_end < bound_end)
            break;
        const uint64_t region_end = uint64_t(region->first) + region->second.size;
        if (bound_end <= region_end) {
            // Every included range contains the binding, so their union has
            // no gaps. Keep the prefix too: GXP offsets can be negative.
            begin = std::min(begin, region->first);
            end = std::max(end, region_end);
            mapped = true;
        }
    }
    return {begin, size_t(end - begin), mapped};
}
size_t MappedGuestRegions::extent(uint32_t address, size_t bound_size) const {
    const auto mapped = range(address, bound_size);
    return mapped.size - (address - mapped.address);
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
UploadBufferSlice DirectGuestBufferCache::get(Device &device, uintptr_t begin, uintptr_t end) {
    if (begin >= end || end - begin > device.native_device().maxBufferLength)
        throw std::invalid_argument("Metal: invalid direct guest buffer range");
    const auto key = std::make_pair(begin, end);
    if (const auto found = buffers.find(key); found != buffers.end())
        return {found->second, 0};
    // Uniform addresses often advance through one large GXM mapping. Reuse a
    // containing resource instead of registering its remaining pages again.
    for (const auto &[range, buffer] : buffers) {
        if (range.first > begin) break;
        if (range.second >= end) return {buffer, begin - range.first};
    }
    id<MTLBuffer> buffer = [device.native_device() newBufferWithBytesNoCopy:reinterpret_cast<void *>(begin)
        length:end - begin options:MTLResourceStorageModeShared deallocator:nil];
    if (!buffer || !buffer.gpuAddress)
        throw std::runtime_error("Metal: cannot map direct guest buffer");
    // Already encoded commands retain their resources through useResource.
    // Drop cache ownership of narrower aliases now covered by this buffer.
    for (auto entry = buffers.lower_bound({begin, 0}); entry != buffers.end();) {
        if (entry->first.first >= end) break;
        if (entry->first.second <= end) entry = buffers.erase(entry);
        else ++entry;
    }
    buffers.emplace(key, buffer);
    return {buffer, 0};
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
        } else mappings.push_back({begin, end, nil, 0, direct, direct});
    }
    for (auto &mapping : mappings) {
        // A register-only binding inside mapped guest pages does not shorten
        // those pages' lifetime. Determine coverage of the merged resource
        // from the direct ranges, instead of rejecting every mixed alias.
        // An adjacent transient page outside that coverage must remain uncached.
        uintptr_t covered = mapping.begin;
        for (const auto &range : pages) {
            if (range.begin > covered) break;
            if (range.direct && range.end > covered) covered = range.end;
            if (covered >= mapping.end) break;
        }
        mapping.cacheable = covered >= mapping.end;
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
        } else if (mapping.cacheable && direct_cache) {
            const auto slice = direct_cache->get(device, mapping.begin, mapping.end);
            mapping.buffer = slice.buffer;
            mapping.offset = slice.offset;
        }
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
