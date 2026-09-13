// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/buffers.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstring>

namespace renderer::metal {
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
GuestBufferBindings::GuestBufferBindings(Device &device, std::span<const GuestBufferRange> ranges, size_t page_size, bool snapshot,
    UploadBufferArena *arena)
    : snapshots(snapshot) {
    if (!page_size || (page_size & (page_size - 1)))
        throw std::invalid_argument("Metal: invalid host page size");
    std::vector<std::pair<uintptr_t, uintptr_t>> pages;
    for (const auto &range : ranges) {
        if (!range.size) continue;
        const auto begin = reinterpret_cast<uintptr_t>(range.data);
        if (!begin || range.size > std::numeric_limits<uintptr_t>::max() - begin
            || begin + range.size > std::numeric_limits<uintptr_t>::max() - (page_size - 1))
            throw std::invalid_argument("Metal: invalid guest buffer range");
        pages.emplace_back(begin & ~(page_size - 1), (begin + range.size + page_size - 1) & ~(page_size - 1));
    }
    std::sort(pages.begin(), pages.end());
    for (const auto &[begin, end] : pages) {
        // Aliases share one Metal resource and hence one GPU address space.
        if (!mappings.empty() && begin < mappings.back().end)
            mappings.back().end = std::max(mappings.back().end, end);
        else mappings.push_back({begin, end, nil});
    }
    for (auto &mapping : mappings) {
        const auto length = mapping.end - mapping.begin;
        if (length > device.native_device().maxBufferLength)
            throw std::runtime_error("Metal: guest buffer exceeds maximum native buffer length");
        if (snapshots) {
            if (arena) {
                const auto slice = arena->allocate(device, length, page_size);
                mapping.buffer = slice.buffer;
                mapping.offset = slice.offset;
            } else mapping.buffer = [device.native_device() newBufferWithLength:length options:MTLResourceStorageModeShared];
            if (mapping.buffer) for (const auto &range:ranges) {
                const auto begin=reinterpret_cast<uintptr_t>(range.data);
                if (range.size && begin>=mapping.begin && begin<mapping.end && range.size<=mapping.end-begin)
                    std::memcpy(static_cast<uint8_t *>(mapping.buffer.contents)+mapping.offset+(begin-mapping.begin),range.data,range.size);
            }
        } else mapping.buffer = [device.native_device() newBufferWithBytesNoCopy:reinterpret_cast<void *>(mapping.begin)
            length:length options:MTLResourceStorageModeShared deallocator:nil];
        if (!mapping.buffer || !mapping.buffer.gpuAddress)
            throw std::runtime_error("Metal: cannot map guest uniform buffer");
    }
}
size_t GuestBufferBindings::allocated_bytes() const {
    size_t bytes=0;
    for(const auto &mapping:mappings) bytes+=mapping.end-mapping.begin;
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
        [encoder useResource:mapping.buffer usage:snapshots ? MTLResourceUsageRead : MTLResourceUsageRead | MTLResourceUsageWrite stages:MTLRenderStageVertex | MTLRenderStageFragment];
}
}
