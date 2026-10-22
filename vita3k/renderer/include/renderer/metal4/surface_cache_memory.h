// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#import <Metal/Metal.h>
#include <renderer/metal4/surface_cache_budget.h>
#include <functional>
#include <cassert>
#include <tuple>
#include <type_traits>

namespace renderer::metal {
inline SurfaceCacheBudget::Allocation surface_allocation(id<MTLTexture> texture) {
    if (!texture) return {};
    while (texture.parentTexture) texture = texture.parentTexture;
    id<MTLResource> allocation = texture.buffer ? (id<MTLResource>)texture.buffer : texture;
    return {reinterpret_cast<uintptr_t>((__bridge void *)allocation), uint64_t(allocation.allocatedSize)};
}

// Call at a completed presentation boundary. Only reproducible derived caches
// are removed. Authoritative native color/depth/MSAA/raw/mask contents, guest
// snapshots, write watches and context bindings remain pinned.
template <class State>
SurfaceCacheBudget::Plan trim_derived_surface_cache(State &state,
    uint64_t bytes = 128ull * 1024 * 1024, size_t entries = 256) {
    SurfaceCacheBudget budget;
    std::vector<std::function<void()>> remove;
    const auto pin = [&](id<MTLTexture> texture) { budget.pin(surface_allocation(texture)); };
    const auto maps = [&](uint64_t age, auto &... caches) {
        const size_t count = (caches.size() + ...);
        if (!count) return;
        std::vector<SurfaceCacheBudget::Allocation> allocations;
        const auto collect = [&](const auto &cache) {
            for (const auto &[key, texture] : cache) allocations.push_back(surface_allocation(texture));
        };
        (collect(caches), ...);
        const uint64_t payload = (uint64_t(caches.size() * sizeof(typename std::remove_reference_t<decltype(caches)>::value_type)) + ...);
        [[maybe_unused]] const auto index = budget.add_group(age, std::move(allocations), payload, count);
        assert(index == remove.size());
        remove.emplace_back([tables = std::tie(caches...)] { std::apply([](auto &... table) { (table.clear(), ...); }, tables); });
    };
    for (auto &[address, surface] : state.surfaces) {
        pin(surface.color); pin(surface.raw_color); pin(surface.multisample_color); pin(surface.raw_multisample_color);
        budget.pin_payload(surface.cpu_snapshot.capacity());
        maps(surface.last_frame_rendered, surface.rgba8_casts, surface.word_casts, surface.word_rect_casts,
            surface.halfword_rect_casts, surface.byte_rect_casts, surface.small_rect_casts, surface.subrectangles);
    }
    for (auto &[key, surface] : state.depth_surfaces) {
        pin(surface.texture); pin(surface.mask); pin(surface.sample_rate_copy); pin(surface.sample_rate_mask);
        budget.pin_payload(surface.guest_depth_snapshot.capacity());
        budget.pin_payload(surface.guest_stencil_snapshot.capacity());
        maps(surface.last_attached_frame, surface.snapshots, surface.subrectangles);
    }
    const auto images = [&](auto &cache) {
        for (const auto &[key, image] : cache) {
            std::vector<SurfaceCacheBudget::Allocation> allocations{surface_allocation(image.texture), surface_allocation(image.uploaded)};
            for (auto source : image.sources) allocations.push_back(surface_allocation(source));
            const uint64_t payload = sizeof(image) + key.capacity() + image.sources.capacity() * sizeof(id<MTLTexture>);
            [[maybe_unused]] const auto index = budget.add_group(image.last_used_frame, std::move(allocations), payload, 1);
            assert(index == remove.size());
            remove.emplace_back([&cache, key] { cache.erase(key); });
        }
    };
    images(state.rendered_images); images(state.rendered_cubes);
    for (const auto &[key, image] : state.cube_aliases) {
        [[maybe_unused]] const auto index = budget.add_group(image.last_used_frame,
            {surface_allocation(image.source), surface_allocation(image.cube)}, sizeof(image), 1);
        assert(index == remove.size());
        remove.emplace_back([&cache = state.cube_aliases, key] { cache.erase(key); });
    }
    auto plan = budget.plan(bytes, entries);
    for (auto index : plan.evict) remove.at(index)();
    return plan;
}
}
