// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// CPU accounting and Metal resource-ownership doubles; no device or GPU.
#import <Foundation/Foundation.h>
#include <renderer/metal4/surface_cache_memory.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#define CHECK(...) do { if (!(__VA_ARGS__)) throw std::runtime_error("check failed at " + std::to_string(__LINE__)); } while (0)
using namespace renderer::metal;

static void planner_tests() {
    std::mt19937 random(0x43414348);
    for (unsigned trial = 0; trial < 10000; ++trial) {
        SurfaceCacheBudget budget;
        struct Root { uint64_t bytes; bool pinned; };
        struct Group { uint64_t age, payload; size_t entries; std::set<uintptr_t> roots; };
        std::map<uintptr_t, Root> roots;
        std::vector<Group> groups;
        const uint64_t pinned_payload = random() % 1024;
        budget.pin_payload(pinned_payload);
        for (uintptr_t identity = 1; identity <= 32; ++identity) {
            roots[identity] = {random() % 8192, random() % 5 == 0};
            if (roots[identity].pinned) budget.pin({identity, roots[identity].bytes});
        }
        for (unsigned n = 0, count = random() % 40; n < count; ++n) {
            Group group{random() % 8, random() % 512, random() % 12, {}};
            std::vector<SurfaceCacheBudget::Allocation> allocations;
            for (unsigned j = 0, count = random() % 15; j < count; ++j) {
                const uintptr_t id = 1 + random() % 32;
                group.roots.insert(id); allocations.push_back({id, roots[id].bytes});
            }
            CHECK(budget.add_group(group.age, allocations, group.payload, group.entries) == groups.size());
            groups.push_back(group);
        }
        std::vector<size_t> order(groups.size()); std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return groups[a].age < groups[b].age; });
        std::set<size_t> deleted;
        const auto measure = [&] {
            std::set<uintptr_t> used;
            uint64_t bytes = 0; size_t entries = 0;
            for (size_t i = 0; i < groups.size(); ++i) if (!deleted.contains(i)) {
                bytes += groups[i].payload; entries += groups[i].entries;
                used.insert(groups[i].roots.begin(), groups[i].roots.end());
            }
            for (auto id : used) if (!roots.at(id).pinned) bytes += roots.at(id).bytes;
            return std::pair{bytes, entries};
        };
        uint64_t pinned = pinned_payload;
        for (const auto &[id, root] : roots) if (root.pinned) pinned += root.bytes;
        const auto before = measure();
        const uint64_t bytes = random() % 40000; const size_t entries = random() % 100;
        const auto plan = budget.plan(bytes, entries);
        std::vector<size_t> expected;
        for (auto index : order) {
            const auto current = measure();
            if (current.first <= bytes && current.second <= entries) break;
            deleted.insert(index); expected.push_back(index);
        }
        CHECK(plan.evict == expected && plan.pinned_bytes == pinned);
        CHECK(plan.derived_before == before.first && plan.entries_before == before.second);
        CHECK(plan.derived_after == measure().first && plan.entries_after == measure().second);
        CHECK(budget.plan(bytes, entries).evict == plan.evict); // Pure, repeatable plan.
    }
    SurfaceCacheBudget overflow;
    overflow.pin_payload(UINT64_MAX);
    bool rejected = false;
    try { overflow.pin_payload(1); } catch (const std::overflow_error &) { rejected = true; }
    CHECK(rejected);
    SurfaceCacheBudget aliased;
    aliased.add_group(1, {{7, 100}, {7, 200}}, 0, 1);
    aliased.add_group(2, {{7, 300}}, 0, 1);
    auto p = aliased.plan(299, 5);
    CHECK(p.derived_before == 300 && p.derived_after == 0 && p.evict == std::vector<size_t>({0,1}));
    aliased.pin({7, 400}); p = aliased.plan(0, 5);
    CHECK(p.pinned_bytes == 400 && !p.derived_before && p.evict.empty());
}

@interface CacheResource : NSObject
@property NSUInteger allocatedSize;
@property(strong) id<MTLTexture> parentTexture;
@property(strong) id<MTLBuffer> buffer;
@end
@implementation CacheResource
@end
static id<MTLTexture> texture(NSUInteger bytes, id<MTLTexture> parent = nil, id<MTLBuffer> buffer = nil) {
    auto r = [CacheResource new]; r.allocatedSize = bytes; r.parentTexture = parent; r.buffer = buffer;
    return (id<MTLTexture>)r;
}
@interface BudgetCommands : NSObject
@property BOOL completed;
@property BOOL presented;
- (void)presentDrawable:(id)d;
@end
@implementation BudgetCommands
- (void)presentDrawable:(id)d { CHECK(d); _presented = YES; }
@end
using TextureMap = std::map<unsigned, id<MTLTexture>>;
struct Color {
    id<MTLTexture> color, raw_color, multisample_color, raw_multisample_color;
    TextureMap rgba8_casts, word_casts, word_rect_casts, halfword_rect_casts, byte_rect_casts, small_rect_casts, subrectangles;
    std::vector<uint8_t> cpu_snapshot;
    uint64_t last_frame_rendered = 0, revision = 123, watch_epoch = 456;
};
struct Depth {
    id<MTLTexture> texture, mask, sample_rate_copy, sample_rate_mask;
    TextureMap snapshots, subrectangles;
    std::vector<uint8_t> guest_depth_snapshot, guest_stencil_snapshot;
    uint64_t last_attached_frame = 0;
    bool depth_content_stored = false;
};
struct Image { id<MTLTexture> texture, uploaded; std::vector<id<MTLTexture>> sources; uint64_t last_used_frame = 0; };
struct Alias { id<MTLTexture> source, cube; uint64_t last_used_frame = 0; };
struct Device {
    bool fail = false;
    bool submit_and_wait(BudgetCommands *c, std::string &) { if (fail) return false; c.completed = YES; return true; }
};
struct Scratch { void completed(BudgetCommands *c) { CHECK(c.completed); } };
struct State {
    std::map<unsigned, Color> surfaces;
    std::map<std::pair<unsigned,unsigned>, Depth> depth_surfaces;
    std::map<std::string, Image> rendered_images, rendered_cubes;
    std::map<uintptr_t, Alias> cube_aliases;
    std::unique_ptr<Device> device = std::make_unique<Device>();
    BudgetCommands *screen_commands;
    id drawable;
    Scratch presentation_scratch;
    SurfaceCacheBudget::Plan surface_cache_budget;
};
struct MetalState { std::unique_ptr<State> impl = std::make_unique<State>(); void swap_window(); };
static void require(bool success, const std::string &) { if (!success) throw std::runtime_error("injected submission error"); }
// SWAP_HOOK

int main() {
    @autoreleasepool {
        planner_tests();
        State state;
        auto base = texture(100), multisample = texture(400), derived = texture(1000);
        auto &color = state.surfaces[1];
        color.color = base; color.raw_color = texture(0, texture(0, base));
        color.multisample_color = multisample; color.raw_multisample_color = texture(0, multisample);
        color.cpu_snapshot.assign(64, 0xA5);
        color.rgba8_casts[1] = derived; color.subrectangles[1] = texture(0, base);
        auto &depth = state.depth_surfaces[{2,3}];
        depth.texture = texture(2000); depth.mask = texture(300);
        depth.sample_rate_copy = texture(100); depth.sample_rate_mask = texture(50);
        depth.guest_depth_snapshot.assign(32, 0x5A); depth.guest_stencil_snapshot.assign(16, 0xCC);
        depth.snapshots[1] = derived;
        state.rendered_images["image"] = {derived, texture(700), {base}, 1};
        state.rendered_cubes["cube"] = {texture(5000), nil, {base, multisample}, 2};
        state.cube_aliases[1] = {base, texture(3000), 3};
        auto buffer = (id<MTLBuffer>)texture(900);
        auto a = texture(0, nil, buffer), b = texture(0, nil, buffer);
        CHECK(surface_allocation(a).identity == surface_allocation(b).identity);
        CHECK(surface_allocation(a).bytes == 900);
        color.word_casts[1] = a; color.word_casts[2] = b;
        CHECK(surface_allocation(color.raw_color).identity == surface_allocation(base).identity);
        auto pending = @[derived, a, state.rendered_cubes.at("cube").texture];
        const auto original_color = color.cpu_snapshot, original_depth = depth.guest_depth_snapshot;
        const auto original_stencil = depth.guest_stencil_snapshot;
        auto plan = trim_derived_surface_cache(state, 0, 0);
        const uint64_t pinned = 2950 + color.cpu_snapshot.capacity() + depth.guest_depth_snapshot.capacity() + depth.guest_stencil_snapshot.capacity();
        CHECK(plan.pinned_bytes == pinned && plan.derived_before >= 10600);
        CHECK(!plan.derived_after && !plan.entries_after && plan.evict.size() == 5);
        CHECK(state.surfaces.size() == 1 && state.depth_surfaces.size() == 1);
        CHECK(color.color == base && color.multisample_color == multisample && color.raw_color && color.raw_multisample_color);
        CHECK(color.cpu_snapshot == original_color && color.revision == 123 && color.watch_epoch == 456);
        CHECK(depth.guest_depth_snapshot == original_depth && depth.guest_stencil_snapshot == original_stencil && !depth.depth_content_stored);
        CHECK(depth.texture && depth.mask && depth.sample_rate_copy && depth.sample_rate_mask);
        CHECK(pending.count == 3 && pending[0] == derived); // Outstanding owner survives eviction.
        CHECK(state.rendered_images.empty() && state.rendered_cubes.empty() && state.cube_aliases.empty());
        CHECK(color.rgba8_casts.empty() && color.word_casts.empty() && depth.snapshots.empty());
        for (unsigned frame = 0; frame < 1000; ++frame) {
            color.last_frame_rendered = frame;
            for (unsigned i = 0; i < 300; ++i) color.subrectangles[i] = texture(0, base);
            const auto next = trim_derived_surface_cache(state, UINT64_MAX, 256);
            CHECK(next.entries_before == 300 && !next.entries_after && color.subrectangles.empty());
            CHECK(next.pinned_bytes == pinned && color.cpu_snapshot == original_color);
        }
        // The actual presentation call site cannot prune before a successful
        // completion. Large native data remain pinned even above the budget.
        MetalState renderer; auto &s = *renderer.impl;
        s.surfaces[1].color = texture(512ull * 1024 * 1024);
        s.surfaces[1].rgba8_casts[1] = texture(129ull * 1024 * 1024);
        renderer.swap_window(); CHECK(s.surfaces[1].rgba8_casts.size() == 1);
        s.screen_commands = [BudgetCommands new]; s.drawable = [NSObject new]; s.device->fail = true;
        bool failed = false;
        try { renderer.swap_window(); } catch (const std::runtime_error &) { failed = true; }
        CHECK(failed && s.surfaces[1].rgba8_casts.size() == 1);
        s.device->fail = false; auto commands = s.screen_commands;
        renderer.swap_window();
        CHECK(commands.completed && commands.presented && !s.screen_commands && !s.drawable);
        CHECK(s.surfaces[1].rgba8_casts.empty() && s.surfaces[1].color);
        CHECK(s.surface_cache_budget.pinned_bytes == 512ull * 1024 * 1024 && !s.surface_cache_budget.derived_after);
        std::cout << "PASS surface cache budget "
#ifdef NDEBUG
                  << "NDEBUG"
#else
                  << "assertions"
#endif
                  << ": 10000 independent ownership/age/budget plans; parent/buffer alias accounting, all derived cache classes, pending owners, 1000 bounded refills, pinned raw/MSAA/depth/masks/snapshots and actual presentation completion/failure gate; CPU doubles only\n";
    }
}
