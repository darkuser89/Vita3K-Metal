// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/texture_lookup.h>
#include <cassert>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>

using renderer::metal::DrawTextureLookup;
using renderer::metal::DrawSurfaceLookup;
using Address = uint32_t;
constexpr unsigned MTLPixelFormatRGB9E5Float = 1;
struct SurfaceRect {
    uint32_t x, y, width, height;
    bool operator==(const SurfaceRect &) const = default;
};
struct Guest {
    uint32_t address, bytes, pitch, layout, rectangles;
    SceGxmColorBaseFormat colorFormat;
};
struct Surface {
    Guest guest;
    struct { unsigned pixelFormat; } color;
    bool eligible;
};
namespace gxm { auto get_base_format(SceGxmColorBaseFormat format) { return format; } }
struct State {
    std::map<uint32_t, Surface> surfaces;
    renderer::metal::SurfaceAddressIndex<decltype(surfaces)> surface_addresses;
};
// This synthetic rectangle predicate also accepts zero-height rectangles at
// the inclusive end. Cover those deliberately broader test-only semantics.
static uint64_t surface_lookup_end(const Guest &s) { return uint64_t(s.address) + s.bytes + 1; }
static size_t overlap_calls, rectangle_calls;
// Model only the geometry boundary here. Actual format/rectangle math is
// unchanged in production; these predicates exercise arbitrary overlaps,
// incompatible layouts, holes and subimages without claiming pixel validation.
bool surface_texture_layout_overlap(const Guest &s, const SceGxmTexture &t) {
    ++overlap_calls;
    const uint64_t address = uint64_t(t.data_addr) << 2;
    return address >= s.address && address < uint64_t(s.address) + s.bytes
        && t.width == s.pitch && t.type == s.layout;
}
std::optional<SurfaceRect> rect(const Guest &s, const SceGxmTexture &t, unsigned kind) {
    ++rectangle_calls;
    const uint64_t address = uint64_t(t.data_addr) << 2;
    if (!(s.rectangles & (1u << kind)) || address < s.address
        || address + uint64_t(t.height) * 4 > uint64_t(s.address) + s.bytes) return std::nullopt;
    return SurfaceRect{uint32_t(address - s.address),kind,t.width,t.height};
}
auto surface_subrectangle(const Guest &s, const SceGxmTexture &t) { return rect(s, t, 0); }
#define RECT(name, kind) auto surface_##name##_subrectangle(const Guest &s, const SceGxmTexture &t) { return rect(s, t, kind); }
RECT(word, 1) RECT(halfword, 2) RECT(byte, 3) RECT(rgba8_byte, 4) RECT(wide_small, 5) RECT(32_small, 6)
#undef RECT
// COLOR_LOOKUP_HOOK
// FORMAT_BOUNDS_HOOK

void geometry_cache() {
    DrawSurfaceLookup<std::optional<SurfaceRect>> cache;
    SceGxmTexture texture{};
    std::array<unsigned,2> owners{};
    size_t calls = 0;
    const SurfaceRect expected{1,2,3,4};
    const auto resolve = [&]() -> std::optional<SurfaceRect> { ++calls; return expected; };
    for (unsigned operation = 0; operation < 8; ++operation) {
        assert(cache.get(&owners[0],texture,operation,resolve) == expected);
        for (unsigned n = 0; n < 10000; ++n)
            assert(cache.get(&owners[0],texture,operation,[]() -> std::optional<SurfaceRect> { assert(false); return {}; }) == expected);
    }
    assert(calls == 8);
    const SurfaceRect other{9,8,7,6};
    for (unsigned operation = 0; operation < 8; ++operation)
        assert(cache.get(&owners[1],texture,operation,[&]() -> std::optional<SurfaceRect> { return other; }) == other);
    assert(cache.get(&owners[0],texture,0,resolve) == expected);
    for (unsigned bit = 0; bit < 128; ++bit) {
        auto words = std::bit_cast<std::array<uint32_t,4>>(texture);
        words[bit/32] ^= 1u << (bit%32);
        auto changed = std::bit_cast<SceGxmTexture>(words);
        assert(cache.get(&owners[0],changed,0,[]() -> std::optional<SurfaceRect> { return {}; }) == std::nullopt);
        assert(cache.get(&owners[0],changed,0,resolve) == std::nullopt);
    }
    DrawSurfaceLookup<int> bounded;
    for (unsigned n = 0; n < 256; ++n) {
        texture.data_addr = n;
        assert(bounded.get(&owners[0],texture,0,[n] { return int(n); }) == int(n));
    }
    texture.data_addr = 256;
    assert(bounded.get(&owners[0],texture,0,[] { return 1; }) == 1);
    assert(bounded.get(&owners[0],texture,0,[] { return 2; }) == 2);
    texture.data_addr = 0;
    assert(bounded.get(&owners[0],texture,0,[] { assert(false); return -1; }) == 0);
    DrawSurfaceLookup<int> failed;
    try { failed.get(&owners[0],texture,0,[]() -> int { throw std::runtime_error("geometry"); }); assert(false); }
    catch (const std::runtime_error &) {}
    assert(failed.get(&owners[0],texture,0,[] { return 7; }) == 7);
    DrawSurfaceLookup<std::optional<SurfaceRect>> next_draw;
    assert(next_draw.get(&owners[0],texture,0,[&]() -> std::optional<SurfaceRect> { return other; }) == other);
    std::cout << "PASS surface geometry cache: 80000 warm results without recomputation, 128 descriptor bits, independent owners and eight operations, negative results, bounded 256-entry bypass, exception retry and new-draw invalidation\n";
}

void address_index() {
    struct Value { uint64_t end; uint32_t kind; };
    std::map<uint32_t, Value> map;
    renderer::metal::SurfaceAddressIndex<decltype(map)> index;
    std::mt19937 rng(0xADD023);
    size_t reads = 0, predicates = 0, queries = 0;
    const auto end = [&](const Value &value) { ++reads; return value.end; };
    const auto check = [&](uint32_t address, unsigned kind) {
        auto expected = map.end(), it = map.upper_bound(address);
        while (it != map.begin()) {
            --it;
            if (it->second.end > address && (it->second.kind & kind)) { expected = it; break; }
        }
        auto actual = index.find(map, address, end, [&](auto candidate) {
            ++predicates;
            assert(candidate->first <= address && candidate->second.end > address);
            return bool(candidate->second.kind & kind);
        });
        assert(actual == expected); ++queries;
    };
    check(0, 1);
    for (unsigned round = 0; round < 2000; ++round) {
        index.invalidate();
        map.clear();
        for (unsigned n = 0; n < round % 129; ++n) {
            const uint32_t base = round % 3 ? rng() % 4096 : rng();
            map[base] = {std::min(uint64_t(1) << 32, uint64_t(base) + rng() % 8192), uint32_t(rng() % 8)};
        }
        for (unsigned q = 0; q < 128; ++q) check(round % 3 ? rng() % 16384 : rng(), 1u << (q % 3));
        const auto after_build = reads;
        // Eligibility/format changes are live; they need no bounds rebuild.
        for (auto &[key, value] : map) value.kind ^= 7;
        for (unsigned q = 0; q < 32; ++q) check(rng() % 8192, 1u << (q % 3));
        assert(reads == after_build);
        // Same-size erase/reinsert and changed bounds must not leave dangling
        // iterators or miss a newly enclosing owner on the following draw.
        index.invalidate();
        if (!map.empty()) map.erase(map.begin());
        map[0] = {uint64_t(1) << 32, 7};
        check(UINT32_MAX, 1); check(0, 1);
    }
    index.clear(); map.clear();
    for (unsigned n = 0; n < 65536; ++n) map[n * 4096] = {uint64_t(n * 4096) + 256, 1};
    const auto before = predicates;
    for (unsigned n = 0; n < 10000; ++n) {
        const uint32_t address = (n * 3571u % 65536) * 4096 + 2048;
        assert(index.find(map,address,end,[&](auto) { ++predicates; return true; }) == map.end());
        ++queries;
    }
    assert(predicates == before); // holes prune every exact predicate
    const auto before_reads = reads;
    for (unsigned n = 0; n < 10000; ++n) {
        const uint32_t address = (n * 3571u % 65536) * 4096;
        assert(index.find(map,address,end,[&](auto) { ++predicates; return true; }) == map.find(address));
        ++queries;
    }
    assert(reads == before_reads && predicates == before + 10000);
    index.invalidate();
    try {
        index.find(map, 0, [](const Value &) -> uint64_t { throw std::runtime_error("rebuild"); }, [](auto) { return true; });
        assert(false);
    } catch (const std::runtime_error &) {}
    check(0, 1);
    index.clear(); map.clear(); check(UINT32_MAX, 7);
    std::cout << "PASS surface address index: " << queries
              << " choices against descending-scan/sparse-map expectations, live filters, mutation/erase/reuse/clear, exception retry; 10000 hole queries with zero exact predicates and 10000 hits without rebuilding a 65536-entry index; ASAN/UBSAN\n";
}

void address_bounds() {
    struct GuestBounds {
        struct Pointer { uint32_t value; uint32_t address() const { return value; } } data;
        uint32_t width, height, strideInPixels;
    };
    size_t cases = 0;
    // Check coverage against independent linear-visible, tiled-band and
    // compact-Morton address extents, including overflowing 32-bit ranges.
    for (uint32_t base : {1u,4u,0x80000000u,0xfffff000u,0xfffffffcu})
    for (uint32_t width : {1u,7u,32u,16384u,UINT32_MAX})
    for (uint32_t height : {1u,31u,32u,33u,16384u,UINT32_MAX})
    for (uint32_t padding : {0u,1u,31u,100u})
    for (uint32_t bytes : {1u,2u,3u,4u,8u}) {
        const uint32_t stride = uint32_t(std::min(uint64_t(UINT32_MAX), uint64_t(width) + padding));
        const GuestBounds surface{{base},width,height,stride};
        const uint64_t bound = renderer::metal::surface_lookup_end(surface);
        using Wide = unsigned __int128;
        const Wide linear = Wide(base) + (Wide(height) - 1) * stride * bytes + Wide(width) * bytes;
        const Wide tiled = Wide(base) + ((Wide(height) + 31) / 32) * 32 * stride * bytes;
        const Wide morton = Wide(base) + Wide(width) * height * bytes;
        for (Wide end : {linear,tiled,morton}) {
            assert(bound >= uint64_t(std::min(Wide(1) << 32,end)));
            assert(bound <= uint64_t(1) << 32); ++cases;
        }
    }
    assert(renderer::metal::surface_lookup_end(GuestBounds{{0},1,1,1}) == 0);
    assert(renderer::metal::surface_lookup_end(GuestBounds{{4},0,1,1}) == 4);
    assert(renderer::metal::surface_lookup_end(GuestBounds{{4},1,0,1}) == 4);
    assert(renderer::metal::surface_lookup_end(GuestBounds{{4},2,1,1}) == 4);
    std::cout << "PASS surface lookup bounds: " << cases
              << " independent 128-bit linear/tiled/Morton extents covered, 1/2/3/4/8-byte formats, padding, zero/invalid and 32-bit saturation edges\n";
}

void generic_cache() {
    DrawTextureLookup<unsigned> cache;
    SceGxmTexture texture{};
    unsigned calls = 0;
    const auto resolve = [&] { return ++calls; };
    assert(cache.get(texture, resolve) == 1);
    for (unsigned n = 0; n < 10000; ++n) assert(cache.get(texture, resolve) == 1);
    assert(calls == 1);
    // Every descriptor bit participates; mutation at the same address is not
    // identity, and different instances with equal bytes reuse the result.
    for (unsigned bit = 0; bit < 128; ++bit) {
        DrawTextureLookup<unsigned> one;
        const auto zero = one.get(texture, resolve);
        auto words = std::bit_cast<std::array<uint32_t, 4>>(texture);
        words[bit / 32] ^= 1u << (bit % 32);
        const auto other = std::bit_cast<SceGxmTexture>(words);
        const auto changed = one.get(other, resolve);
        assert(changed != zero && one.get(other, resolve) == changed);
        assert(one.get(texture, resolve) == zero);
    }
    DrawTextureLookup<int> negative;
    assert(negative.get(texture, [] { return -1; }) == -1);
    assert(negative.get(texture, [] { assert(false); return 0; }) == -1);
    DrawTextureLookup<unsigned> failed;
    try { failed.get(texture, []() -> unsigned { throw std::runtime_error("miss"); }); assert(false); }
    catch (const std::runtime_error &) {}
    assert(failed.get(texture, [] { return 42u; }) == 42);
    DrawTextureLookup<unsigned> bounded;
    for (unsigned n = 0; n < 32; ++n) {
        texture.data_addr = n;
        assert(bounded.get(texture, [n] { return n; }) == n);
    }
    for (unsigned n = 32; n < 64; ++n) {
        texture.data_addr = n;
        assert(bounded.get(texture, [n] { return n; }) == n);
        assert(bounded.get(texture, [] { return 1000u; }) == 1000);
    }
    for (unsigned n = 0; n < 32; ++n) {
        texture.data_addr = n;
        assert(bounded.get(texture, [] { assert(false); return 0u; }) == n);
    }
}

static size_t geometry_results, direct_choices;
void compare_draw(State &state, const std::array<SceGxmTexture, 32> &textures) {
    auto *impl = &state;
    auto texture_surface_usable = [](const Surface &surface) { return surface.eligible; };
    // FINDERS_HOOK
    const auto reference = [&](const SceGxmTexture &texture, unsigned kind) {
        if (!kind) return find_color_subrectangle(impl->surfaces, texture, texture_surface_usable);
        auto found = impl->surfaces.upper_bound(texture.data_addr << 2);
        while (found != impl->surfaces.begin()) {
            --found;
            if (surface_texture_layout_overlap(found->second.guest, texture))
                return texture_surface_usable(found->second)
                    && (kind != 6 || found->second.guest.colorFormat != SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9
                        || found->second.color.pixelFormat == MTLPixelFormatRGB9E5Float)
                    && rect(found->second.guest, texture, kind)
                    ? found : impl->surfaces.end();
        }
        return impl->surfaces.end();
    };
    for (unsigned pass = 0; pass < 3; ++pass) {
        // Native format remains a live filter, never frozen into layout-owner
        // results. Eligibility here models the already-frozen per-draw policy.
        if (pass == 1) for (auto &[key, surface] : impl->surfaces) surface.color.pixelFormat ^= 1;
        for (const auto &texture : textures) {
            const auto expected = [&](unsigned kind) { return reference(texture, kind); };
            const auto actual_rectangle = find_subrectangle(texture), expected_rectangle = expected(0);
            if (actual_rectangle != expected_rectangle) {
                std::cerr << "rectangle mismatch address=" << (uint64_t(texture.data_addr) << 2)
                          << " height=" << texture.height << " actual=" << (actual_rectangle == impl->surfaces.end() ? UINT32_MAX : actual_rectangle->first)
                          << " expected=" << (expected_rectangle == impl->surfaces.end() ? UINT32_MAX : expected_rectangle->first);
                if (expected_rectangle != impl->surfaces.end()) std::cerr << " expected_end=" << surface_lookup_end(expected_rectangle->second.guest);
                std::cerr << '\n';
            }
            assert(actual_rectangle == expected_rectangle);
            assert(find_word_subrectangle(texture) == expected(1));
            assert(find_halfword_subrectangle(texture) == expected(2));
            assert(find_byte_subrectangle(texture) == expected(3));
            assert(find_rgba8_byte_subrectangle(texture) == expected(4));
            assert(find_wide_small_subrectangle(texture) == expected(5));
            assert(find_32_small_subrectangle(texture) == expected(6));
            const auto direct = impl->surfaces.find(texture.data_addr << 2);
            const auto expected_direct = direct != impl->surfaces.end() && direct->second.eligible
                && (surface_texture_layout_overlap(direct->second.guest,texture)
                    || surface_subrectangle(direct->second.guest,texture)) ? direct : impl->surfaces.end();
            assert(find_direct_surface(texture) == expected_direct); ++direct_choices;
            // Simulate later binding/feedback/mip consumers using any owner,
            // not just the winning iterator. Compare full rectangle payloads.
            for (const auto &[address,surface] : impl->surfaces) {
                const auto &guest = surface.guest;
                assert(cached_surface_subrectangle(guest,texture) == surface_subrectangle(guest,texture));
                assert(cached_surface_word_subrectangle(guest,texture) == surface_word_subrectangle(guest,texture));
                assert(cached_surface_halfword_subrectangle(guest,texture) == surface_halfword_subrectangle(guest,texture));
                assert(cached_surface_byte_subrectangle(guest,texture) == surface_byte_subrectangle(guest,texture));
                assert(cached_surface_rgba8_byte_subrectangle(guest,texture) == surface_rgba8_byte_subrectangle(guest,texture));
                assert(cached_surface_wide_small_subrectangle(guest,texture) == surface_wide_small_subrectangle(guest,texture));
                assert(cached_surface_32_small_subrectangle(guest,texture) == surface_32_small_subrectangle(guest,texture));
                assert(cached_surface_texture_layout_overlap(guest,texture) == surface_texture_layout_overlap(guest,texture));
                geometry_results += 8;
            }
        }
    }
    // All 32 descriptors fit; repeating the six layout searches visits zero
    // surface-map candidates, including negative owner decisions.
    const auto before = overlap_calls;
    for (unsigned repeat = 0; repeat < 10; ++repeat) for (const auto &texture : textures) {
        find_word_subrectangle(texture); find_halfword_subrectangle(texture);
        find_byte_subrectangle(texture); find_rgba8_byte_subrectangle(texture);
        find_wide_small_subrectangle(texture); find_32_small_subrectangle(texture);
    }
    assert(overlap_calls == before);
    const auto before_rect = rectangle_calls;
    for (const auto &texture : textures) { find_subrectangle(texture); find_direct_surface(texture); }
    assert(rectangle_calls == before_rect && overlap_calls == before);
}

int main() {
    generic_cache();
    geometry_cache();
    format_bounds();
    address_index();
    address_bounds();
    std::mt19937 rng(0xA023);
    size_t queries = 0;
    for (unsigned draw = 0; draw < 2000; ++draw) {
        State state;
        for (unsigned n = 0; n < draw % 41; ++n) {
            const uint32_t address = uint32_t(rng() % 128) * 4;
            state.surfaces[address] = {{address, uint32_t(rng() % 256), uint32_t(rng() % 4),
                uint32_t(rng() % 3), uint32_t(rng() % 128), (rng() & 1)
                    ? SCE_GXM_COLOR_BASE_FORMAT_SE5M9M9M9 : SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8},
                {unsigned(rng() & 1)}, bool(rng() & 1)};
        }
        std::array<SceGxmTexture, 32> textures{};
        for (auto &t : textures) {
            t.data_addr = rng() % 160; t.width = rng() % 4;
            t.height = rng() % 16; t.type = rng() % 3;
        }
        compare_draw(state, textures);
        // A later draw sees changed eligibility, replaced layouts, new owners,
        // erased nodes and empty maps; caches must not escape the draw scope.
        for (auto &[key, surface] : state.surfaces) { surface.eligible = !surface.eligible; surface.guest.pitch ^= 1; }
        state.surface_addresses.invalidate();
        if (!state.surfaces.empty()) state.surfaces.erase(state.surfaces.begin());
        compare_draw(state, textures);
        queries += textures.size() * 7 * 3 * 2;
    }
    std::cout << "PASS texture owner lookup: " << queries
        << " old/new alias choices, " << direct_choices << " direct-owner choices, " << geometry_results << " exact cached/uncached geometry results, nested/dirty/format-blocked owners, all 128 descriptor bits, bounded negative cache, new-draw invalidation and zero repeat overlap scans (synthetic geometry)\n";
}
