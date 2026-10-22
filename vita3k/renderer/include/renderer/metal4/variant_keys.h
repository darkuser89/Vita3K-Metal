// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace renderer::metal {
// Only a complete, known-one mask is equivalent to no mask test. Shared
// retained depth masks need a full clear in this scene to establish that fact.
inline bool can_omit_mask_test(bool owned_or_cleared, bool valid, bool one,
    bool mask_update, bool depth_only, bool interlocked_fragcolor) {
    return owned_or_cleared && valid && one && !mask_update && !depth_only && !interlocked_fragcolor;
}
// Use the same resource mask as native texture binding. Unreferenced guest
// texture descriptors can contain stale state and must not specialize a shader.
// Canonicalize the translator input too, so key and emitted code always agree.
template <class Hints, class Used, class Format>
void canonicalize_shader_textures(Hints &hints, const Used &vertex_used, const Used &fragment_used,
    Format default_format) {
    for (size_t slot = 0; slot < 16; ++slot) {
        if (!vertex_used[slot]) {
            hints.vertex_textures[slot] = default_format;
            hints.metal_vertex_lod_bias[slot] = 0.f;
        }
        if (!fragment_used[slot]) {
            hints.fragment_textures[slot] = default_format;
            hints.metal_fragment_lod_bias[slot] = 0.f;
        }
    }
}
inline void append_variant_word(std::string &key, uint32_t value) {
    const char bytes[] = {char(value), char(value >> 8), char(value >> 16), char(value >> 24)};
    key.append(bytes, sizeof(bytes));
}
struct ShaderKeyWords {
    bool depth_only = false;
    bool mask_test_omitted = false;
    uint32_t prefix_count = 0;
    std::array<uint32_t, 32> prefix{}, textures{};
    bool operator==(const ShaderKeyWords &) const = default;
};
// Stack-only canonical state. Compare selected fields/float bits, never padding,
// Hints pointers, inactive blend fields, or floating-point numerical equality.
template <class Hints>
ShaderKeyWords shader_key_words(const Hints &hints, bool vertex, bool depth_only,
    uint32_t features, uint32_t samples, bool maskupdate, bool gamma, bool omit_mask = false) {
    ShaderKeyWords words;
    words.depth_only = !vertex && depth_only;
    words.mask_test_omitted = !vertex && !depth_only && !maskupdate && omit_mask;
    const auto put = [&](uint32_t value) { words.prefix.at(words.prefix_count++) = value; };
    if (words.depth_only) { put(samples); return words; }
    put(vertex);
    if (vertex) {
        put(hints.metal_missing_vertex_outputs);
        put(hints.metal_far_clip);
    }
    put(hints.metal_mip_sampling);
    put(hints.metal_float_cube_filter);
    put(features);
    // These target properties affect fragment translation only. Keep
    // one vertex variant when just the fragment attachment changes.
    if (!vertex) put(hints.color_format);
    if (!vertex) {
        put(hints.metal_raw_color_attachment);
        put(hints.metal_raw_color_fetch);
        put(hints.metal_red_alpha_shader_blend);
        if (hints.metal_red_alpha_shader_blend) {
            const auto &blend = hints.metal_red_alpha_blend;
            put(blend.colorFunc); put(blend.alphaFunc);
            put(blend.colorSrc); put(blend.colorDst);
            put(blend.alphaSrc); put(blend.alphaDst);
        }
        put(hints.metal_quantized_color_blend);
        if (hints.metal_quantized_color_blend) {
            const auto &blend = hints.metal_quantized_blend;
            put(blend.colorFunc); put(blend.alphaFunc);
            put(blend.colorSrc); put(blend.colorDst);
            put(blend.alphaSrc); put(blend.alphaDst);
            put(blend.colorMask);
        }
    }
    if (!vertex) put(hints.metal_output_register_size);
    if (!vertex) put(hints.metal_output_register_format);
    put(vertex ? 1u : samples);
    put(!vertex && maskupdate);
    put(!vertex && !maskupdate && gamma);
    static_assert(std::size(hints.vertex_textures) == 16);
    for (size_t slot = 0; slot < 16; ++slot) {
        words.textures[2 * slot] = vertex ? hints.vertex_textures[slot] : hints.fragment_textures[slot];
        words.textures[2 * slot + 1] = std::bit_cast<uint32_t>(vertex
            ? hints.metal_vertex_lod_bias[slot] : hints.metal_fragment_lod_bias[slot]);
    }
    return words;
}
// One variant per stage/context. Immutable owners make identity meaningful and
// prevent pointer reuse. Publish key/state only after successful serialization.
template <class Binding>
class ShaderVariantKeyCache {
    std::shared_ptr<const Binding> program, attributes;
    ShaderKeyWords words;
    std::string key;
    uint64_t generation = 0;
public:
    uint64_t revision() const { return generation; }
    template <class Hash>
    const std::string &get(std::shared_ptr<const Binding> next_program,
        std::shared_ptr<const Binding> next_attributes, const ShaderKeyWords &next, Hash hash) {
        if (next.depth_only) { next_program.reset(); next_attributes.reset(); }
        else if (!next_program || !next_attributes) throw std::runtime_error("Metal: missing shader key binding");
        if (generation && program == next_program && attributes == next_attributes && words == next) return key;
        std::string replacement = next.depth_only ? "metal-depth-only" : hash();
        replacement.reserve(replacement.size() + 4 * (next.prefix_count + next.textures.size()
            + (next_attributes ? 1 + 3 * next_attributes->attributes.size() : 0)));
        for (uint32_t i = 0; i < next.prefix_count; ++i) append_variant_word(replacement, next.prefix.at(i));
        if (!next.depth_only) {
            append_variant_word(replacement, next_attributes->attributes.size());
            for (const auto &a : next_attributes->attributes) {
                append_variant_word(replacement, a.regIndex);
                append_variant_word(replacement, a.format);
                append_variant_word(replacement, a.componentCount);
            }
            for (const auto word : next.textures) append_variant_word(replacement, word);
        }
        // Nine bytes cannot collide with the word-aligned ordinary key.
        // Keep existing shader/pipeline identities for the full-mask path.
        if (next.mask_test_omitted) replacement += "-mask-one";
        key = std::move(replacement); program = std::move(next_program); attributes = std::move(next_attributes);
        words = next; ++generation;
        return key;
    }
};
template <class Binding>
class PipelineVariantKeyCache {
    std::shared_ptr<const Binding> layout_owner;
    std::array<uint64_t, 2> revisions{};
    std::array<uint32_t, 14> target{};
    std::string key;
public:
    const std::string &get(const std::shared_ptr<const Binding> &layout,
        std::array<uint64_t, 2> next_revisions, const std::array<uint32_t, 14> &next_target,
        const std::string &vertex, const std::string &fragment, const std::string &layout_key) {
        if (layout_owner == layout && revisions == next_revisions && target == next_target) return key;
        std::string replacement;
        replacement.reserve(vertex.size() + fragment.size() + layout_key.size() + 4 * next_target.size());
        replacement += vertex; replacement += fragment; replacement += layout_key;
        for (const auto word : next_target) append_variant_word(replacement, word);
        key = std::move(replacement); layout_owner = layout; revisions = next_revisions; target = next_target;
        return key;
    }
};
}
