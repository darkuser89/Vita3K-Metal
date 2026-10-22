// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal4/apple.h>
#include <renderer/metal4/vertex_layout.h>
#include <shader/metal4/msl_recompiler.h>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>

namespace renderer::metal {
inline void require_vertex_layout(bool condition, const char *error) {
    if (!condition) throw std::runtime_error(error);
}
inline MTLVertexFormat native_attribute_format(SceGxmAttributeFormat f, uint32_t count, bool is_signed) {
    require_vertex_layout(count >= 1 && count <= 4, "Metal: invalid vertex component count");
    static const MTLVertexFormat formats[][4] = {
        {MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4},
        {MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4},
        {MTLVertexFormatUShort, MTLVertexFormatUShort2, MTLVertexFormatUShort3, MTLVertexFormatUShort4},
        {MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4},
        {MTLVertexFormatUCharNormalized, MTLVertexFormatUChar2Normalized, MTLVertexFormatUChar3Normalized, MTLVertexFormatUChar4Normalized},
        {MTLVertexFormatCharNormalized, MTLVertexFormatChar2Normalized, MTLVertexFormatChar3Normalized, MTLVertexFormatChar4Normalized},
        {MTLVertexFormatUShortNormalized, MTLVertexFormatUShort2Normalized, MTLVertexFormatUShort3Normalized, MTLVertexFormatUShort4Normalized},
        {MTLVertexFormatShortNormalized, MTLVertexFormatShort2Normalized, MTLVertexFormatShort3Normalized, MTLVertexFormatShort4Normalized},
        {MTLVertexFormatHalf, MTLVertexFormatHalf2, MTLVertexFormatHalf3, MTLVertexFormatHalf4},
        {MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4},
        {MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3, MTLVertexFormatUInt4}
    };
    require_vertex_layout(f <= SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED, "Metal: unsupported vertex format");
    if (f == SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED && is_signed) {
        static const MTLVertexFormat signed_formats[] = {
            MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3, MTLVertexFormatInt4
        };
        return signed_formats[count - 1];
    }
    return formats[f][count - 1];
}
struct NativeVertexLayout {
    struct VertexAttributeLayout {
        MTLVertexFormat format = MTLVertexFormatInvalid;
        NSUInteger offset = 0;
        NSUInteger buffer_index = 0;
        size_t source_offset = 0;
        size_t byte_size = 0;
    };
    struct VertexStreamLayout {
        NSUInteger stride = 0;
        MTLVertexStepFunction step_function = MTLVertexStepFunctionPerVertex;
        NSUInteger step_rate = 0;
    };
    std::array<VertexAttributeLayout, 31> vertex_attributes{};
    std::array<VertexStreamLayout, SCE_GXM_MAX_VERTEX_STREAMS> vertex_stream_layouts{};
    uint32_t streams_used = 0;
    std::array<uint32_t, SCE_GXM_MAX_VERTEX_STREAMS> stream_extents{};
    std::array<bool, SCE_GXM_MAX_VERTEX_STREAMS> repack_stream{};
    std::string key;
};
// Only frozen patcher attributes, stream declarations and shader input metadata
// contribute here. Guest addresses, bytes, indices and instance counts do not.
template <class Binding>
NativeVertexLayout build_native_vertex_layout(const Binding &program) {
    const auto *vp = &program;
    NativeVertexLayout result;
    auto &[vertex_attributes, vertex_stream_layouts, streams_used, stream_extents, repack_stream, key] = result;
    key.reserve(4 * (4 * vp->attributes.size() + 3 * vp->streams.size()));
    const auto append = [](std::string &key, uint32_t value) {
        const char bytes[] = {static_cast<char>(value), static_cast<char>(value >> 8),
            static_cast<char>(value >> 16), static_cast<char>(value >> 24)};
        key.append(bytes, sizeof(bytes));
    };
    for (const auto &a : vp->attributes) {
        auto it = vp->attribute_infos.find(a.regIndex);
        if (it == vp->attribute_infos.end()) continue;
        const auto &info = it->second;
        const auto shape = vertex_attribute_shape(a, info);
        require_vertex_layout(shape.has_value(), "Metal: invalid vertex attribute shape");
        const auto components = shape->components;
        const auto f = shape->format;
        const auto element_size = shape->component_size;
        require_vertex_layout(a.streamIndex < vp->streams.size() && a.streamIndex < SCE_GXM_MAX_VERTEX_STREAMS, "Metal: invalid vertex stream");
        stream_extents[a.streamIndex] = std::max(stream_extents[a.streamIndex], uint32_t(a.offset + components * element_size));
        for (uint32_t element = 0; element < (components + 3) / 4; ++element) {
            const auto location = info.location + element;
            require_vertex_layout(location < 31, "Metal: attribute location exceeds hardware limit");
            const auto n = std::min(components - 4 * element, 4u);
            auto &attribute = vertex_attributes[location];
            attribute.format = native_attribute_format(f, n, !info.regformat && info.is_signed);
            attribute.offset = a.offset + 4 * element * element_size;
            attribute.buffer_index = shader::metal::VERTEX_STREAM_BUFFER_BASE + a.streamIndex;
            attribute.source_offset = attribute.offset;
            attribute.byte_size = n * element_size;
            const size_t stride = vp->streams[a.streamIndex].stride;
            // Whole-stride padding cannot preserve attributes that cross
            // a record boundary. Metal also needs each fetched attribute
            // to fit the native record and its offset to be aligned.
            repack_stream[a.streamIndex] = repack_stream[a.streamIndex]
                || attribute.offset % 4 != 0
                || (stride && attribute.offset + attribute.byte_size > stride);
            append(key, location); append(key, attribute.format);
            append(key, attribute.offset); append(key, a.streamIndex);
        }
        streams_used |= 1u << a.streamIndex;
    }
    require_vertex_layout(vp->streams.size() <= SCE_GXM_MAX_VERTEX_STREAMS, "Metal: too many vertex streams");
    for (uint32_t stream = 0; stream < vp->streams.size(); ++stream) if (streams_used & (1u << stream)) {
        auto &binding = vertex_stream_layouts[stream];
        const auto stride = vp->streams[stream].stride;
        binding.stride = ((stride ? stride : stream_extents[stream]) + 3) & ~size_t(3);
        binding.step_function = !stride ? MTLVertexStepFunctionConstant
            : gxm::is_stream_instancing(static_cast<SceGxmIndexSource>(vp->streams[stream].indexSource))
                ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
        binding.step_rate = stride ? 1 : 0;
        if (repack_stream[stream]) {
            size_t offset = 0;
            for (auto &attribute : vertex_attributes) {
                if (attribute.format == MTLVertexFormatInvalid
                    || attribute.buffer_index != shader::metal::VERTEX_STREAM_BUFFER_BASE + stream)
                    continue;
                attribute.offset = offset;
                offset = (offset + attribute.byte_size + 3) & ~size_t(3);
            }
            binding.stride = offset;
        }
        // Include the layout policy so old cached descriptors cannot be
        // reused when a relocated attribute happens to keep the same stride.
        append(key, repack_stream[stream] ? 1 : 0);
        append(key, binding.stride); append(key, binding.step_function);
    }
    return result;
}
// A single render-context entry. Owning the immutable binding avoids recycled
// pointer hits; another binding rebuilds the complete layout before publication.
template <class Binding>
class FrozenVertexLayoutCache {
    std::shared_ptr<const Binding> owner;
    NativeVertexLayout layout;
public:
    const NativeVertexLayout &get(const std::shared_ptr<const Binding> &program) {
        require_vertex_layout(bool(program), "Metal: missing vertex program binding");
        if (owner != program) {
            auto replacement = build_native_vertex_layout(*program);
            layout = std::move(replacement);
            owner = program;
        }
        return layout;
    }
};
}
