// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <gxm/functions.h>
#include <shader/usse_program_analyzer.h>
#include <optional>

namespace renderer::metal {

struct VertexAttributeShape {
    SceGxmAttributeFormat format;
    uint32_t components;
    uint32_t component_size;

    uint32_t byte_size() const { return components * component_size; }
};

// One definition for guest-thread stream bounds and native vertex descriptors.
// Register-format inputs carry raw bytes derived from the shader, including
// arrays; ordinary inputs keep the patcher's conversion format and width.
inline std::optional<VertexAttributeShape> vertex_attribute_shape(
    const SceGxmVertexAttribute &attribute, const shader::usse::AttributeInformation &info) {
    uint64_t components = attribute.componentCount;
    auto format = attribute.format;
    if (info.regformat) {
        components = uint64_t(info.component_count) * info.array_size;
        switch (info.gxm_type) {
        case SCE_GXM_PARAMETER_TYPE_U8:
        case SCE_GXM_PARAMETER_TYPE_S8:
            format = SCE_GXM_ATTRIBUTE_FORMAT_U8;
            break;
        case SCE_GXM_PARAMETER_TYPE_C10:
            format = SCE_GXM_ATTRIBUTE_FORMAT_U8;
            components = (components * 10 + 7) / 8;
            break;
        case SCE_GXM_PARAMETER_TYPE_U16:
        case SCE_GXM_PARAMETER_TYPE_S16:
        case SCE_GXM_PARAMETER_TYPE_F16:
            format = SCE_GXM_ATTRIBUTE_FORMAT_U16;
            break;
        default:
            format = SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED;
            break;
        }
    }
    // The native descriptor has 31 locations. Check in wide arithmetic before
    // narrowing so malformed array counts cannot wrap into a small upload.
    if (!components || info.location >= 31 || (components + 3) / 4 > 31 - info.location
        || uint32_t(format) > uint32_t(SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED))
        return std::nullopt;
    return VertexAttributeShape{format, uint32_t(components), gxm::attribute_format_size(format)};
}

} // namespace renderer::metal
