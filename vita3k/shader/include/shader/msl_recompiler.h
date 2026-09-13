// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct FeatureState;
struct SceGxmProgram;
namespace shader {
struct Hints;
}

namespace shader::metal {

// Per-stage Metal slots. Vertex streams must not overlap shader buffers.
inline constexpr uint32_t RENDER_INFO_BUFFER = 0;
inline constexpr uint32_t UNIFORM_BUFFER = 1;
inline constexpr uint32_t TEXTURE_INFO_BUFFER = 2;
inline constexpr uint32_t VERTEX_STREAM_BUFFER_BASE = 4;
inline constexpr uint32_t TEXTURE_COUNT = 16;
inline constexpr uint32_t COLOR_ATTACHMENT_TEXTURE = 16;
inline constexpr uint32_t MASK_TEXTURE = 17;
inline constexpr uint32_t RAW_COLOR_ATTACHMENT_TEXTURE = 18;
inline constexpr uint32_t SHADER_ABI_VERSION = 18;

enum class Stage { Vertex, Fragment };

struct Program {
    std::string source;
    std::string entry_point;
    Stage stage;
    bool uses_framebuffer_fetch = false;
    bool uses_raster_order_groups = false;
    bool uses_buffer_addresses = false;
    // Conservative default for manually supplied MSL. convert_spirv proves
    // when a translated shader does not write externally backed buffers.
    bool writes_guest_memory = true;
    uint32_t cube_texture_mask = 0;
};

// Accepts the descriptor-set/clip-space convention of SpirVVulkan, without
// requiring a Vulkan instance or driver. Throws on unsupported resource layouts.
// The result targets MSL 3.0 and uses native framebuffer fetch for subpass inputs.
Program convert_spirv(const std::vector<uint32_t> &spirv);
bool writes_external_memory(const std::vector<uint32_t> &spirv);

// Built-in emulator overlays use their own ABI: push constants at buffer(0),
// vertex data at buffer(1), and image/font textures at texture/sampler(0..1).
Program convert_overlay_spirv(const std::vector<uint32_t> &spirv);

// Recompile the guest program using the native renderer's current format and
// attribute hints. With memory mapping enabled the render-info address table
// must contain native MTLBuffer GPU addresses, never guest or CPU pointers.
Program convert_gxp(const SceGxmProgram &program, const std::string &hash,
    const FeatureState &features, const Hints &hints, bool maskupdate = false);

} // namespace shader::metal
