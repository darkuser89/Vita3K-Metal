// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cstdint>
namespace shader::metal {
// Separate Metal buffer; shared Vulkan/OpenGL render-info layouts are unchanged.
struct alignas(16) TextureMipInfo {
    // enabled, sampler flags, anisotropy, reserved. Flags: min/mag/mip linear in
    // bits 0/1/2, U/V address modes in bits 3..5/6..8, minimum LOD in bits 9..12.
    std::array<uint32_t, 4> control{};
    std::array<std::array<uint32_t, 2>, 16> sizes{};
};
static_assert(sizeof(TextureMipInfo) == 144);
using TextureMipInfos = std::array<TextureMipInfo, 16>;
static_assert(sizeof(TextureMipInfos) == 2304);
} // namespace shader::metal
