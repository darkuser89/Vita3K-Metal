// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>

namespace shader::metal {

// Slot zero is clip position; slots 1..13 are guest varyings and slot 14 is
// point size. Slots 15..22 contain rasterizer clip distances in x: three
// depth-clamp guards followed by up to five declared GXM planes, as in Plus.
inline constexpr uint32_t CAPTURE_CLIP_SLOT = 15;
inline constexpr uint32_t CAPTURE_CLIP_COUNT = 8;
inline constexpr uint32_t CAPTURE_OUTPUT_SLOT_COUNT = CAPTURE_CLIP_SLOT + CAPTURE_CLIP_COUNT;

} // namespace shader::metal
