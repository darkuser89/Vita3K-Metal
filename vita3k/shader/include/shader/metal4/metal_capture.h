// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>

namespace shader::metal3::metal {

// Slot zero is clip position; slots 1..13 are guest varyings and slot 14 is
// point size. Capture retains all three guards plus all eight guest planes.
// Native Metal still accepts eight distances; overflow draws are CPU-clipped
// before replay. Keep the runtime and private translator layouts identical.
inline constexpr uint32_t CAPTURE_CLIP_SLOT = 15;
inline constexpr uint32_t GUEST_CLIP_COUNT = 8;
inline constexpr uint32_t CLIP_GUARD_COUNT = 3;
inline constexpr uint32_t NATIVE_CLIP_COUNT = 8;
inline constexpr uint32_t CAPTURE_CLIP_COUNT = CLIP_GUARD_COUNT + GUEST_CLIP_COUNT;
inline constexpr uint32_t CAPTURE_OUTPUT_SLOT_COUNT = CAPTURE_CLIP_SLOT + CAPTURE_CLIP_COUNT;

} // namespace shader::metal
