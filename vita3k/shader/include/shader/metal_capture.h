// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>

namespace shader::metal {

// Slot zero is clip position; slots 1..13 are guest varyings and slot 14 is
// point size. Each captured vertex occupies this many float4s.
inline constexpr uint32_t CAPTURE_OUTPUT_SLOT_COUNT = 15;

} // namespace shader::metal
