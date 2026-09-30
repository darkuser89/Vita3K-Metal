// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

namespace renderer::metal {
#ifdef __APPLE__
// The native backend uses the system default Metal device.
bool supports_metalfx_spatial();
#else
inline bool supports_metalfx_spatial() { return false; }
#endif
}
