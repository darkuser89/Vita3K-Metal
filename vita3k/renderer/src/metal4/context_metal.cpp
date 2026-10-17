// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/state.h>
namespace renderer::metal {
bool is_native_context(renderer::Context *context) { return dynamic_cast<MetalGuestContext *>(context) != nullptr; }
bool is_native_state(const renderer::State &state) { return state.current_backend == Backend::Metal; }
MetalGuestContext::GxmMetadata &gxm_metadata(renderer::Context &context) { return static_cast<MetalGuestContext &>(context).metal_gxm; }
}
