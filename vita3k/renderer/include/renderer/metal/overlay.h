// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal/device.h>
#include <filesystem>

namespace overlay { class display_manager; }
namespace renderer::metal {
class OverlayRenderer {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    OverlayRenderer(Device &, const std::filesystem::path &assets);
    ~OverlayRenderer();
    void render(id<MTLRenderCommandEncoder>, const overlay::display_manager &, MTLViewport);
};
}
