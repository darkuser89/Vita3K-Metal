// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal/device.h>

namespace renderer::metal {
// Shared by window presentation and offscreen composition validation.
class ScreenRenderer {
    Device &device;
    id<MTLRenderPipelineState> pipeline;
    id<MTLSamplerState> sampler;
public:
    explicit ScreenRenderer(Device &device);
    void set_filter(bool linear);
    void render(id<MTLRenderCommandEncoder>, id<MTLTexture> source, MTLViewport);
};
}
