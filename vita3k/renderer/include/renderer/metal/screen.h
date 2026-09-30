// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal/device.h>
#import <MetalFX/MetalFX.h>
#include <array>
#include <filesystem>
#include <string_view>

namespace renderer::metal {
// Shared by window presentation and offscreen composition validation.
class ScreenRenderer {
    Device &device;
    id<MTLFunction> vertex_function;
    id<MTLRenderPipelineState> pipeline;
    id<MTLRenderPipelineState> bicubic_pipeline;
    id<MTLRenderPipelineState> fxaa_pipeline;
    id<MTLComputePipelineState> easu_pipeline;
    id<MTLComputePipelineState> rcas_pipeline;
    id<MTLSamplerState> sampler;
    id<MTLTexture> fsr_intermediate;
    id<MTLTexture> fsr_output;
    id<MTLFXSpatialScaler> metalfx_scaler;
    id<MTLTexture> metalfx_input;
    id<MTLTexture> metalfx_output;
    std::array<NSUInteger, 4> metalfx_dimensions{};
    std::filesystem::path assets;
    std::string bicubic_source;
    std::string fxaa_source;
    enum class Filter { Nearest, Bilinear, Bicubic, FXAA, FSR, MetalFX } filter = Filter::Bilinear;
    id<MTLRenderPipelineState> create_filter_pipeline(const std::string &source, const char *entry);
    void initialize_fsr();
    id<MTLTexture> prepare_metalfx(id<MTLCommandBuffer> commands, id<MTLTexture> source, uint32_t width, uint32_t height);
public:
    explicit ScreenRenderer(Device &device, std::filesystem::path static_assets = {});
    void set_filter(bool linear);
    void set_filter(std::string_view name);
    id<MTLTexture> prepare(id<MTLCommandBuffer> commands, id<MTLTexture> source, uint32_t width, uint32_t height);
    void render(id<MTLRenderCommandEncoder>, id<MTLTexture> source, MTLViewport);
};
}
