// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

// Establish Metal's query scope and sample counting before mapping it onto
// GXM's per-core, per-face visibility buffers. This is a native API fixture,
// not evidence that the emulator already implements visibility queries.
int main() {
    @autoreleasepool {
        try {
            std::string error;
            auto device = renderer::metal::Device::create(error);
            check(bool(device), error.c_str());
            NSError *native_error = nil;
            auto library = [device->native_device() newLibraryWithSource:@R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 vs(uint id [[vertex_id]]) {
    const float2 points[] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
    return float4(points[id],0.5,1);
}
fragment void fs(float4 p [[position]], constant uint &discard_half [[buffer(0)]]) {
    if (discard_half && uint(p.x) % 2) discard_fragment();
}
)" options:nil error:&native_error];
            check(library != nil, native_error.localizedDescription.UTF8String ?: "Cannot compile query probe");
            constexpr uint32_t width = 32, height = 24;
            constexpr uint64_t guard = 0xbadc0ffee1234567ull;
            unsigned passed = 0;
            for (uint32_t samples : {1u,4u}) {
                check([device->native_device() supportsTextureSampleCount:samples], "Required sample count is unavailable");
                auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                    width:width height:height mipmapped:NO];
                desc.storageMode = MTLStorageModePrivate; desc.usage = MTLTextureUsageRenderTarget;
                desc.sampleCount = samples;
                if (samples > 1) desc.textureType = MTLTextureType2DMultisample;
                auto depth = [device->native_device() newTextureWithDescriptor:desc];
                check(depth != nil, "Cannot allocate query depth texture");
                auto pipeline_desc = [MTLRenderPipelineDescriptor new];
                pipeline_desc.vertexFunction = [library newFunctionWithName:@"vs"];
                pipeline_desc.fragmentFunction = [library newFunctionWithName:@"fs"];
                pipeline_desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
                pipeline_desc.rasterSampleCount = samples;
                auto pipeline = [device->native_device() newRenderPipelineStateWithDescriptor:pipeline_desc error:&native_error];
                check(pipeline != nil, native_error.localizedDescription.UTF8String ?: "Cannot create query pipeline");
                auto depth_desc = [MTLDepthStencilDescriptor new];
                depth_desc.depthWriteEnabled = NO;
                for (unsigned scenario = 0; scenario < 11; ++scenario) {
                    std::array<uint64_t,8> initial;
                    initial.fill(guard);
                    initial[1] = 0;
                    auto results = [device->native_device() newBufferWithBytes:initial.data()
                        length:sizeof(initial) options:MTLResourceStorageModeShared];
                    const bool boolean = scenario == 2 || scenario == 3;
                    const bool hidden = scenario == 3 || scenario == 4;
                    const bool discard = scenario == 6;
                    const unsigned draws = scenario == 1 ? 2 : 1;
                    const unsigned passes = scenario == 8 ? 2 : 1;
                    uint64_t expected = uint64_t(width)*height*samples;
                    if (scenario == 1) expected *= 2;
                    if (boolean) expected = 1;
                    if (hidden || scenario == 7 || scenario == 10) expected = 0;
                    if (scenario == 5) expected = 7*5*samples;
                    if (discard) expected /= 2;
                    auto commands = [device->command_queue() commandBuffer];
                    for (unsigned pass_index = 0; pass_index < passes; ++pass_index) {
                        auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
                        pass.depthAttachment.texture = depth;
                        pass.depthAttachment.loadAction = MTLLoadActionClear;
                        pass.depthAttachment.storeAction = MTLStoreActionDontCare;
                        pass.depthAttachment.clearDepth = hidden ? 0 : 1;
                        pass.visibilityResultBuffer = results;
                        auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
                        check(encoder != nil, "Cannot create query encoder");
                        [encoder setRenderPipelineState:pipeline];
                        depth_desc.depthCompareFunction = MTLCompareFunctionLess;
                        [encoder setDepthStencilState:[device->native_device() newDepthStencilStateWithDescriptor:depth_desc]];
                        [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
                        [encoder setCullMode:scenario == 7 ? MTLCullModeFront : (scenario == 9 ? MTLCullModeBack : MTLCullModeNone)];
                        const uint32_t discard_half = discard;
                        [encoder setFragmentBytes:&discard_half length:4 atIndex:0];
                        [encoder setVisibilityResultMode:scenario == 10 ? MTLVisibilityResultModeDisabled
                            : (boolean ? MTLVisibilityResultModeBoolean : MTLVisibilityResultModeCounting) offset:8];
                        if (scenario == 5 || (scenario == 8 && pass_index == 0))
                            [encoder setScissorRect:MTLScissorRect{3,4,7,5}];
                        for (unsigned draw = 0; draw < draws; ++draw)
                            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        [encoder endEncoding];
                    }
                    check(device->submit_and_wait(commands,error),error.c_str());
                    const auto *actual = static_cast<const uint64_t *>(results.contents);
                    for (size_t i = 0; i < initial.size(); ++i) {
                        const uint64_t wanted = i == 1 ? expected : guard;
                        if (actual[i] != wanted)
                            throw std::runtime_error("Visibility mismatch scenario="+std::to_string(scenario)+" samples="+std::to_string(samples)
                                +" slot="+std::to_string(i)+" actual="+std::to_string(actual[i])+" expected="+std::to_string(wanted));
                    }
                    ++passed;
                    std::cout << "PASS query scenario=" << scenario << " samples=" << samples << " result=" << expected << " guards intact\n";
                }
            }
            std::cout << "PASS " << passed << " native visibility cases: counting, Boolean, repeated draws, depth rejection, scissor, discard, culling, render-pass scope\n";
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
