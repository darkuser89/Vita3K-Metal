// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal4/metal4_pipeline.h>
#include <objc/runtime.h>
#include <stdexcept>

namespace renderer::metal {
namespace { char descriptor_key; }
id<MTLFunction> metal4_function(id<MTLLibrary> library, NSString *name, MTLFunctionConstantValues *constants, NSError **error) {
    if (!library || !name) return nil;
    id<MTLFunction> function;
    if (constants) {
        auto metadata = [MTLFunctionDescriptor new];
        metadata.name = name;
        metadata.specializedName = name;
        metadata.constantValues = constants;
        function = [library newFunctionWithDescriptor:metadata error:error];
    } else function = [library newFunctionWithName:name];
    if (!function) return nil;
    auto base = [MTL4LibraryFunctionDescriptor new];
    base.library = library;
    base.name = name;
    MTL4FunctionDescriptor *descriptor = base;
    if (constants) {
        auto specialized = [MTL4SpecializedFunctionDescriptor new];
        specialized.functionDescriptor = base;
        specialized.constantValues = constants;
        // Device gives every source/stage/constant variant its own stable entry
        // name. Use it in both descriptors so archive harvesting and lookup agree.
        specialized.specializedName = name;
        descriptor = specialized;
    }
    objc_setAssociatedObject(function, &descriptor_key, descriptor, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    return function;
}
MTL4FunctionDescriptor *metal4_function_descriptor(id<MTLFunction> function) {
    if (!function) return nil;
    auto descriptor = (MTL4FunctionDescriptor *)objc_getAssociatedObject(function, &descriptor_key);
    if (!descriptor) throw std::runtime_error("Metal 4 shader lacks its library/specialization descriptor");
    return descriptor;
}
MTL4RenderPipelineDescriptor *metal4_render_descriptor(MTLRenderPipelineDescriptor *source) {
    auto desc = [MTL4RenderPipelineDescriptor new];
    desc.label = source.label;
    desc.vertexFunctionDescriptor = metal4_function_descriptor(source.vertexFunction);
    desc.fragmentFunctionDescriptor = metal4_function_descriptor(source.fragmentFunction);
    desc.vertexDescriptor = source.vertexDescriptor;
    desc.rasterSampleCount = source.rasterSampleCount;
    desc.alphaToCoverageState = source.alphaToCoverageEnabled ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
    desc.alphaToOneState = source.alphaToOneEnabled ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
    desc.rasterizationEnabled = source.rasterizationEnabled;
    desc.inputPrimitiveTopology = source.inputPrimitiveTopology;
    desc.maxVertexAmplificationCount = source.maxVertexAmplificationCount;
    desc.supportIndirectCommandBuffers = source.supportIndirectCommandBuffers ? MTL4IndirectCommandBufferSupportStateEnabled : MTL4IndirectCommandBufferSupportStateDisabled;
    for (NSUInteger i = 0; i < 8; ++i) {
        auto from = source.colorAttachments[i]; auto to = desc.colorAttachments[i];
        to.pixelFormat = from.pixelFormat;
        to.blendingState = from.blendingEnabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
        to.sourceRGBBlendFactor = from.sourceRGBBlendFactor;
        to.destinationRGBBlendFactor = from.destinationRGBBlendFactor;
        to.rgbBlendOperation = from.rgbBlendOperation;
        to.sourceAlphaBlendFactor = from.sourceAlphaBlendFactor;
        to.destinationAlphaBlendFactor = from.destinationAlphaBlendFactor;
        to.alphaBlendOperation = from.alphaBlendOperation;
        to.writeMask = from.writeMask;
    }
    // Metal 4 selects depth/stencil formats with the render pass, rather than
    // baking them into the graphics pipeline. Depth state is still preserved.
    return desc;
}
MTL4ComputePipelineDescriptor *metal4_compute_descriptor(id<MTLFunction> function) {
    auto desc = [MTL4ComputePipelineDescriptor new];
    desc.computeFunctionDescriptor = metal4_function_descriptor(function);
    return desc;
}
id<MTLRenderPipelineState> metal4_render_pipeline(id<MTL4Compiler> compiler, MTLRenderPipelineDescriptor *source, MTL4CompilerTaskOptions *options, NSError **error) {
    return [compiler newRenderPipelineStateWithDescriptor:metal4_render_descriptor(source) compilerTaskOptions:options error:error];
}
id<MTLComputePipelineState> metal4_compute_pipeline(id<MTL4Compiler> compiler, id<MTLFunction> function, MTL4CompilerTaskOptions *options, NSError **error) {
    return [compiler newComputePipelineStateWithDescriptor:metal4_compute_descriptor(function) compilerTaskOptions:options error:error];
}
}
