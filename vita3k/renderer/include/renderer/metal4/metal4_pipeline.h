// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal4/apple.h>

namespace renderer::metal {
// Retain the existing shader metadata and descriptor construction, while all
// executable pipelines are compiled by the Metal 4 compiler.
id<MTLFunction> metal4_function(id<MTLLibrary>, NSString *, MTLFunctionConstantValues * = nil, NSError ** = nullptr);
MTL4FunctionDescriptor *metal4_function_descriptor(id<MTLFunction>);
MTL4RenderPipelineDescriptor *metal4_render_descriptor(MTLRenderPipelineDescriptor *);
MTL4ComputePipelineDescriptor *metal4_compute_descriptor(id<MTLFunction>);
id<MTLRenderPipelineState> metal4_render_pipeline(id<MTL4Compiler>, MTLRenderPipelineDescriptor *, MTL4CompilerTaskOptions *, NSError **);
id<MTLComputePipelineState> metal4_compute_pipeline(id<MTL4Compiler>, id<MTLFunction>, MTL4CompilerTaskOptions *, NSError **);
}
