// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later

#include <shader/msl_recompiler.h>
#include <shader/metal_capture.h>

#define SPV_ENABLE_UTILITY_CODE
#include <spirv_msl.hpp>

#include <stdexcept>
#include <unordered_map>
#include <cstdlib>
#include <iostream>

namespace shader::metal {

bool writes_external_memory(const std::vector<uint32_t> &words) {
    if (words.size()<5 || words[0]!=spv::MagicNumber) throw std::invalid_argument("Invalid SPIR-V memory analysis input");
    std::unordered_map<uint32_t,spv::StorageClass> pointer_types, pointers;
    bool legacy_storage_buffer=false;
    const auto visit = [&](auto &&function) {
        for (size_t at=5;at<words.size();) {
            const uint32_t count=words[at]>>16;
            if (!count || count>words.size()-at) throw std::invalid_argument("Truncated SPIR-V memory analysis input");
            function(static_cast<spv::Op>(words[at]&65535),words.data()+at+1,count-1);
            at+=count;
        }
    };
    visit([&](spv::Op op,const uint32_t *args,uint32_t count) {
        if(op==spv::OpDecorate && count>=2 && args[1]==spv::DecorationBufferBlock) legacy_storage_buffer=true;
        if (op==spv::OpTypePointer) {
            if(count!=3) throw std::invalid_argument("Invalid SPIR-V pointer type");
            pointer_types[args[0]]=static_cast<spv::StorageClass>(args[1]);
        }
    });
    if(legacy_storage_buffer) return true;
    visit([&](spv::Op op,const uint32_t *args,uint32_t count) {
        bool result=false,typed=false;spv::HasResultAndType(op,&result,&typed);
        if (typed) {
            if(count<2) throw std::invalid_argument("Invalid typed SPIR-V instruction");
            if(auto found=pointer_types.find(args[0]);found!=pointer_types.end()) pointers[args[1]]=found->second;
        }
    });
    const auto external = [](spv::StorageClass storage) {
        return storage!=spv::StorageClassFunction && storage!=spv::StorageClassPrivate && storage!=spv::StorageClassOutput;
    };
    bool writes=false;
    auto mark_write=[&](const char *reason,spv::Op op,uint32_t id) {
        if(!writes && std::getenv("VITA3K_METAL_TRACE_MEMORY_ANALYSIS"))
            std::cerr<<"Metal memory analysis: "<<reason<<" op="<<spv::OpToString(op)<<" pointer="<<id<<'\n';
        writes=true;
    };
    visit([&](spv::Op op,const uint32_t *args,uint32_t count) {
        // Core module declarations contain literals (capability numbers, array
        // lengths, constant bits) which must never be mistaken for pointer IDs.
        if(uint32_t(op)<uint32_t(spv::OpFunction) && op!=spv::OpExtInst) return;
        if(op==spv::OpStore || op==spv::OpCopyMemory || op==spv::OpCopyMemorySized) {
            if(!count) throw std::invalid_argument("Invalid SPIR-V memory write");
            const auto pointer=pointers.find(args[0]);
            if(pointer==pointers.end() || external(pointer->second)) mark_write("store",op,args[0]);
        }
        // Permit only known non-writing uses of external pointers. Unknown
        // pointer-consuming extensions, calls and atomics keep synchronization.
        switch(op) {
        case spv::OpLoad: case spv::OpStore: case spv::OpVariable: case spv::OpFunctionParameter:
        case spv::OpAccessChain: case spv::OpInBoundsAccessChain: case spv::OpPtrAccessChain: case spv::OpInBoundsPtrAccessChain:
        case spv::OpConvertUToPtr: case spv::OpConvertPtrToU: case spv::OpBitcast: case spv::OpCopyObject:
        case spv::OpPhi: case spv::OpSelect: case spv::OpPtrEqual: case spv::OpPtrNotEqual: case spv::OpPtrDiff:
        case spv::OpName: case spv::OpDecorate: case spv::OpMemberDecorate: case spv::OpEntryPoint:
        case spv::OpLine: case spv::OpNoLine: case spv::OpModuleProcessed: case spv::OpExecutionModeId: case spv::OpDecorateId:
            return;
        default: break;
        }
        for(uint32_t i=op==spv::OpExtInst?4u:0u;i<count;++i)
            if(auto pointer=pointers.find(args[i]);pointer!=pointers.end()) {
                const auto storage=pointer->second;
                if(storage==spv::StorageClassPhysicalStorageBuffer || storage==spv::StorageClassStorageBuffer
                    || storage==spv::StorageClassCrossWorkgroup || storage==spv::StorageClassGeneric || storage==spv::StorageClassAtomicCounter)
                    mark_write("unproven pointer use",op,args[i]);
            }
    });
    return writes;
}

Program convert_overlay_spirv(const std::vector<uint32_t> &spirv) {
    if (spirv.size() < 5 || spirv.front() != spv::MagicNumber)
        throw std::invalid_argument("Invalid built-in overlay SPIR-V");
    spirv_cross::CompilerMSL compiler(spirv);
    const auto entries = compiler.get_entry_points_and_stages();
    if (entries.size() != 1)
        throw std::invalid_argument("Overlay must have one shader entry point");
    const auto &entry = entries.front();
    const bool vertex = entry.execution_model == spv::ExecutionModelVertex;
    if (!vertex && entry.execution_model != spv::ExecutionModelFragment)
        throw std::invalid_argument("Invalid overlay shader stage");
    auto common = compiler.get_common_options();
    common.vertex.flip_vert_y = vertex;
    compiler.set_common_options(common);
    spirv_cross::CompilerMSL::Options options;
    options.platform = spirv_cross::CompilerMSL::Options::macOS;
    options.set_msl_version(3, 0);
    compiler.set_msl_options(options);
    spirv_cross::MSLResourceBinding constants;
    constants.stage = entry.execution_model;
    constants.desc_set = spirv_cross::kPushConstDescSet;
    constants.binding = spirv_cross::kPushConstBinding;
    constants.msl_buffer = 0;
    compiler.add_msl_resource_binding(constants);
    for (const auto &resource : compiler.get_shader_resources().sampled_images) {
        const auto binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
        if (compiler.get_decoration(resource.id, spv::DecorationDescriptorSet) != 0 || binding > 1)
            throw std::invalid_argument("Invalid overlay texture binding");
        spirv_cross::MSLResourceBinding texture;
        texture.stage = entry.execution_model;
        texture.desc_set = 0; texture.binding = binding;
        texture.msl_texture = texture.msl_sampler = binding;
        compiler.add_msl_resource_binding(texture);
    }
    Program result;
    result.stage = vertex ? Stage::Vertex : Stage::Fragment;
    result.source = compiler.compile();
    result.entry_point = compiler.get_cleansed_entry_point_name(entry.name, entry.execution_model);
    return result;
}

Program convert_builtin_compute_spirv(const std::vector<uint32_t> &spirv) {
    if (spirv.size() < 5 || spirv.front() != spv::MagicNumber)
        throw std::invalid_argument("Invalid built-in compute SPIR-V");
    spirv_cross::CompilerMSL compiler(spirv);
    const auto entries = compiler.get_entry_points_and_stages();
    if (entries.size() != 1 || entries.front().execution_model != spv::ExecutionModelGLCompute)
        throw std::invalid_argument("Built-in compute shader must have one compute entry point");
    const auto &entry = entries.front();
    const auto resources = compiler.get_shader_resources();
    if (resources.push_constant_buffers.size() != 1 || resources.separate_images.size() != 1
        || resources.storage_images.size() != 1 || resources.separate_samplers.size() != 1
        || !resources.sampled_images.empty() || !resources.uniform_buffers.empty()
        || !resources.storage_buffers.empty())
        throw std::invalid_argument("Unexpected built-in compute resource layout");
    const auto bind = [&](const spirv_cross::Resource &resource, uint32_t expected, uint32_t texture, uint32_t sampler) {
        if (compiler.get_decoration(resource.id, spv::DecorationDescriptorSet) != 0
            || compiler.get_decoration(resource.id, spv::DecorationBinding) != expected)
            throw std::invalid_argument("Unexpected built-in compute binding");
        spirv_cross::MSLResourceBinding binding;
        binding.stage = entry.execution_model;
        binding.desc_set = 0;
        binding.binding = expected;
        binding.msl_texture = texture;
        binding.msl_sampler = sampler;
        compiler.add_msl_resource_binding(binding);
    };
    bind(resources.separate_images.front(), 1, 0, 0);
    bind(resources.storage_images.front(), 2, 1, 0);
    bind(resources.separate_samplers.front(), 3, 0, 0);
    spirv_cross::MSLResourceBinding constants;
    constants.stage = entry.execution_model;
    constants.desc_set = spirv_cross::kPushConstDescSet;
    constants.binding = spirv_cross::kPushConstBinding;
    constants.msl_buffer = 0;
    compiler.add_msl_resource_binding(constants);
    spirv_cross::CompilerMSL::Options options;
    options.platform = spirv_cross::CompilerMSL::Options::macOS;
    options.set_msl_version(3, 0);
    compiler.set_msl_options(options);
    Program result;
    result.stage = Stage::Compute;
    result.source = compiler.compile();
    result.entry_point = compiler.get_cleansed_entry_point_name(entry.name, entry.execution_model);
    result.writes_guest_memory = false;
    return result;
}

Program convert_spirv(const std::vector<uint32_t> &spirv) {
    if (spirv.size() < 5 || spirv.front() != spv::MagicNumber)
        throw std::invalid_argument("Metal shader is not a SPIR-V module");

    spirv_cross::CompilerMSL compiler(spirv);
    bool uses_interlock = false;
    bool uses_buffer_addresses = false;
    for (const auto capability : compiler.get_declared_capabilities()) {
        uses_buffer_addresses |= capability == spv::CapabilityPhysicalStorageBufferAddresses;
        uses_interlock |= capability == spv::CapabilityFragmentShaderSampleInterlockEXT
            || capability == spv::CapabilityFragmentShaderPixelInterlockEXT;
    }
    const auto entries = compiler.get_entry_points_and_stages();
    if (entries.size() != 1)
        throw std::invalid_argument("Metal GXM shader must have exactly one entry point");
    const auto &entry = entries.front();
    const bool vertex = entry.execution_model == spv::ExecutionModelVertex;
    if (!vertex && entry.execution_model != spv::ExecutionModelFragment)
        throw std::invalid_argument("Metal GXM shader must be a vertex or fragment shader");

    spirv_cross::CompilerMSL::Options options;
    options.platform = spirv_cross::CompilerMSL::Options::macOS;
    options.set_msl_version(3, 0);
    options.use_framebuffer_fetch_subpasses = true;
    compiler.set_msl_options(options);

    const auto resources = compiler.get_shader_resources();
    const auto bind = [&](const spirv_cross::Resource &resource, bool buffer, bool attachment, bool framebuffer_fetch = false) {
        if (!compiler.has_decoration(resource.id, spv::DecorationDescriptorSet)
            || !compiler.has_decoration(resource.id, spv::DecorationBinding))
            throw std::invalid_argument("Metal GXM resource is missing its explicit descriptor binding");
        const uint32_t set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
        const uint32_t binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
        spirv_cross::MSLResourceBinding remap;
        remap.stage = entry.execution_model;
        remap.desc_set = set;
        remap.binding = binding;
        if (buffer) {
            const uint32_t render_binding = vertex ? 0 : 1;
            const uint32_t uniform_binding = vertex ? 2 : 3;
            if (set != 0 || (binding != render_binding && binding != uniform_binding && binding != 4
                && !(vertex && binding == 5)))
                throw std::invalid_argument("Unexpected GXM Metal buffer binding");
            remap.msl_buffer = binding == 5 ? VERTEX_OUTPUT_CAPTURE_BUFFER
                : binding == 4 ? TEXTURE_INFO_BUFFER : binding == render_binding ? RENDER_INFO_BUFFER : UNIFORM_BUFFER;
        } else if (attachment) {
            if (vertex || set != 1 || binding > 2)
                throw std::invalid_argument("Unexpected GXM Metal attachment binding");
            // SPIRV-Cross interprets msl_texture as [[color(n)]] for native
            // framebuffer fetch, not as a sampled/storage texture slot.
            remap.msl_texture = framebuffer_fetch ? 0 : COLOR_ATTACHMENT_TEXTURE + binding;
        } else {
            if (set != (vertex ? 2u : 3u) || binding >= TEXTURE_COUNT)
                throw std::invalid_argument("Unexpected GXM Metal texture binding");
            remap.msl_texture = binding;
            remap.msl_sampler = binding;
        }
        compiler.add_msl_resource_binding(remap);
    };

    for (const auto &resource : resources.uniform_buffers)
        bind(resource, true, false);
    for (const auto &resource : resources.storage_buffers)
        bind(resource, true, false);
    uint32_t cube_texture_mask = 0;
    for (const auto &resource : resources.sampled_images) {
        bind(resource, false, false);
        if (compiler.get_type(resource.base_type_id).image.dim == spv::DimCube)
            cube_texture_mask |= 1u << compiler.get_decoration(resource.id, spv::DecorationBinding);
    }
    for (const auto &resource : resources.storage_images)
        bind(resource, false, true);
    for (const auto &resource : resources.subpass_inputs) {
        if (compiler.get_decoration(resource.id, spv::DecorationInputAttachmentIndex) != 0)
            throw std::invalid_argument("Unexpected GXM Metal framebuffer attachment index");
        bind(resource, false, true, true);
    }
    if (!resources.push_constant_buffers.empty() || !resources.separate_images.empty() || !resources.separate_samplers.empty())
        throw std::invalid_argument("Unsupported GXM Metal shader resource layout");

    Program result;
    result.stage = vertex ? Stage::Vertex : Stage::Fragment;
    result.uses_framebuffer_fetch = !resources.subpass_inputs.empty();
    result.uses_raster_order_groups = uses_interlock;
    result.uses_buffer_addresses = uses_buffer_addresses;
    result.writes_guest_memory = writes_external_memory(spirv);
    result.cube_texture_mask = cube_texture_mask;
    result.source = compiler.compile();
    result.entry_point = compiler.get_cleansed_entry_point_name(entry.name, entry.execution_model);
    return result;
}

Program point_replay_vertex_program() {
    Program result;
    result.stage = Stage::Vertex;
    result.entry_point = "point_replay_vs";
    result.writes_guest_memory = false;
    result.source = R"(
#include <metal_stdlib>
using namespace metal;

struct PointReplayOutput {
    float4 gl_Position [[position]];
    float gl_PointSize [[point_size]];
    float4 v_Color0 [[user(locn1)]];
    float4 v_Color1 [[user(locn2)]];
    float4 v_Fog [[user(locn3)]];
    float4 v_TexCoord0 [[user(locn4)]];
    float4 v_TexCoord1 [[user(locn5)]];
    float4 v_TexCoord2 [[user(locn6)]];
    float4 v_TexCoord3 [[user(locn7)]];
    float4 v_TexCoord4 [[user(locn8)]];
    float4 v_TexCoord5 [[user(locn9)]];
    float4 v_TexCoord6 [[user(locn10)]];
    float4 v_TexCoord7 [[user(locn11)]];
    float4 v_TexCoord8 [[user(locn12)]];
    float4 v_TexCoord9 [[user(locn13)]];
};

vertex PointReplayOutput point_replay_vs(const device float4* outputs [[buffer(4)]],
    uint vertex_id [[vertex_id]]) {
    const uint base = vertex_id * )" + std::to_string(CAPTURE_OUTPUT_SLOT_COUNT) + R"(u;
    PointReplayOutput result;
    result.gl_Position = outputs[base];
    result.gl_PointSize = outputs[base + 14].x;
    result.v_Color0 = outputs[base + 1];
    result.v_Color1 = outputs[base + 2];
    result.v_Fog = outputs[base + 3];
    result.v_TexCoord0 = outputs[base + 4];
    result.v_TexCoord1 = outputs[base + 5];
    result.v_TexCoord2 = outputs[base + 6];
    result.v_TexCoord3 = outputs[base + 7];
    result.v_TexCoord4 = outputs[base + 8];
    result.v_TexCoord5 = outputs[base + 9];
    result.v_TexCoord6 = outputs[base + 10];
    result.v_TexCoord7 = outputs[base + 11];
    result.v_TexCoord8 = outputs[base + 12];
    result.v_TexCoord9 = outputs[base + 13];
    return result;
}
)";
    return result;
}

} // namespace shader::metal
