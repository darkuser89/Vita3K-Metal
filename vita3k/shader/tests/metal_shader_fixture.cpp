// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include "metal_shader_fixture.h"
#include <shader/spirv_recompiler.h>
#include <shader/gxp_parser.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>

// Keep guest headers in C++: Apple's legacy Ptr typedef conflicts with GXM Ptr.
shader::metal::Program compile_gxp_fixture(const std::vector<uint32_t> &words, size_t size,
    const std::string &name, const std::string &mode, float lod_bias) {
    if (size < sizeof(SceGxmProgram))
        throw std::runtime_error("Truncated GXP header");
    const auto &gxp = *reinterpret_cast<const SceGxmProgram *>(words.data());
    if (gxp.magic != 0x00505847 || gxp.size > size || gxp.size < sizeof(SceGxmProgram))
        throw std::runtime_error("Invalid GXP header");
    const auto inputs = shader::get_program_input(gxp);
    uint32_t packed_c10 = 0, typed_c10 = 0, cube_samplers = 0;
    for (const auto &input : inputs.inputs) {
        if (input.type != shader::usse::DataType::C10) continue;
        if (const auto *attribute = std::get_if<shader::usse::AttributeInputSource>(&input.source)) {
            if (attribute->regformat) ++packed_c10;
            else ++typed_c10;
        }
    }
    for (const auto &sampler : inputs.samplers) cube_samplers += sampler.is_cube;
    std::cout << "META stage=" << (gxp.is_vertex() ? "vertex" : "fragment")
              << " packed_c10=" << packed_c10 << " typed_c10=" << typed_c10
              << " samplers=" << inputs.samplers.size() << " cube_samplers=" << cube_samplers
              << " thread_buffer_bytes=" << gxp.thread_buffer_count << '\n';
    FeatureState features;
    features.support_shader_interlock = mode == "gxp-interlock" || mode == "gxp-interlock-f16";
    features.direct_fragcolor = !features.support_shader_interlock;
    features.preserve_f16_nan_as_u16 = mode == "gxp-interlock-f16";
    features.support_unknown_format = true;
    features.support_scaled_attribute_formats = false;
    const bool msaa = mode == "gxp-msaa2" || mode == "gxp-msaa4";
    const bool unit13_shadow = mode == "gxp-mapped-unit13-shadow" || mode == "gxp-mapped-unit13-shadow-mips";
    const bool unit13_cube = mode == "gxp-mapped-unit13-cube" || mode == "gxp-mapped-unit13-mips";
    const bool shadow = mode == "gxp-mapped-shadow" || mode == "gxp-mapped-shadow-mips" || unit13_shadow;
    const bool half_attachment = mode == "gxp-mapped-half" || mode == "gxp-mapped-projected-half";
    const bool mip_sampling = mode == "gxp-mapped-shadow-mips" || mode == "gxp-mapped-unit13-shadow-mips" || mode == "gxp-mapped-mips" || mode == "gxp-mapped-unit13-mips" || mode == "gxp-mapped-projected-half";
    const bool mapped = mip_sampling || unit13_cube || half_attachment || shadow || msaa || mode == "gxp-mapped" || mode == "gxp-mapped-gbuffer" || mode == "gxp-mapped-viewport" || mode == "gxp-native64";
    features.use_mask_bit = mode == "gxp-mask" || mode == "gxp-mask-update" || mapped;
    features.enable_memory_mapping = mapped;
    features.use_texture_viewport = mip_sampling || unit13_cube || half_attachment || shadow || msaa || mode == "gxp-mapped-viewport";
    shader::Hints hints{};
    hints.metal_mip_sampling = mip_sampling;
    std::fill_n(hints.metal_vertex_lod_bias, SCE_GXM_MAX_TEXTURE_UNITS, lod_bias);
    std::fill_n(hints.metal_fragment_lod_bias, SCE_GXM_MAX_TEXTURE_UNITS, lod_bias);
    hints.metal_samples = msaa ? (mode == "gxp-msaa2" ? 2 : 4) : 1;
    // Explicit offline target assumption; real draws supply outputRegisterSize.
    hints.metal_output_register_size = unit13_cube || half_attachment || mode == "gxp-native64" || mode == "gxp-mapped-gbuffer" ? SCE_GXM_OUTPUT_REGISTER_SIZE_64BIT : SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT;
    // Offline dumps do not contain the draw's vertex layout. Validate shader
    // compilation with an explicit empty layout; runtime validation must use
    // the actual draw attributes and texture/color formats.
    const std::vector<SceGxmVertexAttribute> attributes;
    hints.attributes = &attributes;
    hints.color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
    if (half_attachment)
        hints.color_format = SCE_GXM_COLOR_FORMAT_F16F16F16F16_ABGR;
    if (unit13_cube)
        hints.color_format = SCE_GXM_COLOR_FORMAT_F32F32_GR;
    std::fill_n(hints.vertex_textures, SCE_GXM_MAX_TEXTURE_UNITS, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR);
    std::fill_n(hints.fragment_textures, SCE_GXM_MAX_TEXTURE_UNITS, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR);
    if (shadow)
        hints.fragment_textures[14] = SCE_GXM_TEXTURE_FORMAT_F32_R;
    if (unit13_shadow) {
        hints.fragment_textures[10] = SCE_GXM_TEXTURE_FORMAT_F32_R;
        hints.fragment_textures[11] = SCE_GXM_TEXTURE_FORMAT_F32_R;
    }
    if (mode == "gxp-mapped-gbuffer") {
        hints.color_format = SCE_GXM_COLOR_FORMAT_F32F32_GR;
        hints.fragment_textures[0] = SCE_GXM_TEXTURE_FORMAT_UBC3_ABGR;
        hints.fragment_textures[1] = SCE_GXM_TEXTURE_FORMAT_PVRTII4BPP_ABGR;
    }
    auto result=shader::metal::convert_gxp(gxp, name, features, hints, mode == "gxp-mask-update" && gxp.is_fragment());
    std::cout << "NATIVE_MEMORY writes_guest=" << result.writes_guest_memory << '\n';
    std::cout << "NATIVE_CUBES mask=" << result.cube_texture_mask << '\n';
    return result;
}
