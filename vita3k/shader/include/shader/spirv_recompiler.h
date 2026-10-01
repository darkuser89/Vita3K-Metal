// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#pragma once

#include <gxm/types.h>
#include <shader/usse_translator_types.h>
#include <shader/usse_types.h>

#include <features/state.h>

#include <string>
#include <vector>

namespace shader {

static constexpr int COLOR_ATTACHMENT_TEXTURE_SLOT_IMAGE = 0;
static constexpr int MASK_TEXTURE_SLOT_IMAGE = 1;
static constexpr int COLOR_ATTACHMENT_RAW_TEXTURE_SLOT_IMAGE = 3;
static constexpr uint32_t CURRENT_VERSION = 17;
// fragment shader using the rendering surface as a storage image (because of shader interlock) have a line
// layout (constant_id = GAMMA_CORRECTION_SPECIALIZATIO_ID) const bool is_srgb = false;
// Setting this constant to true performs gamma correction in the shader
static constexpr uint32_t GAMMA_CORRECTION_SPECIALIZATION_ID = 0;

enum struct Target {
    GLSLOpenGL,
    SpirVOpenGL,
    SpirVVulkan,
    SpirVMetal
};

// Hints given while compiling the shader
// For example, the gxp shader knows at runtime the swizzle of the color surface
// or the format / number of components of the sampled images, but we can't
struct Hints {
    // used when symbols are stripped
    // also used when scaled attribute formats are not supported and conversion must be done in the shader
    const std::vector<SceGxmVertexAttribute> *attributes;
    // color format of the surface, used for the opengl renderer when the GPU does not support features.support_unknown_format
    // I'm asking for the format instead of the base format as the swizzle might be used in future updates for surfaces with only 1 or 2 components
    SceGxmColorFormat color_format;
    // RG8 backs Metal R/A color targets. Their independent GXM red/alpha
    // blend equations are evaluated with framebuffer fetch in the fragment
    // shader because Metal applies one RGB equation to both RG channels.
    bool metal_red_alpha_shader_blend = false;
    SceGxmBlendInfo metal_red_alpha_blend = {};
    // The packed 3/3/2 and mixed signed targets use expanded Metal attachments.
    // Apply guest channel precision between draws, including masked channels.
    bool metal_quantized_color_blend = false;
    SceGxmBlendInfo metal_quantized_blend = {};

    // we need to have the texture formats of the sampled textures in two cases:
    // - when doing dependent texture queries with unknown format we need to know the format
    // - when sampling, we need to know the component count of a texture
    SceGxmTextureFormat vertex_textures[SCE_GXM_MAX_TEXTURE_UNITS];
    SceGxmTextureFormat fragment_textures[SCE_GXM_MAX_TEXTURE_UNITS];
    // Native Metal mask attachment sample count; ignored by other targets.
    uint32_t metal_samples = 1;
    // Missing varyings requested by the paired fragment program. Use (0,0,0,1)
    // for unwritten values; declared outputs retain their register layout.
    uint32_t metal_missing_vertex_outputs = 0;
    // Plus's far-plane policy, specialized per draw without changing shared uniforms.
    bool metal_far_clip = false;
    // Emit final clip-space position and varyings to a separate Metal buffer
    // for post-vertex polygon routing. Ordinary shader variants leave this off.
    bool metal_capture_vertex_outputs = false;
    // Store every captured vertex at slot zero. The renderer can bind a
    // different buffer offset per sparse guest index to avoid a huge buffer.
    bool metal_capture_vertex_outputs_compact = false;
    bool metal_mip_sampling = false;
    // Generate software Cube filtering only on devices that lack F32 filtering.
    bool metal_float_cube_filter = false;
    bool metal_raw_color_attachment = false;
    bool metal_raw_color_fetch = false;
    // Tile register width is independent of the stored surface format. Keep
    // the declared shader type when no native draw descriptor is available.
    uint32_t metal_output_register_size = SCE_GXM_OUTPUT_REGISTER_SIZE_64BIT;
    SceGxmOutputRegisterFormat metal_output_register_format = SCE_GXM_OUTPUT_REGISTER_FORMAT_DECLARED;
    // Metal samplers have no descriptor LOD-bias field. Native translation
    // applies these values to sampling operations; other targets ignore them.
    float metal_vertex_lod_bias[SCE_GXM_MAX_TEXTURE_UNITS] = {};
    float metal_fragment_lod_bias[SCE_GXM_MAX_TEXTURE_UNITS] = {};
};

struct GeneratedShader {
    std::string glsl;
    usse::SpirvCode spirv;
};

// Dump generated SPIR-V disassembly up to this point
void spirv_disasm_print(const usse::SpirvCode &spirv_binary, std::string *spirv_dump = nullptr);

// the returned object will only have its glsl or spirv field non-empty depending on the target
GeneratedShader convert_gxp(const SceGxmProgram &program, const std::string &shader_hash, const FeatureState &features, const Target target, const Hints &hints, bool maskupdate = false,
    bool force_shader_debug = false, const std::function<bool(const std::string &ext, const std::string &dump)> &dumper = nullptr);

void convert_gxp_to_glsl_from_filepath(const std::string &shader_filepath_utf8);

} // namespace shader
