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

#include <shader/usse_types.h>
#include <gxm/types.h>

#include <map>
#include <vector>

namespace spv {
typedef unsigned int Id;
} // namespace spv

namespace shader::usse {

using SpirvCode = std::vector<uint32_t>;
using SpirvVarRegBank = spv::Id;

struct SpirvUniformBufferInfo {
    std::uint32_t base;
    std::uint32_t size;
    std::uint32_t index_in_container;
};

struct SamplerInfo {
    spv::Id id;
    uint32_t index;
    DataType component_type;
    uint8_t component_count;
    bool is_cube;
};
using SamplerMap = std::map<uint32_t, SamplerInfo>;

struct SpirvShaderParameters {
    bool native_metal = false;
    // Fragment PA words initialized by vertex linkage before USSE execution.
    std::uint32_t metal_fragment_input_pa_regs = 0;
    bool native_cube_float_filter = false;
    spv::Id native_texture_info = 0;
    spv::Id native_frag_coord = 0;
    float native_texture_lod_bias[SCE_GXM_MAX_TEXTURE_UNITS] = {};
    // Native fragment emulation of SGX global registers 23, 24 and 43.
    spv::Id native_global_regs = 0;
    // Effective front face, including point replay, for the typed g16 load.
    spv::Id native_effective_front_facing_id = 0;
    // Mapped to 'pa' (primary attribute) USSE registers
    // for vertex: vertex inputs (vertex attributes)
    // for fragment: fragment inputs (linkage from vertex stage)
    SpirvVarRegBank ins;

    // Mapped to 'sa' (secondary attribute) USSE registers
    SpirvVarRegBank uniforms;

    // Mapped to 'r' (temporary) USSE registers
    SpirvVarRegBank temps;
    spv::Id frag_output_holds_declared_type = 0;

    // Mapped to 'i' (internal) usse registers
    SpirvVarRegBank internals;

    // Mapped to 'o' (output) USSE registers
    // for vertex: vertex outputs (linkage into fragment stage)
    // for fragment: fragment outputs (color outputs)
    SpirvVarRegBank outs;

    // Mapped to 'idx' (index) USSE registers
    // Use for indexing bank registers dynamically
    SpirvVarRegBank indexes;

    // Mapped to 'p' (predicates) USSE registers
    // Store boolean, use for conditional execution.
    SpirvVarRegBank predicates;

    // Sampler map. Since all banks are a flat array, sampler must be in an explicit bank.
    SamplerMap samplers;

    // Uniform buffer map contains layout info of a UBO inside the big SSBO.
    // only used if memory mapping is not enabled
    std::map<std::uint32_t, SpirvUniformBufferInfo> buffers;

    // when not using buffer device address, contains the storage buffer type
    spv::Id buffer_container;

    // ids for the given fields in the uniform block container
    int buffer_addresses_id;
    uint32_t buffer_count = 0;
    int viewport_ratio_id;
    int viewport_offset_id;

    // when using a thread, texture or literal buffer, if not -1, this fields contain the sa register
    // with the matching address, this assumes of course that this address is not copied somewhere
    // else and that this register is not overwritten
    // the base field is the offset to be applied when reading this buffer (almost always -4)
    int texture_buffer_sa_offset = -1;
    int texture_buffer_base;

    int literal_buffer_sa_offset = -1;
    int literal_buffer_base;

    int thread_buffer_sa_offset = -1;
    int thread_buffer_base;
    spv::Id thread_buffer;
    uint32_t thread_buffer_f32_count = 0;

    spv::Id render_info_id;

    // When using shader interlock, specialization constant telling us if the texture is gamma corrected
    spv::Id is_srgb_constant;
};

using Coord = std::pair<spv::Id, int>;

struct NonDependentTextureQueryCallInfo {
    spv::Id sampler;
    Coord coord;
    int coord_index = 0;
    int sampler_index = 0;

    std::uint32_t dest_offset = 0;
    int store_type = 0; ///< For sampling method later
    int prod_pos = -1;
    int dim = 2;

    DataType component_type;
    uint8_t component_count;
    uint8_t store_component_count = 0; // Metal: program-derived query write width, zero keeps the hint.
};

using NonDependentTextureQueryCallInfos = std::vector<NonDependentTextureQueryCallInfo>;

} // namespace shader::usse
