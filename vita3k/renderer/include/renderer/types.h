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
#include <renderer/commands.h>
#include <renderer/gxm_types.h>
#include <shader/spirv_recompiler.h>
#include <shader/usse_program_analyzer.h>
#include <util/hash.h>

#include <array>
#include <bitset>
#include <map>
#include <memory>
#include <vector>

static constexpr auto DEFAULT_RES_WIDTH = 960;
static constexpr auto DEFAULT_RES_HEIGHT = 544;

struct SceGxmProgram;
struct GxmState;

using UniformBufferSizes = std::array<std::uint32_t, 15>;

namespace renderer {
namespace metal { struct ProgramBinding; }

// State types
typedef std::map<Sha256Hash, const SceGxmProgram *> GXPPtrMap;

struct UniformSetRequest {
    const SceGxmProgramParameter *parameter;
    const void *data;
};

enum class Backend : uint32_t {
    OpenGL,
    Vulkan,
    Metal
};

enum class GXMState : std::uint16_t {
    RegionClip,
    Program,
    Viewport,
    DepthBias,
    DepthFunc,
    DepthWriteEnable,
    PolygonMode,
    PointLineWidth,
    StencilFunc,
    Texture,
    YuvProfile,
    StencilRef,
    VertexStream,
    TwoSided,
    CullMode,
    UniformBuffer,
    FragmentProgramEnable,
    VisibilityBuffer,
    VisibilityIndex,
    VisibilityBackIndex,
    TotalState
};

struct RenderTarget;

struct GXMStreamInfo {
    Ptr<const uint8_t> data = Ptr<const uint8_t>(0);
    size_t size = 0;
};

// We separate the following two parts of the stencil state because the first is part of the pipeline creation
// while the second is dynamic
struct GxmStencilStateOp {
    SceGxmStencilFunc func = SCE_GXM_STENCIL_FUNC_ALWAYS;
    SceGxmStencilOp stencil_fail = SCE_GXM_STENCIL_OP_KEEP;
    SceGxmStencilOp depth_fail = SCE_GXM_STENCIL_OP_KEEP;
    SceGxmStencilOp depth_pass = SCE_GXM_STENCIL_OP_KEEP;
};

struct GxmStencilStateValues {
    uint8_t compare_mask = 0xff; // TODO What's the default value?
    uint8_t write_mask = 0xff; // TODO What's the default value?
    uint8_t ref = 0; // TODO What's the default value?
};

// we hash the first part of this state as a key for the pipeline cache in vulkan
struct GxmRecordState {
    Sha256Hash vertex_program_hash;
    Sha256Hash fragment_program_hash;

    SceGxmColorBaseFormat color_base_format;

    SceGxmCullMode cull_mode = SCE_GXM_CULL_NONE;
    SceGxmTwoSidedMode two_sided = SCE_GXM_TWO_SIDED_DISABLED;
    SceGxmRegionClipMode region_clip_mode = SCE_GXM_REGION_CLIP_OUTSIDE;
    SceGxmPolygonMode front_polygon_mode = SCE_GXM_POLYGON_MODE_TRIANGLE_FILL;
    SceGxmPolygonMode back_polygon_mode = SCE_GXM_POLYGON_MODE_TRIANGLE_FILL;

    GxmStencilStateOp front_stencil_state_op;
    GxmStencilStateOp back_stencil_state_op;

    SceGxmDepthFunc front_depth_func = SCE_GXM_DEPTH_FUNC_LESS_EQUAL;
    SceGxmDepthFunc back_depth_func = SCE_GXM_DEPTH_FUNC_LESS_EQUAL;

    SceGxmDepthWriteMode front_depth_write_mode = SCE_GXM_DEPTH_WRITE_ENABLED;
    SceGxmDepthWriteMode back_depth_write_mode = SCE_GXM_DEPTH_WRITE_ENABLED;

    SceGxmFragmentProgramMode front_side_fragment_program_mode = SCE_GXM_FRAGMENT_PROGRAM_ENABLED;
    SceGxmFragmentProgramMode back_side_fragment_program_mode = SCE_GXM_FRAGMENT_PROGRAM_ENABLED;

    bool is_maskupdate = false;
    bool is_gamma_corrected = false;
    bool has_depth_stencil_surface = true;

    uint8_t _padding[5] = {};

    // Do not put any state not used for the Vulkan pipeline creation before vertex_streams
    std::array<GXMStreamInfo, SCE_GXM_MAX_VERTEX_STREAMS> vertex_streams;

    // Programs.
    Ptr<SceGxmFragmentProgram> fragment_program;
    Ptr<SceGxmVertexProgram> vertex_program;

    SceGxmColorSurface color_surface;
    SceGxmDepthStencilSurface depth_stencil_surface;

    bool viewport_flat = false;
    std::array<float, 4> viewport_flip = { 1.0f, 1.0f, 1.0f, 1.0f };
    float z_offset = 0.5f;
    float z_scale = 0.5f;

    GxmStencilStateValues front_stencil_state_values;
    GxmStencilStateValues back_stencil_state_values;

    uint32_t line_width = 1;
    uint32_t back_line_width = 1;

    int depth_bias_unit = 0;
    int depth_bias_slope = 0;
    int back_depth_bias_unit = 0;
    int back_depth_bias_slope = 0;

    SceIVector2 region_clip_min;
    SceIVector2 region_clip_max;

    float writing_mask = 0.0f;
};

struct Context {
    RenderTarget *current_render_target{};
    GxmRecordState record;

    CommandList command_list;
    CommandAllocFunc alloc_func;
    CommandFreeFunc free_func;

    int render_finish_status = 0;
    int notification_finish_status = 0;

    Sha256Hash last_draw_fragment_program_hash;
    Sha256Hash last_draw_vertex_program_hash;

    std::map<int, std::vector<uint8_t>> ubo_data;

    shader::Hints shader_hints;

    // Guest-thread bookkeeping for Metal's deferred GXM callbacks. Keep it
    // here so the guest-allocated SceGxmContext retains its size and layout.
    struct {
        bool enabled = false;
        bool recording_failed = false;
        // Non-owning: GXM destroys its contexts before its host state.
        GxmState *owner = nullptr;
        size_t vertex_reserved_bytes = 0;
        size_t fragment_reserved_bytes = 0;
    } metal_gxm;

    virtual ~Context() = default;
};

typedef std::bitset<SCE_GXM_MAX_TEXTURE_UNITS> TextureInfo;

struct ShaderProgram {
    // Populated only by Metal. The queued binding owns independent metadata
    // and GXP bytes, so guest program destruction cannot invalidate a draw.
    std::shared_ptr<const metal::ProgramBinding> metal_binding;
    Sha256Hash hash;
    UniformBufferSizes uniform_buffer_sizes; // Size of the buffer in 4-bytes unit
    UniformBufferSizes uniform_buffer_data_offsets; // Offset of the buffer in 4-bytes unit
    size_t max_total_uniform_buffer_storage;
    uint16_t buffer_count; // max buffer index used by the shader + 1
    uint32_t dynamic_uniform_buffers = 0; // Metal may need bytes past the declared register span

    uint16_t texture_count; // max texture index used by the shader + 1
    TextureInfo textures_used; // textures_used[i] is true if and only if the i-th texture is used by the shader
};

struct FragmentProgram : ShaderProgram {
};

struct VertexProgram : ShaderProgram {
    shader::usse::AttributeInformationMap attribute_infos;
};

struct ShadersHash {
    Sha256Hash frag;
    Sha256Hash vert;
};

struct RenderTarget {
    int holder;
    SceGxmMultisampleMode multisample_mode;
    bool has_macroblock_sync;
    uint16_t macroblock_width;
    uint16_t macroblock_height;
    virtual ~RenderTarget() = default;
};

} // namespace renderer
