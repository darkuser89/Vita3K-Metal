// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/functions.h>
#include <renderer/driver_functions.h>
#include <renderer/shaders.h>
#include <renderer/state.h>
namespace renderer {
void cmd_handle_create_context_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_destroy_context_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_create_render_target_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_destroy_render_target_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_memory_map_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_memory_unmap_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
bool create_metal3(std::unique_ptr<FragmentProgram> &fp, State &state, const SceGxmProgram &program, const SceGxmBlendInfo *blend, GXPPtrMap &gxp_ptr_map);
bool create_metal3(std::unique_ptr<VertexProgram> &vp, State &state, const SceGxmProgram &program, GXPPtrMap &gxp_ptr_map, const std::vector<SceGxmVertexAttribute> &attributes);
bool init_metal3(FrameHost &frame, std::unique_ptr<State> &state, Backend backend, const Config &config, const Root &root_paths);
void set_program_metal3(State &state, Context *ctx, Ptr<const void> program, const bool is_fragment,
    const std::shared_ptr<const metal::ProgramBinding> &metal_binding);
void set_context_metal3(State &state, Context *ctx, RenderTarget *target, SceGxmColorSurface *color_surface, SceGxmDepthStencilSurface *depth_stencil_surface);
void set_vertex_stream_metal3(State &state, Context *ctx, const std::size_t index, const std::size_t data_len, const Ptr<const void> stream);
void destroy_context_during_shutdown_metal3(State &state, std::unique_ptr<Context> &context);
void destroy_render_target_during_shutdown_metal3(State &state, std::unique_ptr<RenderTarget> &rt);
void set_uniform_buffer_metal3(State &state, Context *ctx, const bool is_vertex_uniform, const int block_number, const std::uint32_t block_size, const Ptr<const void> buffer);
void cmd_set_state_region_clip_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_program_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_uniform_buffer_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_viewport_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_depth_bias_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_depth_func_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_depth_write_enable_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_polygon_mode_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_point_line_width_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_stencil_func_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_stencil_ref_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_texture_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_yuv_profile(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_two_sided_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_cull_mode_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_vertex_stream_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_visibility_buffer_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_visibility_index_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_set_state_visibility_back_index(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, Context *render_context);
void cmd_handle_set_state_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void destroy_command_payload_metal3(Command &cmd);
void cmd_handle_set_context_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_sync_surface_data_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_mid_scene_flush_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_draw_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_transfer_copy_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_transfer_downscale_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_transfer_fill_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_sync_guest_range(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_handle_set_screen_filter_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void cmd_new_frame_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context);
void submit_command_list_metal3(State &state, renderer::Context *context, CommandList &command_list);
void generic_command_free_metal3(Command *cmd);
void process_batch_metal3(renderer::State &state, const FeatureState &features, MemState &mem, Config &config, CommandList &command_list);
void process_batches_metal3(renderer::State &state, const FeatureState &features, MemState &mem, Config &config, int64_t max_wait_ms);
void render_loop_metal3(renderer::State &state, DisplayState &display, GxmState &gxm, MemState &mem, Config &config);
void stop_render_thread_metal3(State &state);
bool get_shaders_cache_hashs_metal3(State &renderer);
void save_shaders_cache_hashs_metal3(State &renderer, std::vector<ShadersHash> &shaders_cache_hashs);
}
