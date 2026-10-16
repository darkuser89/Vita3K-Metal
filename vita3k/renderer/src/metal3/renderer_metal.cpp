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

#include <renderer/functions.h>
#include <renderer/state.h>
#include <renderer/types.h>

#include <dialog/state.h>
#include <overlay/common_dialog.h>
#include <overlay/display_manager.h>
#include <overlay/font.h>
#include <overlay/pause_overlay.h>
#include <overlay/perf_overlay.h>
#include <overlay/shader_compile_notice.h>
#include <renderer/gl/state.h>
#include <renderer/gl/types.h>
#include <renderer/vulkan/functions.h>
#ifdef __APPLE__
#include <renderer/metal3/state.h>
#endif

#include <gxm/functions.h>
#include <util/log.h>

#include <cstring>

#include <renderer/metal3/integration.h>

namespace renderer {

bool append_deferred_command_list(Context &destination, const CommandList &source) {
    if (!source.first)
        return source.last == nullptr;
    if (!source.last)
        return false;

    CommandList copied{};
    const auto discard = [&]() {
        for (Command *cmd = copied.first; cmd;) {
            Command *next = cmd->next;
            destroy_command_payload(*cmd);
            destination.free_func(cmd);
            cmd = next;
        }
    };

    // A deferred list may be executed more than once before the immediate
    // batch is consumed. Linking its original nodes would overwrite `next`
    // pointers in earlier executions and can create a cycle.
    for (const Command *original = source.first; original; original = original->next) {
        // Single-use payloads and status waits cannot be shallow-copied into
        // a reusable deferred list. SetContext is immediate-scene-only.
        if (original->opcode == CommandOpcode::SetContext || original->status
            || (original->flags & Command::FLAG_METAL_RAW_PAYLOAD)) {
            discard();
            return false;
        }
        Command *copy = destination.alloc_func();
        if (!copy) {
            discard();
            return false;
        }
        copy->opcode = original->opcode;
        copy->status = nullptr;
        std::memcpy(copy->data, original->data, sizeof(copy->data));
        copy->next = nullptr;
#ifdef __APPLE__
        if (original->flags & Command::FLAG_METAL_PROGRAM_BINDING) {
            try {
                CommandHelper helper(copy);
                helper.pop<GXMState>();
                helper.pop<Ptr<const void>>();
                helper.pop<bool>();
                const auto position = helper.point;
                const auto *original_payload = helper.pop<std::shared_ptr<const metal::ProgramBinding> *>();
                auto payload = std::make_unique<std::shared_ptr<const metal::ProgramBinding>>(*original_payload);
                helper.point = position;
                auto *pointer = payload.get();
                helper.push(pointer);
                copy->flags |= Command::FLAG_METAL_PROGRAM_BINDING;
                payload.release();
            } catch (...) {
                destination.free_func(copy);
                discard();
                throw;
            }
        }
#endif
        if (copied.last)
            copied.last->next = copy;
        else
            copied.first = copy;
        copied.last = copy;
        if (original == source.last) {
            if (destination.command_list.last)
                destination.command_list.last->next = copied.first;
            else
                destination.command_list.first = copied.first;
            destination.command_list.last = copied.last;
            return true;
        }
    }

    discard();
    return false;
}



















void set_program_metal3(State &state, Context *ctx, Ptr<const void> program, const bool is_fragment,
    const std::shared_ptr<const metal::ProgramBinding> &metal_binding) {
#ifdef __APPLE__
    if (state.current_backend == Backend::Metal) {
        auto payload = std::make_unique<std::shared_ptr<const metal::ProgramBinding>>(metal_binding);
        if (renderer::add_state_set_command(ctx, renderer::GXMState::Program, program, is_fragment, payload.get())) {
            ctx->command_list.last->flags |= Command::FLAG_METAL_PROGRAM_BINDING;
            payload.release();
        }
        return;
    }
#endif
    renderer::add_state_set_command(ctx, renderer::GXMState::Program, program, is_fragment);
    }





void set_yuv_profile(State &state, Context *ctx, uint32_t index, SceGxmYuvProfile profile) {
    renderer::add_state_set_command(ctx, renderer::GXMState::YuvProfile, index, profile);
}











void set_context_metal3(State &state, Context *ctx, RenderTarget *target, SceGxmColorSurface *color_surface, SceGxmDepthStencilSurface *depth_stencil_surface) {
    if (state.current_backend == Backend::Metal) {
        std::unique_ptr<SceGxmColorSurface> color(color_surface);
        std::unique_ptr<SceGxmDepthStencilSurface> depth(depth_stencil_surface);
        if (renderer::add_command(ctx, renderer::CommandOpcode::SetContext, nullptr, target, color_surface, depth_stencil_surface)) {
            color.release();
            depth.release();
        }
        return;
    }
    renderer::add_command(ctx, renderer::CommandOpcode::SetContext, nullptr, target, color_surface, depth_stencil_surface);
    }

void set_vertex_stream_metal3(State &state, Context *ctx, const std::size_t index, const std::size_t data_len, const Ptr<const void> stream) {
#ifdef __APPLE__
    if (state.current_backend == Backend::Metal) {
        static_assert(sizeof(GXMState) + sizeof(stream) + sizeof(index) + sizeof(data_len) + sizeof(uint64_t) <= MAX_COMMAND_DATA_SIZE);
        renderer::add_state_set_command(ctx, renderer::GXMState::VertexStream, stream, index, data_len, uint64_t(~0ull));
        return;
    }
#endif
    renderer::add_state_set_command(ctx, renderer::GXMState::VertexStream, stream, index, data_len);
    }















void destroy_context_during_shutdown_metal3(State &state, std::unique_ptr<Context> &context) {
    assert(!state.render_thread);

#ifdef __APPLE__
    if (state.current_backend == Backend::Metal && context)
        static_cast<metal::MetalState &>(state).finish_for_shutdown(static_cast<metal::MetalContext &>(*context));
#endif

    if (state.current_backend == Backend::OpenGL) {
        state.set_current();
    }

    if (state.context == context.get()) {
        state.context = nullptr;
    }

    context.reset();
    }





void destroy_render_target_during_shutdown_metal3(State &state, std::unique_ptr<RenderTarget> &rt) {
    assert(!state.render_thread);
    if (!rt)
        return;

    switch (state.current_backend) {
    case Backend::OpenGL:
        state.set_current();
        break;

    case Backend::Vulkan:
        vulkan::destroy(dynamic_cast<vulkan::VKState &>(state), rt);
        break;
    case Backend::Metal:
        break;
    }

    rt.reset();
    }

void set_uniform_buffer_metal3(State &state, Context *ctx, const bool is_vertex_uniform, const int block_number, const std::uint32_t block_size, const Ptr<const void> buffer) {
    // Calculate the number of bytes
    // Preserve the legacy backends' upload behavior. Native Metal needs the
    // complete range for writable buffers, which can exceed 64 KiB.
    const uint32_t effective_size = state.current_backend == Backend::Metal ? block_size : static_cast<uint16_t>(block_size);
    std::uint32_t bytes_to_copy_and_pad = ((effective_size + 15) / 16) * 16;

    renderer::add_state_set_command(ctx, renderer::GXMState::UniformBuffer, buffer, is_vertex_uniform, block_number, bytes_to_copy_and_pad);
    }





void set_back_visibility_index(State &state, Context *ctx, bool enable, uint32_t index, bool is_increment) {
    renderer::add_state_set_command(ctx, renderer::GXMState::VisibilityBackIndex, index, enable, is_increment);
}

int sync_guest_range(State &state, Address address, uint32_t size) {
    if (state.current_backend != Backend::Metal || !size) return 0;
    return send_single_command(state, nullptr, CommandOpcode::SyncGuestRange, true, address, size);
}

} // namespace renderer
