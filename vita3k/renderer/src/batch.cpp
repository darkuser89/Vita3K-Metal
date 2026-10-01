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

#include <renderer/commands.h>
#include <renderer/driver_functions.h>
#include <renderer/functions.h>
#include <renderer/state.h>
#include <renderer/types.h>

#include <renderer/vulkan/types.h>
#ifdef __APPLE__
#include <renderer/metal/state.h>
#endif

#include <config/state.h>
#include <display/state.h>
#include <gxm/functions.h>
#include <gxm/state.h>
#include <functional>
#include <overlay/display_manager.h>
#include <overlay/shader_precompile_progress.h>
#include <util/log.h>

#include <memory>
#include <exception>
#include <thread>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

struct FeatureState;

namespace renderer {
void discard_metal_commands(Context *context, Command *first, Command *last) {
    for (Command *cmd = first; cmd;) {
        Command *next = cmd == last ? nullptr : cmd->next;
        release_metal_command_status(*cmd);
        destroy_metal_command_payload(*cmd);
        // A rejected producer can run ahead of an active consumer. Returning
        // its ring slot via free_func would advance the FIFO allocator past
        // still-live earlier slots. Ring storage stays owned by its context;
        // only independent host nodes can be deleted here.
        if (!(cmd->flags & Command::FLAG_NO_FREE)
            && (!context || (cmd->flags & Command::FLAG_FROM_HOST)))
            generic_command_free(cmd);
        cmd = next;
    }
}

Command *generic_command_allocate() {
    return new Command;
}

void generic_command_free(Command *cmd) {
    release_metal_command_status(*cmd);
    destroy_metal_command_payload(*cmd);
    delete cmd;
}

void complete_command(State &state, CommandHelper &helper, const int code) {
    auto lock = std::unique_lock(state.command_finish_one_mutex);
    helper.complete(code);
    state.command_finish_one.notify_all();
}

bool is_cmd_ready(MemState &mem, CommandList &command_list) {
    // we check if the cmd starts with a WaitSyncObject and if this is the case if it is ready
    if (!command_list.first || command_list.first->opcode != CommandOpcode::WaitSyncObject)
        return true;

    SceGxmSyncObject *sync = reinterpret_cast<Ptr<SceGxmSyncObject> *>(&command_list.first->data[0])->get(mem);
    const uint32_t timestamp = *reinterpret_cast<uint32_t *>(&command_list.first->data[sizeof(uint32_t)]);

    return sync->timestamp_current >= timestamp;
}

static renderer::SyncWaitResult wait_cmd(MemState &mem, CommandList &command_list) {
    // we assume here that the cmd starts with a WaitSyncObject

    SceGxmSyncObject *sync = reinterpret_cast<Ptr<SceGxmSyncObject> *>(&command_list.first->data[0])->get(mem);
    const uint32_t timestamp = *reinterpret_cast<uint32_t *>(&command_list.first->data[sizeof(uint32_t)]);

    // wait 500 micro seconds and then return in case should_display is set to true
    return renderer::wishlist(sync, timestamp, 500);
}

static void process_batch(renderer::State &state, const FeatureState &features, MemState &mem, Config &config, CommandList &command_list) {
    using CommandHandlerFunc = decltype(cmd_handle_set_context);

    Command *cmd = command_list.first;
    struct MetalBatchOwners {
        State &state;
        Command *&remaining;
        Command *last;
        Context *context;
        ~MetalBatchOwners() {
            if (state.current_backend == Backend::Metal)
                discard_metal_commands(context, remaining, last);
        }
    } owners{state, cmd, command_list.last, command_list.context};

    const static std::map<CommandOpcode, CommandHandlerFunc *> handlers = {
        { CommandOpcode::SetContext, cmd_handle_set_context },
        { CommandOpcode::SyncSurfaceData, cmd_handle_sync_surface_data },
        { CommandOpcode::MidSceneFlush, cmd_handle_mid_scene_flush },
        { CommandOpcode::CreateContext, cmd_handle_create_context },
        { CommandOpcode::CreateRenderTarget, cmd_handle_create_render_target },
        { CommandOpcode::MemoryMap, cmd_handle_memory_map },
        { CommandOpcode::MemoryUnmap, cmd_handle_memory_unmap },
        { CommandOpcode::Draw, cmd_handle_draw },
        { CommandOpcode::TransferCopy, cmd_handle_transfer_copy },
        { CommandOpcode::TransferDownscale, cmd_handle_transfer_downscale },
        { CommandOpcode::TransferFill, cmd_handle_transfer_fill },
        { CommandOpcode::SyncGuestRange, cmd_handle_sync_guest_range },
        { CommandOpcode::Nop, cmd_handle_nop },
        { CommandOpcode::SetState, cmd_handle_set_state },
        { CommandOpcode::SignalSyncObject, cmd_handle_signal_sync_object },
        { CommandOpcode::WaitSyncObject, cmd_handle_wait_sync_object },
        { CommandOpcode::SignalNotification, cmd_handle_notification },
        { CommandOpcode::SetScreenFilter, cmd_handle_set_screen_filter },
        { CommandOpcode::NewFrame, cmd_new_frame },
        { CommandOpcode::DestroyRenderTarget, cmd_handle_destroy_render_target },
        { CommandOpcode::DestroyContext, cmd_handle_destroy_context }
    };

    if (command_list.context)
        state.context = command_list.context;

    // Take a batch, and execute it. Hope it's not too large
    do {
        if (cmd == nullptr) {
            break;
        }
        if (state.current_backend == Backend::Metal && !begin_metal_command(state, *cmd))
            return;

        auto handler = handlers.find(cmd->opcode);
        if (handler == handlers.end()) {
            LOG_ERROR("Unimplemented command opcode {}", static_cast<int>(cmd->opcode));
        } else {
#ifdef __APPLE__
            // Metal may queue snapshot-backed read-only draws. Publish GPU work
            // before notifications, transfers, frees and other CPU-visible
            // commands. SyncSurfaceData finishes in its handler, where the
            // scene's depth store can join the pending draw command buffer
            // before notifications are signaled.
            if(state.current_backend==Backend::Metal && state.context
                && cmd->opcode!=CommandOpcode::Draw && cmd->opcode!=CommandOpcode::SetState
                && cmd->opcode!=CommandOpcode::SyncSurfaceData)
                static_cast<metal::MetalState &>(state).finish(*static_cast<metal::MetalContext *>(state.context));
#endif
            CommandHelper helper(cmd);
            handler->second(state, mem, config, helper, features, command_list.context);
        }

        Command *last_cmd = cmd;
        cmd = cmd->next;

        auto status_owner = detach_metal_command_status(*last_cmd);
        if (command_list.context) {
            command_list.context->free_func(last_cmd);
        } else {
            generic_command_free(last_cmd);
        }
    } while (true);
}

void process_batches(renderer::State &state, const FeatureState &features, MemState &mem, Config &config, int64_t max_wait_ms) {
    auto max_time = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() + max_wait_ms;

    while (!state.should_display) {
        if (state.render_abort.load(std::memory_order_relaxed))
            return;

        // overlay requested an async present
        if (state.async_flip_requested.load(std::memory_order_relaxed))
            return;

        // Try to wait for a batch (about 2 or 3ms, game should be fast for this)
        auto cmd_list = state.command_buffer_queue.top(state.current_backend == Backend::Metal ? 3000 : 3);

        if (!cmd_list || !is_cmd_ready(mem, *cmd_list)) {
            // beginning of the game or homebrew not using gxm
            if (state.context == nullptr)
                return;

            // keep the old behavior for opengl with vsync as it looks like the new one causes some issues
            if (state.current_backend == Backend::OpenGL && config.current_config.v_sync)
                return;

            renderer::SyncWaitResult wait_result = renderer::SyncWaitResult::TimedOut;
            if (cmd_list)
                wait_result = wait_cmd(mem, *cmd_list);
            if (!cmd_list || wait_result != renderer::SyncWaitResult::Ready) {
                if (wait_result == renderer::SyncWaitResult::Shutdown)
                    return;

                if (state.async_flip_requested.load(std::memory_order_relaxed))
                    return;

                auto curr_time = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                if (curr_time >= max_time)
                    // display a frame even though the game is not diplaying anything
                    return;

                // this mean the command is still not ready, check if we can display it again
                continue;
            }
        }

        if (state.current_backend == Backend::Metal) {
            // top() is only a peek. Abort can win before pop(), in which case
            // the queue still owns the nodes and shutdown must discard them.
            CommandList owned_list{};
            if (!state.command_buffer_queue.try_pop(owned_list))
                return;
            process_batch(state, features, mem, config, owned_list);
        } else {
            state.command_buffer_queue.pop();
            process_batch(state, features, mem, config, *cmd_list);
        }
    }
}

void reset_command_list(CommandList &command_list) {
    command_list.first = nullptr;
    command_list.last = nullptr;
}

static void render_loop_body(renderer::State &state, DisplayState &display, GxmState &gxm, MemState &mem, Config &config) {
    if (state.precompile_requested) {
        auto progress_overlay = state.overlay_manager
            ? state.overlay_manager->create<overlay::shader_precompile_progress>()
            : std::shared_ptr<overlay::shader_precompile_progress>();

        if (progress_overlay && !state.precompile_bg_path.empty()) {
            auto bg = std::make_unique<overlay::image_info>(state.precompile_bg_path);
            if (bg->get_data())
                progress_overlay->set_background_image(std::move(bg));
        }

        const int total = static_cast<int>(state.precompile_queue.size());
        state.precompile_total = total;

        for (int i = 0; i < total && !state.render_abort.load(std::memory_order_relaxed); ++i) {
            if (!state.set_current())
                break;

            state.precompile_shader(state.precompile_queue[i]);
            state.precompile_progress = i + 1;

            if (progress_overlay) {
                progress_overlay->set_progress(i + 1, total);
                state.render_frame(display, gxm, mem);
                state.swap_window();
            }
        }

        state.precompile_queue.clear();

        if (progress_overlay) {
            if (!state.set_current()) {
                state.done_current();
                return;
            }
            progress_overlay->set_background_only();
            state.render_frame(display, gxm, mem);
            state.swap_window();
        }

        state.precompile_complete.store(true, std::memory_order_release);
    }

    if (!state.precompile_requested
        && state.overlay_manager
        && !state.precompile_bg_path.empty()) {
        auto loading = state.overlay_manager->create<overlay::shader_precompile_progress>();
        if (loading) {
            if (!state.set_current()) {
                state.done_current();
                return;
            }
            auto bg = std::make_unique<overlay::image_info>(state.precompile_bg_path);
            if (bg->get_data())
                loading->set_background_image(std::move(bg));
            loading->set_background_only();
            state.render_frame(display, gxm, mem);
            state.swap_window();
        }
    }
    while (!state.render_abort.load(std::memory_order_relaxed)) {
#ifdef TRACY_ENABLE
        ZoneScopedN("Game rendering");
#endif
        if (!state.set_current())
            break;

        process_batches(state, state.features, mem, config, 500);

        if (state.render_abort.load(std::memory_order_relaxed))
            break;

        if (state.overlay_manager) {
            auto precompile = state.overlay_manager->get<overlay::shader_precompile_progress>();
            if (precompile) {
                DisplayFrameInfo peek;
                {
                    std::lock_guard<std::mutex> guard(display.display_info_mutex);
                    peek = display.next_rendered_frame;
                }
                if (peek.base) {
                    state.overlay_manager->remove<overlay::shader_precompile_progress>();
                }
            }
        }

        state.render_frame(display, gxm, mem);
        state.swap_window();
        state.async_flip_requested.store(false, std::memory_order_relaxed);

#ifdef TRACY_ENABLE
        FrameMark;
#endif
    }

    state.done_current();
}

static void render_loop(renderer::State &state, DisplayState &display, GxmState &gxm, MemState &mem, Config &config) {
    if (state.current_backend != Backend::Metal) {
        render_loop_body(state, display, gxm, mem, config);
        return;
    }
    try {
        // Include precompilation and loading overlays in the Metal boundary.
        render_loop_body(state, display, gxm, mem, config);
    } catch (const std::exception &error) {
        request_metal_render_abort(state);
        // Invalidating sync objects can stop the display consumer with queued
        // entries still present. Wake producers and GXM termination as well.
        gxm.display_queue.abort_synchronized();
        gxm::invalidate_sync_objects(gxm);
        auto pending = state.command_buffer_queue.take_pending();
        while (!pending.empty()) {
            const auto &list = pending.front();
            discard_metal_commands(list.context, list.first, list.last);
            pending.pop();
        }
        LOG_CRITICAL("Metal render thread stopped: {}", error.what());
        // Metal's current-context hooks are no-ops. Keep the normal teardown
        // contract; device resources are released by the app shutdown path.
        state.done_current();
    }
}

void start_render_thread(State &state, DisplayState &display, GxmState &gxm, MemState &mem, Config &config) {
    state.render_abort = false;
    state.render_thread = std::make_unique<std::thread>(render_loop, std::ref(state), std::ref(display), std::ref(gxm), std::ref(mem), std::ref(config));
}

void stop_render_thread(State &state) {
    if (state.current_backend == Backend::Metal)
        request_metal_render_abort(state);
    else {
        state.render_abort = true;
        state.command_buffer_queue.abort();
    }
    if (state.render_thread && state.render_thread->joinable())
        state.render_thread->join();
    state.render_thread.reset();
#ifdef __APPLE__
    if (state.current_backend == Backend::Metal) {
        auto pending = state.command_buffer_queue.take_pending();
        while (!pending.empty()) {
            const auto &list = pending.front();
            discard_metal_commands(list.context, list.first, list.last);
            pending.pop();
        }
    }
#endif
    state.command_buffer_queue.reset();
}

} // namespace renderer
