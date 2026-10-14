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

#include <atomic>
#include <chrono>
#include <future>
#include <type_traits>
#include <renderer/commands.h>
#include <renderer/driver_functions.h>
#include <renderer/functions.h>
#include <renderer/state.h>
#include <renderer/types.h>

#include <display/state.h>
#include <gxm/functions.h>
#include <gxm/state.h>
#include <renderer/gl/functions.h>
#include <renderer/gl/state.h>
#include <renderer/vulkan/functions.h>
#include <renderer/vulkan/state.h>
#include <renderer/vulkan/types.h>
#ifdef __APPLE__
#include <renderer/metal3/state.h>
#endif

#include <renderer/functions.h>
#include <util/tracy.h>



#include <renderer/metal3/integration.h>

namespace renderer {
namespace {
struct MetalCommandStatus {
    enum class Phase { Queued, Running, Cancelled, Retired };
    int value = CommandErrorCodePending;
    std::atomic<unsigned> owners{1};
    State *renderer = nullptr;
    // Protected by renderer->command_finish_one_mutex, like value.
    Phase phase = Phase::Queued;
};
// Command::status points to the first member. The standard-layout guarantee
// makes conversion back to its owning allocation valid.
static_assert(std::is_standard_layout_v<MetalCommandStatus>);
}

void MetalCommandStatusDeleter::operator()(int *status) const {
    auto *owned = reinterpret_cast<MetalCommandStatus *>(status);
    if (retire) {
        {
            const std::lock_guard<std::mutex> lock(owned->renderer->command_finish_one_mutex);
            owned->phase = MetalCommandStatus::Phase::Retired;
        }
        owned->renderer->command_finish_one.notify_all();
    }
    if (owned->owners.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete owned;
}

MetalCommandStatusOwner make_metal_command_status(State &state, bool wait) {
    if (!wait || state.current_backend != Backend::Metal)
        return {};
    auto *owned = new MetalCommandStatus;
    owned->renderer = &state;
    return MetalCommandStatusOwner(&owned->value);
}

void retain_metal_command_status(Command &cmd) {
    assert(cmd.status && !(cmd.flags & Command::FLAG_METAL_STATUS_OWNER));
    auto *owned = reinterpret_cast<MetalCommandStatus *>(cmd.status);
    owned->owners.fetch_add(1, std::memory_order_relaxed);
    cmd.flags |= Command::FLAG_METAL_STATUS_OWNER;
}

MetalCommandStatusOwner detach_metal_command_status(Command &cmd) {
    if (!(cmd.flags & Command::FLAG_METAL_STATUS_OWNER))
        return {};
    int *status = cmd.status;
    cmd.flags &= ~Command::FLAG_METAL_STATUS_OWNER;
    cmd.status = nullptr;
    return MetalCommandStatusOwner(status, MetalCommandStatusDeleter{true});
}

void release_metal_command_status(Command &cmd) {
    auto owner = detach_metal_command_status(cmd);
}

bool begin_metal_command(State &state, Command &cmd) {
    if (!(cmd.flags & Command::FLAG_METAL_STATUS_OWNER))
        return !state.render_abort.load(std::memory_order_relaxed);
    const std::lock_guard<std::mutex> lock(state.command_finish_one_mutex);
    auto *owned = reinterpret_cast<MetalCommandStatus *>(cmd.status);
    if (state.render_abort.load(std::memory_order_relaxed)
        || owned->phase != MetalCommandStatus::Phase::Queued)
        return false;
    owned->phase = MetalCommandStatus::Phase::Running;
    return true;
}

int wait_for_metal_command(State &state, int *status) {
    auto *owned = reinterpret_cast<MetalCommandStatus *>(status);
    std::unique_lock<std::mutex> lock(state.command_finish_one_mutex);
    state.command_finish_one.wait(lock, [&]() {
        if (owned->phase == MetalCommandStatus::Phase::Retired)
            return true;
        if (state.render_abort.load(std::memory_order_relaxed)
            && owned->phase == MetalCommandStatus::Phase::Queued) {
            // The caller may now drop borrowed arguments. The consumer uses
            // this same mutex before starting, and will refuse this command.
            owned->phase = MetalCommandStatus::Phase::Cancelled;
            return true;
        }
        // An already-running handler may still access caller-owned arguments,
        // even after complete_command wrote its result. Wait for retirement.
        return false;
    });
    return *status;
}

void request_metal_render_abort(State &state) {
    assert(state.current_backend == Backend::Metal);
    // Change the predicate under its wait mutex so a waiter cannot miss the
    // transition between checking it and entering condition_variable::wait.
    {
        const std::lock_guard<std::mutex> lock(state.command_finish_one_mutex);
        state.render_abort = true;
    }
    state.command_finish_one.notify_all();
    {
        // NotificationWait uses the same abort predicate under this mutex.
        const std::lock_guard<std::mutex> lock(state.notification_mutex);
    }
    state.notification_ready.notify_all();
    state.command_buffer_queue.abort_synchronized();
}









void cmd_handle_set_screen_filter_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context) {
    TRACY_FUNC_COMMANDS(handle_set_screen_filter);
    std::unique_ptr<std::string> filter(helper.pop<std::string *>());
    if (renderer.current_backend == Backend::Metal)
        helper.cmd->flags &= ~Command::FLAG_METAL_RAW_PAYLOAD;

    switch (renderer.current_backend) {
    case Backend::Metal:
        renderer.set_screen_filter(*filter);
        break;
    case Backend::OpenGL:
        dynamic_cast<gl::GLState &>(renderer).set_screen_filter(*filter);
        break;

    case Backend::Vulkan:
        dynamic_cast<vulkan::VKState &>(renderer).screen_renderer.set_filter(*filter);
        break;
    }
    }

void cmd_new_frame_metal3(renderer::State &renderer, MemState &mem, Config &config, CommandHelper &helper, const FeatureState &features, Context *render_context) {
    TRACY_FUNC_COMMANDS(new_frame);
    DisplayFrameInfo *next_frame = helper.pop<DisplayFrameInfo *>();
    DisplayState *display = helper.pop<DisplayState *>();

    if (next_frame) {
        // set the predicted frame as the next one to render
        std::lock_guard<std::mutex> guard(display->display_info_mutex);
        display->next_rendered_frame = *next_frame;
        delete next_frame;
        if (renderer.current_backend == Backend::Metal)
            helper.cmd->flags &= ~Command::FLAG_METAL_RAW_PAYLOAD;

        renderer.should_display = true;
    }

    if (renderer.current_backend == Backend::Vulkan) {
        renderer::Context *active_context = helper.pop<renderer::Context *>();
        if (active_context) {
            vulkan::new_frame(*reinterpret_cast<vulkan::VKContext *>(active_context));
        }
    }
#ifdef __APPLE__
    if (renderer.current_backend == Backend::Metal)
        static_cast<metal::MetalState &>(renderer).new_frame();
#endif
    }

// Client side function








void submit_command_list_metal3(State &state, renderer::Context *context, CommandList &command_list) {
    command_list.context = context;
    [[maybe_unused]] const bool submitted = state.command_buffer_queue.push(std::move(command_list));
#ifdef __APPLE__
    if (!submitted && state.current_backend == Backend::Metal) {
        discard_metal_commands(context, command_list.first, command_list.last);
        reset_command_list(command_list);
    }
#endif
    }
} // namespace renderer
