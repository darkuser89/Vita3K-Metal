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
#include <renderer/metal/state.h>
#endif

#include <renderer/functions.h>
#include <util/tracy.h>

namespace gxm {
void invalidate_sync_objects(GxmState &gxm) {
    std::lock_guard<std::mutex> lock(gxm.sync_objects_mutex);
    for (SceGxmSyncObject *sync_object : gxm.sync_objects) {
        {
            std::lock_guard<std::mutex> sync_lock(sync_object->lock);
            sync_object->being_deleted = true;
        }
        sync_object->cond.notify_all();
    }
}
} // namespace gxm

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

COMMAND(handle_nop) {
    TRACY_FUNC_COMMANDS(handle_nop);
    // Signal back to client
    int code_to_finish = helper.pop<int>();
    complete_command(renderer, helper, code_to_finish);
}

COMMAND(handle_signal_sync_object) {
    TRACY_FUNC_COMMANDS(handle_signal_sync_object);
    SceGxmSyncObject *sync = helper.pop<Ptr<SceGxmSyncObject>>().get(mem);
    const uint32_t timestamp = helper.pop<uint32_t>();

    if (features.enable_memory_mapping && config.current_config.high_accuracy) {
        assert(renderer.current_backend == renderer::Backend::Vulkan);
        vulkan::signal_sync_object(dynamic_cast<vulkan::VKState &>(renderer), sync, timestamp);
    } else {
        renderer::subject_done(sync, timestamp);
    }
}

COMMAND(handle_wait_sync_object) {
    TRACY_FUNC_COMMANDS(handle_wait_sync_object);
    SceGxmSyncObject *sync = helper.pop<Ptr<SceGxmSyncObject>>().get(mem);
    const uint32_t timestamp = helper.pop<uint32_t>();

    renderer::wishlist(sync, timestamp);
}

COMMAND(handle_notification) {
    TRACY_FUNC_COMMANDS(handle_notification);
    SceGxmNotification notif = helper.pop<SceGxmNotification>();

    {
        std::unique_lock<std::mutex> lock(renderer.notification_mutex);
        uint32_t *val = notif.address.get(mem);
        if (val) // Ratchet and clank Trilogy request this
            *val = notif.value;
    }
    renderer.notification_ready.notify_all();
}

COMMAND(handle_set_screen_filter) {
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

COMMAND(new_frame) {
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
void finish(State &state, Context *context) {
    // Add NOP then wait for it
    renderer::send_single_command(state, context, renderer::CommandOpcode::Nop, true, 1);

    // unblock game threads if shutting down
    if (state.render_abort.load(std::memory_order_relaxed))
        return;

    // Wait for the VK wait thread to finish processing all pending requests.
    // Push a callback request on the queue and wait for it to be treated
    if (state.current_backend == Backend::Vulkan && state.features.enable_memory_mapping) {
        auto &vk_state = static_cast<vulkan::VKState &>(state);
        std::promise<void> promise;
        auto callback = [&]() {
            promise.set_value();
        };
        vk_state.request_queue.push(vulkan::CallbackRequest{ new vulkan::CallbackRequestFunction(callback) });
        promise.get_future().wait();
    }
}

int wait_for_status(State &state, int *status, int signal, bool wake_on_equal) {
    std::unique_lock<std::mutex> lock(state.command_finish_one_mutex);
    const bool wake_on_unequal = !wake_on_equal;
    if ((*status == signal) ^ wake_on_unequal) {
        // Signaled, return
        return *status;
    }

    // unblock threads if shutting down
    state.command_finish_one.wait(lock, [&]() {
        return state.render_abort.load(std::memory_order_relaxed)
            || ((*status == signal) ^ wake_on_unequal);
    });
    return *status;
}

SyncWaitResult wishlist(SceGxmSyncObject *sync_object, const uint32_t timestamp, const int32_t timeout_micros) {
    std::unique_lock<std::mutex> lock(sync_object->lock);
    if (sync_object->timestamp_current < timestamp) {
        const auto &pred = [&]() {
            return sync_object->being_deleted || sync_object->timestamp_current >= timestamp;
        };

        if (timeout_micros == -1) {
            sync_object->cond.wait(lock, pred);
        } else if (!sync_object->cond.wait_for(lock, std::chrono::microseconds(timeout_micros), pred)) {
            return SyncWaitResult::TimedOut;
        }
    }
    if (sync_object->being_deleted)
        return SyncWaitResult::Shutdown;
    return SyncWaitResult::Ready;
}

void subject_done(SceGxmSyncObject *sync_object, const uint32_t timestamp) {
    assert(sync_object->timestamp_ahead >= timestamp);
    {
        std::unique_lock<std::mutex> lock(sync_object->lock);
        sync_object->timestamp_current = std::max(sync_object->timestamp_current.load(), timestamp);
    }
    // maybe notify_one is enough
    sync_object->cond.notify_all();
}

void submit_command_list(State &state, renderer::Context *context, CommandList &command_list) {
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
