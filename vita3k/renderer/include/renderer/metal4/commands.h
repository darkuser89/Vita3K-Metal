// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <gxm/metal4/types.h>
#include <gxm/metal4/formats.h>
#include <renderer/commands.h>
#include <renderer/types.h>
#include <renderer/state.h>
#include <memory>
namespace renderer {
void submit_command_list(State &, Context *, CommandList &);
Command *generic_command_allocate();
void generic_command_free(Command *);
int wait_for_status(State &, int *, int, bool);
// Metal waits may return on shutdown before the command is consumed. Keep
// their status alive independently of the waiter, without enlarging Command
// or adding a nontrivial destructor to guest command-ring storage.
struct MetalCommandStatusDeleter {
    // The detached command owner retires the operation after its allocator
    // callback returns. The waiting caller only releases its reference.
    bool retire = false;
    void operator()(int *status) const;
};
using MetalCommandStatusOwner = std::unique_ptr<int, MetalCommandStatusDeleter>;
MetalCommandStatusOwner make_metal_command_status(State &state, bool wait);
void retain_metal_command_status(Command &cmd);
MetalCommandStatusOwner detach_metal_command_status(Command &cmd);
void release_metal_command_status(Command &cmd);
bool begin_metal_command(State &state, Command &cmd);
int wait_for_metal_command(State &state, int *status);

bool append_deferred_command_list(Context &destination, const CommandList &source);
void request_metal_render_abort(State &state);
void set_yuv_profile(State &state, Context *ctx, uint32_t index, SceGxmYuvProfile profile);
void set_back_visibility_index(State &state, Context *ctx, bool enable, uint32_t index, bool is_increment);
int sync_guest_range(State &state, Address address, uint32_t size);
void mark_metal_command_payload(Command &cmd);
void mark_metal_command_payload(const State &state, Command &cmd);
void destroy_metal_command_payload(Command &cmd);
void discard_metal_commands(Context *context, Command *first, Command *last);
namespace metal {
bool is_native_context(Context *);
bool is_native_state(const State &);
template <typename... Args>
Command *make_command_metal3(CommandAllocFunc alloc_func, CommandFreeFunc free_func, const CommandOpcode opcode, int *status, Args... arguments) {
    Command *new_command = alloc_func();
    if (!new_command)
        return nullptr;

    new_command->opcode = opcode;
    new_command->status = status;
    new_command->next = nullptr;

    CommandHelper helper(new_command);

    if constexpr (sizeof...(arguments) > 0) {
        if (!do_command_push_data(helper, arguments...)) {
            free_func(new_command);
            return nullptr;
        }
    }

    return new_command;
}

template <typename... Args>
bool add_command_metal3(Context *ctx, const CommandOpcode opcode, int *status, Args... arguments) {
    if (!ctx) {
        return false;
    }
    auto cmd_maked = make_command_metal3(ctx->alloc_func, ctx->free_func, opcode, status, arguments...);

    if (!cmd_maked) {
        return false;
    }
    if (is_native_context(ctx))
        mark_metal_command_payload(*cmd_maked);

    if (!ctx->command_list.first) {
        ctx->command_list.first = cmd_maked;
        ctx->command_list.last = cmd_maked;
    } else {
        ctx->command_list.last->next = cmd_maked;
        ctx->command_list.last = cmd_maked;
    }

    return true;
}

template <typename... Args>
int send_single_command_metal3(State &state, Context *ctx, const CommandOpcode opcode, bool wait, Args... arguments) {
    // Make a temporary command list
    int status = CommandErrorCodePending; // Pending.
    auto metal_status = make_metal_command_status(state, wait);
    int *status_ptr = wait ? (metal_status ? metal_status.get() : &status) : nullptr;
    auto cmd = ctx && is_native_context(ctx)
        ? ::renderer::metal::make_command_metal3(ctx->alloc_func, ctx->free_func, opcode, status_ptr, arguments...)
        : ::renderer::metal::make_command_metal3(ctx ? ctx->alloc_func : generic_command_allocate, ctx ? ctx->free_func : generic_command_free,
              opcode, status_ptr, arguments...);

    if (!cmd) {
        return CommandErrorArgumentsTooLarge;
    }
    mark_metal_command_payload(state, *cmd);
    if (metal_status)
        retain_metal_command_status(*cmd);

    CommandList list;
    list.first = cmd;
    list.last = cmd;

    // Submit it
    submit_command_list(state, ctx, list);
    if (metal_status)
        return wait_for_metal_command(state, status_ptr);
    if (wait)
        return wait_for_status(state, status_ptr, CommandErrorCodePending, false);
    else
        return 0;
}
}
}
