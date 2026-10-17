// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <mem/functions.h>
bool is_valid_addr_range_metal3(const MemState &state, Address start, Address end);
Address alloc_aligned_metal3(MemState &state, uint32_t size, const char *name, unsigned int alignment, Address start_addr);
bool handle_access_violation_metal3(MemState &state, uint8_t *addr, bool write) noexcept;
bool add_protect_metal3(MemState &state, Address addr, const uint32_t size, const MemPerm perm, const ProtectCallback &callback);
bool is_protecting_metal3(MemState &state, Address addr, MemPerm *perm);
void free_metal3(MemState &state, Address address);
