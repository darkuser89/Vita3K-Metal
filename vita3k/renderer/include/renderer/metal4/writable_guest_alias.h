// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <cstdint>
#include <limits>
#include <optional>

namespace renderer::metal {
// A shared, non-copying mapping. Changing this alias's protection does not
// unprotect the original guest address while a publication owns its lock.
class WritableGuestAlias {
    mach_vm_address_t mapping = 0;
    mach_vm_size_t mapping_size = 0;
    uint8_t *data = nullptr;
public:
    static std::optional<vm_prot_t> protection(const void *source) {
        const auto address = reinterpret_cast<uintptr_t>(source);
        if (!address) return {};
        mach_vm_address_t region = address;
        mach_vm_size_t extent = 0;
        vm_region_basic_info_data_64_t info{};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        const auto result = mach_vm_region(mach_task_self(), &region, &extent, VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info), &count, &object);
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        if (result != KERN_SUCCESS || region > address || !extent || region + extent <= address) return {};
        return info.protection;
    }
    static bool original_is_writable(const void *source) {
        const auto permission = protection(source);
        return permission && (*permission & VM_PROT_WRITE);
    }
    static bool original_is_read_only(const void *source, size_t size) {
        mach_vm_address_t cursor = reinterpret_cast<uintptr_t>(source);
        if (!source || !size || size > std::numeric_limits<mach_vm_address_t>::max() - cursor) return false;
        const auto end = cursor + size;
        while (cursor < end) {
            auto region = cursor;
            mach_vm_size_t extent = 0;
            vm_region_basic_info_data_64_t info{};
            mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object = MACH_PORT_NULL;
            const auto result = mach_vm_region(mach_task_self(), &region, &extent, VM_REGION_BASIC_INFO_64,
                reinterpret_cast<vm_region_info_t>(&info), &count, &object);
            if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
            if (result != KERN_SUCCESS || region > cursor || !extent || region + extent <= cursor
                || (info.protection & VM_PROT_WRITE)) return false;
            cursor = region + extent;
        }
        return true;
    }
    WritableGuestAlias(const WritableGuestAlias &) = delete;
    WritableGuestAlias &operator=(const WritableGuestAlias &) = delete;
    WritableGuestAlias(void *source, size_t size) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(source);
        const uintptr_t page = vm_page_size;
        if (!source || !size || !page || size > std::numeric_limits<uintptr_t>::max() - address
            || address + size > std::numeric_limits<uintptr_t>::max() - (page - 1)) return;
        const auto begin = address / page * page;
        mapping_size = (address + size + page - 1) / page * page - begin;
        vm_prot_t current = 0, maximum = 0;
        if (mach_vm_remap(mach_task_self(), &mapping, mapping_size, 0, VM_FLAGS_ANYWHERE,
                mach_task_self(), begin, FALSE, &current, &maximum, VM_INHERIT_NONE) != KERN_SUCCESS) {
            mapping = 0; return;
        }
        if (!(maximum & VM_PROT_WRITE) || mach_vm_protect(mach_task_self(), mapping,
                mapping_size, FALSE, VM_PROT_READ | VM_PROT_WRITE) != KERN_SUCCESS) {
            mach_vm_deallocate(mach_task_self(), mapping, mapping_size);
            mapping = 0; return;
        }
        data = reinterpret_cast<uint8_t *>(mapping) + (address - begin);
    }
    ~WritableGuestAlias() { if (mapping) mach_vm_deallocate(mach_task_self(), mapping, mapping_size); }
    uint8_t *contents() const { return data; }
};
}
