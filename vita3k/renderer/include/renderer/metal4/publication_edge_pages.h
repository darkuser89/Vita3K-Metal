// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <renderer/metal4/writable_guest_alias.h>
#include <array>
#include <exception>
#include <optional>
#include <stdexcept>

namespace renderer::metal {
// Used only under the guest generation AND protection mutexes. Edge pages
// remain writable during GPU waiting, including unrelated neighboring data.
class PublicationEdgePages {
    class Page {
        mach_vm_address_t address;
        vm_prot_t restore;
    public:
        Page(const Page &) = delete;
        Page &operator=(const Page &) = delete;
        explicit Page(uintptr_t value) : address(value) {
            const auto original = WritableGuestAlias::protection(reinterpret_cast<void *>(value));
            if (!original) throw std::runtime_error("Metal: cannot inspect publication edge protection");
            restore = *original;
            if (mach_vm_protect(mach_task_self(), address, vm_page_size, FALSE, VM_PROT_READ) != KERN_SUCCESS)
                throw std::runtime_error("Metal: cannot protect publication edge");
        }
        void published() { restore = VM_PROT_READ | VM_PROT_WRITE; }
        ~Page() {
            // The generation lock still prevents unmapping. Failing to restore
            // a successfully protected page cannot leave the guest running.
            if (mach_vm_protect(mach_task_self(), address, vm_page_size, FALSE, restore) != KERN_SUCCESS)
                std::terminate();
        }
    };
    std::array<std::optional<Page>,2> pages;
public:
    PublicationEdgePages(void *source, size_t size) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(source), page = vm_page_size;
        const uintptr_t end = address + size;
        const uintptr_t first = address / page * page, last = (end - 1) / page * page;
        if (address % page) pages[0].emplace(first);
        if (end % page && (!pages[0] || first != last)) pages[1].emplace(last);
    }
    void published() { for (auto &page : pages) if (page) page->published(); }
};
}
