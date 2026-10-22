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

#include <mem/functions.h>
#include <mem/state.h>
#include <mem/metal4.h>
#include "../../include/renderer/metal4/guest_publication_queue.h"
#ifdef __APPLE__
#include "../../include/renderer/metal4/writable_guest_alias.h"
#endif

#include <util/align.h>
#include <util/log.h>

#include <algorithm>
#include <cassert>
#include <bit>
#include <cstring>
#include <mutex>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <csignal>
#include <sys/mman.h>
#include <unistd.h>
#endif

constexpr uint32_t STANDARD_PAGE_SIZE = KiB(4);
constexpr size_t TOTAL_MEM_SIZE = GiB(4);
constexpr bool LOG_PROTECT = false;
#ifdef NDEBUG
constexpr bool PAGE_NAME_TRACKING = false;
#else
constexpr bool PAGE_NAME_TRACKING = true;
#endif

// TODO: support multiple handlers
static AccessViolationHandler access_violation_handler;
static void register_access_violation_handler(const AccessViolationHandler &handler);

static Address alloc_inner(MemState &state, uint32_t start_page, uint32_t page_count, const char *name, const bool force);
static void delete_memory(uint8_t *memory);

#ifdef _WIN32
static std::string get_error_msg() {
    return std::system_category().message(GetLastError());
}
#else
static std::string get_error_msg() {
    return strerror(errno);
}
#endif



static void delete_memory(uint8_t *memory) {
    if (memory != nullptr) {
#ifdef _WIN32
        const BOOL ret = VirtualFree(memory, 0, MEM_RELEASE);
        assert(ret);
#else
        munmap(memory, TOTAL_MEM_SIZE);
#endif
    }
}



bool is_valid_addr_range_metal3(const MemState &state, Address start, Address end) {
    if (end < start)
        return false;
    const uint32_t start_page = start / STANDARD_PAGE_SIZE;
    const uint32_t end_page = uint32_t((uint64_t(end) + STANDARD_PAGE_SIZE - 1) / STANDARD_PAGE_SIZE);
    return state.allocator.free_slot_count(start_page, end_page) == 0;
}

static Address alloc_inner(MemState &state, uint32_t start_page, uint32_t page_count, const char *name, const bool force) {
    int page_num;
    if (force) {
        if (state.allocator.allocate_at(start_page, page_count) < 0) {
            return 0;
        }
        page_num = start_page;
    } else {
        page_num = state.allocator.allocate_from(start_page, page_count, false);
        if (page_num < 0)
            return 0;
    }

    const uint32_t size = page_count * STANDARD_PAGE_SIZE;
    const Address addr = page_num * STANDARD_PAGE_SIZE;

    const Address commit_start = align_down(addr, state.host_page_size);
    const Address commit_end = align(addr + size, state.host_page_size);
    const uint32_t commit_size = commit_end - commit_start;
    uint8_t *const commit_ptr = &state.memory[commit_start];

    // Make memory chunk available to access
#ifdef _WIN32
    const void *const ret = VirtualAlloc(commit_ptr, commit_size, MEM_COMMIT, PAGE_READWRITE);
    LOG_CRITICAL_IF(!ret, "VirtualAlloc failed: {}", get_error_msg());
#else
    const int ret = mprotect(commit_ptr, commit_size, PROT_READ | PROT_WRITE);
    LOG_CRITICAL_IF(ret == -1, "mprotect failed: {}", get_error_msg());
#endif
    std::memset(&state.memory[addr], 0, size);

    AllocMemPage &page = state.alloc_table[page_num];
    assert(!page.allocated);
    page.allocated = 1;
    page.size = page_count;

    if (PAGE_NAME_TRACKING) {
        state.page_name_map.emplace(page_num, name);
    }

    return addr;
}

static void decommit_free_host_pages(MemState &state, uint64_t region_start, uint64_t region_end) {
    if (region_start >= region_end)
        return;

    uint64_t host_page = align_down(region_start, uint64_t(state.host_page_size));
    uint64_t batch_start = 0;
    size_t batch_size = 0;
    const auto flush_batch = [&]() {
        if (!batch_size)
            return;
        uint8_t *memory = &state.memory[batch_start];
#ifdef _WIN32
        const BOOL ret = VirtualFree(memory, batch_size, MEM_DECOMMIT);
        LOG_CRITICAL_IF(!ret, "VirtualFree failed: {}", get_error_msg());
#else
        int ret = mprotect(memory, batch_size, PROT_NONE);
        LOG_CRITICAL_IF(ret == -1, "mprotect failed: {}", get_error_msg());
        ret = madvise(memory, batch_size, MADV_DONTNEED);
        LOG_CRITICAL_IF(ret == -1, "madvise failed: {}", get_error_msg());
#endif
        batch_size = 0;
    };

    while (host_page < region_end) {
        const uint64_t host_page_end = host_page + state.host_page_size;
        const uint32_t first_guest = uint32_t(host_page / STANDARD_PAGE_SIZE);
        const uint32_t last_guest = uint32_t(host_page_end / STANDARD_PAGE_SIZE);
        if (state.allocator.free_slot_count(first_guest, last_guest) == last_guest - first_guest) {
            if (!batch_size)
                batch_start = host_page;
            batch_size += state.host_page_size;
        } else {
            flush_batch();
        }
        host_page = host_page_end;
    }
    flush_batch();
}

Address alloc_aligned_metal3(MemState &state, uint32_t size, const char *name, unsigned int alignment, Address start_addr) {
    if (alignment == 0)
        return alloc(state, size, name, start_addr);
    if (alignment > (1u << 31))
        return 0;
    alignment = std::bit_ceil(alignment);

    const uint64_t requested_pages = (uint64_t(size) + STANDARD_PAGE_SIZE - 1) / STANDARD_PAGE_SIZE;
    const uint64_t alignment_pages = std::max<uint64_t>(1, (uint64_t(alignment) + STANDARD_PAGE_SIZE - 1) / STANDARD_PAGE_SIZE);
    if (!requested_pages || requested_pages + alignment_pages - 1 > UINT32_MAX)
        return 0;

    const std::lock_guard<std::mutex> lock(state.generation_mutex);
    const uint32_t page_count = uint32_t(requested_pages + alignment_pages - 1);
    const Address addr = alloc_inner(state, start_addr / STANDARD_PAGE_SIZE, page_count, name, false);
    if (!addr)
        return 0;

    const Address aligned = align(addr, alignment);
    const uint32_t page_num = addr / STANDARD_PAGE_SIZE;
    const uint32_t aligned_page = aligned / STANDARD_PAGE_SIZE;
    const uint32_t front_pages = aligned_page - page_num;
    const uint32_t back_pages = page_count - front_pages - uint32_t(requested_pages);
    if (front_pages) {
        state.allocator.free(page_num, front_pages);
        state.alloc_table[page_num].allocated = 0;
        if (PAGE_NAME_TRACKING) {
            state.page_name_map.erase(page_num);
            state.page_name_map.emplace(aligned_page, name);
        }
    }
    if (back_pages)
        state.allocator.free(aligned_page + uint32_t(requested_pages), back_pages);

    state.alloc_table[aligned_page].allocated = 1;
    state.alloc_table[aligned_page].size = uint32_t(requested_pages);
    decommit_free_host_pages(state, addr, aligned);
    decommit_free_host_pages(state, uint64_t(aligned) + requested_pages * STANDARD_PAGE_SIZE,
        uint64_t(addr) + uint64_t(page_count) * STANDARD_PAGE_SIZE);
    return aligned;
}

static void align_to_page(MemState &state, Address &addr, Address &size) {
    const Address end = align(addr + size, STANDARD_PAGE_SIZE);
    addr = align_down(addr, STANDARD_PAGE_SIZE);
    size = end - addr;
}





static void consume_protection(MemState &state, ProtectSegmentTrees::iterator it, Address address, bool write) {
    for (auto &[block_addr, block] : it->second.blocks)
        block.callback(address, write);
    unprotect_inner(state, it->first, it->second.size);
    state.protect_tree.erase(it);
}

bool prepare_host_write(MemState &state, Address address, size_t size) {
    if (state.use_page_table) return false;
    const uint64_t end = uint64_t(address) + size;
    if (!size || address < state.host_page_size || size > UINT32_MAX
        || end > UINT32_MAX || !is_valid_addr_range_metal3(state, address, Address(end))) return false;
    const std::lock_guard lock(state.protect_mutex);
    // Select and consume under one lock: another thread cannot retire/rearm
    // the selected segment between lookup and callback delivery.
    for (auto it = state.protect_tree.begin(); it != state.protect_tree.end();) {
        const auto &segment = it->second;
        if (uint64_t(it->first) >= end || uint64_t(it->first) + segment.size <= address
            || (segment.perm != MemPerm::None && segment.perm != MemPerm::ReadOnly)
            || std::any_of(segment.blocks.begin(), segment.blocks.end(), [](const auto &entry) {
                   return !entry.second.retain_on_read;
               })) {
            // Guest fault handlers can observe the precise first store address
            // and execution context. Only write watches opt into eager delivery.
            ++it;
            continue;
        }
        Address touched = std::max(address, it->first);
        if (segment.write_watch_ranges_only) {
            // A consumed read trap can leave writable holes between retained
            // watches. Writing a hole must not consume its neighbours.
            touched = 0;
            for (const auto &[base, block] : segment.blocks) {
                if (uint64_t(base) < end && uint64_t(base) + block.size > address) {
                    touched = std::max(address, base);
                    break;
                }
            }
            if (!touched) { ++it; continue; }
        }
        const auto current = it++;
        consume_protection(state, current, touched, true);
    }
    return true;
}

bool handle_access_violation_metal3(MemState &state, uint8_t *addr, bool write) noexcept {
    const uintptr_t memory_addr = reinterpret_cast<uintptr_t>(state.memory.get());
    const uintptr_t fault_addr = reinterpret_cast<uintptr_t>(addr);

    Address vaddr = 0;
    const std::unique_lock<std::mutex> lock(state.protect_mutex);
    if (fault_addr < memory_addr || fault_addr >= memory_addr + TOTAL_MEM_SIZE) {
        if (state.use_page_table) {
            // this may come from an external mapping
            uint64_t addr_val = std::bit_cast<uint64_t>(addr);
            auto it = state.external_mapping.lower_bound(addr_val);
            if (it != state.external_mapping.end() && addr_val < it->first + it->second.size) {
                vaddr = static_cast<Address>(addr_val - it->first + it->second.address);
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        vaddr = static_cast<Address>(fault_addr - memory_addr);
    }

    if (!is_valid_addr(state, vaddr)) {
        return false;
    }
    // init() reserves the entire first host page as the null guard. On macOS
    // this is often larger than one guest page, so an untracked fault in its
    // later 4 KiB slots must never make the host page writable.
    if (vaddr < state.host_page_size)
        return false;
    if (LOG_PROTECT) {
        fmt::print("Access: {}\n", log_hex(vaddr));
    }

    auto it = state.protect_tree.lower_bound(vaddr);
    if (it == state.protect_tree.end()) {
#ifdef __APPLE__
        // A publication edge (or another completed watch) may have been
        // restored while this write fault was waiting for protect_mutex.
        if (write && renderer::metal::WritableGuestAlias::original_is_writable(addr)) return true;
#endif
        // HACK: keep going
        unprotect_inner(state, align_down(vaddr, state.host_page_size), state.host_page_size);
        LOG_CRITICAL("Unhandled write protected region was valid. Address=0x{:X}", vaddr);
        return true;
    }

    ProtectSegmentInfo &info = it->second;
    if (vaddr < it->first || vaddr >= it->first + info.size) {
#ifdef __APPLE__
        // A publication edge (or another completed watch) may have been
        // restored while this write fault was waiting for protect_mutex.
        if (write && renderer::metal::WritableGuestAlias::original_is_writable(addr)) return true;
#endif
        // HACK: keep going
        unprotect_inner(state, align_down(vaddr, state.host_page_size), state.host_page_size);
        LOG_CRITICAL("Unhandled write protected region was valid. Address=0x{:X}", vaddr);
        return true;
    }

    if (!write && std::any_of(info.blocks.begin(), info.blocks.end(), [](const auto &entry) {
            return entry.second.retain_on_read;
        })) {
        // A legacy read trap may overlap a Metal write watch. Consume the
        // read callbacks while retaining the write registrations and their
        // tokens. Do not invoke them for a read or allocate replacement nodes
        // from the signal handler.
        for (auto block = info.blocks.begin(); block != info.blocks.end();) {
            if (block->second.retain_on_read) {
                ++block;
            } else {
                block->second.callback(vaddr, false);
                block = info.blocks.erase(block);
            }
        }
        info.perm = MemPerm::ReadOnly;
        info.write_watch_ranges_only = true;
        // Downgrade watched pages directly from None to ReadOnly. Temporarily
        // making the whole merged segment writable would lose a concurrent
        // CPU store between the two mprotect calls. Only gaps lose protection.
        uint64_t cursor = it->first;
        for (const auto &[address, block] : info.blocks) {
            if (address > cursor)
                unprotect_inner(state, Address(cursor), uint32_t(address - cursor));
            protect_inner(state, address, block.size, MemPerm::ReadOnly);
            cursor = std::max(cursor, uint64_t(address) + block.size);
        }
        const uint64_t end = uint64_t(it->first) + info.size;
        if (cursor < end)
            unprotect_inner(state, Address(cursor), uint32_t(end - cursor));
        return true;
    }

    consume_protection(state, it, vaddr, write);

    return true;
}

static bool add_protect_impl(MemState &state, Address addr, const uint32_t size, const MemPerm perm,
    const ProtectCallback &callback, bool retain_on_read) {
    const std::lock_guard<std::mutex> lock(state.protect_mutex);
    ProtectSegmentInfo protect(size, perm);
    align_to_page(state, addr, protect.size);

    ProtectBlockInfo block;
    block.size = size;
    block.callback = callback;
    block.retain_on_read = retain_on_read;

    protect.blocks.emplace(addr, std::move(block));

    auto it = state.protect_tree.lower_bound(addr);
    if (it == state.protect_tree.end() || it->first + it->second.size <= addr) {
        if (it == state.protect_tree.begin())
            it = state.protect_tree.end();
        else
            --it;
    }

    while (it != state.protect_tree.end() && it->first < addr + size) {
        const Address start = std::min(it->first, addr);
        protect.size = std::max(it->first + it->second.size, addr + protect.size) - start;
        addr = start;
        protect.blocks.merge(it->second.blocks); // transfer blocks to the new protect
        protect.perm = most_restrictive_perm(protect.perm, it->second.perm);

        if (it == state.protect_tree.begin()) {
            state.protect_tree.erase(it);
            break;
        }

        // protect tree is in reverse order, so decrease it
        state.protect_tree.erase(it--);
    }

    protect_inner(state, addr, protect.size, protect.perm);

    state.protect_tree.emplace(addr, std::move(protect));
    return true;
}

bool add_protect_metal3(MemState &state, Address addr, const uint32_t size, const MemPerm perm, const ProtectCallback &callback) {
    return add_protect_impl(state, addr, size, perm, callback, false);
}

bool add_write_watch(MemState &state, Address addr, uint32_t size, const ProtectCallback &callback) {
    // A partial host page can fault outside its logical tree interval. Reject
    // it rather than silently removing another owner's protection there.
    const uint32_t page = state.host_page_size;
    const uint64_t end = uint64_t(addr) + size;
    if (state.use_page_table || !page || !addr || !size || addr % page || size % page
        || end > uint64_t(UINT32_MAX) - page || !callback
        || !is_valid_addr_range_metal3(state, addr, Address(end))) return false;
    state.native_metal = true;
    return add_protect_impl(state, addr, size, MemPerm::ReadOnly, callback, true);
}

bool is_protecting_metal3(MemState &state, Address addr, MemPerm *perm) {
    const std::lock_guard<std::mutex> lock(state.protect_mutex);
    auto ite = state.protect_tree.lower_bound(addr);

    if (ite != state.protect_tree.end() && addr < ite->first + ite->second.size) {
        if (ite->second.write_watch_ranges_only
            && std::none_of(ite->second.blocks.begin(), ite->second.blocks.end(), [addr](const auto &entry) {
                return addr >= entry.first && uint64_t(addr) < uint64_t(entry.first) + entry.second.size;
            })) return false;
        if (perm)
            *perm = ite->second.perm;

        return true;
    }

    return false;
}













void free_metal3(MemState &state, Address address) {
    const auto lock = renderer::metal::lock_guest_memory_after_publication(&state, state.generation_mutex);
    const uint32_t page_num = address / STANDARD_PAGE_SIZE;
    assert(page_num >= 0);

    if (!state.alloc_table) {
        LOG_CRITICAL("Freeing an address before guest memory initialization");
        return;
    }

    AllocMemPage &page = state.alloc_table[page_num];
    if (!page.allocated) {
        LOG_CRITICAL("Freeing unallocated page at address 0x{:X}; ignoring stale allocation size {}", address, static_cast<uint32_t>(page.size));
        return;
    }
    page.allocated = 0;

    state.allocator.free(page_num, page.size);
    if (PAGE_NAME_TRACKING) {
        state.page_name_map.erase(page_num);
    }

    assert(!state.use_page_table || state.page_table[address / KiB(4)] == state.memory.get());
    const uint64_t region_start = uint64_t(page_num) * STANDARD_PAGE_SIZE;
    decommit_free_host_pages(state, region_start, region_start + uint64_t(page.size) * STANDARD_PAGE_SIZE);
}







#ifdef _WIN32

static LONG WINAPI exception_handler(PEXCEPTION_POINTERS pExp) noexcept {
    if (pExp->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT && IsDebuggerPresent()) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const auto ptr = reinterpret_cast<uint8_t *>(pExp->ExceptionRecord->ExceptionInformation[1]);
    const bool is_writing = pExp->ExceptionRecord->ExceptionInformation[0] == 1;
    const bool is_executing = pExp->ExceptionRecord->ExceptionInformation[0] == 8;

    if (pExp->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && !is_executing) {
        if (access_violation_handler(ptr, is_writing)) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

static void register_access_violation_handler(const AccessViolationHandler &handler) {
    access_violation_handler = handler;
    if (!AddVectoredExceptionHandler(1, exception_handler)) {
        LOG_CRITICAL("Failed to register an exception handler");
    }
}

#else

static void signal_handler(int sig, siginfo_t *info, void *uct) noexcept {
    auto context = static_cast<ucontext_t *>(uct);

#ifdef __aarch64__
#ifdef __APPLE__
    const uint32_t esr = context->uc_mcontext->__es.__esr;
#else
    _aarch64_ctx *ctx = reinterpret_cast<_aarch64_ctx *>(context->uc_mcontext.__reserved);
    // get the ESR register
    while (ctx->magic != ESR_MAGIC) {
        if (ctx->magic == 0)
            [[unlikely]]
            raise(SIGTRAP);
        else
            [[likely]]
            ctx = reinterpret_cast<_aarch64_ctx *>(reinterpret_cast<uint8_t *>(ctx) + ctx->size);
    }

    const uint64_t esr = reinterpret_cast<esr_context *>(ctx)->esr;
#endif
    // https://developer.arm.com/documentation/ddi0595/2021-03/AArch64-Registers/ESR-EL1--Exception-Syndrome-Register--EL1-
    const uint32_t exception_class = static_cast<uint32_t>(esr) >> 26;
    const bool is_executing = (exception_class == 0b100000) || (exception_class == 0b100001);
    const bool is_data_abort = (exception_class == 0b100100) || (exception_class == 0b100101);
    const bool is_writing = is_data_abort && (esr & (1 << 6));
#else
#ifdef __APPLE__
    const uint64_t err = context->uc_mcontext->__es.__err;
#else
    const uint64_t err = context->uc_mcontext.gregs[REG_ERR];
#endif
    const bool is_executing = err & 0x10;
    const bool is_writing = err & 0x2;
#endif

    if (!is_executing) {
        if (access_violation_handler(reinterpret_cast<uint8_t *>(info->si_addr), is_writing)) {
            return;
        }
    }

    LOG_CRITICAL("Unhandled access to 0x{:X}", reinterpret_cast<uintptr_t>(info->si_addr));
    raise(SIGTRAP);
    return;
}

static void register_access_violation_handler(const AccessViolationHandler &handler) {
    access_violation_handler = handler;
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = signal_handler;
    if (sigaction(SIGSEGV, &sa, NULL) == -1) {
        LOG_CRITICAL("Failed to register an exception handler");
    }
#ifdef __APPLE__
    // When accessing memory region which is PROT_NONE on macOS, it is raising SIGBUS not SIGSEGV.
    // So apply same signal handler to SIGBUS
    if (sigaction(SIGBUS, &sa, NULL) == -1) {
        LOG_CRITICAL("Failed to register an exception handler to SIGBUS");
    }
#endif
}

#endif
