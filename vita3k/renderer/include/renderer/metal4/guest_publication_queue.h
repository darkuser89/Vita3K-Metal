// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace renderer::metal {
// One ordered completion worker. Capacity includes reserved, queued and active
// jobs, so retained GPU/guest owners stay bounded even during a stalled fence.
class GuestPublicationQueue : public std::enable_shared_from_this<GuestPublicationQueue> {
    const size_t capacity;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::function<void()>> work;
    size_t outstanding = 0;
    bool stopping = false;
    bool poisoned = false;
    std::exception_ptr failure;
    std::thread worker;
    void release() {
        { std::lock_guard lock(mutex); --outstanding; }
        changed.notify_all();
    }
    void run() {
        for (;;) {
            std::function<void()> task;
            bool canceled = false;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return stopping || !work.empty(); });
                if (work.empty()) return;
                task = std::move(work.front()); work.pop_front();
                canceled = poisoned;
            }
            try { if (!canceled) task(); }
            catch (...) { std::lock_guard lock(mutex); poisoned = true; if (!failure) failure = std::current_exception(); }
            // Drop captured GPU/guest owners before an idle observer may free
            // their memory or tear down the device and notification mutex.
            task = {};
            release();
        }
    }
    explicit GuestPublicationQueue(size_t capacity) : capacity(capacity) {
        if (!capacity) throw std::invalid_argument("Metal: empty publication capacity");
        worker = std::thread([this] { run(); });
    }
public:
    static std::shared_ptr<GuestPublicationQueue> create(size_t capacity = 2) {
        return std::shared_ptr<GuestPublicationQueue>(new GuestPublicationQueue(capacity));
    }
    ~GuestPublicationQueue() {
        { std::lock_guard lock(mutex); stopping = true; }
        changed.notify_all();
        if (worker.joinable()) worker.join();
    }
    class Reservation {
        std::shared_ptr<GuestPublicationQueue> queue;
        friend class GuestPublicationQueue;
        explicit Reservation(std::shared_ptr<GuestPublicationQueue> queue) : queue(std::move(queue)) {}
    public:
        Reservation(const Reservation &) = delete;
        Reservation &operator=(const Reservation &) = delete;
        Reservation(Reservation &&other) noexcept : queue(std::move(other.queue)) {}
        ~Reservation() { if (queue) queue->release(); }
        void submit(std::function<void()> task) {
            if (!queue || !task) throw std::logic_error("Metal: invalid publication reservation");
            {
                std::lock_guard lock(queue->mutex);
                if (queue->stopping) throw std::logic_error("Metal: publication queue stopped");
                queue->work.push_back(std::move(task));
            }
            queue->changed.notify_one();
            queue.reset(); // The worker now owns the outstanding slot.
        }
    };
    void wait_for_capacity() {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return stopping || poisoned || outstanding < capacity; });
        if (stopping) throw std::logic_error("Metal: publication queue stopped");
        if (poisoned) throw std::runtime_error("Metal: publication queue failed");
    }
    std::optional<Reservation> try_reserve() {
        std::lock_guard lock(mutex);
        if (stopping) throw std::logic_error("Metal: publication queue stopped");
        if (poisoned) throw std::runtime_error("Metal: publication queue failed");
        if (outstanding == capacity) return {};
        ++outstanding;
        return Reservation(shared_from_this());
    }
    Reservation reserve() {
        for (;;) {
            wait_for_capacity();
            if (auto reservation = try_reserve()) return std::move(*reservation);
        }
    }
    bool idle() const { std::lock_guard lock(mutex); return !outstanding; }
    void check() {
        std::exception_ptr error;
        { std::lock_guard lock(mutex); error = std::exchange(failure, {}); }
        if (error) std::rethrow_exception(error);
    }
    void drain(bool report_error = true) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return !outstanding; });
        lock.unlock();
        if (report_error) check();
    }
};

template<class Mutex>
GuestPublicationQueue::Reservation reserve_guest_publication(GuestPublicationQueue &queue, Mutex &generation) {
    for (;;) {
        queue.wait_for_capacity(); // Never wait for GPU capacity under the allocator mutex.
        const std::lock_guard lock(generation);
        if (auto reservation = queue.try_reserve()) return std::move(*reservation);
        // Another producer consumed capacity between the wait and this lock.
    }
}

// The memory allocator need not know a renderer/context. A weak registration
// lets the private Metal free path wait outside generation_mutex, then check
// again under it. A reserved job counts as pending before it is published.
inline std::mutex guest_publication_registry_mutex;
inline std::map<const void *, std::vector<std::weak_ptr<GuestPublicationQueue>>> guest_publication_registry;
inline void register_guest_publications(const void *memory, const std::shared_ptr<GuestPublicationQueue> &queue) {
    std::lock_guard lock(guest_publication_registry_mutex);
    auto &list = guest_publication_registry[memory];
    std::erase_if(list, [](const auto &entry) { return entry.expired(); });
    if (std::none_of(list.begin(), list.end(), [&](const auto &entry) { return entry.lock() == queue; })) list.push_back(queue);
}
inline std::vector<std::shared_ptr<GuestPublicationQueue>> guest_publications_for(const void *memory) {
    std::lock_guard lock(guest_publication_registry_mutex);
    std::vector<std::shared_ptr<GuestPublicationQueue>> queues;
    if (auto found = guest_publication_registry.find(memory); found != guest_publication_registry.end()) {
        for (auto &entry : found->second) if (auto queue = entry.lock()) queues.push_back(std::move(queue));
        std::erase_if(found->second, [](const auto &entry) { return entry.expired(); });
        if (found->second.empty()) guest_publication_registry.erase(found);
    }
    return queues;
}
inline void unregister_guest_publications(const void *memory, const std::shared_ptr<GuestPublicationQueue> &queue) {
    std::lock_guard lock(guest_publication_registry_mutex);
    if (auto found = guest_publication_registry.find(memory); found != guest_publication_registry.end()) {
        std::erase_if(found->second, [&](const auto &entry) { return entry.expired() || entry.lock() == queue; });
        if (found->second.empty()) guest_publication_registry.erase(found);
    }
}
inline std::unique_lock<std::mutex> lock_guest_memory_after_publication(const void *memory, std::mutex &generation) {
    for (;;) {
        for (auto &queue : guest_publications_for(memory)) queue->drain(false);
        std::unique_lock lock(generation);
        const auto queues = guest_publications_for(memory);
        if (std::all_of(queues.begin(), queues.end(), [](const auto &queue) { return queue->idle(); })) return lock;
        // A producer reserved a slot between the drain and this lock. It
        // requires generation_mutex to enqueue; wait only after releasing it.
    }
}
}
