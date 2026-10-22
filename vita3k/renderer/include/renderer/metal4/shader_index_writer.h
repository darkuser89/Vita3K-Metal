// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include <unistd.h>

namespace renderer::metal {
class ShaderIndexWriter {
public:
    using Pair = std::array<uint8_t, 64>; // fragment SHA256, then vertex SHA256
    using Sink = std::function<void(const std::filesystem::path &, std::span<const uint8_t>)>;
    // Same bound as the private reader; never generate an unreadable index.
    static constexpr size_t max_pairs = 1'000'000;
    struct Stats { size_t writes = 0, bytes = 0, pairs = 0; std::string error; };
    static void atomic_write(const std::filesystem::path &path, std::span<const uint8_t> bytes) {
        std::filesystem::create_directories(path.parent_path());
        std::string temporary = path.string() + ".tmp.XXXXXX";
        const int fd = mkstemp(temporary.data());
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "create shader index");
        struct File {
            int fd; std::string path;
            ~File() { ::close(fd); if (!path.empty()) ::unlink(path.c_str()); }
        } file{fd, temporary};
        while (!bytes.empty()) {
            const auto size = ::write(fd, bytes.data(), bytes.size());
            if (size < 0 && errno == EINTR) continue;
            if (size <= 0) throw std::system_error(size < 0 ? errno : EIO, std::generic_category(), "write shader index");
            bytes = bytes.subspan(size_t(size));
        }
        if (::fsync(fd) || ::rename(temporary.c_str(), path.c_str()))
            throw std::system_error(errno, std::generic_category(), "publish shader index");
        file.path.clear();
    }
private:
    std::filesystem::path path;
    uint32_t version, features;
    Sink sink;
    std::chrono::milliseconds debounce;
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<Pair> pending;
    size_t submitted = 0;
    bool stopping = false;
    bool failed = false;
    Stats statistics;
    std::thread worker;

    void work() {
        std::vector<Pair> history;
        for (;;) {
            std::vector<Pair> additions;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return stopping || !pending.empty(); });
                if (pending.empty() && stopping) return;
                // Fixed window, not a reset-on-every-draw timer: continuous
                // shader discovery cannot postpone persistence indefinitely.
                if (!stopping) wake.wait_for(lock, debounce, [&] { return stopping; });
                additions.swap(pending);
            }
            std::string error;
            size_t byte_count = 0;
            std::vector<uint8_t> bytes;
            try {
                const size_t count = history.size() + additions.size();
                if (history.capacity() < count)
                    history.reserve(std::min(max_pairs, std::max(count, history.capacity() * 2)));
                history.insert(history.end(), additions.begin(), additions.end());
                const size_t header = sizeof(size_t) + 2 * sizeof(uint32_t);
                bytes.resize(header + count * sizeof(Pair));
                std::memcpy(bytes.data(), &count, sizeof(count));
                std::memcpy(bytes.data() + sizeof(count), &version, sizeof(version));
                std::memcpy(bytes.data() + sizeof(count) + sizeof(version), &features, sizeof(features));
                if (count) std::memcpy(bytes.data() + header, history.data(), count * sizeof(Pair));
            } catch (const std::exception &failure) {
                // A failed in-memory merge must not be followed by a later
                // apparently successful index with missing accepted pairs.
                std::lock_guard lock(mutex);
                statistics.error = failure.what();
                failed = true;
                return;
            }
            try { sink(path, bytes); byte_count = bytes.size(); }
            catch (const std::exception &failure) { error = failure.what(); }
            {
                std::lock_guard lock(mutex);
                ++statistics.writes;
                statistics.bytes += byte_count;
                if (error.empty()) statistics.pairs = history.size();
                statistics.error = std::move(error);
            }
        }
    }
    void stop() {
        { std::lock_guard lock(mutex); stopping = true; }
        wake.notify_one();
        if (worker.joinable()) worker.join();
    }
public:
    ShaderIndexWriter(std::filesystem::path destination, uint32_t version, uint32_t features,
        std::vector<Pair> initial, Sink sink = atomic_write,
        std::chrono::milliseconds debounce = std::chrono::milliseconds(100))
        : path(std::move(destination)), version(version), features(features), sink(std::move(sink)),
          debounce(debounce), pending(std::move(initial)), submitted(pending.size()) {
        if (submitted > max_pairs) throw std::invalid_argument("Metal: shader index exceeds reader limit");
        worker = std::thread([this] { work(); });
    }
    ShaderIndexWriter(const ShaderIndexWriter &) = delete;
    ShaderIndexWriter &operator=(const ShaderIndexWriter &) = delete;
    ~ShaderIndexWriter() { stop(); }
    bool append(const Pair &pair) {
        {
            std::lock_guard lock(mutex);
            if (stopping || failed || submitted == max_pairs) return false;
            if (pending.size() == pending.capacity())
                pending.reserve(std::min(max_pairs, std::max(size_t(16), pending.capacity() * 2)));
            pending.push_back(pair);
            ++submitted;
        }
        wake.notify_one();
        return true;
    }
    Stats finish() {
        stop();
        std::lock_guard lock(mutex);
        return statistics;
    }
};
}
