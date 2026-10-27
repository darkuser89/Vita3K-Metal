// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
// Production cache readers and archive lookup prefixes are injected verbatim.
// File streams pause at an observed boundary; archive creation and descriptor
// restoration are doubles. No Metal device, app or GPU is created.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <shader/metal4/msl_recompiler.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
using namespace std::chrono_literals;
namespace fs = std::filesystem;
struct Gate {
    std::promise<void> entered, release;
    std::shared_future<void> released = release.get_future().share();
    void pause() { entered.set_value(); released.wait(); }
    void observed() { assert(entered.get_future().wait_for(5s) == std::future_status::ready); }
};
static thread_local Gate *read_gate = nullptr, *archive_gate = nullptr, *digest_gate = nullptr;
static thread_local std::string_view digest_input;
static thread_local bool reject_archive = false;
static std::atomic<unsigned> archive_calls = 0;
struct ObservedInput {
    std::ifstream file;
    ObservedInput(const fs::path &path, std::ios::openmode mode) : file(path, mode) {
        if (read_gate) read_gate->pause();
    }
    ObservedInput &read(char *bytes, std::streamsize count) { file.read(bytes, count); return *this; }
    explicit operator bool() const { return bool(file); }
};
@protocol MTL4Archive;
@interface ObservedArchiveDevice : NSObject
- (id<MTL4Archive>)newArchiveWithURL:(NSURL *)path error:(NSError **)error;
@end
@implementation ObservedArchiveDevice
- (id<MTL4Archive>)newArchiveWithURL:(NSURL *)path error:(NSError **)error {
    ++archive_calls;
    if (archive_gate) archive_gate->pause();
    if (reject_archive) {
        if (error) *error = [NSError errorWithDomain:@"test" code:1 userInfo:nil];
        return nil;
    }
    return (id<MTL4Archive>)[NSObject new];
}
@end
struct Device {
    ObservedArchiveDevice *object = [ObservedArchiveDevice new];
    ObservedArchiveDevice *native_device() { return object; }
};
#define LOG_WARN(...) ((void)0)
constexpr size_t max_program_bytes = 8 * 1024 * 1024, max_variant_key_bytes = 4096,
    max_pipeline_template_bytes = 16 * 1024;
constexpr std::string_view magic = "V3KMSL01", variant_magic = "V3KVAR02", pipeline_magic = "V3KPIPE1";
static std::string digest(std::string_view bytes) {
    if (digest_gate && bytes == digest_input) digest_gate->pause();
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes.data(), CC_LONG(bytes.size()), hash);
    std::string result;
    for (auto c : hash) { result += "0123456789abcdef"[c >> 4]; result += "0123456789abcdef"[c & 15]; }
    return result;
}
// HELPERS_HOOK
// STATS_HOOK
struct CachedVariant { std::string key; bool gamma_correction = false; };
struct CachedPipeline { std::string key, vertex_key, fragment_key; };
static std::optional<CachedPipeline> restore_render_template(std::string_view descriptor) {
    return descriptor == "descriptor" ? std::optional<CachedPipeline>(CachedPipeline{}) : std::nullopt;
}
struct PersistentCache {
    struct Impl {
        std::mutex mutex;
        CacheStats counters;
        fs::path programs, native;
        Device device;
        std::map<fs::path, std::shared_ptr<const std::string>> pending_programs;
        std::map<fs::path, id> pending_render, pending_compute;
        std::atomic<unsigned> drains = 0;
        void drain_writes() { ++drains; }
        // REJECT_HOOK
        // ARCHIVE_HOOK
    };
    std::unique_ptr<Impl> impl = std::make_unique<Impl>();
    explicit PersistentCache(const fs::path &root) {
        impl->programs = root / "programs"; impl->native = root / "native";
        fs::create_directories(impl->programs); fs::create_directories(impl->native);
        impl->counters.archive_writable = true;
    }
    std::optional<shader::metal::Program> load_program(std::string_view);
    std::vector<CachedVariant> variants(std::string_view);
    std::vector<CachedPipeline> render_pipeline_templates(std::string_view, std::string_view);
    CacheStats stats() { std::lock_guard lock(impl->mutex); return impl->counters; }
    id open_render(const fs::path &path) {
        // RENDER_OPEN_HOOK
        return archive;
    }
    id open_compute(const fs::path &path) {
        // COMPUTE_OPEN_HOOK
        return archive;
    }
};
// READERS_HOOK
static void write(const fs::path &path, std::string_view bytes) {
    fs::create_directories(path.parent_path()); std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), bytes.size()); assert(file);
}
static std::string program_body(uint32_t flags = 16, uint32_t cube = 3,
    std::string_view entry = "main_vs", std::string_view source = "valid source") {
    std::string body; put32(body, flags); put32(body, cube);
    put32(body, uint32_t(entry.size())); put32(body, uint32_t(source.size()));
    body += entry; body += source; return body;
}
static std::string program_file(std::string_view key, const std::string &body) {
    std::string bytes(magic); bytes += digest(key); bytes += digest(body);
    put32(bytes, uint32_t(body.size())); bytes += body; return bytes;
}
template <typename Read> static bool independent_progress(PersistentCache &cache, Gate *&boundary, Read read) {
    Gate gate;
    const bool native = &boundary == &archive_gate;
    auto blocked = std::async(std::launch::async, [&] { @autoreleasepool {
        auto &slot = native ? archive_gate : read_gate;
        slot = &gate; read(); slot = nullptr;
    } });
    gate.observed();
    auto other = std::async(std::launch::async, [&] {
        auto stats = cache.stats();
        assert(cache.load_program("pending"));
        return stats;
    });
    const bool ready = other.wait_for(2s) == std::future_status::ready;
    // Release and join before asserting, including when reproducing the old lock.
    gate.release.set_value(); blocked.get(); (void)other.get(); return ready;
}
int main(int argc, char **argv) {
    assert(argc == 2); PersistentCache cache(fs::path(argv[1]) / "ü-cache");
    const auto key = "disk";
    const auto body = program_body();
    const auto disk = cache.impl->programs / (digest(key) + ".mslcache");
    write(disk, program_file(key, body));
    cache.impl->pending_programs[cache.impl->programs / (digest("pending") + ".mslcache")] =
        std::make_shared<const std::string>(program_file("pending", body));
    bool progress = independent_progress(cache, read_gate, [&] { assert(cache.load_program(key)); });
    const auto vertex = digest("vertex"), fragment = digest("fragment");
    const auto vk = vertex + "vertex-key", fk = fragment + "fragment-key", pk = vk + fk + "layout";
    std::string variant_body; put32(variant_body, uint32_t(vk.size())); variant_body += char(1); variant_body += vk;
    const auto variant = cache.impl->programs / "variants" / vertex / (digest(vk) + ".variant");
    write(variant, std::string(variant_magic) + digest(vk) + digest(variant_body) + variant_body);
    progress &= independent_progress(cache, read_gate, [&] {
        const auto result = cache.variants(vertex); assert(result.size() == 1 && result[0].key == vk && result[0].gamma_correction);
    });
    std::string pipeline_body; for (size_t n : {pk.size(), vk.size(), fk.size(), size_t(10)}) put32(pipeline_body, uint32_t(n));
    pipeline_body += pk; pipeline_body += vk; pipeline_body += fk; pipeline_body += "descriptor";
    const auto pipeline = cache.impl->native / "warmup" / fragment / vertex / (digest(pk) + ".pipeline");
    write(pipeline, std::string(pipeline_magic) + digest(pipeline_body) + pipeline_body);
    progress &= independent_progress(cache, read_gate, [&] {
        const auto result = cache.render_pipeline_templates(fragment, vertex);
        assert(result.size() == 1 && result[0].key == pk && result[0].vertex_key == vk && result[0].fragment_key == fk);
    });
    const auto archive = cache.impl->native / "archive.metalarc"; write(archive, "archive boundary");
    progress &= independent_progress(cache, archive_gate, [&] { assert(cache.open_render(archive)); });
    progress &= independent_progress(cache, archive_gate, [&] { assert(cache.open_compute(archive)); });
    assert(progress && "Unrelated cache progress blocked behind disk/archive I/O");
    assert(cache.stats().archive_loaded && cache.impl->drains == 2);
    // Snapshot ownership must outlive removal by the completing writer.
    const auto pending_path = cache.impl->programs / (digest("pending") + ".mslcache");
    std::weak_ptr<const std::string> owner = cache.impl->pending_programs.at(pending_path);
    Gate gate;
    auto reading = std::async(std::launch::async, [&] { digest_gate = &gate; digest_input = body; return cache.load_program("pending"); });
    gate.observed();
    { std::lock_guard lock(cache.impl->mutex); cache.impl->pending_programs.erase(pending_path); assert(!owner.expired()); }
    gate.release.set_value(); assert(reading.get()->source == "valid source" && owner.expired());
    // Real file validation and statistics remain intact for all stage/feature bits.
    for (unsigned stage : {0u, 1u, 32u}) for (unsigned features = 0; features < 16; ++features) {
        write(disk, program_file(key, program_body(stage | (features << 1 & 30), 65535)));
        auto result = cache.load_program(key); assert(result && result->cube_texture_mask == 65535);
    }
    const auto before = cache.stats();
    for (auto invalid : {program_body(33), program_body(64), program_body(16, 65536), program_body(16, 0, ""),
            program_body(16, 0, "main", ""), program_body(16, 0, std::string("a\0b", 3))}) {
        write(disk, program_file(key, invalid)); assert(!cache.load_program(key));
    }
    write(disk, "short"); assert(!cache.load_program(key));
    auto corrupt = program_file(key, body); corrupt.back() ^= 1; write(disk, corrupt); assert(!cache.load_program(key));
    fs::remove(disk); assert(!cache.load_program(key));
    auto after = cache.stats(); assert(after.program_misses == before.program_misses + 9 && after.rejected_files == before.rejected_files + 8);
    write(disk, program_file(key, body));
    write(variant.parent_path() / "bad.variant", "short");
    write(pipeline.parent_path() / "bad.pipeline", "short");
    assert(cache.render_pipeline_templates(fragment, vertex).size() == 1);
    const auto before_parallel = cache.stats();
    std::vector<std::thread> readers;
    for (unsigned t = 0; t < 8; ++t) readers.emplace_back([&] {
        for (unsigned n = 0; n < 64; ++n) { assert(cache.load_program(key)); assert(cache.variants(vertex).size() == 1); }
    });
    for (auto &reader : readers) reader.join();
    assert(cache.stats().program_hits == after.program_hits + 512);
    assert(cache.stats().rejected_files == before_parallel.rejected_files + 512);
    const auto calls = archive_calls.load();
    { std::lock_guard lock(cache.impl->mutex); cache.impl->pending_render[archive] = [NSObject new]; cache.impl->pending_compute[archive] = [NSObject new]; }
    assert(cache.open_render(archive) && cache.open_compute(archive) && archive_calls == calls);
    cache.impl->pending_render.clear(); cache.impl->pending_compute.clear();
    reject_archive = true; assert(!cache.open_render(archive)); reject_archive = false;
    const auto rejected = cache.stats().rejected_files;
    fs::resize_file(archive, 512ull * 1024 * 1024 + 1); assert(!cache.open_compute(archive));
    assert(cache.stats().rejected_files == rejected + 1 && archive_calls == calls + 1);
    std::cout << "PASS cache readers: five observed I/O gates permit independent statistics/pending reads; immutable snapshot survives writer removal; stage/feature/invalid file validation; 512 concurrent disk reads and variant scans; pending render/compute fast hits and archive errors; no device/GPU/app\n";
}
