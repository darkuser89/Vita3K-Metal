// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <metal_cache_revision.h>
#include <mutex>
#include <renderer/metal/cache.h>
#include <renderer/metal/device.h>
#include <sys/file.h>
#include <unistd.h>
#include <util/hash.h>
#include <util/log.h>

namespace renderer::metal {
namespace {
constexpr size_t max_program_bytes = 8 * 1024 * 1024;
constexpr std::string_view magic = "V3KMSL01";
std::string digest(std::string_view s) { return hex_string(sha256(s.data(), s.size())); }
NSURL *url(const std::filesystem::path &path) {
    return [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
}
std::string error_text(NSError *e) { return e.localizedDescription.UTF8String ?: "Unknown Metal cache error"; }
void put32(std::string &s, uint32_t v) {
    for (unsigned i = 0; i < 4; ++i)
        s.push_back(char(v >> (i * 8)));
}
uint32_t get32(std::string_view s, size_t at) {
    uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i)
        v |= uint32_t(uint8_t(s[at + i])) << (i * 8);
    return v;
}
// Same-directory rename keeps a previous complete cache valid until publication.
struct TemporaryFile {
    std::filesystem::path path;
    int fd = -1;
    explicit TemporaryFile(const std::filesystem::path &destination) {
        std::string pattern = destination.string() + ".tmp.XXXXXX";
        fd = mkstemp(pattern.data());
        if (fd >= 0)
            path = pattern;
    }
    ~TemporaryFile() {
        if (fd >= 0)
            close(fd);
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
    bool publish(const std::filesystem::path &destination) {
        if (fd < 0 || fsync(fd) != 0 || ::rename(path.c_str(), destination.c_str()) != 0)
            return false;
        path.clear();
        return true;
    }
};
bool atomic_write(const std::filesystem::path &path, std::string_view data) {
    TemporaryFile file(path);
    if (file.fd < 0)
        return false;
    while (!data.empty()) {
        const ssize_t n = ::write(file.fd, data.data(), data.size());
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        data.remove_prefix(size_t(n));
    }
    return file.publish(path);
}
} // namespace
// Each pipeline has its own archive. Some drivers fail to serialize aggregate
// archives containing several optimized variants ("expecting fragment stage").
// Files are only an index: Metal still validates the complete descriptor on load.
static std::string render_key(MTLRenderPipelineDescriptor *desc) {
    std::string key = "render\n";
    key += desc.vertexFunction.name.UTF8String ?: "";
    key += "\n";
    key += desc.fragmentFunction.name.UTF8String ?: "";
    auto add = [&](uint64_t value) { put32(key, uint32_t(value)); put32(key, uint32_t(value >> 32)); };
    add(desc.rasterSampleCount);
    add(desc.alphaToCoverageEnabled);
    add(desc.alphaToOneEnabled);
    add(desc.rasterizationEnabled);
    add(desc.depthAttachmentPixelFormat);
    add(desc.stencilAttachmentPixelFormat);
    add(desc.inputPrimitiveTopology);
    add(desc.supportIndirectCommandBuffers);
    add(desc.maxVertexAmplificationCount);
    for (unsigned i = 0; i < 8; ++i) {
        auto a = desc.colorAttachments[i];
        add(a.pixelFormat);
        add(a.writeMask);
        add(a.blendingEnabled);
        add(a.rgbBlendOperation);
        add(a.alphaBlendOperation);
        add(a.sourceRGBBlendFactor);
        add(a.destinationRGBBlendFactor);
        add(a.sourceAlphaBlendFactor);
        add(a.destinationAlphaBlendFactor);
    }
    for (unsigned i = 0; i < 31; ++i) {
        auto a = desc.vertexDescriptor.attributes[i];
        auto layout = desc.vertexDescriptor.layouts[i];
        add(a.format);
        add(a.offset);
        add(a.bufferIndex);
        add(layout.stride);
        add(layout.stepFunction);
        add(layout.stepRate);
        add(desc.vertexBuffers[i].mutability);
        add(desc.fragmentBuffers[i].mutability);
    }
    return key;
}
struct PersistentCache::Impl {
    Device &device;
    std::filesystem::path root, programs, native;
    mutable std::mutex mutex;
    CacheStats counters;
    int lock_fd = -1;
    Impl(Device &d, const std::filesystem::path &path)
        : device(d) {
        root = path / ("v2-abi" + std::to_string(shader::metal::SHADER_ABI_VERSION) + "-" + METAL_CACHE_REVISION);
        programs = root / "programs";
        // Registry IDs can change across reboots; model/architecture names do not.
        std::string architecture = "metal3";
        if (@available(macOS 14.0, *))
            architecture = d.native_device().architecture.name.UTF8String ?: "metal3";
        const std::string hardware = std::string(d.native_device().name.UTF8String ?: "unknown") + "\n" + architecture + "\n"
            + std::string(NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String ?: "unknown") + "\nmsl3.0;fast-math=off;archive-v2";
        native = root / "native" / digest(hardware);
        std::error_code ec;
        std::filesystem::create_directories(programs, ec);
        if (ec)
            io_error("Cannot create shader cache directory");
        ec.clear();
        std::filesystem::create_directories(native, ec);
        if (ec)
            io_error("Cannot create native archive directory");
        lock_fd = open((native / ".lock").c_str(), O_CREAT | O_RDWR, 0600);
        counters.archive_writable = lock_fd >= 0 && flock(lock_fd, LOCK_EX | LOCK_NB) == 0;
    }
    ~Impl() {
        if (lock_fd >= 0)
            close(lock_fd);
    }
    void io_error(const char *message) {
        if (counters.io_errors++ == 0)
            LOG_WARN("Metal cache: {} (rendering continues without cache writes)", message);
    }
    id<MTLBinaryArchive> load_archive(const std::filesystem::path &path) {
        auto desc = [MTLBinaryArchiveDescriptor new];
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        const bool exists = !ec;
        if (exists && size <= 512ull * 1024 * 1024)
            desc.url = url(path);
        else if (exists)
            ++counters.rejected_files;
        NSError *error = nil;
        auto archive = [device.native_device() newBinaryArchiveWithDescriptor:desc error:&error];
        if (archive && desc.url)
            counters.archive_loaded = true;
        if (!archive && desc.url) {
            ++counters.rejected_files;
            LOG_WARN("Metal cache: ignoring invalid native archive: {}", error_text(error));
            desc.url = nil;
            error = nil;
            archive = [device.native_device() newBinaryArchiveWithDescriptor:desc error:&error];
        }
        return archive;
    }
    void save(id<MTLBinaryArchive> archive, const std::filesystem::path &path) {
        if (!archive || !counters.archive_writable)
            return;
        TemporaryFile file(path);
        NSError *error = nil;
        if (file.fd < 0 || ![archive serializeToURL:url(file.path) error:&error]) {
            LOG_WARN("Metal native archive serialization failed: {}", error_text(error));
            counters.archive_writable = false;
            io_error("Cannot serialize native archive");
            return;
        }
        close(file.fd);
        file.fd = open(file.path.c_str(), O_RDONLY);
        if (!file.publish(path)) {
            counters.archive_writable = false;
            io_error("Cannot publish native archive");
            return;
        }
        ++counters.archive_writes;
    }
};
PersistentCache::PersistentCache(Device &device, const std::filesystem::path &root)
    : impl(std::make_unique<Impl>(device, root)) {}
PersistentCache::~PersistentCache() = default;
std::optional<shader::metal::Program> PersistentCache::load_program(std::string_view key) {
    std::lock_guard lock(impl->mutex);
    const auto hash = digest(key);
    const auto path = impl->programs / (hash + ".mslcache");
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    auto miss = [&](bool rejected) -> std::optional<shader::metal::Program> {
        ++impl->counters.program_misses;
        if (rejected)
            ++impl->counters.rejected_files;
        return std::nullopt;
    };
    if (ec)
        return miss(false);
    constexpr size_t header = 8 + 64 + 64 + 4;
    if (size < header + 16 || size > max_program_bytes + header)
        return miss(true);
    std::string bytes(size, '\0');
    std::ifstream file(path, std::ios::binary);
    if (!file.read(bytes.data(), bytes.size()))
        return miss(true);
    std::string_view view(bytes), body = view.substr(header);
    if (view.substr(0, 8) != magic || view.substr(8, 64) != hash || view.substr(72, 64) != digest(body)
        || get32(view, 136) != body.size())
        return miss(true);
    const uint32_t flags = get32(body, 0), cube = get32(body, 4), entry = get32(body, 8), source = get32(body, 12);
    if (flags & ~31u || cube & ~65535u || !entry || entry > 256 || !source || source > max_program_bytes
        || uint64_t(entry) + source + 16 != body.size())
        return miss(true);
    shader::metal::Program program;
    program.stage = flags & 1 ? shader::metal::Stage::Fragment : shader::metal::Stage::Vertex;
    program.uses_framebuffer_fetch = flags & 2;
    program.uses_raster_order_groups = flags & 4;
    program.uses_buffer_addresses = flags & 8;
    program.writes_guest_memory = flags & 16;
    program.cube_texture_mask = cube;
    program.entry_point = body.substr(16, entry);
    program.source = body.substr(16 + entry, source);
    if (program.entry_point.find('\0') != std::string::npos || program.source.find('\0') != std::string::npos)
        return miss(true);
    ++impl->counters.program_hits;
    return program;
}
void PersistentCache::store_program(std::string_view key, const shader::metal::Program &program) {
    std::lock_guard lock(impl->mutex);
    if (program.source.empty() || program.source.size() + program.entry_point.size() + 16 > max_program_bytes
        || program.entry_point.empty() || program.entry_point.size() > 256)
        return;
    std::string body;
    put32(body, uint32_t(program.stage == shader::metal::Stage::Fragment) | (uint32_t(program.uses_framebuffer_fetch) << 1) | (uint32_t(program.uses_raster_order_groups) << 2) | (uint32_t(program.uses_buffer_addresses) << 3) | (uint32_t(program.writes_guest_memory) << 4));
    put32(body, program.cube_texture_mask);
    put32(body, uint32_t(program.entry_point.size()));
    put32(body, uint32_t(program.source.size()));
    body += program.entry_point;
    body += program.source;
    const auto hash = digest(key);
    std::string file(magic);
    file += hash;
    file += digest(body);
    put32(file, uint32_t(body.size()));
    file += body;
    if (atomic_write(impl->programs / (hash + ".mslcache"), file))
        ++impl->counters.program_writes;
    else
        impl->io_error("Cannot save translated shader");
}
id<MTLRenderPipelineState> PersistentCache::create_render_pipeline(MTLRenderPipelineDescriptor *descriptor, std::string &error) {
    std::lock_guard lock(impl->mutex);
    error.clear();
    const auto path = impl->native / (digest(render_key(descriptor)) + ".metalarc");
    auto archive = impl->load_archive(path);
    MTLRenderPipelineDescriptor *desc = [descriptor copy];
    NSError *native_error = nil;
    if (archive) {
        desc.binaryArchives = @[ archive ];
        auto cached = [impl->device.native_device() newRenderPipelineStateWithDescriptor:desc options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&native_error];
        if (cached) {
            ++impl->counters.pipeline_hits;
            return cached;
        }
    }
    ++impl->counters.pipeline_misses;
    // A mismatching indexed file must not retain an unrelated pipeline: even
    // collisions are safe, because Metal performs the authoritative match above.
    auto empty = [MTLBinaryArchiveDescriptor new];
    archive = [impl->device.native_device() newBinaryArchiveWithDescriptor:empty error:&native_error];
    desc.binaryArchives = nil;
    bool added = archive && [archive addRenderPipelineFunctionsWithDescriptor:desc error:&native_error];
    if (added)
        desc.binaryArchives = @[ archive ];
    native_error = nil;
    auto result = [impl->device.native_device() newRenderPipelineStateWithDescriptor:desc error:&native_error];
    if (!result && added) {
        desc.binaryArchives = nil;
        native_error = nil;
        result = [impl->device.native_device() newRenderPipelineStateWithDescriptor:desc error:&native_error];
    }
    if (!result)
        error = error_text(native_error);
    else if (added)
        impl->save(archive, path);
    return result;
}
id<MTLComputePipelineState> PersistentCache::create_compute_pipeline(id<MTLFunction> function, std::string &error) {
    std::lock_guard lock(impl->mutex);
    error.clear();
    const auto path = impl->native / (digest(std::string("compute\n") + (function.name.UTF8String ?: "")) + ".metalarc");
    auto archive = impl->load_archive(path);
    auto desc = [MTLComputePipelineDescriptor new];
    desc.computeFunction = function;
    NSError *native_error = nil;
    if (archive) {
        desc.binaryArchives = @[ archive ];
        auto cached = [impl->device.native_device() newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&native_error];
        if (cached) {
            ++impl->counters.pipeline_hits;
            return cached;
        }
    }
    ++impl->counters.pipeline_misses;
    auto empty = [MTLBinaryArchiveDescriptor new];
    archive = [impl->device.native_device() newBinaryArchiveWithDescriptor:empty error:&native_error];
    desc.binaryArchives = nil;
    bool added = archive && [archive addComputePipelineFunctionsWithDescriptor:desc error:&native_error];
    if (added)
        desc.binaryArchives = @[ archive ];
    native_error = nil;
    auto result = [impl->device.native_device() newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&native_error];
    if (!result && added) {
        desc.binaryArchives = nil;
        native_error = nil;
        result = [impl->device.native_device() newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&native_error];
    }
    if (!result)
        error = error_text(native_error);
    else if (added)
        impl->save(archive, path);
    return result;
}
void PersistentCache::flush() {} // Each completed pipeline is published immediately.
CacheStats PersistentCache::stats() const {
    std::lock_guard lock(impl->mutex);
    return impl->counters;
}
std::filesystem::path PersistentCache::directory() const { return impl->root; }
} // namespace renderer::metal
