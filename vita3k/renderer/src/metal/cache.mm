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
constexpr std::string_view variant_magic = "V3KVAR02";
constexpr std::string_view pipeline_magic = "V3KPIPE1";
constexpr size_t max_variant_key_bytes = 4096;
constexpr size_t max_pipeline_template_bytes = 16 * 1024;
std::string digest(std::string_view s) { return hex_string(sha256(s.data(), s.size())); }
bool valid_guest_hash(std::string_view hash) {
    return hash.size() == 64 && std::all_of(hash.begin(), hash.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
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
uint64_t get64(std::string_view s, size_t at) {
    return uint64_t(get32(s, at)) | (uint64_t(get32(s, at + 4)) << 32);
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
static std::optional<CachedPipeline> restore_render_template(std::string_view bytes) {
    constexpr size_t field_bytes=(9+8*9+31*8)*8;
    if (bytes.size()<7+3+field_bytes || bytes.substr(0,7)!="render\n") return std::nullopt;
    const auto names=bytes.substr(7,bytes.size()-7-field_bytes);
    const auto separator=names.find('\n');
    if (separator==std::string_view::npos || !separator || separator+1==names.size()
        || names.find('\n',separator+1)!=std::string_view::npos
        || names.size()>512) return std::nullopt;
    CachedPipeline result;
    result.vertex_function=names.substr(0,separator);
    result.fragment_function=names.substr(separator+1);
    auto desc=[MTLRenderPipelineDescriptor new];
    size_t at=bytes.size()-field_bytes;
    const auto next=[&]() { const uint64_t value=get64(bytes,at);at+=8;return value; };
    const uint64_t samples=next();
    if (samples!=1 && samples!=2 && samples!=4) return std::nullopt;
    desc.rasterSampleCount=samples;
    desc.alphaToCoverageEnabled=next();
    desc.alphaToOneEnabled=next();
    desc.rasterizationEnabled=next();
    desc.depthAttachmentPixelFormat=MTLPixelFormat(next());
    desc.stencilAttachmentPixelFormat=MTLPixelFormat(next());
    desc.inputPrimitiveTopology=MTLPrimitiveTopologyClass(next());
    desc.supportIndirectCommandBuffers=next();
    desc.maxVertexAmplificationCount=next();
    for (unsigned i=0;i<8;++i) {
        auto a=desc.colorAttachments[i];
        a.pixelFormat=MTLPixelFormat(next());
        a.writeMask=MTLColorWriteMask(next());
        a.blendingEnabled=next();
        a.rgbBlendOperation=MTLBlendOperation(next());
        a.alphaBlendOperation=MTLBlendOperation(next());
        a.sourceRGBBlendFactor=MTLBlendFactor(next());
        a.destinationRGBBlendFactor=MTLBlendFactor(next());
        a.sourceAlphaBlendFactor=MTLBlendFactor(next());
        a.destinationAlphaBlendFactor=MTLBlendFactor(next());
    }
    auto layout=[MTLVertexDescriptor vertexDescriptor];
    for (unsigned i=0;i<31;++i) {
        auto a=layout.attributes[i];auto stream=layout.layouts[i];
        const uint64_t format=next(),offset=next(),buffer=next(),stride=next(),step=next(),rate=next();
        const uint64_t vertex_mutability=next(),fragment_mutability=next();
        if (format>255 || offset>65536 || buffer>=31 || stride>65536 || step>8 || rate>65536
            || vertex_mutability>2 || fragment_mutability>2) return std::nullopt;
        a.format=MTLVertexFormat(format);a.offset=offset;a.bufferIndex=buffer;
        stream.stride=stride;stream.stepFunction=MTLVertexStepFunction(step);stream.stepRate=rate;
        desc.vertexBuffers[i].mutability=MTLMutability(vertex_mutability);
        desc.fragmentBuffers[i].mutability=MTLMutability(fragment_mutability);
    }
    if (at!=bytes.size()) return std::nullopt;
    desc.vertexDescriptor=layout;
    const std::string reconstructed=render_key(desc);
    if (reconstructed.substr(8)!=bytes.substr(bytes.size()-field_bytes)) {
        const auto original=bytes.substr(bytes.size()-field_bytes);
        const auto actual=std::string_view(reconstructed).substr(8);
        size_t mismatch=0;
        while (mismatch<original.size() && original[mismatch]==actual[mismatch]) ++mismatch;
        LOG_WARN("Metal pipeline warmup descriptor differs at field byte {}",mismatch);
        return std::nullopt;
    }
    result.descriptor=desc;
    return result;
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
    if (flags & ~63u || (flags & 33u) == 33u || cube & ~65535u || !entry || entry > 256 || !source || source > max_program_bytes
        || uint64_t(entry) + source + 16 != body.size())
        return miss(true);
    shader::metal::Program program;
    program.stage = flags & 32 ? shader::metal::Stage::Compute
        : flags & 1 ? shader::metal::Stage::Fragment : shader::metal::Stage::Vertex;
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
void PersistentCache::store_program(std::string_view key, const shader::metal::Program &program,
    std::string_view guest_hash, bool gamma_correction) {
    std::lock_guard lock(impl->mutex);
    if (program.source.empty() || program.source.size() + program.entry_point.size() + 16 > max_program_bytes
        || program.entry_point.empty() || program.entry_point.size() > 256)
        return;
    std::string body;
    put32(body, uint32_t(program.stage == shader::metal::Stage::Fragment) | (uint32_t(program.uses_framebuffer_fetch) << 1) | (uint32_t(program.uses_raster_order_groups) << 2) | (uint32_t(program.uses_buffer_addresses) << 3) | (uint32_t(program.writes_guest_memory) << 4)
        | (uint32_t(program.stage == shader::metal::Stage::Compute) << 5));
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
    if (atomic_write(impl->programs / (hash + ".mslcache"), file)) {
        ++impl->counters.program_writes;
        if (guest_hash.empty()) return;
        if (!valid_guest_hash(guest_hash) || key.size() > max_variant_key_bytes
            || key.substr(0, guest_hash.size()) != guest_hash) return;
        const auto directory = impl->programs / "variants" / std::string(guest_hash);
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        if (ec) {
            impl->io_error("Cannot create shader variant index");
            return;
        }
        std::string body;
        put32(body, uint32_t(key.size()));
        body.push_back(char(gamma_correction));
        body += key;
        std::string variant(variant_magic);
        variant += hash;
        variant += digest(body);
        variant += body;
        if (!atomic_write(directory / (hash + ".variant"), variant))
            impl->io_error("Cannot save shader variant index");
    } else
        impl->io_error("Cannot save translated shader");
}
std::vector<CachedVariant> PersistentCache::variants(std::string_view guest_hash) {
    std::lock_guard lock(impl->mutex);
    std::vector<CachedVariant> result;
    if (!valid_guest_hash(guest_hash)) return result;
    std::error_code ec;
    const auto directory = impl->programs / "variants" / std::string(guest_hash);
    std::filesystem::directory_iterator it(directory, ec), end;
    if (ec) return result;
    for (; it != end; it.increment(ec)) {
        if (ec) break;
        const auto path = it->path();
        if (path.extension() != ".variant") continue;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec || size < 141 || size > 141 + max_variant_key_bytes) {
            ++impl->counters.rejected_files;
            ec.clear();
            continue;
        }
        std::string bytes(size, '\0');
        std::ifstream file(path, std::ios::binary);
        if (!file.read(bytes.data(), bytes.size())) {
            ++impl->counters.rejected_files;
            continue;
        }
        const std::string_view view(bytes), key = view.substr(141);
        if (view.substr(0, 8) != variant_magic || uint8_t(view[140]) > 1
            || get32(view, 136) != key.size() || key.substr(0, 64) != guest_hash
            || view.substr(8, 64) != digest(key)
            || view.substr(72, 64) != digest(view.substr(136))
            || path.filename() != std::string(view.substr(8, 64)) + ".variant") {
            ++impl->counters.rejected_files;
            continue;
        }
        result.push_back({std::string(key), view[140] != 0});
    }
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.key < b.key; });
    return result;
}
void PersistentCache::store_render_pipeline_template(std::string_view fragment_hash, std::string_view vertex_hash,
    std::string_view key, std::string_view vertex_key, std::string_view fragment_key,
    MTLRenderPipelineDescriptor *descriptor) {
    std::lock_guard lock(impl->mutex);
    if (!valid_guest_hash(fragment_hash) || !valid_guest_hash(vertex_hash) || !descriptor
        || key.empty() || key.size()>max_variant_key_bytes
        || vertex_key.empty() || vertex_key.size()>max_variant_key_bytes
        || fragment_key.empty() || fragment_key.size()>max_variant_key_bytes
        || vertex_key.substr(0,64)!=vertex_hash
        || (fragment_key.substr(0,64)!=fragment_hash && !fragment_key.starts_with("metal-depth-only"))
        || !key.starts_with(vertex_key) || !key.substr(vertex_key.size()).starts_with(fragment_key)) return;
    const std::string descriptor_key=render_key(descriptor);
    const size_t payload=key.size()+vertex_key.size()+fragment_key.size()+descriptor_key.size();
    if (payload+16>max_pipeline_template_bytes) return;
    std::string body;
    for (size_t length:{key.size(),vertex_key.size(),fragment_key.size(),descriptor_key.size()})
        put32(body,uint32_t(length));
    body.append(key);body.append(vertex_key);body.append(fragment_key);body+=descriptor_key;
    std::string file(pipeline_magic);
    file+=digest(body);file+=body;
    const auto directory=impl->native/"warmup"/std::string(fragment_hash)/std::string(vertex_hash);
    std::error_code ec;
    std::filesystem::create_directories(directory,ec);
    if (ec || !atomic_write(directory/(digest(key)+".pipeline"),file))
        impl->io_error("Cannot save Metal pipeline warmup descriptor");
}
std::vector<CachedPipeline> PersistentCache::render_pipeline_templates(std::string_view fragment_hash,
    std::string_view vertex_hash) {
    std::lock_guard lock(impl->mutex);
    std::vector<CachedPipeline> result;
    if (!valid_guest_hash(fragment_hash) || !valid_guest_hash(vertex_hash)) return result;
    const auto directory=impl->native/"warmup"/std::string(fragment_hash)/std::string(vertex_hash);
    std::error_code ec;
    std::filesystem::directory_iterator it(directory,ec),end;
    if (ec) return result;
    for (size_t count=0;it!=end && count<4096;it.increment(ec),++count) {
        if (ec) break;
        const auto path=it->path();
        if (path.extension()!=".pipeline") continue;
        const auto size=std::filesystem::file_size(path,ec);
        if (ec || size<72+16 || size>72+max_pipeline_template_bytes) {
            ++impl->counters.rejected_files;ec.clear();continue;
        }
        std::string bytes(size,'\0');
        std::ifstream input(path,std::ios::binary);
        if (!input.read(bytes.data(),bytes.size())) {++impl->counters.rejected_files;continue;}
        const std::string_view view(bytes),body=view.substr(72);
        if (view.substr(0,8)!=pipeline_magic || view.substr(8,64)!=digest(body)) {
            ++impl->counters.rejected_files;continue;
        }
        const size_t key_size=get32(body,0),vertex_size=get32(body,4),fragment_size=get32(body,8),descriptor_size=get32(body,12);
        if (!key_size || !vertex_size || !fragment_size || !descriptor_size
            || key_size>max_variant_key_bytes || vertex_size>max_variant_key_bytes
            || fragment_size>max_variant_key_bytes || size_t(16)+key_size+vertex_size+fragment_size+descriptor_size!=body.size()) {
            ++impl->counters.rejected_files;continue;
        }
        const auto key=body.substr(16,key_size);
        const auto vertex=body.substr(16+key_size,vertex_size);
        const auto fragment=body.substr(16+key_size+vertex_size,fragment_size);
        const auto descriptor=body.substr(16+key_size+vertex_size+fragment_size,descriptor_size);
        if (path.filename()!=digest(key)+".pipeline" || vertex.substr(0,64)!=vertex_hash
            || (fragment.substr(0,64)!=fragment_hash && !fragment.starts_with("metal-depth-only"))
            || !key.starts_with(vertex) || !key.substr(vertex.size()).starts_with(fragment)) {
            ++impl->counters.rejected_files;continue;
        }
        auto restored=restore_render_template(descriptor);
        if (!restored) {++impl->counters.rejected_files;continue;}
        restored->key=key;
        restored->vertex_key=vertex;
        restored->fragment_key=fragment;
        result.push_back(std::move(*restored));
    }
    std::sort(result.begin(),result.end(),[](const auto &a,const auto &b){return a.key<b.key;});
    return result;
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
