// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <renderer/metal/device.h>
#include <util/hash.h>
#include <util/log.h>

namespace renderer::metal {

static std::string describe_error(NSError *error, const char *fallback) {
    return error.localizedDescription.UTF8String ?: fallback;
}

Device::~Device() = default;
void Device::configure_cache(const std::filesystem::path &root) {
    cache.reset();
    if (root.empty())
        return;
    try {
        cache = std::make_unique<PersistentCache>(*this, root);
        LOG_INFO("Metal persistent cache: {}", cache->directory().string());
    } catch (const std::exception &e) {
        LOG_WARN("Metal cache unavailable: {}", e.what());
    }
}
std::optional<shader::metal::Program> Device::load_cached_program(std::string_view key) const {
    return cache ? cache->load_program(key) : std::nullopt;
}
void Device::store_cached_program(std::string_view key, const shader::metal::Program &program,
    std::string_view guest_hash, bool gamma_correction) const {
    if (cache)
        cache->store_program(key, program, guest_hash, gamma_correction);
}
std::vector<CachedVariant> Device::cached_variants(std::string_view guest_hash) const {
    return cache ? cache->variants(guest_hash) : std::vector<CachedVariant>{};
}
void Device::store_cached_pipeline_template(std::string_view fragment_hash, std::string_view vertex_hash,
    std::string_view key, std::string_view vertex_key, std::string_view fragment_key,
    MTLRenderPipelineDescriptor *descriptor) const {
    if (cache) cache->store_render_pipeline_template(fragment_hash,vertex_hash,key,vertex_key,fragment_key,descriptor);
}
std::vector<CachedPipeline> Device::cached_pipeline_templates(std::string_view fragment_hash, std::string_view vertex_hash) const {
    return cache ? cache->render_pipeline_templates(fragment_hash,vertex_hash) : std::vector<CachedPipeline>{};
}
void Device::flush_cache() const {
    if (cache)
        cache->flush();
}
CacheStats Device::cache_stats() const { return cache ? cache->stats() : CacheStats{}; }
std::filesystem::path Device::cache_directory() const { return cache ? cache->directory() : std::filesystem::path{}; }
id<MTLComputePipelineState> Device::create_compute_pipeline(id<MTLFunction> function, std::string &error) const {
    if (cache)
        return cache->create_compute_pipeline(function, error);
    error.clear();
    NSError *native_error = nil;
    auto result = [device newComputePipelineStateWithFunction:function error:&native_error];
    if (!result)
        error = describe_error(native_error, "Could not create Metal compute pipeline");
    return result;
}

std::unique_ptr<Device> Device::create(std::string &error) {
    error.clear();
    auto result = std::unique_ptr<Device>(new Device);
    result->device = MTLCreateSystemDefaultDevice();
    if (!result->device || ![result->device supportsFamily:MTLGPUFamilyMetal3]) {
        error = "Vita3K requires a Metal 3 capable GPU for the native Metal renderer";
        return nullptr;
    }
    result->queue = [result->device newCommandQueue];
    if (!result->queue) {
        error = "Could not create the Metal command queue";
        return nullptr;
    }
    result->queue.label = @"Vita3K Metal graphics queue";
    return result;
}

std::unique_ptr<CompiledProgram> Device::compile(const shader::metal::Program &program,
    bool gamma_correction, std::string &error) const {
    error.clear();
    if (program.uses_raster_order_groups && !supports_raster_order_groups()) {
        error = "Metal shader requires raster order groups, which this GPU does not support";
        return nullptr;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_0;
    // Guest code packs values into floating-point registers and can observe NaNs.
    // Fast math would permit transformations that discard those semantics.
    options.fastMathEnabled = NO;
    // Stable symbols distinguish source/stage/gamma variants in the persistent
    // pipeline index. Metal additionally verifies the complete descriptor.
    const std::string identity = program.entry_point + "\n" + std::to_string(unsigned(program.stage)) + "\n" + std::to_string(gamma_correction) + "\n" + program.source;
    const std::string entry = "vita3k_" + hex_string(sha256(identity.data(), identity.size()));
    NSString *name = [NSString stringWithUTF8String:entry.c_str()];
    options.preprocessorMacros = @{ [NSString stringWithUTF8String:program.entry_point.c_str()] : name };
    NSError *native_error = nil;
    auto result = std::make_unique<CompiledProgram>();
    result->library = [device newLibraryWithSource:[NSString stringWithUTF8String:program.source.c_str()]
                                           options:options
                                             error:&native_error];
    if (!result->library) {
        error = describe_error(native_error, "Could not compile Metal shader");
        return nullptr;
    }
    result->function = [result->library newFunctionWithName:name];
    if (result->function) {
        for (MTLFunctionConstant *constant in result->function.functionConstantsDictionary.allValues) {
            if (constant.index != 0)
                continue;
            if (constant.type != MTLDataTypeBool) {
                error = "Unexpected type for Metal gamma-correction function constant";
                return nullptr;
            }
            MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&gamma_correction type:MTLDataTypeBool atIndex:0];
            result->function = [result->library newFunctionWithName:name constantValues:constants error:&native_error];
            break;
        }
    }
    if (!result->function) {
        error = describe_error(native_error, "Could not create Metal shader entry point");
        return nullptr;
    }
    const auto expected_type = program.stage == shader::metal::Stage::Vertex ? MTLFunctionTypeVertex
        : program.stage == shader::metal::Stage::Fragment ? MTLFunctionTypeFragment : MTLFunctionTypeKernel;
    if (result->function.functionType != expected_type) {
        error = "Metal shader entry point has the wrong stage";
        return nullptr;
    }
    result->stage = program.stage;
    result->uses_framebuffer_fetch = program.uses_framebuffer_fetch;
    result->writes_guest_memory = program.writes_guest_memory;
    result->cube_texture_mask = program.cube_texture_mask;
    return result;
}

id<MTLRenderPipelineState> Device::create_pipeline(MTLRenderPipelineDescriptor *descriptor,
    std::string &error) const {
    for (NSUInteger i = 0; i < 8; ++i) {
        auto *color = descriptor.colorAttachments[i];
        if (color.pixelFormat != MTLPixelFormatRGB9E5Float)
            continue;
        // RGB9E5 has no stored alpha. Metal nevertheless requires All for
        // a complete RGB write, including when the guest masks alpha out.
        const auto rgb = color.writeMask & (MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue);
        if (rgb == (MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue))
            color.writeMask = MTLColorWriteMaskAll;
        else if (rgb == MTLColorWriteMaskNone)
            color.writeMask = MTLColorWriteMaskNone;
    }
    if (cache)
        return cache->create_render_pipeline(descriptor, error);
    error.clear();
    NSError *native_error = nil;
    id<MTLRenderPipelineState> result = [device newRenderPipelineStateWithDescriptor:descriptor error:&native_error];
    if (!result)
        error = describe_error(native_error, "Could not create Metal render pipeline");
    return result;
}

bool Device::submit_and_wait(id<MTLCommandBuffer> commands, std::string &error) const {
    error.clear();
    if (!commands || commands.status != MTLCommandBufferStatusNotEnqueued) {
        error = "Metal command buffer must be valid and unsubmitted";
        return false;
    }
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted) {
        error = describe_error(commands.error, "Metal command buffer did not complete successfully");
        return false;
    }
    return true;
}

} // namespace renderer::metal
