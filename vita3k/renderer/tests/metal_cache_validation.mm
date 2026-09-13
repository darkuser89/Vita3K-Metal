// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
// SPDX-License-Identifier: GPL-2.0-or-later
#include <array>
#include <iostream>
#include <renderer/metal/device.h>
#include <stdexcept>
using renderer::metal::Device;
using shader::metal::Program;
static void check(bool ok, const std::string &error) {
    if (!ok)
        throw std::runtime_error(error);
}
static bool equal(const Program &a, const Program &b) {
    return a.source == b.source && a.entry_point == b.entry_point && a.stage == b.stage
        && a.uses_framebuffer_fetch == b.uses_framebuffer_fetch && a.uses_raster_order_groups == b.uses_raster_order_groups
        && a.uses_buffer_addresses == b.uses_buffer_addresses && a.writes_guest_memory == b.writes_guest_memory
        && a.cube_texture_mask == b.cube_texture_mask;
}
static Program persisted(Device &device, std::string key, const Program &expected) {
    if (auto cached = device.load_cached_program(key)) {
        check(equal(*cached, expected), "Cached program lost source or semantic metadata");
        return *cached;
    }
    device.store_cached_program(key, expected);
    return expected;
}
int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: metal-cache-validation <cache directory> <cold|warm|fallback|disabled|readonly>\n";
        return 2;
    }
    @autoreleasepool {
        try {
            std::string error, mode = argv[2];
            auto device = Device::create(error);
            check(bool(device), error);
            if (mode != "disabled")
                device->configure_cache(argv[1]);
            // Exercise all metadata independently, including both guest-memory
            // write values: losing this bit would break GPU/CPU synchronization.
            for (unsigned bits = 0; bits < 32; ++bits) {
                Program p{ .source = "metadata fixture", .entry_point = "entry", .stage = bits & 1 ? shader::metal::Stage::Fragment : shader::metal::Stage::Vertex, .uses_framebuffer_fetch = bool(bits & 2), .uses_raster_order_groups = bool(bits & 4), .uses_buffer_addresses = bool(bits & 8), .writes_guest_memory = bool(bits & 16), .cube_texture_mask = 0xa55a };
                persisted(*device, "metadata-" + std::to_string(bits), p);
            }
            auto vertex = persisted(*device, "fullscreen", Program{ .source = R"(#include <metal_stdlib>
using namespace metal;
vertex float4 fullscreen(uint i [[vertex_id]]) { const float2 p[]={float2(-1,-1),float2(3,-1),float2(-1,3)}; return float4(p[i],0,1); })",
                                                               .entry_point = "fullscreen",
                                                               .stage = shader::metal::Stage::Vertex,
                                                               .writes_guest_memory = false });
            auto fragment = persisted(*device, "color", Program{ .source = R"(#include <metal_stdlib>
using namespace metal;
constant bool gamma [[function_constant(0)]];
fragment float4 color(float4 p [[position]]) { return float4(gamma ? 1.0 : 0.0, uint(p.x)%2, uint(p.y)%2, 1); })",
                                                            .entry_point = "color",
                                                            .stage = shader::metal::Stage::Fragment,
                                                            .writes_guest_memory = false });
            auto vs = device->compile(vertex, false, error);
            check(bool(vs), error);
            unsigned checked = 0;
            for (bool gamma : { false, true })
                for (auto format : { MTLPixelFormatRGBA8Unorm, MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB }) {
                    auto fs = device->compile(fragment, gamma, error);
                    check(bool(fs), error);
                    auto desc = [MTLRenderPipelineDescriptor new];
                    desc.vertexFunction = vs->function;
                    desc.fragmentFunction = fs->function;
                    desc.colorAttachments[0].pixelFormat = format;
                    auto pipeline = device->create_pipeline(desc, error);
                    check(pipeline != nil, error);
                    auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:16 height:16 mipmapped:NO];
                    td.storageMode = MTLStorageModeShared;
                    td.usage = MTLTextureUsageRenderTarget;
                    auto texture = [device->native_device() newTextureWithDescriptor:td];
                    check(texture != nil, "No texture");
                    auto pass = [MTLRenderPassDescriptor new];
                    pass.colorAttachments[0].texture = texture;
                    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                    auto commands = [device->command_queue() commandBuffer];
                    auto encoder = [commands renderCommandEncoderWithDescriptor:pass];
                    [encoder setRenderPipelineState:pipeline];
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                    [encoder endEncoding];
                    check(device->submit_and_wait(commands, error), error);
                    std::array<uint8_t, 16 * 16 * 4> bytes;
                    [texture getBytes:bytes.data() bytesPerRow:64 fromRegion:MTLRegionMake2D(0, 0, 16, 16) mipmapLevel:0];
                    for (unsigned y = 0; y < 16; ++y)
                        for (unsigned x = 0; x < 16; ++x) {
                            std::array<unsigned, 4> expected = { gamma ? 255u : 0u, (x % 2) * 255, (y % 2) * 255, 255 };
                            if (format == MTLPixelFormatBGRA8Unorm)
                                std::swap(expected[0], expected[2]);
                            for (unsigned c = 0; c < 4; ++c) {
                                check(bytes[(y * 16 + x) * 4 + c] == expected[c], "Cached pipeline changed output pixels");
                                ++checked;
                            }
                        }
                }
            NSError *native_error = nil;
            auto lib = [device->native_device() newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal; kernel void fill(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) { out[i]=(i*7919u)^0x12345678u; }" options:nil error:&native_error];
            check(lib != nil, "Compute library failed");
            auto pipeline = device->create_compute_pipeline([lib newFunctionWithName:@"fill"], error);
            check(pipeline != nil, error);
            auto buffer = [device->native_device() newBufferWithLength:256 * 4 options:MTLResourceStorageModeShared];
            auto commands = [device->command_queue() commandBuffer];
            auto encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer offset:0 atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
            check(device->submit_and_wait(commands, error), error);
            for (unsigned i = 0; i < 256; ++i)
                check(static_cast<uint32_t *>(buffer.contents)[i] == ((i * 7919u) ^ 0x12345678u), "Cached compute pipeline changed output");
            device->flush_cache();
            const auto stats = device->cache_stats();
            std::cout << "CACHE_RESULT {\"program_hits\":" << stats.program_hits << ",\"program_misses\":" << stats.program_misses
                      << ",\"pipeline_hits\":" << stats.pipeline_hits << ",\"pipeline_misses\":" << stats.pipeline_misses
                      << ",\"archive_writes\":" << stats.archive_writes << ",\"rejected_files\":" << stats.rejected_files
                      << ",\"io_errors\":" << stats.io_errors << ",\"archive_loaded\":" << stats.archive_loaded
                      << ",\"archive_writable\":" << stats.archive_writable << ",\"checked_bytes\":" << checked + 1024 << "}\n";
            if (mode == "cold")
                check(stats.program_misses == 34 && stats.pipeline_misses == 7 && stats.archive_writes > 0 && stats.io_errors == 0, "Cold cache did not populate");
            if (mode == "warm")
                check(stats.program_hits == 34 && stats.program_misses == 0 && stats.pipeline_hits == 7 && stats.pipeline_misses == 0 && stats.archive_loaded, "Separate process did not reuse native archive and shader metadata");
            if (mode == "disabled")
                check(device->cache_directory().empty() && stats.archive_writes == 0, "Disabled cache was accessed");
            if (mode == "readonly")
                check(!stats.archive_writable && stats.archive_writes == 0, "Concurrent reader acquired writer lock");
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "FAIL " << e.what() << '\n';
            return 1;
        }
    }
}
